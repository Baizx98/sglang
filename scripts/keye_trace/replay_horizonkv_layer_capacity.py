"""M4 offline cache curves from continuous, returned layer-capture steps."""

import argparse
import csv
import ctypes
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from horizonkv_belady import compile_belady, run
from horizonkv_capacity_replay import compile_replay
from horizonkv_single_runtime import notify, save, update


def ordered_pack(slots, writes):
    # The public within-batch tie rule is request order, then the stored Top-K
    # order. All hits are counted before any admission/eviction.
    unique = []
    for values in slots:
        values = values.reshape(-1)
        _, first = np.unique(values, return_index=True)
        unique.append(values[np.sort(first)])
        assert unique[-1].size == values.size
    offsets = np.array([0] + list(np.cumsum([v.size for v in slots])), dtype=np.int64)
    uoffsets = np.array([0] + list(np.cumsum([v.size for v in unique])), dtype=np.int64)
    woffsets = np.array([0] + list(np.cumsum([v.size for v in writes])), dtype=np.int64)
    flat = np.concatenate(slots).reshape(-1).astype(np.int32)
    written = np.concatenate(writes).astype(np.int32)
    return max(int(flat.max()), int(written.max(initial=0))) + 1, [offsets, flat, uoffsets, np.concatenate(unique).astype(np.int32),
                                                        woffsets, written]


def load_series(root):
    responses = {}
    for path in sorted(Path(root).rglob("input-wave*.json")):
        for row in json.loads(path.read_text()):
            responses[row["rid"]] = row
    series = defaultdict(dict)
    records = [json.loads(line) for line in (Path(root) / "fixtures/manifest_rank00.jsonl").read_text().splitlines()]
    for meta in records:
        if not any(r in responses for r in meta["request_ids"]):
            continue
        value = torch.load(Path(root) / "fixtures" / meta["file"], weights_only=False)
        for i, (rid, step) in enumerate(zip(value["request_ids"], value["steps"])):
            if rid not in responses:
                continue
            request = responses[rid]
            returned = len(request["response"]["output_ids"])
            if step >= returned - 1:
                continue
            expected_length = request["prompt_tokens"] + step + 1
            assert int(value["seq_lens"][i]) == expected_length
            ids = value["topk"][i].numpy().copy()
            assert ids.size == 2048 and ids.min() >= 0 and ids.max() < expected_length
            assert np.unique(ids).size == 2048
            key = (rid, value["layer"])
            assert step not in series[key]
            series[key][step] = ids
    for (rid, _), steps in series.items():
        expected = len(responses[rid]["response"]["output_ids"]) - 1
        assert sorted(steps) == list(range(expected)), (rid, len(steps), expected)
    return responses, series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    out = args.output_dir.resolve() if args.output_dir else root / "m4-capacity-v01"
    out.mkdir(parents=True, exist_ok=True)
    update(root, phase="m4_offline_cache_replay", detail="离线回放连续有效步；不测GPU时延。")
    responses, series = load_series(root)
    ordinary = compile_replay(out)
    lru = ctypes.CDLL(str(out / "capacity_lru.so")).capacity_replay_refined
    lru.argtypes, lru.restype = ordinary.argtypes, ordinary.restype
    belady = compile_belady(out)
    by_wave = defaultdict(list)
    for rid in responses:
        by_wave[rid.split("__")[1]].append(rid)
    groups = [(f"c1-{rid}", [rid]) for rid in sorted(responses)]
    groups.extend((f"c4-{w}", sorted(rs)) for w, rs in sorted(by_wave.items()) if len(rs) == 4)
    by_task = defaultdict(list)
    for w, rs in sorted(by_wave.items()):
        if len(rs) == 4:
            by_task[responses[rs[0]]["task"]].append((w, sorted(rs)))
    for task, waves in by_task.items():
        for start in range(0, len(waves) - 1, 2):
            a, b = waves[start:start+2]
            groups.append((f"c8-{a[0]}-{b[0]}", a[1] + b[1]))
    rows = []
    for group_id, rids in groups:
        for layer in (11, 12, 23, 24):
            n = min(len(series[r, layer]) for r in rids)
            if not n:
                continue
            stride = max(responses[r]["prompt_tokens"] + n + 1 for r in rids)
            packs = []
            for rid in rids:
                slots = [series[rid, layer][t].reshape(1, -1) for t in range(n)]
                writes = [np.array([responses[rid]["prompt_tokens"] + t], dtype=np.int32) for t in range(n)]
                packs.append(ordered_pack(slots, writes))
            shared_slots = [np.stack([series[rid, layer][t] + i * stride for i, rid in enumerate(rids)]) for t in range(n)]
            shared_writes = [np.array([responses[rid]["prompt_tokens"] + t + i * stride for i, rid in enumerate(rids)], dtype=np.int32) for t in range(n)]
            shared = ordered_pack(shared_slots, shared_writes)
            for allocation, multiples in (("per_request", (1, 2, 4, 8)), ("shared_total", (8, 16, 32))):
                for multiple in multiples:
                    capacity = multiple * 2048
                    for policy, fn in (("lru_hits_above_misses", lru), ("belady_bypass", belady)):
                        if allocation == "per_request":
                            flags = np.stack([run(fn, capacity, packed).reshape(n, 2048) for packed in packs], axis=1)
                            total = len(rids) * capacity
                        else:
                            flags = run(fn, capacity, shared).reshape(n, len(rids), 2048)
                            total = capacity
                        for view, start in (("all", 0), ("after32", 32)):
                            selected = max(n - start, 0) * len(rids) * 2048
                            hit = int(flags[start:].sum())
                            rows.append({"group": group_id, "task": responses[rids[0]]["task"],
                                         "layer": layer, "concurrency": len(rids), "allocation": allocation,
                                         "policy": policy, "view": view, "capacity_tokens": capacity,
                                         "layer_total_tokens": total, "B_over_k": total / 2048,
                                         "B_over_Ck": total / (len(rids) * 2048),
                                         "request_steps": max(n-start, 0) * len(rids),
                                         "selected_tokens": selected, "hit_tokens": hit,
                                         "hit_ratio": hit / selected if selected else None,
                                         "logical_miss_bytes": (selected-hit) * 2048,
                                         "source_batch_note": "C8 is offline concatenation of two distinct C4 traces" if len(rids)==8 else "model-derived trace"})
    with (out / "capacity-by-group-layer.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    audit = {"status": "completed", "requests": len(responses), "request_layer_series": len(series),
             "layers": [11, 12, 23, 24], "rows": len(rows), "trace_gpu": "A6000 rank0; full-model inputs from A6000+L40S",
             "valid_decode_steps": sum(len(v) for v in series.values()),
             "contract": "offline simulation, returned steps, fixed active common prefix; not latency",
             "lru": "same hit-above-miss refinement and stored Top-K order as GPU replay",
             "belady": "future-aware admission with bypass; same batch snapshot and token granularity",
             "future": "clipped to each common prefix; after32 is a view of the same optimized trajectory"}
    save(out / "audit.json", audit)
    update(root, phase="m4_offline_cache_completed", detail=f"{len(rows)}行离线缓存计数；{len(series)}条连续层轨迹通过核对。")
    notify(f"HorizonKV M4：{len(responses)}请求、{len(series)}条连续层轨迹、{len(rows)}行LRU/Belady结果完成（离线模拟，4层）。")


if __name__ == "__main__":
    main()
