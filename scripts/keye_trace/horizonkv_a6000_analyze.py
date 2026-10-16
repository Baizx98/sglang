"""Audit the dual-A6000 queue and derive capacity, overlap and timing tables."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from horizonkv_belady import compile_belady, pack, run
from horizonkv_capacity_replay import access_groups, compile_replay


def export(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    output = root / "processed"
    output.mkdir(exist_ok=True)
    plan = json.loads((root / "plan.json").read_text())
    functions = {"lru": compile_replay(output), "belady": compile_belady(output)}
    rows, source_files, audit_rows = [], [], []
    for directory in sorted(root.glob("trace-p*")) + sorted(root.glob("audit-p*")):
        if not (directory / "completed.json").exists():
            continue
        requests = jsonlines(directory / "requests.jsonl")
        lookup = {r["rid"]: r for r in requests}
        summaries = {}
        previous, prev_wave = {}, None
        for rank, (wave, layer), events in access_groups(directory, requests, allow_padding=True):
            if (rank, wave) != prev_wave:
                previous, prev_wave = {}, (rank, wave)
            info = lookup[events[0][0]["request_ids"][0]]
            concurrency = info["concurrency"]
            # Replay the same finite prefix, retaining initial partial batches
            # as real cache warmup. Full-batch and all-returned views differ.
            prefix, full_flags = [], []
            started_full = False
            for event in events:
                record = event[0]
                full = len(record["request_ids"]) == concurrency
                if not all(record["valid_returned"]) or (started_full and not full):
                    break
                started_full = started_full or full
                prefix.append(event)
                full_flags.append(full)
            if not prefix:
                continue
            slots, writes = [v[2] for v in prefix], [v[3] for v in prefix]
            packed = pack(slots, writes)
            shape_offsets = packed[1][0]
            hints = []
            for record, logical, _, _ in prefix:
                present = []
                for i, (rid, step) in enumerate(zip(record["request_ids"], record["decode_steps"])):
                    prior = previous.get((rid, step))
                    if layer:
                        assert prior is not None and prior[0] == layer - 1
                    present.append((logical[i] >= 0) & np.isin(logical[i], prior[1][prior[1] >= 0]) if prior is not None else np.zeros(2048, dtype=bool))
                hints.append(np.array(present))
            for capacity in plan["capacities"]:
                totals = {}
                for policy, fn in functions.items():
                    flat_hits = run(fn, capacity, packed)
                    counters = defaultdict(lambda: np.zeros(8, dtype=np.int64))
                    for i, (record, logical, physical, _) in enumerate(prefix):
                        hit = flat_hits[shape_offsets[i]:shape_offsets[i+1]].reshape(physical.shape).astype(bool)
                        valid = physical >= 0
                        miss = valid & ~hit
                        actual_capacity = int(info["storage"][3:]) if info["storage"].startswith("lru") else 0
                        if policy == "lru" and record["online_hits"] is not None and capacity == actual_capacity:
                            assert hit.sum(axis=1).tolist() == record["online_hits"]
                            audit_rows.append({"source": directory.name, "rank": rank, "wave": wave, "layer": layer,
                                               "decode_step": min(record["decode_steps"]), "status": "exact_online_hits_match"})
                        values = np.array([hit.sum(), valid.sum(), hit.shape[0],
                            hints[i].sum() if layer else 0, valid.sum() if layer else 0,
                            (miss & hints[i]).sum() if layer else 0, miss.sum() if layer else 0,
                            np.count_nonzero(miss.sum(axis=1) == 0) if layer else 0], dtype=np.int64)
                        counters["all_returned"] += values
                        if full_flags[i]:
                            counters["all_valid"] += values
                        if full_flags[i] and min(record["decode_steps"]) >= 32:
                            counters["after32"] += values
                    for view, counts in counters.items():
                        values = [int(v) for v in counts]
                        row = {"source": directory.name, "dataset": info["cell"][0], "task": info["cell"][1],
                            "length_bucket_tokens": info["cell"][2], "concurrency": concurrency,
                            "rank": rank, "wave": wave, "layer": layer, "capacity_tokens": capacity,
                            "capacity_over_k": capacity / 2048, "capacity_over_batch_k": capacity / (concurrency*2048),
                            "persistent_main_kv_mib_per_gpu": capacity*48/1024, "policy": policy, "view": view}
                        row.update(initial_partial_events=full_flags.index(True) if True in full_flags else len(full_flags),
                                   full_batch_events=sum(full_flags))
                        row.update(zip(["hit_tokens", "selected_tokens", "request_steps", "overlap_tokens", "overlap_selected_tokens",
                                        "covered_miss_tokens", "miss_tokens_with_prior_layer", "zero_miss_request_steps_with_prior_layer"], values))
                        row.update(hit_ratio=values[0]/values[1], topk_overlap=values[3]/values[4] if values[4] else "",
                            miss_coverage=values[5]/values[6] if values[6] else "")
                        rows.append(row)
                        if view == "all_returned":
                            totals[policy] = values[0]
                assert totals["belady"] >= totals["lru"], (rank, wave, layer, capacity, totals)
            for record, logical, *_ in prefix:
                for i, (rid, step) in enumerate(zip(record["request_ids"], record["decode_steps"])):
                    previous[(rid, step)] = (layer, logical[i].copy())
            if layer == 47:
                print(json.dumps({"replayed": directory.name, "rank": rank, "wave": wave}), flush=True)
        for path in sorted((directory / "trace").glob("*")):
            if path.suffix in (".jsonl", ".bin"):
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                source_files.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size})
    export(output / "capacity-layer-counts.csv", rows)
    export(output / "online-audit.csv", audit_rows)
    profiles, controls = [], []
    captured_windows = []
    for directory in sorted(root.glob("breakdown-*")):
        if not (directory / "requests.jsonl").exists():
            continue
        requests = jsonlines(directory / "requests.jsonl")
        keys = defaultdict(list)
        for request in requests:
            keys["__".join(request["rid"].split("__")[:3])].append(request)
            if request["phase"] == "control":
                arrivals = request["stream_arrivals"]
                first, last = arrivals[0], arrivals[-1]
                n = (last.get("completion_tokens") or 0) - (first.get("completion_tokens") or 0)
                controls.append({"source": directory.name, "case_id": request["case_id"],
                    "task": request["cell"][1], "length_bucket_tokens": request["cell"][2],
                    "concurrency": request["concurrency"], "repetition": request["repetition"],
                    "storage": request["storage"], "output_tokens": len(request["response"]["output_ids"]),
                    "client_decode_interval_s": last["elapsed_s"]-first["elapsed_s"],
                    "client_intertoken_mean_ms": (last["elapsed_s"]-first["elapsed_s"])*1000/n if n else ""})
        for path in sorted((directory / "profiles").glob("*.summary.json")):
            value = json.loads(path.read_text())
            batch = keys[value["wave"]]
            info = batch[0]
            model_ranges = [r for r in value["ranges"] if r["component"] == "model_total"]
            actual_batches = [r["active_batch"] for r in model_ranges]
            captured_windows.append((directory.name, value["rank"], value["captured_steps"]))
            totals = defaultdict(lambda: [0., 0., 0])
            transfers = defaultdict(lambda: [0., 0., 0., 0., 0., 0.])
            by_id = {r["id"]: r["component"] for r in value["ranges"]}
            for event in jsonlines(path.with_name(path.name.replace(".summary.json", ".activities.jsonl"))):
                if event["kind"] == "metadata":
                    continue
                component = by_id.get(event["external_id"], "unattributed")
                duration = (event["end_ns"]-event["start_ns"])/1000
                if event["kind"] == "kernel":
                    transfers[component][0] += duration
                    if "nccl" in event["name"].lower():
                        transfers[component][5] += duration
                elif event["kind"] == "memcpy" and event["copy_kind"] in (1, 2):
                    ix = event["copy_kind"]
                    transfers[component][ix] += duration
                    transfers[component][ix+2] += event["bytes"]
            for row in value["ranges"]:
                assert row["gpu_activity_us"] >= -1, (path, row)
                totals[row["component"]][0] += row["gpu_activity_us"]
                totals[row["component"]][1] += row["cpu_range_exclusive_us"]
                totals[row["component"]][2] += 1
            for component, (gpu, cpu, count) in totals.items():
                profiles.append({"source": directory.name, "task": info["cell"][1], "length_bucket_tokens": info["cell"][2],
                    "concurrency": info["concurrency"], "storage": info["storage"], "repetition": info["repetition"],
                    "actual_active_batch_min": min(actual_batches), "actual_active_batch_max": max(actual_batches),
                    "rank": value["rank"], "component": component, "profiled_decode_steps": value["captured_steps"],
                    "gpu_activity_ms_per_step": gpu/1000/value["captured_steps"],
                    "cpu_exclusive_range_ms_per_step": cpu/1000/value["captured_steps"], "range_count": count,
                    "kernel_ms_per_step": transfers[component][0]/1000/value["captured_steps"],
                    "h2d_copy_ms_per_step": transfers[component][1]/1000/value["captured_steps"],
                    "d2h_copy_ms_per_step": transfers[component][2]/1000/value["captured_steps"],
                    "h2d_bytes_per_step": transfers[component][3]/value["captured_steps"],
                    "d2h_bytes_per_step": transfers[component][4]/value["captured_steps"],
                    "nccl_kernel_ms_per_step": transfers[component][5]/1000/value["captured_steps"],
                    "allocated_mib": value["allocated_bytes"]/1024**2, "reserved_mib": value["reserved_bytes"]/1024**2})
    export(output / "breakdown-activities.csv", profiles)
    export(output / "breakdown-controls.csv", controls)
    if plan.get("smoke_only"):
        assert rows and audit_rows, "Smoke must produce capacity statistics and exact online LRU matches"
        expected_sources = {f"breakdown-p140000-{m}" for m in ("hbm", "dram", "lru8192")}
        assert {(source, rank) for source, rank, steps in captured_windows if steps == 8} == {
            (source, rank) for source in expected_sources for rank in (0,1)
        }, ("Incomplete real-model GPU capture", captured_windows)
        for source in expected_sources:
            for rank in (0, 1):
                components = {r["component"] for r in profiles if r["source"] == source and r["rank"] == rank}
                assert {"indexer_score", "indexer_topk", "attention_compute", "moe_ffn_compute", "qkv_projection"} <= components, (source, rank, components)
    (output / "provenance.json").write_text(json.dumps({"source_commit": plan["source_commit"], "gpu_uuids": plan["gpu_uuids"],
        "raw_files": source_files, "capacity_rows": len(rows), "online_audit_rows": len(audit_rows),
        "profile_rows": len(profiles), "control_rows": len(controls),
        "paper_ready": False, "reason": "Human audit of quality/output pairing, profiler perturbation, completeness and infeasible points still required"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
