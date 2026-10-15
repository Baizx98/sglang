"""Replay both policies on frozen C4 prefixes; --monitor sends ntfy updates.

python horizonkv_belady.py --source data/.../2026-10-15_capacity-c4-v01 \
    --output data/.../2026-10-15_belady-c4-v01
No GPU is used. The source CSV and all four trace hashes must match old audits.
"""

import argparse
import csv
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from horizonkv_capacity_replay import CAPACITIES, access_groups, compile_replay


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compile_belady(directory):
    source = Path(__file__).with_name("horizonkv_capacity_belady.cpp")
    output = Path(directory) / "capacity_belady.so"
    subprocess.run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
    )
    fn = ctypes.CDLL(str(output.resolve())).capacity_belady
    fn.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int64] + [
        ctypes.c_void_p
    ] * 7
    fn.restype = ctypes.c_int
    return fn


def pack(slots, writes):
    offsets = np.array([0] + list(np.cumsum([v.size for v in slots])), dtype=np.int64)
    unique = [np.unique(v[v >= 0]) for v in slots]
    assert all(u.size == np.count_nonzero(s >= 0) for u, s in zip(unique, slots)), (
        "Unit-cost oracle requires unique KV identities in a simultaneous batch"
    )
    sorted_offsets = np.array(
        [0] + list(np.cumsum([v.size for v in unique])), dtype=np.int64
    )
    write_offsets = np.array(
        [0] + list(np.cumsum([v.size for v in writes])), dtype=np.int64
    )
    flat = np.concatenate([v.reshape(-1) for v in slots]).astype(np.int32)
    ordered = np.concatenate(unique).astype(np.int32)
    written = np.concatenate(writes).astype(np.int32)
    universe = max(int(flat.max(initial=0)), int(written.max(initial=0))) + 1
    return universe, [offsets, flat, sorted_offsets, ordered, write_offsets, written]


def run(fn, capacity, packed):
    universe, values = packed
    result = np.empty(values[1].size, dtype=np.uint8)
    args = values + [result]
    error = fn(capacity, universe, len(values[0]) - 1, *(v.ctypes.data for v in args))
    if error:
        raise ValueError(f"Replay rejected input: {error}")
    return result


def export(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def monitor(root, owner):
    next_notice = 0
    while True:
        state = json.loads((root / "state.json").read_text())
        process = Path(f"/proc/{owner}/cmdline")
        alive = (
            process.exists() and "horizonkv_belady.py" in process.read_bytes().decode()
        )
        terminal = state["status"] in ("completed", "failed")
        if not alive and not terminal:
            state = dict(state, status="abnormal_exit")
            terminal = True
        if terminal or time.time() >= next_notice:
            text = (
                f"HorizonKV Belady: {state['status']}; phase={state.get('phase')}; "
                f"groups={state.get('completed_groups', 0)}/480; "
                f"rank={state.get('rank', '-')}, wave={state.get('wave', '-')}, "
                f"layer={state.get('layer', '-')}."
            )
            delivered = False
            # Keep credentials out of argv, subprocess environment and logs.
            for _ in range(3):
                try:
                    config = json.loads(
                        Path("/home10T/bzx/.codex/ntfy.json").read_text()
                    )
                    request = urllib.request.Request(
                        config["url"],
                        data=text.encode(),
                        method="POST",
                        headers={
                            "Authorization": "Bearer " + config["token"],
                            "Content-Type": "text/plain; charset=utf-8",
                            "Title": "HorizonKV Belady " + state["status"],
                        },
                    )
                    with urllib.request.urlopen(request, timeout=20) as response:
                        delivered = response.status == 200
                    if delivered:
                        break
                except (OSError, ValueError, KeyError) as error:
                    failure_type = type(error).__name__
                    time.sleep(5)
            item = {
                "checked_unix_s": time.time(),
                "state": state,
                "ntfy_sent": delivered,
            }
            if not delivered:
                item["notification_error_type"] = locals().get(
                    "failure_type", "HTTPStatus"
                )
            with (root / "monitor.jsonl").open("a") as stream:
                stream.write(json.dumps(item) + "\n")
            save(root / "monitor-status.json", item)
            next_notice = time.time() + 1200
            if terminal:
                return
        time.sleep(15)


def analyze(source, root):
    state = {
        "status": "running",
        "phase": "source_audit",
        "completed_groups": 0,
        "total_groups": 480,
        "pid": os.getpid(),
    }
    save(root / "state.json", state)
    original_audit = json.loads((source / "analysis/audit.json").read_text())
    for key, expected in original_audit.items():
        if key.endswith("_sha256"):
            assert sha(source / "hbm" / key.removesuffix("_sha256")) == expected
    old_layer = {}
    with (source / "analysis/capacity_by_layer.csv").open() as stream:
        for r in csv.DictReader(stream):
            old_layer[
                (
                    int(r["rank"]),
                    r["wave"],
                    int(r["layer"]),
                    int(r["capacity_tokens"]),
                    r["view"],
                )
            ] = r
    old_wave = {}
    with (source / "analysis/capacity_by_wave.csv").open() as stream:
        for r in csv.DictReader(stream):
            old_wave[
                (int(r["rank"]), r["wave"], int(r["capacity_tokens"]), r["view"])
            ] = r
    requests = [json.loads(line) for line in (source / "hbm/requests.jsonl").open()]
    assert len(requests) == 20
    functions = {"lru": compile_replay(root), "belady_bypass": compile_belady(root)}
    totals, layers = defaultdict(lambda: [0, 0, 0]), []
    previous_hits = {}
    state["phase"] = "replay"
    for rank, (wave, layer), raw in access_groups(source / "hbm", requests):
        events = []
        for item in raw:
            r = item[0]
            if len(r["request_ids"]) != 4 or not all(r["valid_returned"]):
                break  # source has no partial-batch warmup before its C4 prefix
            assert r["decode_steps"] == [len(events)] * 4
            events.append(item)
        assert len(events) in (250, 255)
        packed = pack([v[2] for v in events], [v[3] for v in events])
        offsets = packed[1][0]
        counts = np.diff(offsets)
        for capacity in CAPACITIES:
            policy_hits = {}
            for policy, function in functions.items():
                mask = run(function, capacity, packed)
                hits = np.add.reduceat(mask.astype(np.int64), offsets[:-1])
                policy_hits[policy] = int(hits.sum())
                assert np.all(hits <= min(capacity, 8192))
                views = {
                    "all_valid": np.arange(len(events)),
                    "after32": np.arange(32, len(events)),
                }
                views.update(
                    {
                        f"step_bin_{i // 16:02d}": np.arange(
                            i, min(i + 16, len(events))
                        )
                        for i in range(0, len(events), 16)
                    }
                )
                for view, indices in views.items():
                    h, n, steps = (
                        int(hits[indices].sum()),
                        int(counts[indices].sum()),
                        len(indices) * 4,
                    )
                    target = totals[(policy, rank, wave, capacity, view)]
                    for i, value in enumerate((h, n, steps)):
                        target[i] += value
                    if view in ("all_valid", "after32"):
                        if policy == "lru":
                            old = old_layer[(rank, wave, layer, capacity, view)]
                            assert (h, n, steps) == tuple(
                                int(old[k])
                                for k in (
                                    "hit_tokens",
                                    "selected_tokens",
                                    "request_steps",
                                )
                            )
                        layers.append(
                            {
                                "policy": policy,
                                "rank": rank,
                                "wave": wave,
                                "layer": layer,
                                "capacity_tokens": capacity,
                                "persistent_main_kv_mib": capacity * 48 / 1024,
                                "alpha": capacity / 2048,
                                "rho": capacity / 8192,
                                "view": view,
                                "hit_tokens": h,
                                "selected_tokens": n,
                                "miss_tokens": n - h,
                                "request_steps": steps,
                                "hit_ratio": h / n,
                            }
                        )
            assert policy_hits["belady_bypass"] >= policy_hits["lru"]
            key = rank, wave, layer
            assert policy_hits["belady_bypass"] >= previous_hits.get(key, 0)
            previous_hits[key] = policy_hits["belady_bypass"]
        state.update(
            completed_groups=state["completed_groups"] + 1,
            rank=rank,
            wave=wave,
            layer=layer,
            updated_unix_s=time.time(),
        )
        save(root / "state.json", state)
        if layer % 8 == 7:
            print(json.dumps(state), flush=True)
    assert state["completed_groups"] == 480
    result = []
    for (policy, rank, wave, capacity, view), (h, n, layer_steps) in sorted(
        totals.items()
    ):
        steps = layer_steps // 48
        assert n == steps * 48 * 2048
        if policy == "lru":
            old = old_wave[(rank, wave, capacity, view)]
            assert (h, n, steps) == tuple(
                int(old[k]) for k in ("hit_tokens", "selected_tokens", "request_steps")
            )
        result.append(
            {
                "policy": policy,
                "rank": rank,
                "wave": wave,
                "capacity_tokens": capacity,
                "persistent_main_kv_mib": capacity * 48 / 1024,
                "alpha": capacity / 2048,
                "rho": capacity / 8192,
                "view": view,
                "hit_tokens": h,
                "selected_tokens": n,
                "miss_tokens": n - h,
                "request_steps": steps,
                "miss_bytes": (n - h) * 1024,
                "hit_ratio": h / n,
            }
        )
    export(root / "capacity_by_wave.csv", result)
    export(root / "capacity_by_layer.csv", layers)
    audit = {
        "source_run": str(source),
        "source_audit": original_audit,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "groups": 480,
        "requests": 20,
        "capacities": CAPACITIES,
        "lru_exact_old_counts": True,
        "belady_full_window_dominates_lru": True,
        "belady_full_window_capacity_monotone": True,
        "future_truncated_at_valid_c4_end": True,
        "writes_end_old_versions": True,
        "tie_break": "incumbent wins; latest inserted evicted on equal end; sorted physical insertion",
        "timing_claims": False,
        "by_wave_sha256": sha(root / "capacity_by_wave.csv"),
        "by_layer_sha256": sha(root / "capacity_by_layer.csv"),
    }
    save(root / "audit.json", audit)
    save(root / "state.json", dict(state, status="completed", phase="replay_completed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--owner-pid", type=int)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.monitor:
        monitor(root, args.owner_pid)
        return
    assert not (root / "capacity_by_layer.csv").exists(), "Preserve existing results"
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Commit implementation before production replay")
    save(
        root / "state.json",
        {"status": "running", "phase": "starting", "completed_groups": 0},
    )
    with (root / "monitor.log").open("a") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(root),
                "--monitor",
                "--owner-pid",
                str(os.getpid()),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    save(
        root / "launch.json",
        {
            "runner_pid": os.getpid(),
            "monitor_pid": proc.pid,
            "source": str(args.source.resolve()),
            "output": str(root),
        },
    )
    try:
        analyze(args.source.resolve(), root)
    except BaseException as error:
        old = json.loads((root / "state.json").read_text())
        save(root / "state.json", dict(old, status="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
