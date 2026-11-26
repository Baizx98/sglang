"""Replay fixed Top-M prefetch requests with alternative HBM eviction policies."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from horizonkv_eviction_inputs import load_sequences


K = 2048
DEFAULT_TRACE = Path("data/horizonkv-predictive/2026-11-16_2106_processed330/combined330")
COUNTERS = (
    "requests", "demand_loads", "prefetch_loads", "repeated_h2d", "reload_after_eviction",
    "evictions", "prefetched_consumed", "unused_evicted", "unused_at_end",
    "displaced_demand", "rescued_demand", "reference_dense_state_bytes", "cpu_ms",
)
STEP_POLICIES = (
    ("lru", 1, 0), ("lru-touch", 1, 0), ("lru2", 1, 0),
    ("lfu", 4, 0), ("lfu", 16, 0), ("lfu", 64, 0), ("pred", 1, 0),
    ("lru-pred", 1, 2), ("lru-pred", 1, 8),
    ("lfu-pred", 4, 0.5), ("lfu-pred", 4, 2),
    ("lfu-pred", 16, 0.5), ("lfu-pred", 16, 2),
)
POLICIES = tuple((*policy, 0) for policy in STEP_POLICIES)
ISSUED_POLICIES = (
    ("lru-issued", 1, 2, 0), ("lru-issued", 1, 8, 0),
    ("lfu-issued", 4, 0.5, 0), ("lfu-issued", 4, 2, 0),
    ("lfu-issued", 16, 0.5, 0), ("lfu-issued", 16, 2, 0),
)
TURN_POLICIES = (
    ("turn", 1, 0, 0), ("role", 1, 0, 0),
    *tuple((name, half, hint, bonus)
           for half, bonus in itertools.product((4, 16), (0.5, 2))
           for prior in ("turn", "role")
           for name, hint in ((f"lfu-{prior}", 0),
                              (f"lfu-pred-{prior}", 0.5), (f"lfu-pred-{prior}", 2))),
)


def policy_label(policy):
    name, decay, bonus, prior = policy
    return f"{name}:{decay:g}:{bonus:g}" + (f":{prior:g}" if prior else "")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(count, steps, priors=()):
    data = [np.asarray([count, len(steps)], dtype="<i4").tobytes()]
    for prior in priors or ((), ()):
        data.append(np.asarray([len(prior)], dtype="<i4").tobytes())
        data.append(np.asarray(prior, dtype="<f4").tobytes())
    for step in steps:
        for array in step:
            data.append(np.asarray([len(array)], dtype="<i4").tobytes())
            data.append(np.asarray(array, dtype="<i4").tobytes())
    return b"".join(data)


def call(executable, payload, configs, limit=0):
    suffix = "".join(f"{capacity} {mode} {decay} {bonus} {prior} {limit}\n"
                     for capacity, (mode, decay, bonus, prior) in configs)
    outcome = subprocess.run([executable], input=payload + suffix.encode(), capture_output=True, check=False)
    if outcome.returncode:
        raise RuntimeError(outcome.stderr.decode())
    rows = []
    for line in outcome.stdout.decode().splitlines():
        metrics = [int(x) for x in line.split()[:-1]] + [float(line.split()[-1])]
        rows.append(dict(zip(COUNTERS, metrics, strict=True)))
    if len(configs) != len(rows):
        raise RuntimeError("replay did not return all configurations")
    return rows


def verify_protocol(executable):
    lru, touch = POLICIES[:2]
    # Prefetch the wrong row, evict one consumed row, reload it, and account
    # for the correct speculative transfer only when its row is consumed.
    p = [
        ([1], [1], [0], [], []),
        ([2], [2], [0, 2], [], []),
        ([], [], [0], [3], [3]),
        ([], [], [3], [], []),
    ]
    a, b, zero = call(executable, frame(4, p), [(1, lru), (1, touch), (0, lru)])
    assert a["demand_loads"] == b["demand_loads"] == 3
    assert a["prefetched_consumed"] == 1 and a["unused_evicted"] == 1
    assert a["displaced_demand"] == a["rescued_demand"] == 1
    assert a["repeated_h2d"] == a["reload_after_eviction"] == 2 and zero["prefetch_loads"] == 0
    assert zero["demand_loads"] == zero["requests"] == 5
    configs = [(2, policy) for policy in POLICIES]
    frames = [
        ([1], [1], [0], [], []),
        ([0], [0], [0], [], []),
        ([2], [2], [0], [], []),
    ]
    for timing in (1, 2):
        a = call(executable, frame(3, frames), configs, timing)
        # Alternative future demand cannot change prefix counters.
        b = call(executable, frame(3, [*frames[:timing], ([], [], [1], [], [])]), configs, timing)
        for x, y in zip(a, b, strict=True):
            assert all(x[k] == y[k] for k in COUNTERS if k != "cpu_ms")
    events = [([0, 1], [0, 1], [], [], []), ([2], [2], [0], [], [])]
    a, b = call(executable, frame(3, events, ([0.9, 0.1, 0], [0.1, 0.9, 0])),
                [(2, policy) for policy in TURN_POLICIES[:2]])
    assert a["demand_loads"] == 0 and b["demand_loads"] == 1


def pack_steps(batch, prefetch_width):
    offsets = np.cumsum([0] + [int(seq["current"][-1]) + 1 for seq in batch])
    count = len(batch[0]["demands"])
    if any(len(seq["demands"]) != count for seq in batch):
        raise ValueError("batched sequences must use the same number of steps")
    for step in range(count):
        predicted, prefetch, demand, writes, used = [], [], [], [], []
        for seq, offset in zip(batch, offsets[:-1], strict=True):
            p = seq["predictions"][step] + offset
            predicted.append(p)
            prefetch.append(p[:prefetch_width])
            demand.append(np.sort(seq["demands"][step] + offset))
            current = int(seq["current"][step] + offset)
            writes.append(current)
            if seq["used_current"][step]:
                used.append(current)
        predicted_order = interleave(predicted)
        prefetch_order = interleave(prefetch)
        yield predicted_order, prefetch_order, np.concatenate(demand), writes, used


def interleave(arrays):
    if not arrays or max(map(len, arrays)) == 0:
        return np.empty(0, dtype=np.int32)
    stacked = np.full((max(map(len, arrays)), len(arrays)), -1, dtype=np.int32)
    for col, ids in enumerate(arrays):
        stacked[:len(ids), col] = ids
    return stacked[stacked >= 0]


def replay_batch(executable, pack, args):
    group, bucket, layer, split, batch, members, partition, budget = pack
    policies = POLICIES + TURN_POLICIES if args.turn_trace else POLICIES
    if args.issued_baseline:
        policies += ISSUED_POLICIES
    configs = [(round(ratio * K * len(partition)), policy)
               for ratio, policy in itertools.product(args.capacity, policies)]
    count = sum(int(seq["current"][-1]) + 1 for seq in partition)
    priors = tuple(np.concatenate([seq[name] for seq in partition]) for name in ("turn_prior", "role_prior")) if args.turn_trace else ()
    data = frame(count, list(pack_steps(partition, budget)), priors)
    results = call(executable, data, configs)
    rows = []
    for ((capacity, policy), stats) in zip(configs, results, strict=True):
        assert stats["demand_loads"] <= stats["requests"]
        rows.append(dict(
            dataset=group, bucket=bucket, layer=layer, split=split, batch=batch,
            requests_in_group="|".join(members), requests_in_partition=len(partition),
            views=args.group_by, placement=args.placement, capacity=capacity,
            capacity_over_batch_k=capacity / (len(partition) * K), prefetch_per_request=budget,
            policy=policy_label(policy), family=policy[0], **stats,
            h2d_bytes=(stats["demand_loads"] + stats["prefetch_loads"]) * args.kv_bytes,
            repeated_h2d_bytes=stats["repeated_h2d"] * args.kv_bytes,
            supplied_prior_bytes=sum(p.nbytes for p in priors),
        ))
    return rows


def replay_groups(args, executable, pool):
    source = {}
    grid = []
    for split in ("calibration", "test"):
        if args.turn_trace:
            from horizonkv_eviction_turn_inputs import load_terminal_sequences

            cohorts = load_terminal_sequences(args.turn_trace, args.prior_trace, args.layers, split, args.seed)
        else:
            cohorts = load_sequences(args.trace, args.layers, split, args.max_per_cell, args.seed, args.group_by)
        for (group, bucket, layer), traces in sorted(cohorts.items()):
            if args.only_bucket and bucket not in args.only_bucket:
                continue
            if args.only_dataset and group not in args.only_dataset:
                continue
            for trace in traces:
                source[str(trace["source"])] = trace["source_sha256"]
            for batch in args.batch:
                for start in range(0, len(traces) - batch + 1, batch):
                    members = traces[start:start + batch]
                    ids = [item["rid"] for item in members]
                    partitions = [[item] for item in members] if args.placement == "private" else [members]
                    for partition, budget in itertools.product(partitions, args.prefetch):
                        grid.append((group, bucket, layer, split, batch, ids, partition, budget))
    if not grid:
        raise RuntimeError("no complete batch available for replay")
    futures = pool.map(lambda pack: replay_batch(executable, pack, args), grid)
    with (args.output / "cells.csv").open("x") as stream:
        writer = None
        for i, rows in enumerate(futures, 1):
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
                writer.writeheader()
            writer.writerows(rows)
            if i % 10 == 0 or i == len(grid):
                stream.flush()
                print(f"{i}/{len(grid)} partitions", flush=True)
    return source, len(grid)


def summarize(root):
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    by_bucket = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    with (root / "cells.csv").open() as stream:
        for row in csv.DictReader(stream):
            view = tuple(row[k] for k in ("batch", "capacity_over_batch_k", "prefetch_per_request"))
            policy = row["policy"]
            keys = (row["split"], *view)
            stats = [int(row[k]) for k in ("h2d_bytes", "demand_loads", "requests", "repeated_h2d_bytes")]
            for table in (counts[keys], by_bucket[(*keys, row["dataset"], row["bucket"], row["layer"])]):
                for i, value in enumerate(stats):
                    table[policy][i] += value
    emitted = []
    for key, test in sorted(counts.items()):
        if key[0] != "test" or ("calibration", *key[1:]) not in counts:
            continue
        calib = counts[("calibration", *key[1:])]
        def best(*families):
            return min((p for p in calib if p.split(":")[0] in families), key=lambda p: (calib[p][0], p))

        baseline = best("lru", "lru-touch", "lru2", "lfu")
        blended = best("lru-pred", "lfu-pred")
        tuned_lfu = min((p for p in calib if p.startswith("lfu:")), key=lambda p: (calib[p][0], p))
        policies = [("baseline", baseline), ("lru-demand", "lru:1:0"), ("lfu", tuned_lfu),
                    ("prediction", "pred:1:0"), ("combined", blended)]
        if "turn:1:0" in calib:
            policies += [(name, best(name)) for name in (
                "turn", "role", "lfu-turn", "lfu-role", "lfu-pred-turn", "lfu-pred-role",
            )]
        cohort = [("all", "all", "all", test)]
        cohort += [(*part[-3:], values) for part, values in sorted(by_bucket.items()) if part[:4] == key]
        for dataset, bucket, layer, values in cohort:
            reference = values[baseline]
            for role, policy in policies:
                transferred, misses, requests, repeated = values[policy]
                emitted.append(dict(batch=key[1], capacity_over_batch_k=key[2], prefetch_per_request=key[3],
                                    dataset=dataset, bucket=bucket, layer=layer, role=role, policy=policy,
                                    reference=baseline, h2d_bytes=transferred, demand_loads=misses, requests=requests,
                                    repeated_h2d_bytes=repeated, h2d_saved_fraction=1 - transferred/reference[0]))
    with (root / "summary.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(emitted[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(emitted)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cpp = Path(__file__).with_suffix(".cpp")
    executable = (args.output / "replay").resolve()
    sources = (cpp, Path(__file__), cpp.with_name("horizonkv_eviction_inputs.py"))
    if args.turn_trace:
        sources += tuple(cpp.with_name(name) for name in ("horizonkv_eviction_turn_inputs.py", "horizonkv_turn_reuse.py"))
    source_sha = {x.name: file_sha(x) for x in sources}
    subprocess.run(["g++", "-O3", "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(executable)], check=True)
    verify_protocol(executable)
    if args.check_only:
        print("fixed-prefetch replay protocol checks passed")
        return
    torch.set_num_threads(1)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        traces, partitions = replay_groups(args, executable, pool)
    summarize(args.output)
    metadata = dict(args=vars(args), trace_sha256=traces, partitions=partitions,
                    code_sha256=source_sha,
                    compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0])
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    cli.add_argument("--turn-trace", type=Path)
    cli.add_argument("--prior-trace", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--layers", type=int, nargs="+", default=[1, 12, 24, 47])
    cli.add_argument("--batch", type=int, nargs="+", default=[4])
    cli.add_argument("--capacity", type=float, nargs="+", default=[0.5, 1, 2])
    cli.add_argument("--prefetch", type=int, nargs="+", default=[0, 512, 2048])
    cli.add_argument("--issued-baseline", action="store_true")
    cli.add_argument("--only-bucket", type=int, nargs="+")
    cli.add_argument("--only-dataset", nargs="+")
    cli.add_argument("--placement", choices=["private", "shared"], default="private")
    cli.add_argument("--group-by", choices=["dataset", "length"], default="dataset")
    cli.add_argument("--max-per-cell", type=int, default=8)
    cli.add_argument("--seed", type=int, default=20261126)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--kv-bytes", type=int, default=1024)
    cli.add_argument("--check-only", action="store_true")
    opts = cli.parse_args()
    if min(opts.batch) < 1 or min(opts.capacity) <= 0 or not all(0 <= p <= 3071 for p in opts.prefetch):
        cli.error("expected positive batch and capacity and prefetch width from 0 to 3071")
    if bool(opts.turn_trace) != bool(opts.prior_trace):
        cli.error("--turn-trace and --prior-trace must be used together")
    run(opts)
