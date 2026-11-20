"""Freeze a predictor policy on calibration only, then audit it on holdouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(paths):
    for source in paths:
        with source.open() as stream:
            for line, text in enumerate(stream, 1):
                yield str(source), line, json.loads(text)


def bucket_key(record):
    return str(record["bucket"]), str(record["layer"]), str(record["width"])


def percentile95(values):
    return sorted(values)[math.ceil(.95 * len(values)) - 1]


def measure(record, name):
    initial = record["exact_candidate_intersection"]
    method = record["methods"].get(name)
    if method is None or "failed" in method or name not in record["timings"]:
        raise ValueError(f"unverified calibrated method {name}")
    speed = record["timings"][record["fastest_equal_layout_bf16"]]["median_us"]
    speed /= record["timings"][name]["median_us"]
    misses = [
        (a - b) * 100 / 2048
        for a, b in zip(initial, method["true_top2048_intersection"], strict=True)
    ]
    return speed, misses


def summarize(records, name):
    pairs = [measure(record, name) for record in records]
    individual = [x for _, miss in pairs for x in miss]
    return dict(
        cells=len(records), request_rows=len(individual),
        geomean_speedup=statistics.geometric_mean(speed for speed, _ in pairs),
        mean_extra_miss_pp=statistics.mean(individual),
        p95_extra_miss_pp=percentile95(individual),
        max_extra_miss_pp=max(individual),
    )


def freeze(args):
    by_key = defaultdict(list)
    for _, _, record in read(args.inputs):
        if record["split"] != "calibration":
            raise ValueError("only calibration rows may select the predictor implementation")
        if record["batch"] >= args.minimum_batch:
            by_key[bucket_key(record)].append(record)
    if not by_key:
        raise ValueError("no calibration rows satisfy the specified batch minimum")
    table = []
    for (length, layer, width), records in sorted(by_key.items()):
        valid = []
        for record in records:
            methods = {
                name for name, method in record["methods"].items()
                if (method.get("exact") is not None and "failed" not in method
                    and name in record["timings"] and "_temporal_" not in name)
            }
            valid.append(methods)
        eligible = set.intersection(*valid)
        prior = []
        for name in sorted(eligible):
            stats = summarize(records, name)
            if stats["mean_extra_miss_pp"] <= args.mean_limit and stats["p95_extra_miss_pp"] <= args.tail_limit:
                prior.append((stats["geomean_speedup"], name, stats))
        if not prior:
            raise ValueError(f"no qualified path for {length}, layer {layer}, M={width}")
        _, chosen, stats = max(prior, key=lambda item: (item[0], item[1]))
        table.append(dict(
            length_bucket=int(length), layer=int(layer), M=int(width), method=chosen,
            exact=records[0]["methods"][chosen]["exact"],
            calibration_batch_sizes=sorted({record["batch"] for record in records}),
            quality_calibration=stats,
        ))
    result = dict(
        metric="equal-layout fastest BF16 per condition",
        cutoffs=dict(mean_extra_miss_pp=args.mean_limit, p95_extra_miss_pp=args.tail_limit),
        gate_scope="same-state prediction coverage only",
        lower_batch_bound=args.minimum_batch,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in args.inputs},
        calibration_request_ids=sorted({
            rid for records in by_key.values() for record in records for rid in record["request_ids"]
        }),
        routes=table,
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


def test(args):
    policy = json.loads(args.policy.read_text())
    route = {
        (str(r["length_bucket"]), str(r["layer"]), str(r["M"])):r["method"]
        for r in policy["routes"]
    }
    results = []
    calibration_requests = set(policy["calibration_request_ids"])
    measured = set()
    for source, line, record in read(args.inputs):
        if record["split"] != "test":
            raise ValueError("test audit requires held-out request rows")
        identity = bucket_key(record)
        overlap = calibration_requests.intersection(record["request_ids"])
        if overlap:
            raise ValueError(f"held-out requests also used for calibration: {sorted(overlap)}")
        cell = (*identity, record["batch"], record["step"])
        if cell in measured:
            raise ValueError(f"duplicate holdout cell {cell}")
        measured.add(cell)
        if identity not in route:
            raise ValueError(f"missing pre-registered route for {identity}")
        selected = route[identity]
        speed, miss = measure(record, selected)
        results.append(dict(
            source=source, line=line, length_bucket=record["bucket"], layer=record["layer"],
            batch=record["batch"], step=record["step"], M=record["width"], method=selected,
            speedup_same_layout=speed, extra_miss_pp=miss,
        ))
    expected = {
        (*identity, batch, step)
        for identity in route for batch in args.expect_batch for step in args.expect_step
    }
    if expected != measured:
        raise ValueError(f"holdout matrix mismatch: {len(expected - measured)} missing, "
                         f"{len(measured - expected)} extra")
    individual = [delta for row in results for delta in row["extra_miss_pp"]]
    summary = dict(
        cells=len(results), request_rows=len(individual),
        geomean_speedup=statistics.geometric_mean(row["speedup_same_layout"] for row in results),
        mean_extra_miss_pp=statistics.mean(individual),
        p95_extra_miss_pp=percentile95(individual),
        max_extra_miss_pp=max(individual),
        policy_not_reselected=True,
    )
    summary["target_reached"] = (
        summary["geomean_speedup"] >= 1.5
        and summary["mean_extra_miss_pp"] <= policy["cutoffs"]["mean_extra_miss_pp"]
        and summary["p95_extra_miss_pp"] <= policy["cutoffs"]["p95_extra_miss_pp"]
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(dict(summary=summary, results=results), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    p_freeze = subs.add_parser("freeze")
    p_freeze.add_argument("--mean-limit", type=float, default=1)
    p_freeze.add_argument("--tail-limit", type=float, default=3)
    p_freeze.add_argument("--minimum-batch", type=int, default=4)
    p_test = subs.add_parser("test")
    p_test.add_argument("--policy", type=Path, required=True)
    p_test.add_argument("--expect-batch", type=int, nargs="+", default=[4, 16])
    p_test.add_argument("--expect-step", type=int, nargs="+", default=[2, 4])
    for sub in (p_freeze, p_test):
        sub.add_argument("--inputs", type=Path, nargs="+", required=True)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (freeze if args.mode == "freeze" else test)(args)
