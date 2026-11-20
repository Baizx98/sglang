"""Summarize same-server RULER outcomes and paired decode latency."""
from __future__ import annotations

import argparse
import csv
from itertools import takewhile
import json
from pathlib import Path
from statistics import mean, median


def token_ids(row):
    return row["response"]["output_ids"]


def cycles(timed, period):
    refresh, reuse = [], []
    for first in range(2, max(timed) + 1, period):
        if any(n not in timed for n in range(first - 1, first + period)):
            continue
        refresh.append(timed[first] - timed[first - 1])
        reuse.extend(timed[n] - timed[n - 1] for n in range(first + 1, first + period))
    return refresh, reuse


def compare(row, exact, paired_tokens, period):
    output = token_ids(row)
    reference = token_ids(exact)
    prefix = sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(output, reference)))
    refresh = row["refresh_parity_s"]
    reuse = row["reuse_parity_s"]
    values = row["decode_s_per_token"]
    baseline = exact["decode_s_per_token"]
    timed = dict(row["stream_timestamps_s"])
    exact_timed = dict(exact["stream_timestamps_s"])
    first_decode = timed[2] - timed[1] if {1, 2} <= timed.keys() else None
    trailing_steps = len(output) - 2
    trailing_decode = (
        (timed[len(output)] - timed[2]) / trailing_steps
        if trailing_steps > 0 and {2, len(output)} <= timed.keys() else None
    )
    partial = timed[paired_tokens] - timed[1] if {1, paired_tokens} <= timed.keys() else None
    exact_partial = exact_timed[paired_tokens] - exact_timed[1] if {1, paired_tokens} <= exact_timed.keys() else None
    paired_cycle_observed = paired_tokens >= period + 1
    cycle_refresh, cycle_reuse = cycles(timed, period)
    cycle = cycle_refresh + cycle_reuse
    return {
        **{name: row[name] for name in (
            "case_id", "task", "length_bucket", "prompt_tokens", "mode", "repetition",
            "end_of_think_observed", "unfinished_think",
        )},
        "task_raw_score": float(row["task_raw_score"]),
        "task_final_score": float(row["task_final_score"]),
        "completion_tokens": len(output),
        "finish_reason": row["response"]["meta_info"]["finish_reason"]["type"],
        "output_ids_match": output == reference,
        "matching_output_prefix_tokens": prefix,
        "final_task_score_delta": row["task_final_score"] - exact["task_final_score"],
        "ttft_s": row["ttft_s"],
        "full_request_s": row["elapsed_s"],
        "decode_ms_per_token": values * 1000 if values is not None else None,
        "first_decode_ms": first_decode * 1000 if first_decode is not None else None,
        "later_decode_ms_per_token": trailing_decode * 1000 if trailing_decode is not None else None,
        "paired_decode_speedup": baseline / values if baseline is not None and values else None,
        "paired_tokens": paired_tokens,
        "paired_cycle_observed": paired_cycle_observed,
        "full_prefix_s": timed.get(paired_tokens),
        "paired_prefix_decode_ms_per_token": (
            partial * 1000 / (paired_tokens - 1) if partial is not None and paired_tokens > 1 else None
        ),
        "paired_prefix_baseline_ms_per_token": (
            exact_partial * 1000 / (paired_tokens - 1)
            if exact_partial is not None and paired_tokens > 1 else None
        ),
        "paired_prefix_speedup": (
            exact_partial / partial
            if exact_partial is not None and partial and paired_cycle_observed else None
        ),
        "cycle_period": period,
        "complete_cycles": len(cycle_refresh),
        "cycle_token_mean_ms": mean(cycle) * 1000 if cycle else None,
        "cycle_refresh_intervals": len(cycle_refresh),
        "cycle_refresh_mean_ms": mean(cycle_refresh) * 1000 if cycle_refresh else None,
        "cycle_reuse_intervals": len(cycle_reuse),
        "cycle_reuse_mean_ms": mean(cycle_reuse) * 1000 if cycle_reuse else None,
        "refresh_parity_intervals": len(refresh),
        "reuse_parity_intervals": len(reuse),
        "refresh_parity_mean_ms": mean(refresh) * 1000 if refresh else None,
        "reuse_parity_mean_ms": mean(reuse) * 1000 if reuse else None,
        "refresh_parity_median_ms": median(refresh) * 1000 if refresh else None,
        "reuse_parity_median_ms": median(reuse) * 1000 if reuse else None,
        "all_decode_intervals_observed": len(refresh) + len(reuse) == len(output) - 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Summarize only completed exact+layer+step case groups.")
    parser.add_argument("--audit-dir", type=Path,
                        help="Also aggregate exact shadow metrics, without timing comparisons.")
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / "outputs.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError("no completed generations")
    exact = {(row["case_id"], row["repetition"]): row for row in rows if row["mode"] == "exact"}
    groups = {}
    for row in rows:
        groups.setdefault((row["case_id"], row["repetition"]), []).append(row)
    modes = json.loads((args.run / "config.json").read_text())["arguments"]["modes"]
    period = 4 if "skip4" in modes else (2 if "step" in modes else 1)
    comparisons = []
    for key, group in groups.items():
        if sorted(row["mode"] for row in group) != sorted(modes):
            if args.allow_incomplete:
                continue
            raise ValueError(f"incomplete exact/candidate pair for {key}")
        for row in group:
            paired_tokens = min(32, len(token_ids(row)), len(token_ids(exact[key])))
            if paired_tokens > period:
                paired_tokens = 1 + ((paired_tokens - 1) // period) * period
            comparisons.append(compare(row, exact[key], paired_tokens, period))
    if not comparisons:
        raise ValueError("no completed paired request groups")
    table = args.run / "paired.csv"
    with table.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    per_mode = {}
    for mode in sorted({row["mode"] for row in comparisons}):
        group = [row for row in comparisons if row["mode"] == mode]
        speedups = [row["paired_decode_speedup"] for row in group
                    if row["paired_decode_speedup"] is not None]
        prefix_speedups = [row["paired_prefix_speedup"] for row in group
                           if row["paired_prefix_speedup"] is not None]
        per_mode[mode] = {
            "requests": len(group),
            "raw_task_score_mean": mean(row["task_raw_score"] for row in group),
            "final_task_score_mean": mean(row["task_final_score"] for row in group),
            "unfinished_think_count": sum(row["unfinished_think"] for row in group),
            "output_ids_match_count": sum(row["output_ids_match"] for row in group),
            "all_intervals_observed_count": sum(row["all_decode_intervals_observed"] for row in group),
            "paired_decode_speedup_median": median(speedups) if speedups else None,
            "paired_prefix_speedup_median": median(prefix_speedups) if prefix_speedups else None,
            "refresh_parity_weighted_mean_ms": weighted_mean(group, "refresh_parity"),
            "reuse_parity_weighted_mean_ms": weighted_mean(group, "reuse_parity"),
            "complete_cycles": sum(row["complete_cycles"] for row in group),
            "cycle_refresh_weighted_mean_ms": weighted_mean(group, "cycle_refresh"),
            "cycle_reuse_weighted_mean_ms": weighted_mean(group, "cycle_reuse"),
        }
    (args.run / "summary.json").write_text(json.dumps(per_mode, indent=2) + "\n")
    if args.audit_dir is not None:
        summarize_shadow(args.audit_dir)


def weighted_mean(group, term):
    counts = [row[f"{term}_intervals"] for row in group]
    total = sum(row[f"{term}_mean_ms"] * count
                for row, count in zip(group, counts) if count)
    return total / sum(counts) if sum(counts) else None


def summarize_shadow(directory):
    data = {}
    for path in sorted(directory.glob("candidate-shadow-pp*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            length_bucket = 1 << (row["length"] - 1).bit_length()
            data.setdefault((row["request_id"], length_bucket), []).append(row)
    if not data:
        raise ValueError("no actual service shadow observations")
    result = []
    for (request, length_bucket), rows in sorted(data.items()):
        overlap = [row["overlap"] / 2048 for row in rows]
        key_checks = [row["k_only_matches_qkw"] for row in rows if row["k_only_matches_qkw"] is not None]
        writes = [row["current_write_matches_projection"] for row in rows
                  if row.get("current_write_matches_projection") is not None]
        native_keys = [row["fused_key_matches_native"] for row in rows
                       if row.get("fused_key_matches_native") is not None]
        contract_rows = [row for row in rows if "invalid_or_missing" in row]
        result.append({
            "request_id": request, "mode": request.split("__")[1], "length_bucket": length_bucket,
            "layer_step_rows": len(rows), "mean_recall": mean(overlap),
            "min_recall": min(overlap), "strictly_higher_missing": sum(row["strictly_higher_missing"] for row in rows),
            "selected_below_threshold": sum(row["selected_below_threshold"] for row in rows),
            "k_only_matches_qkw": sum(key_checks), "k_only_checks": len(key_checks),
            "cache_write_matches": sum(writes), "cache_write_checks": len(writes),
            "fused_keys_match_native": sum(native_keys), "fused_key_checks": len(native_keys),
            "contract_rows": len(contract_rows),
            "invalid_or_missing": sum(row["invalid_or_missing"] for row in contract_rows),
            "duplicate_indices": sum(row["duplicate_indices"] for row in contract_rows),
            "native_current_selected": sum(row["native_selects_current"] for row in contract_rows),
            "result_current_selected": sum(row["result_selects_current"] for row in contract_rows),
        })
    with (directory / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(result[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    main()
