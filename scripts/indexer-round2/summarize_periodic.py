"""Tabulate real five-step periodic Top-K refresh replay."""

import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path


LAYOUTS = ("contiguous", "shuffle_128_token_blocks", "shuffle_1_token_blocks")


def summarize(cases):
    result = []
    for case in cases:
        meta = case["input"]
        steps = case["steps"]
        checks = case["checks"]
        timing = {name: value["median_us"] for name, value in case["timings"].items()}
        reference = min(
            (name for name, checks_by_step in checks.items()
             if name.startswith(("native_", "old_lossless_")) and all(row["valid"] for row in checks_by_step)),
            key=timing.get,
        )
        for period in (3, 4):
            method = f"period{period}"
            refreshed = case["refresh_steps"][method]
            reuse = [row for step, row in zip(steps, checks[method]) if step not in refreshed]
            selected = sum(row["set_overlap"] for row in reuse)
            total = sum(row["total_selected"] for row in reuse)
            result.append({
                "length_bucket": meta["length_bucket"], "batch": meta["batch"],
                "source_slots": meta["physical_layout"], "method": method,
                "baseline": reference, "baseline_window_us": timing[reference],
                "same_slab_exact_window_us": timing["always_slab"],
                "window_us": timing[method], "per_step_us": timing[method] / len(steps),
                "window_mad_us": case["timings"][method]["mad_us"],
                "first_cycle_us": timing[f"{method}_cycle"],
                "first_cycle_steps": period, "terminal_fragment_us": timing[f"{method}_tail"],
                "terminal_fragment_steps": len(steps) - period,
                "speedup_vs_baseline": timing[reference] / timing[method],
                "speedup_vs_same_slab": timing["always_slab"] / timing[method],
                "reuse_selected": selected, "reuse_total": total, "reuse_overlap": selected / total,
                "reuse_strict_missed": sum(row["missing_strictly_above_threshold"] for row in reuse),
                "refresh_exact": all(checks[method][step]["valid"] for step in refreshed),
                "final_step_exact": checks[method][-1]["valid"],
                "final_step_overlap": checks[method][-1]["set_overlap"] /
                                      checks[method][-1]["total_selected"],
                "all_steps_exact": all(check["valid"] for check in checks[method]),
                "all_current_keys_bitwise_equal": not any(
                    check["key_only_diff_bf16"] + check.get("refresh_diff_bf16", check["fused_diff_bf16"])
                    for check in case["key_checks"]
                ),
                "index_slab_bytes": case["state_bytes"]["key_slab"],
                "topk_state_bytes": case["state_bytes"]["topk_cache"],
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.input.read_text().splitlines()]
    measured = [(case["input"]["length_bucket"], case["input"]["batch"], case["input"]["physical_layout"])
                for case in cases]
    expected = list(itertools.product((32768, 65536, 131072), (4, 16), LAYOUTS))
    if not args.allow_partial and sorted(expected) != sorted(measured):
        raise ValueError(f"expected 18 conditions, got {measured}")
    rows = summarize(cases)
    with args.input.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metrics = {}
    for method in ("period3", "period4"):
        subset = [row for row in rows if row["method"] == method]
        metrics[method] = {
            "conditions": len(subset),
            "geomean_over_baseline": statistics.geometric_mean(row["speedup_vs_baseline"] for row in subset),
            "geomean_over_same_slab": statistics.geometric_mean(row["speedup_vs_same_slab"] for row in subset),
            "worst_speedup": min(row["speedup_vs_baseline"] for row in subset),
            "all_refresh_exact": all(row["refresh_exact"] for row in subset),
            "all_keys_bitwise_exact": all(row["all_current_keys_bitwise_equal"] for row in subset),
            "all_steps_exact_conditions": sum(row["all_steps_exact"] for row in subset),
            "reuse_token_overlap": sum(row["reuse_selected"] for row in subset) /
                                   sum(row["reuse_total"] for row in subset),
            "worst_case_reuse_overlap": min(row["reuse_overlap"] for row in subset),
            "worst_case_final_step_overlap": min(row["final_step_overlap"] for row in subset),
        }
    args.input.with_suffix(".summary.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
