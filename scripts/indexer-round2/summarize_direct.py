"""Summarize complete INT4-direct timings, layout controls, and selection fidelity."""

import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path

from summarize_full import LAYOUTS, summarize

POLICY = json.loads(Path(__file__).with_name("int4-direct-policy.json").read_text())
METHODS = [POLICY[name] for name in ("primary", "selector_sensitivity",
                                    "serving_projection_control", "intermediate_refinement_ablation")]


def build_summary(rows):
    summary = {}
    for name in METHODS:
        selected = [row for row in rows if row["candidate"] == name]
        if not selected:
            continue
        metrics = {}
        groups = {"all": selected}
        groups.update({layout: [row for row in selected if row["source_slots"] == layout]
                       for layout in LAYOUTS})
        for group, subset in groups.items():
            if not subset:
                continue
            metrics[group] = {
                "conditions": len(subset),
                "against_old_placement": statistics.geometric_mean(r["over_strongest"] for r in subset),
                "against_same_placement": statistics.geometric_mean(r["over_same_placement"] for r in subset),
                "weakest_against_old_placement": min(r["over_strongest"] for r in subset),
                "weakest_against_same_placement": min(r["over_same_placement"] for r in subset),
                "conditions_at_least_1_5x": sum(r["over_strongest"] >= 1.5 for r in subset),
                "same_placement_conditions_at_least_1_5x": sum(r["over_same_placement"] >= 1.5 for r in subset),
                "native_exact_conditions": sum(r["exact"] for r in subset),
                "coarse_exact_conditions": sum(r["coarse_topk_exact"] is True for r in subset),
                "mean_cell_overlap": statistics.mean(r["topk_overlap"] for r in subset),
                "minimum_cell_overlap": min(r["topk_overlap"] for r in subset),
                "mean_resident_to_raw": statistics.mean(
                    r["slab_resident_bytes"] / r["raw_index_bytes"] for r in subset),
            }
            per_request = [r["worst_request_overlap"] for r in subset
                           if r["worst_request_overlap"] is not None]
            if per_request:
                metrics[group]["minimum_request_overlap"] = min(per_request)
        summary[name] = metrics
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.input.read_text().splitlines()]
    measured = [(case["input"]["length_bucket"], case["input"]["batch"], case["input"]["physical_layout"])
                for case in cases]
    expected = list(itertools.product((32768, 65536, 131072), (4, 16), LAYOUTS))
    if sorted(expected) != sorted(measured):
        raise ValueError(f"expected exactly the frozen 18 cells, found {measured}")
    rows = [row for row in summarize(cases) if row["candidate"] in METHODS]
    with args.input.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    payload = build_summary(rows)
    args.input.with_suffix(".summary.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
