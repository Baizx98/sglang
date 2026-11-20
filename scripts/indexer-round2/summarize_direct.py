"""Summarize complete INT4-direct timings, layout controls, and selection fidelity."""

import argparse
import csv
import hashlib
import itertools
import json
import statistics
from pathlib import Path

from summarize_full import LAYOUTS, summarize

DEFAULT_POLICY = Path(__file__).with_name("int4-direct-policy.json")
METHOD_FIELDS = ("primary", "selector_sensitivity", "serving_projection_control",
                 "auxiliary_projection_control", "intermediate_refinement_ablation")


def build_summary(rows, methods):
    summary = {}
    for name in methods:
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
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    registered = policy["acceptance"]
    methods = [policy[name] for name in METHOD_FIELDS if name in policy]
    provenance = json.loads(args.input.with_suffix(".metadata.json").read_text())
    policy_digest = hashlib.sha256(args.policy.read_bytes()).hexdigest()
    if not any(Path(name).name == args.policy.name and digest == policy_digest
               for name, digest in provenance["source_sha256"].items()):
        raise ValueError("the submitted policy does not match the measured policy hash")
    run_args = provenance["args"]
    for field, recorded in (("layer", "layer"), ("decode_step", "step"),
                            ("request_offset", "request_offset"), ("seed", "seed"),
                            ("trials", "trials"), ("graph_iterations", "graph_iterations"),
                            ("warmups", "warmups")):
        if registered[field] != run_args[recorded]:
            raise ValueError(f"{recorded} differs from the registered policy")
    cases = [json.loads(line) for line in args.input.read_text().splitlines()]
    measured = [(case["input"]["length_bucket"], case["input"]["batch"], case["input"]["physical_layout"])
                for case in cases]
    expected = list(itertools.product(
        registered["length_buckets"], registered["batch_sizes"], registered["source_slot_layouts"],
    ))
    if sorted(expected) != sorted(measured):
        raise ValueError(f"expected exactly the frozen 18 cells, found {measured}")
    rows = [row for row in summarize(cases) if row["candidate"] in methods]
    with args.input.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    payload = build_summary(rows, methods)
    pair_checks = {}
    for case in cases:
        for methods_pair, matches in case.get("direct_pair_exact_sets", {}).items():
            counts = pair_checks.setdefault(methods_pair, {"same_set_rows": 0, "rows": 0})
            counts["same_set_rows"] += matches
            counts["rows"] += case["input"]["batch"]
    if pair_checks:
        payload["paired_selector_and_projection"] = pair_checks
    args.input.with_suffix(".summary.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
