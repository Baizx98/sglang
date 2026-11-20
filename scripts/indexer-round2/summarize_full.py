"""Tabulate the frozen Full Indexer acceptance matrix."""

import argparse
import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path


LAYOUTS = ("contiguous", "shuffle_128_token_blocks", "shuffle_1_token_blocks")
CANDIDATES = (
    "fixed_layout_fused_norm", "fixed_layout_lossless_fused_norm", "fixed_layout_fused_epilogue",
    "fixed_layout_lossless_fused_epilogue", "slab_bf16", "slab_bf16_fused_norm",
    "slab_bf16_fused_epilogue", "slab_lossless", "slab_lossless_fused_norm",
    "slab_lossless_fused_epilogue", "int4_scan_bf16_refine_4096", "int4_scan_bf16_refine_8192",
    "int4_direct_flashinfer", "int4_direct_vllm", "int4_scan_bf16_refine_2560",
    "int4_direct_flashinfer_native_projection",
)
POLICY = json.loads(Path(__file__).with_name("full-policy.json").read_text())
POLICY_NAMES = ("frozen_independent_layout", "frozen_unchanged_layout",
                "frozen_approximate", "posthoc_fastest_correct")


def exact(check):
    return not check.get("overflow_rows") and all(
        report["valid"] for kind in ("native", "torch") for report in check[kind]
    )


def group_stats(rows, candidates):
    result = {}
    for name in candidates:
        matching = [row for row in rows if row["candidate"] == name]
        if not matching:
            continue
        valid = [row for row in matching if row["exact"]]
        result[name] = {
            "conditions": len(matching), "exact_conditions": len(valid),
            "over_strongest_geomean_all": statistics.geometric_mean(
                row["over_strongest"] for row in matching),
            "worst_overlap": min(row["topk_overlap"] for row in matching),
            "failed_conditions": [
                [row["length_bucket"], row["batch"], row["source_slots"]]
                for row in matching if not row["exact"]
            ],
        }
        if valid:
            for field in ("over_native", "over_tuned_original", "over_old_lossless", "over_strongest",
                          "over_same_placement"):
                numbers = [row[field] for row in valid]
                result[name][f"{field}_geomean_valid"] = statistics.geometric_mean(numbers)
                result[name][f"{field}_wins"] = sum(number > 1 for number in numbers)
                result[name][f"{field}_min_valid"] = min(numbers)
    return result


def summarize(data):
    rows = []
    for case in data:
        metadata = case["input"]
        values = {k: v["median_us"] for k, v in case["timings"].items()}
        checks = case["checks"]

        def best(prefix):
            return min((name for name, value in checks.items()
                        if name.startswith(prefix) and exact(value)), key=values.get)

        native = best("original_sgl_")
        tuned = best("original_tuned_dense_")
        lossless = best("old_lossless_")
        strongest = min((native, tuned, lossless), key=values.get)
        same_placement = best("same_placement_")
        case_rows = {}
        for candidate in CANDIDATES:
            if candidate not in values:
                continue
            duration = values[candidate]
            details = checks[candidate]["native"]
            slabs = case["slab"]
            slab_key = "slab_lossless" if candidate.startswith("slab_lossless") else "slab_bf16"
            if candidate.startswith("slab"):
                allocated = slabs[slab_key]
            elif candidate.startswith("int4_direct"):
                allocated = slabs["int4_direct"]
            elif candidate.startswith("int4"):
                allocated = slabs["int4"]
            else:
                allocated = None
            projection = ("fused_epilogue" if candidate.endswith("fused_epilogue")
                          else "fused_norm"
                          if candidate.endswith("fused_norm") or (
                              candidate.startswith("int4") and not candidate.endswith("native_projection")
                          ) else None)
            proj_check = case["projection_check"][projection] if projection else None
            case_rows[candidate] = {
                "length_bucket": metadata["length_bucket"], "batch": metadata["batch"],
                "source_slots": metadata["physical_layout"], "candidate": candidate,
                "exact": exact(checks[candidate]),
                "all_native_sets_equal": all(result["set_overlap"] == result["total_selected"]
                                             for result in details),
                "coarse_topk_exact": all(item["valid"] for item in checks[candidate]["coarse_topk"])
                if "coarse_topk" in checks[candidate] else None,
                "topk_overlap": min(result["set_overlap"] / result["total_selected"]
                                    for result in details),
                "worst_request_overlap": min(min(values) / 2048
                                             for values in checks[candidate]["native_overlap_by_request"])
                if "native_overlap_by_request" in checks[candidate] else None,
                "strict_missed": max(report["missing_strictly_above_threshold"] for report in details),
                "native_name": native, "native_us": values[native],
                "tuned_original_name": tuned, "tuned_original_us": values[tuned],
                "old_lossless_name": lossless, "old_lossless_us": values[lossless],
                "strongest_name": strongest, "strongest_us": values[strongest],
                "same_placement_name": same_placement,
                "same_placement_us": values[same_placement],
                "candidate_us": duration, "candidate_mad_us": case["timings"][candidate]["mad_us"],
                "over_native": values[native] / duration,
                "over_tuned_original": values[tuned] / duration,
                "over_old_lossless": values[lossless] / duration,
                "over_strongest": values[strongest] / duration,
                "over_same_placement": values[same_placement] / duration,
                "raw_index_bytes": case["raw_index_k_bytes"],
                "slab_resident_bytes": allocated["bytes"] if allocated else None,
                "slab_init_wall_us": allocated.get("initial_build_wall_median_us") if allocated else None,
                "workspace_bytes_before_selector": allocated.get("workspace_bytes_before_selector")
                if allocated else None,
                "native_project_us": values["project_native"],
                "candidate_project_us": values[f"project_{projection or 'native'}"],
                "query_different_bf16": proj_check["query"]["different_bf16"] if proj_check else 0,
                "key_different_bf16": proj_check["current_key"]["different_bf16"] if proj_check else 0,
            }
        rows.extend(case_rows.values())
        fixed = POLICY["primary_fixed_index_storage"]
        use_fused_original = (metadata["length_bucket"] == 32768 or
                              metadata["physical_layout"] == "shuffle_1_token_blocks")
        fixed_method = fixed["32K_or_per_token_random_slots"] if use_fused_original else fixed["other"]
        allowed = [row for name, row in case_rows.items()
                   if not name.startswith("int4") and row["exact"]]
        aliases = {
            "frozen_independent_layout": POLICY["primary_independent_index_storage"],
            "frozen_unchanged_layout": fixed_method,
            "frozen_approximate": POLICY["primary_exploratory_approximate"],
        }
        if allowed:
            aliases["posthoc_fastest_correct"] = min(allowed, key=lambda row: row["candidate_us"])["candidate"]
        for alias, method in aliases.items():
            if method in case_rows:
                rows.append({**case_rows[method], "candidate": alias, "selected_method": method})
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("input", type=Path)
    cli.add_argument("--allow-partial", action="store_true")
    args = cli.parse_args()
    with args.input.open() as stream:
        cases = [json.loads(line) for line in stream]
    matrix = list(itertools.product((32768, 65536, 131072), (4, 16), LAYOUTS))
    measured = [(case["input"]["length_bucket"], case["input"]["batch"],
                 case["input"]["physical_layout"]) for case in cases]
    if not args.allow_partial and sorted(matrix) != sorted(measured):
        raise ValueError(f"expected the frozen 18-case matrix, got {measured}")
    rows = summarize(cases)
    csv_path = args.input.with_suffix(".csv")
    with csv_path.open("w") as stream:
        fields = dict.fromkeys(key for row in rows for key in row)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    layouts = defaultdict(list)
    for row in rows:
        layouts[row["source_slots"]].append(row)
    summary = {
        "frozen_matrix": len(cases),
        "all_layouts": group_stats(rows, (*CANDIDATES, *POLICY_NAMES)),
        "by_layout": {name: group_stats(items, (*CANDIDATES, *POLICY_NAMES))
                      for name, items in layouts.items()},
    }
    args.input.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
