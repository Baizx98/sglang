"""Extract complete-Indexer and append checks from the registered A6000 holdout runs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def passed(row, name):
    checks = row["evaluations"][name]
    return all(check["valid"] for check in checks["sgl"]) and not checks.get("append_overflow_rows", 0)


def select_names(metadata):
    batch = metadata["batch"]
    bucket = metadata["length_bucket"]
    suffix = "vllm_persistent" if batch <= 4 or bucket == 32768 else "flashinfer_topk"
    block = 64 if bucket == 32768 else 128
    dense = f"full_direct_dense_n{block}w4_{suffix}"
    packed = f"full_direct_pack12_n128w4_{suffix}"
    return suffix, dense, packed


def summarize_row(row, profile, version):
    timing = row["timings"]
    metadata = row["input"]
    suffix, dense, packed = select_names(metadata)
    policy = dense if metadata["length_bucket"] == 32768 else packed
    baseline = min(
        (name for name in timing if name.startswith("full_sgl_score_") and passed(row, name)),
        key=lambda name: timing[name]["median_us"],
    )
    same_selector = f"full_sgl_score_{suffix}"
    speed = {name: timing[name]["median_us"] for name in (policy, baseline, dense, same_selector)}
    storage = row["storage"]
    packed_storage = row["evaluations"][packed]["packed_index_k_bytes"]
    initial = storage["packed_initial_build_wall_median_us"]
    raw = storage["raw_index_k_bytes"]
    saving = speed[baseline] - speed[policy]
    layout = storage.get("page_layout_stats", {})
    return {
        "profile": profile, "flashinfer": version, "layer": metadata["layer"],
        "step": metadata["step"], "length_bucket": metadata["length_bucket"],
        "batch": metadata["batch"], "layout": metadata["physical_layout"],
        "max_sequence_length": max(metadata["sequence_lengths"]),
        "baseline": baseline, "baseline_us": speed[baseline],
        "baseline_mad_us": timing[baseline]["mad_us"],
        "policy": policy, "policy_us": speed[policy],
        "policy_mad_us": timing[policy]["mad_us"],
        "policy_kept_exact": passed(row, policy),
        "speedup_vs_best_baseline": speed[baseline] / speed[policy],
        "native_same_selector_us": speed[same_selector],
        "tuned_dense_same_selector_us": speed[dense],
        "speedup_vs_tuned_dense_same_selector": speed[dense] / speed[policy],
        "fast_topk_v2_valid": passed(row, "full_sgl_score_fast_topk_v2"),
        "device_projection_and_write_us": timing["project_and_write"]["median_us"],
        "direct_packed_write_us": timing["direct_pack12_write"]["median_us"],
        "descriptor_rebuild_us": timing["run_describe_full_n128"]["median_us"],
        "descriptor_tail_us": timing["run_describe_tail_n128"]["median_us"],
        "raw_index_k_bytes": raw, "packed_index_k_bytes": packed_storage,
        "packed_exception_slots": storage["packed_exception_slots"],
        "packed_size_ratio": packed_storage / raw,
        "physical_edge_adjacency": layout["consecutive_neighbor_pairs"] / layout["valid_neighbor_pairs"]
        if layout.get("valid_neighbor_pairs") else "",
        "consecutive_tile_share": layout["consecutive_tiles_128"] / layout["live_tiles_128"]
        if layout.get("live_tiles_128") else "",
        "pack_initial_build_wall_us": initial,
        "wall_encode_over_gpu_step_savings": initial / saving if saving > 0 and policy == packed else "",
    }


def cross_release(rows):
    matched = defaultdict(list)
    for row in rows:
        case = tuple(row[key] for key in ("layer", "step", "length_bucket", "batch", "layout"))
        matched[case].append(row)
    for versions in matched.values():
        if len(versions) != 2:
            continue
        baseline = min(versions, key=lambda row: row["baseline_us"])
        for row in versions:
            row["matched_release_fastest_baseline"] = baseline["baseline"]
            row["matched_release_fastest_baseline_flashinfer"] = baseline["flashinfer"]
            row["matched_release_fastest_baseline_us"] = baseline["baseline_us"]
            row["speedup_vs_matched_release_fastest_baseline"] = baseline["baseline_us"] / row["policy_us"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    profiles = (
        "heldout-old-full128", "heldout-old-full-short", "heldout-old-mixed",
        "heldout-new-full128", "heldout-new-full-short", "heldout-new-mixed",
    )
    summaries = []
    for profile in profiles:
        path = args.results / f"{profile}.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            continue
        version = json.loads(path.with_suffix(".metadata.json").read_text())["software"]["flashinfer_imported"]
        summaries.extend(summarize_row(row, profile, version) for row in read_rows(path))
    if summaries:
        cross_release(summaries)
        fields = dict.fromkeys(key for row in summaries for key in row)
        with (args.results / "frozen-complete-indexer.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == "__main__":
    main()
