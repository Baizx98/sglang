#!/usr/bin/env python3
"""Summarize Gate E0 K=4096 reuse across datasets and layer sampling.

Full-48 traces are treated as measurements. Expanded seven-layer traces use
nearest-layer weighting and remain explicitly labelled as estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

LAYERS = [0, 7, 15, 23, 31, 39, 47]
WEIGHTS = {0: 4, 7: 8, 15: 8, 23: 8, 31: 8, 39: 8, 47: 4}
METRICS = [
    "previous_top4096_token_recall",
    "previous_top4096_page_recall",
    "top4096_page_amplification",
]
SEED = 20260810
BOOTSTRAP_SAMPLES = 2000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def pages(indices: np.ndarray, page_size: int) -> np.ndarray:
    return np.unique(indices[indices >= 0] // page_size)


def intersection_ratio(left: np.ndarray, right: np.ndarray) -> float:
    if not right.size:
        raise ValueError("empty target set")
    return float(np.intersect1d(left, right, assume_unique=True).size / right.size)


def analyze_chunk(path: Path, page_size: int) -> list[dict[str, float | int]]:
    record = torch.load(path, weights_only=False)
    if int(record["schema_version"]) != 5 or record["trace_mode"] != "compact":
        raise ValueError(f"{path}: expected compact trace schema v5")
    if int(record["compact_k"]) != 4096:
        raise ValueError(f"{path}: expected compact_k=4096")
    steps = [int(value) for value in record["decode_step_ids"]]
    if steps != list(range(32)):
        raise ValueError(f"{path}: expected decode steps 0..31")
    canonical = record["indices"].numpy(force=True).astype(np.int32)
    candidates = record["candidate_indices"].numpy(force=True).astype(np.int32)
    valid_counts = record["score_valid_counts"].numpy(force=True).astype(np.int32)
    rows: list[dict[str, float | int]] = []
    for step in range(1, 32):
        common = min(int(valid_counts[step - 1]), int(valid_counts[step]))
        target = np.unique(canonical[step][(canonical[step] >= 0) & (canonical[step] < common)])
        previous = np.unique(
            candidates[step - 1][
                (candidates[step - 1] >= 0) & (candidates[step - 1] < common)
            ]
        )
        target_pages = pages(target, page_size)
        previous_pages = pages(previous, page_size)
        rows.append(
            {
                "step": step,
                "previous_top4096_token_recall": intersection_ratio(previous, target),
                "previous_top4096_page_recall": intersection_ratio(
                    previous_pages, target_pages
                ),
                "top4096_page_amplification": float(
                    previous_pages.size / target_pages.size
                ),
            }
        )
    return rows


def expanded_metadata(run_dir: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*/prepared_requests.jsonl")):
        if path.parent.name == "full48-paired":
            continue
        for row in read_jsonl(path):
            rid = str(row["rid"])
            if rid in metadata:
                raise ValueError(f"duplicate request ID: {rid}")
            metadata[rid] = {
                "dataset": str(row["dataset"]),
                "length_config": int(row["length_config"]),
                "task": str(row["task"]),
                "prompt_len": int(row["prompt_len"]),
            }
    if not metadata:
        raise ValueError(f"no expanded prepared requests below {run_dir}")
    return metadata


def expanded_rows(
    run_dir: Path,
    page_size: int,
    allow_incomplete: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = expanded_metadata(run_dir)
    manifest = read_jsonl(run_dir / "events" / "manifest.jsonl")
    lookup: dict[tuple[str, int], Path] = {}
    for row in manifest:
        rid = str(row["request_id"])
        if rid not in metadata:
            continue
        key = (rid, int(row["layer_id"]))
        if key in lookup:
            raise ValueError(f"duplicate expanded chunk: {key}")
        lookup[key] = run_dir / "events" / str(row["file"])

    complete_rids = sorted(
        rid
        for rid in metadata
        if all((rid, layer) in lookup for layer in LAYERS)
    )
    partial_rids = sorted(
        rid
        for rid in metadata
        if 0 < sum((rid, layer) in lookup for layer in LAYERS) < len(LAYERS)
    )
    missing_rids = sorted(set(metadata) - set(complete_rids) - set(partial_rids))
    if partial_rids:
        raise ValueError(f"partially written requests: {partial_rids}")
    if not allow_incomplete and missing_rids:
        raise ValueError(f"requests without complete traces: {len(missing_rids)}")

    rows: list[dict[str, Any]] = []
    for rid in complete_rids:
        for layer in LAYERS:
            for row in analyze_chunk(lookup[(rid, layer)], page_size):
                rows.append({"rid": rid, "layer": layer, **metadata[rid], **row})
    return pd.DataFrame(rows), {
        "prepared_requests": len(metadata),
        "complete_requests": len(complete_rids),
        "missing_requests": len(missing_rids),
        "partial_requests": len(partial_rids),
        "chunks": len(complete_rids) * len(LAYERS),
    }


def request_means(layer_steps: pd.DataFrame, estimator: str) -> pd.DataFrame:
    keys = ["rid", "dataset", "length_config", "task", "prompt_len"]
    by_layer = layer_steps.groupby(keys + ["layer"], as_index=False)[METRICS].mean()
    rows: list[dict[str, Any]] = []
    for values, part in by_layer.groupby(keys, sort=True):
        layers = sorted(part.layer.astype(int).tolist())
        if estimator == "full48":
            if layers != list(range(48)):
                raise ValueError(f"{values[0]}: incomplete full48 layers")
            weights = np.ones(len(part))
        else:
            if layers != LAYERS:
                raise ValueError(f"{values[0]}: incomplete sampled layers")
            weights = part.layer.map(WEIGHTS).to_numpy(float)
        base = dict(zip(keys, values))
        rows.append(
            {
                **base,
                "estimator": estimator,
                **{
                    metric: float(np.average(part[metric], weights=weights))
                    for metric in METRICS
                },
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(
        values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True
    ).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def group_summary(requests: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    keys = ["estimator", "dataset", "length_config"]
    for values, part in requests.groupby(keys, sort=True):
        for metric in METRICS:
            vector = part[metric].to_numpy(float)
            low, high = bootstrap_mean(vector, rng)
            task_vector = part.groupby("task", sort=True)[metric].mean().to_numpy(float)
            task_low, task_high = bootstrap_mean(task_vector, rng)
            rows.append(
                {
                    **dict(zip(keys, values)),
                    "metric": metric,
                    "requests": len(part),
                    "tasks": part.task.nunique(),
                    "mean": float(vector.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "median": float(np.median(vector)),
                    "p10": float(np.quantile(vector, 0.10)),
                    "p90": float(np.quantile(vector, 0.90)),
                    "task_balanced_mean": float(task_vector.mean()),
                    "task_cluster_ci95_low": task_low,
                    "task_cluster_ci95_high": task_high,
                }
            )
    return pd.DataFrame(rows)


def calibration_table(full48_steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = request_means(full48_steps, "full48")
    sampled = request_means(
        full48_steps[full48_steps.layer.isin(LAYERS)].copy(),
        "seven_layer_nearest_weighted",
    )
    keys = ["rid", "dataset", "length_config", "task", "prompt_len"]
    paired = actual.merge(sampled, on=keys, suffixes=("_full48", "_estimate"))
    long_rows: list[dict[str, Any]] = []
    for _, row in paired.iterrows():
        for metric in METRICS:
            long_rows.append(
                {
                    **{key: row[key] for key in keys},
                    "metric": metric,
                    "full48": row[f"{metric}_full48"],
                    "estimate": row[f"{metric}_estimate"],
                    "error": row[f"{metric}_estimate"] - row[f"{metric}_full48"],
                }
            )
    errors = pd.DataFrame(long_rows)
    rng = np.random.default_rng(SEED)
    summary: list[dict[str, Any]] = []
    for values, part in errors.groupby(
        ["dataset", "length_config", "metric"], sort=True
    ):
        vector = part.error.to_numpy(float)
        low, high = bootstrap_mean(vector, rng)
        summary.append(
            {
                "dataset": values[0],
                "length_config": values[1],
                "metric": values[2],
                "requests": len(part),
                "bias_mean": float(vector.mean()),
                "bias_ci95_low": low,
                "bias_ci95_high": high,
                "request_mae": float(np.abs(vector).mean()),
            }
        )
    return errors, pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument(
        "--full48-table",
        type=Path,
        help="p4 all-layer by-request/layer/step table used for calibration",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    expanded_steps, coverage = expanded_rows(
        args.run_dir, args.page_size, args.allow_incomplete
    )
    if expanded_steps.empty:
        raise ValueError("no complete expanded requests")
    full48_path = args.full48_table or (
        args.run_dir
        / "full48-paired/analysis/layer-sampling-bias-p4-v01/tables/by_request_layer_step.parquet"
    )
    full48_steps = pd.read_parquet(full48_path)
    full48_requests = request_means(full48_steps, "full48")
    expanded_requests = request_means(
        expanded_steps, "seven_layer_nearest_weighted"
    )
    requests = pd.concat([full48_requests, expanded_requests], ignore_index=True)
    calibration, calibration_summary = calibration_table(full48_steps)

    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    expanded_steps.to_parquet(table_dir / "expanded_by_request_layer_step.parquet", index=False)
    requests.to_csv(table_dir / "request_metrics.csv", index=False)
    group_summary(requests).to_csv(table_dir / "dataset_length_summary.csv", index=False)
    calibration.to_csv(table_dir / "layer_sampling_calibration_by_request.csv", index=False)
    calibration_summary.to_csv(table_dir / "layer_sampling_calibration_summary.csv", index=False)
    summary = {
        "schema_version": 1,
        "page_size": args.page_size,
        "sampled_layers": LAYERS,
        "nearest_layer_weights": WEIGHTS,
        "metrics": METRICS,
        "bootstrap": {
            "seed": SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "reported_intervals": [
                "request bootstrap",
                "task-cluster bootstrap over task means",
            ],
        },
        "coverage": coverage,
        "full48_paired_requests": int(full48_requests.rid.nunique()),
        "expanded_estimator": "seven_layer_nearest_weighted",
        "interpretation": {
            "full48": "measurement",
            "seven_layer_nearest_weighted": "estimate; consult calibration tables",
        },
        "structural_contracts_passed": not coverage["partial_requests"],
        "coverage_complete": not coverage["missing_requests"],
        "all_contracts_passed": not coverage["partial_requests"]
        and not coverage["missing_requests"],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
