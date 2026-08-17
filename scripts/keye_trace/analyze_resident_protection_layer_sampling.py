#!/usr/bin/env python3
"""Compare seven-layer and full-48 Gate E1 policy deltas on paired requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEED = 20260810
BOOTSTRAP_SAMPLES = 2000
METRICS = [
    "hbm_token_recall_delta_pp",
    "hbm_page_recall_delta_pp",
    "correction_mib_per_token_per_gpu_delta",
    "total_pcie_mib_per_token_per_gpu_delta",
]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full48-analysis", type=Path, required=True)
    parser.add_argument("--sampled-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    relative = Path("tables/paired_by_request.csv")
    full = pd.read_csv(args.full48_analysis / relative)
    sampled = pd.read_csv(args.sampled_analysis / relative)
    keys = ["dataset", "context_config", "task", "request_id"]
    required = keys + METRICS
    merged = full[required].merge(
        sampled[required], on=keys, suffixes=("_full48", "_seven_layer"), validate="one_to_one"
    )
    if len(merged) != len(full) or len(merged) != len(sampled):
        raise ValueError("full48 and seven-layer request sets differ")
    long_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        for metric in METRICS:
            full_value = float(row[f"{metric}_full48"])
            estimate = float(row[f"{metric}_seven_layer"])
            long_rows.append(
                {
                    **{key: row[key] for key in keys},
                    "metric": metric,
                    "full48": full_value,
                    "seven_layer_estimate": estimate,
                    "sampling_error": estimate - full_value,
                    "absolute_sampling_error": abs(estimate - full_value),
                }
            )
    errors = pd.DataFrame(long_rows)
    rng = np.random.default_rng(SEED)
    summary_rows: list[dict[str, Any]] = []
    for values, part in errors.groupby(["dataset", "context_config", "metric"], sort=True):
        vector = part.sampling_error.to_numpy(float)
        low, high = bootstrap_ci(vector, rng)
        task_vector = part.groupby("task", sort=True).sampling_error.mean().to_numpy(float)
        task_low, task_high = bootstrap_ci(task_vector, rng)
        summary_rows.append(
            {
                "dataset": values[0],
                "context_config": values[1],
                "metric": values[2],
                "requests": part.request_id.nunique(),
                "tasks": part.task.nunique(),
                "request_bias_mean": float(vector.mean()),
                "request_bias_ci95_low": low,
                "request_bias_ci95_high": high,
                "request_mae": float(np.abs(vector).mean()),
                "task_balanced_bias_mean": float(task_vector.mean()),
                "task_cluster_bias_ci95_low": task_low,
                "task_cluster_bias_ci95_high": task_high,
            }
        )
    summary = pd.DataFrame(summary_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.output_dir / "sampling_error_by_request.csv", index=False)
    summary.to_csv(args.output_dir / "sampling_error_summary.csv", index=False)
    payload = {
        "schema_version": 1,
        "analysis_kind": "Gate E1 seven-layer estimator error against full48 paired replay",
        "requests": int(merged.request_id.nunique()),
        "metrics": METRICS,
        "full48_analysis": str(args.full48_analysis.resolve()),
        "sampled_analysis": str(args.sampled_analysis.resolve()),
        "bootstrap": {"seed": SEED, "samples": BOOTSTRAP_SAMPLES},
        "all_request_pairs_matched": len(merged) == len(full) == len(sampled),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
