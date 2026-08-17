#!/usr/bin/env python3
"""Summarize Gate E1 policy deltas by DSA layer without retuning the policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260810
BOOTSTRAP_SAMPLES = 2000
TP = 2
METHODS = ["baseline-rank4096", "resident-protection-v1"]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(
        values,
        size=(BOOTSTRAP_SAMPLES, len(values)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    detail = pd.read_parquet(
        args.analysis_dir / "tables/by_request_layer_step.parquet"
    )
    if set(detail.method) != set(METHODS):
        raise ValueError("expected baseline and resident-protection methods")
    keys = ["dataset", "context_config", "task", "request_id", "layer", "method"]
    request_method = detail.groupby(keys, as_index=False).agg(
        transitions=("step", "size"),
        target_pages=("target_pages", "sum"),
        hbm_hit_pages=("hbm_hit_pages", "sum"),
        target_tokens=("target_tokens", "sum"),
        hbm_hit_tokens=("hbm_hit_tokens", "sum"),
        critical_miss_bytes=("critical_miss_bytes", "sum"),
        total_pcie_bytes=("total_pcie_bytes", "sum"),
    )
    request_method["hbm_page_recall"] = (
        request_method.hbm_hit_pages / request_method.target_pages
    )
    request_method["hbm_token_recall"] = (
        request_method.hbm_hit_tokens / request_method.target_tokens
    )
    request_method["correction_mib_per_token_per_gpu"] = (
        request_method.critical_miss_bytes
        / request_method.transitions
        / TP
        / 2**20
    )
    request_method["total_pcie_mib_per_token_per_gpu"] = (
        request_method.total_pcie_bytes
        / request_method.transitions
        / TP
        / 2**20
    )

    pair_keys = ["dataset", "context_config", "task", "request_id", "layer"]
    metrics = [
        "hbm_page_recall",
        "hbm_token_recall",
        "correction_mib_per_token_per_gpu",
        "total_pcie_mib_per_token_per_gpu",
    ]
    baseline = request_method[request_method.method == METHODS[0]][pair_keys + metrics]
    protected = request_method[request_method.method == METHODS[1]][pair_keys + metrics]
    paired = baseline.merge(
        protected,
        on=pair_keys,
        suffixes=("_baseline", "_protected"),
        validate="one_to_one",
    )
    for metric in metrics:
        paired[f"{metric}_delta"] = (
            paired[f"{metric}_protected"] - paired[f"{metric}_baseline"]
        )
    paired["hbm_page_recall_delta_pp"] = 100 * paired.hbm_page_recall_delta
    paired["hbm_token_recall_delta_pp"] = 100 * paired.hbm_token_recall_delta

    summary_metrics = [
        "hbm_page_recall_delta_pp",
        "hbm_token_recall_delta_pp",
        "correction_mib_per_token_per_gpu_delta",
        "total_pcie_mib_per_token_per_gpu_delta",
    ]
    rng = np.random.default_rng(SEED)
    rows = []
    for layer, part in paired.groupby("layer", sort=True):
        for metric in summary_metrics:
            values = part[metric].to_numpy(float)
            low, high = bootstrap_ci(values, rng)
            rows.append(
                {
                    "layer": int(layer),
                    "metric": metric,
                    "requests": len(part),
                    "mean": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "median": float(np.median(values)),
                    "no_worse_fraction": float(
                        (values >= 0).mean()
                        if "recall" in metric
                        else (values <= 0).mean()
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_by_request_layer.csv", index=False)
    summary.to_csv(args.output_dir / "layer_summary.csv", index=False)
    page = summary[summary.metric == "hbm_page_recall_delta_pp"]
    result = {
        "schema_version": 1,
        "analysis_kind": "post-hoc full48 layer diagnosis; frozen policy unchanged",
        "source_analysis": str(args.analysis_dir.resolve()),
        "requests": int(paired.request_id.nunique()),
        "layers": sorted(paired.layer.astype(int).unique().tolist()),
        "request_layer_pairs": len(paired),
        "bootstrap": {"seed": SEED, "samples": BOOTSTRAP_SAMPLES},
        "page_recall_mean_nonnegative_layers": int((page["mean"] >= 0).sum()),
        "page_recall_mean_negative_layers": page.loc[
            page["mean"] < 0, "layer"
        ].astype(int).tolist(),
        "all_contracts_passed": bool(
            paired.request_id.nunique() * paired.layer.nunique() == len(paired)
            and paired.layer.nunique() == 48
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
