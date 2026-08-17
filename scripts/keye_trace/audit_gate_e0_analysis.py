#!/usr/bin/env python3
"""Independently audit Gate E0 p4 aggregation and layer calibration tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LAYERS = [0, 7, 15, 23, 31, 39, 47]
WEIGHTS = {0: 4, 7: 8, 15: 8, 23: 8, 31: 8, 39: 8, 47: 4}
METRICS = [
    "previous_top4096_token_recall",
    "previous_top4096_page_recall",
    "top4096_page_amplification",
]
REQUEST_KEYS = ["rid", "dataset", "length_config", "task", "prompt_len"]


def request_means(rows: pd.DataFrame, estimator: str) -> pd.DataFrame:
    by_layer = rows.groupby(REQUEST_KEYS + ["layer"], as_index=False)[METRICS].mean()
    output = []
    for values, part in by_layer.groupby(REQUEST_KEYS, sort=True):
        layer_ids = sorted(part.layer.astype(int).tolist())
        if estimator == "full48":
            if layer_ids != list(range(48)):
                raise ValueError(f"{values[0]}: incomplete full48 layer grid")
            weights = np.ones(len(part), dtype=float)
        else:
            if layer_ids != LAYERS:
                raise ValueError(f"{values[0]}: incomplete seven-layer grid")
            weights = part.layer.map(WEIGHTS).to_numpy(float)
        output.append(
            {
                **dict(zip(REQUEST_KEYS, values)),
                "estimator": estimator,
                **{
                    metric: float(np.average(part[metric], weights=weights))
                    for metric in METRICS
                },
            }
        )
    return pd.DataFrame(output)


def frames_close(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    keys: list[str],
) -> tuple[bool, list[str]]:
    expected = expected.sort_values(keys).reset_index(drop=True)
    actual = actual.sort_values(keys).reset_index(drop=True)
    failures: list[str] = []
    if expected[keys].to_dict("records") != actual[keys].to_dict("records"):
        return False, ["key rows differ"]
    for column in expected.columns:
        if column in keys or column not in actual:
            continue
        if pd.api.types.is_numeric_dtype(expected[column]):
            if not np.allclose(
                expected[column].to_numpy(float),
                actual[column].to_numpy(float),
                rtol=1e-10,
                atol=1e-10,
            ):
                failures.append(column)
        elif not expected[column].astype(str).equals(actual[column].astype(str)):
            failures.append(column)
    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--full48-table", type=Path, required=True)
    parser.add_argument("--full48-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = json.loads((args.analysis_dir / "summary.json").read_text())
    full48_summary = json.loads(args.full48_summary.read_text())
    tables = args.analysis_dir / "tables"
    expanded = pd.read_parquet(tables / "expanded_by_request_layer_step.parquet")
    full48 = pd.read_parquet(args.full48_table)
    saved_requests = pd.read_csv(tables / "request_metrics.csv")
    saved_calibration = pd.read_csv(
        tables / "layer_sampling_calibration_by_request.csv"
    )
    failures: list[str] = []

    complete_requests = int(summary["coverage"]["complete_requests"])
    expected_rows = complete_requests * len(LAYERS) * 31
    duplicate_rows = int(expanded.duplicated(["rid", "layer", "step"]).sum())
    if len(expanded) != expected_rows:
        failures.append(f"expanded rows={len(expanded)} expected={expected_rows}")
    if duplicate_rows:
        failures.append(f"duplicate expanded request/layer/step rows={duplicate_rows}")
    if set(expanded.layer.astype(int)) != set(LAYERS):
        failures.append("expanded layer set differs from frozen seven layers")
    if set(expanded.step.astype(int)) != set(range(1, 32)):
        failures.append("expanded transitions are not exactly steps 1..31")
    if not np.isfinite(expanded.select_dtypes(include="number").to_numpy(float)).all():
        failures.append("expanded table contains non-finite numeric values")
    if int(summary.get("page_size", -1)) != 4:
        failures.append("Gate E0 summary is not p4")
    if int(full48_summary.get("page_size", -1)) != 4:
        failures.append("full48 layer calibration is not p4")
    if not bool(full48_summary.get("all_contracts_passed")):
        failures.append("full48 layer calibration contracts failed")

    expanded_requests = request_means(
        expanded, "seven_layer_nearest_weighted"
    )
    full48_requests = request_means(full48, "full48")
    recomputed_requests = pd.concat(
        [full48_requests, expanded_requests], ignore_index=True
    )
    requests_match, request_columns = frames_close(
        recomputed_requests,
        saved_requests,
        REQUEST_KEYS + ["estimator"],
    )
    if not requests_match:
        failures.append(f"request_metrics mismatch: {request_columns}")

    sampled_full48 = request_means(
        full48[full48.layer.isin(LAYERS)].copy(),
        "seven_layer_nearest_weighted",
    )
    paired = full48_requests.merge(
        sampled_full48,
        on=REQUEST_KEYS,
        suffixes=("_full48", "_estimate"),
        validate="one_to_one",
    )
    calibration_rows = []
    for _, row in paired.iterrows():
        for metric in METRICS:
            calibration_rows.append(
                {
                    **{key: row[key] for key in REQUEST_KEYS},
                    "metric": metric,
                    "full48": row[f"{metric}_full48"],
                    "estimate": row[f"{metric}_estimate"],
                    "error": row[f"{metric}_estimate"]
                    - row[f"{metric}_full48"],
                }
            )
    recomputed_calibration = pd.DataFrame(calibration_rows)
    calibration_match, calibration_columns = frames_close(
        recomputed_calibration,
        saved_calibration,
        REQUEST_KEYS + ["metric"],
    )
    if not calibration_match:
        failures.append(f"calibration mismatch: {calibration_columns}")

    contracts = {
        "expanded_complete_grid": len(expanded) == expected_rows
        and duplicate_rows == 0,
        "expanded_finite": bool(
            np.isfinite(expanded.select_dtypes(include="number").to_numpy(float)).all()
        ),
        "expanded_page_size_is_four": int(summary.get("page_size", -1)) == 4,
        "full48_page_size_is_four": int(full48_summary.get("page_size", -1)) == 4,
        "request_metrics_recomputed": requests_match,
        "layer_calibration_recomputed": calibration_match,
    }
    result = {
        "schema_version": 1,
        "analysis_dir": str(args.analysis_dir.resolve()),
        "full48_table": str(args.full48_table.resolve()),
        "coverage": {
            "expanded_requests": complete_requests,
            "expanded_rows": len(expanded),
            "full48_requests": int(full48.rid.nunique()),
            "full48_rows": len(full48),
        },
        "contracts": contracts,
        "failures": failures,
        "passed": not failures and all(contracts.values()),
    }
    output = args.output or args.analysis_dir / "independent_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
