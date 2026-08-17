#!/usr/bin/env python3
"""Independently audit Gate E1 resident-page protection analysis outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 20260810
BOOTSTRAP_SAMPLES = 2000
METHODS = ["baseline-rank4096", "resident-protection-v1"]
REQUEST_KEYS = ["dataset", "context_config", "task", "request_id", "method"]
PAIR_KEYS = ["dataset", "context_config", "task", "request_id"]
TRANSFER_COLUMNS = [
    ("prefetch_bytes", "prefetch_mib_per_token_per_gpu"),
    ("critical_miss_bytes", "correction_mib_per_token_per_gpu"),
    ("total_pcie_bytes", "total_pcie_mib_per_token_per_gpu"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: Any, right: Any, *, atol: float = 1e-10) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=1e-10, atol=atol))


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(
        values,
        size=(BOOTSTRAP_SAMPLES, len(values)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def recompute_requests(detail: pd.DataFrame, layers: list[int], tp: int) -> pd.DataFrame:
    sums = detail.groupby(REQUEST_KEYS, as_index=False).agg(
        transitions=("step", "size"),
        target_tokens=("target_tokens", "sum"),
        target_pages=("target_pages", "sum"),
        hbm_hit_tokens=("hbm_hit_tokens", "sum"),
        hbm_hit_pages=("hbm_hit_pages", "sum"),
        prefetch_bytes=("prefetch_bytes", "sum"),
        critical_miss_bytes=("critical_miss_bytes", "sum"),
        total_pcie_bytes=("total_pcie_bytes", "sum"),
        protected_resident_pages=("protected_resident_pages", "sum"),
        protected_pages_outside_baseline=("protected_pages_outside_baseline", "sum"),
        protection_swaps=("protection_swaps", "sum"),
        unused_prefetch_pages=("unused_prefetch_pages", "sum"),
        pollution_miss_pages=("pollution_miss_pages", "sum"),
    )
    sums["hbm_token_recall"] = sums.hbm_hit_tokens / sums.target_tokens
    sums["hbm_page_recall"] = sums.hbm_hit_pages / sums.target_pages
    scale = 48 / len(layers) / tp / 2**20
    decode_tokens = sums.transitions / len(layers)
    for source, target in TRANSFER_COLUMNS:
        sums[target] = sums[source] * scale / decode_tokens
    return sums


def compare_frames(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    keys: list[str],
    failures: list[str],
    label: str,
) -> None:
    expected = expected.sort_values(keys).reset_index(drop=True)
    actual = actual.sort_values(keys).reset_index(drop=True)
    if expected[keys].to_dict("records") != actual[keys].to_dict("records"):
        failures.append(f"{label}: key rows differ")
        return
    common = [column for column in expected.columns if column in actual.columns and column not in keys]
    for column in common:
        left = expected[column]
        right = actual[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=1e-10, atol=1e-10):
                failures.append(f"{label}: numeric mismatch in {column}")
        elif not left.astype(str).equals(right.astype(str)):
            failures.append(f"{label}: value mismatch in {column}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    summary = json.loads((args.analysis_dir / "summary.json").read_text())
    policy = json.loads(args.policy.read_text())
    tables = args.analysis_dir / "tables"
    detail = pd.read_parquet(tables / "by_request_layer_step.parquet")
    requests = pd.read_csv(tables / "by_request_method.csv")
    paired = pd.read_csv(tables / "paired_by_request.csv")

    layers = [int(value) for value in summary["setting"]["layers"]]
    tp = int(summary["setting"]["tensor_parallel_size"])
    expected_requests = int(summary["coverage"]["complete_trace_requests"])
    expected_rows = expected_requests * len(layers) * 31 * len(METHODS)
    if len(detail) != expected_rows:
        failures.append(f"detail rows={len(detail)} expected={expected_rows}")
    if set(detail.method) != set(METHODS):
        failures.append(f"methods={sorted(set(detail.method))} expected={METHODS}")
    if set(detail.layer.astype(int)) != set(layers):
        failures.append("detail layers differ from summary")
    if set(detail.step.astype(int)) != set(range(31)):
        failures.append("detail steps are not exactly 0..30")
    duplicate_count = int(
        detail.duplicated(["request_id", "layer", "step", "method"]).sum()
    )
    if duplicate_count:
        failures.append(f"duplicate request/layer/step/method rows={duplicate_count}")
    numeric = detail.select_dtypes(include="number").to_numpy(float)
    if not np.isfinite(numeric).all():
        failures.append("detail contains non-finite numeric values")

    method_pairs = detail.pivot(
        index=["request_id", "layer", "step"],
        columns="method",
        values=["target_tokens", "target_pages"],
    )
    demand_equal = True
    for metric in ["target_tokens", "target_pages"]:
        if not (
            method_pairs[(metric, METHODS[0])]
            == method_pairs[(metric, METHODS[1])]
        ).all():
            demand_equal = False
            failures.append(f"{metric} differs between baseline and protected replay")

    baseline = detail[detail.method == METHODS[0]]
    protected = detail[detail.method == METHODS[1]]
    for column in [
        "protected_resident_pages",
        "protected_pages_outside_baseline",
        "protection_swaps",
    ]:
        if bool((baseline[column] != 0).any()):
            failures.append(f"baseline has nonzero {column}")
    page_budget_ok = bool(
        (protected.prefetch_admit_pages <= protected.counterfactual_prefetch_page_budget).all()
    )
    byte_budget_ok = bool(
        (protected.prefetch_bytes <= protected.counterfactual_prefetch_byte_budget).all()
    )
    if not page_budget_ok:
        failures.append("protected replay exceeds same-state K4096 page budget")
    if not byte_budget_ok:
        failures.append("protected replay exceeds same-state K4096 byte budget")

    recomputed_requests = recompute_requests(detail, layers, tp)
    request_failure_start = len(failures)
    compare_frames(
        recomputed_requests,
        requests,
        REQUEST_KEYS,
        failures,
        "by_request_method",
    )
    requests_match = len(failures) == request_failure_start

    baseline_requests = recomputed_requests[
        recomputed_requests.method == METHODS[0]
    ].drop(columns="method")
    protected_requests = recomputed_requests[
        recomputed_requests.method == METHODS[1]
    ].drop(columns="method")
    recomputed_paired = baseline_requests.merge(
        protected_requests,
        on=PAIR_KEYS,
        suffixes=("_baseline", "_protected"),
        validate="one_to_one",
    )
    paired_metrics = [
        "hbm_token_recall",
        "hbm_page_recall",
        "prefetch_mib_per_token_per_gpu",
        "correction_mib_per_token_per_gpu",
        "total_pcie_mib_per_token_per_gpu",
        "protected_resident_pages",
        "protected_pages_outside_baseline",
        "protection_swaps",
        "unused_prefetch_pages",
        "pollution_miss_pages",
    ]
    for metric in paired_metrics:
        recomputed_paired[f"{metric}_delta"] = (
            recomputed_paired[f"{metric}_protected"]
            - recomputed_paired[f"{metric}_baseline"]
        )
    recomputed_paired["hbm_token_recall_delta_pp"] = (
        100 * recomputed_paired.hbm_token_recall_delta
    )
    recomputed_paired["hbm_page_recall_delta_pp"] = (
        100 * recomputed_paired.hbm_page_recall_delta
    )
    paired_failure_start = len(failures)
    compare_frames(
        recomputed_paired,
        paired,
        PAIR_KEYS,
        failures,
        "paired_by_request",
    )
    paired_match = len(failures) == paired_failure_start

    correction = recomputed_paired.correction_mib_per_token_per_gpu_delta.to_numpy(float)
    correction_ci = bootstrap_ci(correction)
    overall = summary["overall"]
    recomputed_overall = {
        "requests": len(recomputed_paired),
        "hbm_page_recall_delta_pp_mean": float(
            recomputed_paired.hbm_page_recall_delta_pp.mean()
        ),
        "hbm_page_recall_no_worse_fraction": float(
            (recomputed_paired.hbm_page_recall_delta_pp >= 0).mean()
        ),
        "hbm_page_recall_delta_pp_worst": float(
            recomputed_paired.hbm_page_recall_delta_pp.min()
        ),
        "correction_mib_per_token_per_gpu_delta_mean": float(correction.mean()),
        "total_pcie_mib_per_token_per_gpu_delta_mean": float(
            recomputed_paired.total_pcie_mib_per_token_per_gpu_delta.mean()
        ),
        "requests_with_any_protection_swap": int(
            (recomputed_paired.protection_swaps_protected > 0).sum()
        ),
    }
    for key, value in recomputed_overall.items():
        if key not in overall or not close(overall[key], value):
            failures.append(f"summary overall mismatch: {key}")
    if not all(
        close(left, right)
        for left, right in zip(overall["correction_delta_ci95"], correction_ci)
    ):
        failures.append("summary correction CI mismatch")

    gate = policy["gate"]
    recomputed_gate = {
        "mean_correction_ci95_high": correction_ci[1]
        <= float(gate["mean_correction_byte_delta_request_bootstrap_ci95_high_max"]),
        "mean_total_pcie_not_higher": recomputed_overall[
            "total_pcie_mib_per_token_per_gpu_delta_mean"
        ]
        <= float(gate["mean_total_pcie_byte_delta_max"]),
        "request_hbm_page_recall_no_worse_fraction": recomputed_overall[
            "hbm_page_recall_no_worse_fraction"
        ]
        >= float(gate["request_hbm_page_recall_no_worse_fraction_min"]),
        "worst_request_hbm_page_recall_delta_pp": recomputed_overall[
            "hbm_page_recall_delta_pp_worst"
        ]
        >= float(gate["worst_request_hbm_page_recall_delta_pp_min"]),
    }
    if recomputed_gate != summary["gate_checks"]:
        failures.append("summary gate checks do not match independent recomputation")
    expected_gate_passed = bool(
        all(recomputed_gate.values())
        and bool(summary["coverage_complete"])
        and page_budget_ok
        and byte_budget_ok
    )
    if bool(summary["gate_passed"]) != expected_gate_passed:
        failures.append("summary gate_passed does not match audited contracts")
    policy_hash_ok = sha256(args.policy) == summary["policy_sha256"]
    if not policy_hash_ok:
        failures.append("policy SHA-256 differs from summary")

    result = {
        "schema_version": 1,
        "analysis_dir": str(args.analysis_dir.resolve()),
        "policy": str(args.policy.resolve()),
        "coverage": {
            "requests": expected_requests,
            "layers": len(layers),
            "transitions_per_layer": 31,
            "methods": len(METHODS),
            "detail_rows": len(detail),
        },
        "contracts": {
            "unique_complete_grid": duplicate_count == 0 and len(detail) == expected_rows,
            "finite_numeric_values": bool(np.isfinite(numeric).all()),
            "identical_exact_demand_between_methods": demand_equal,
            "same_state_page_budget_not_exceeded": page_budget_ok,
            "same_state_byte_budget_not_exceeded": byte_budget_ok,
            "request_tables_recomputed": requests_match,
            "paired_deltas_recomputed": paired_match,
            "gate_recomputed": recomputed_gate == summary["gate_checks"],
            "policy_hash_matched": policy_hash_ok,
        },
        "recomputed_overall": recomputed_overall,
        "recomputed_correction_ci95": correction_ci,
        "recomputed_gate_checks": recomputed_gate,
        "failures": failures,
        "passed": not failures,
    }
    output = args.output or args.analysis_dir / "independent_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
