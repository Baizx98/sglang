#!/usr/bin/env python3
"""Analyze previous-step score-ranked K versus next-step top-k coverage."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from cycler import cycler
from matplotlib.lines import Line2D

TARGET_K = 2048
ABSOLUTE_K = [512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072]
FRACTIONS = [0.25, 0.35, 0.50, 0.65, 0.80]
REQUIRED_Q = [0.80, 0.90, 0.95, 0.99]
CDF_K = [1024, 1536, 2048, 2560, 3072]
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_SAMPLES = 2000
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
METHOD_COLORS = {
    "previous_score_rank": "#0072B2",
    "recency": "#D55E00",
    "random_expectation": "#9E9E9E",
    "current_score_oracle": "#4D4D4D",
}
METHOD_LABELS = {
    "previous_score_rank": "Previous score rank",
    "recency": "Recency",
    "random_expectation": "Random expectation",
    "current_score_oracle": "Current-score oracle",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def load_chunk(path: Path) -> dict[str, np.ndarray]:
    record = torch.load(path, weights_only=False)
    if int(record["schema_version"]) != 4:
        raise ValueError(f"{path}: expected schema v4")
    if record["decode_step_ids"] != list(range(32)):
        raise ValueError(f"{path}: expected decode steps 0..31")
    return {
        "scores": record["scores"].numpy(force=True).astype(np.float32),
        "indices": record["indices"].numpy(force=True).astype(np.int32),
        "score_lens": record["score_valid_counts"]
        .numpy(force=True)
        .astype(np.int32),
    }


def build_lookup(
    run_dir: Path, requests: list[dict[str, Any]]
) -> dict[tuple[str, int], Path]:
    expected_rids = {row["rid"] for row in requests}
    rows = [
        row
        for row in read_jsonl(run_dir / "events" / "manifest.jsonl")
        if row.get("request_id") in expected_rids
    ]
    lookup: dict[tuple[str, int], Path] = {}
    for row in rows:
        key = (row["request_id"], int(row["layer_id"]))
        if key in lookup:
            raise ValueError(f"duplicate chunk: {key}")
        lookup[key] = run_dir / "events" / row["file"]
    expected = len(requests) * 48
    if len(lookup) != expected:
        raise ValueError(f"expected {expected} chunks, found {len(lookup)}")
    return lookup


def tie_audit(
    scores: np.ndarray, order: np.ndarray, stored: np.ndarray
) -> tuple[int, int, int, float]:
    common = len(scores)
    stored = stored[(stored >= 0) & (stored < common)].astype(np.int64)
    if len(stored) != TARGET_K or len(np.unique(stored)) != TARGET_K:
        raise ValueError("stored previous top-k must contain 2048 unique common tokens")
    stored_mask = np.zeros(common, dtype=bool)
    stored_mask[stored] = True
    derived = order[:TARGET_K]
    missing_from_stored = derived[~stored_mask[derived]]
    mismatch = int(2 * len(missing_from_stored))
    if mismatch == 0:
        threshold = scores[derived[-1]]
        return 0, 0, int(np.count_nonzero(scores == threshold)), 0.0

    derived_mask = np.zeros(common, dtype=bool)
    derived_mask[derived] = True
    missing_from_derived = stored[~derived_mask[stored]]
    threshold = scores[derived[-1]]
    non_tie = int(
        np.count_nonzero(scores[missing_from_stored] != threshold)
        + np.count_nonzero(scores[missing_from_derived] != threshold)
    )
    # Positive regret means the online fast_topk result retained a lower-score
    # token than the exact FP32 ranking. The CUDA kernel's own test permits a
    # small number of boundary replacements, so record this divergence instead
    # of silently treating the saved indices as exact score order.
    boundary_regret = max(
        0.0,
        float(scores[missing_from_stored].max() - scores[missing_from_derived].min()),
    )
    return (
        mismatch,
        non_tie,
        int(np.count_nonzero(scores == threshold)),
        boundary_regret,
    )


def required_k(sorted_target_ranks: np.ndarray, quantile: float) -> int:
    count = len(sorted_target_ranks)
    needed = max(1, math.ceil(quantile * count))
    return int(sorted_target_ranks[needed - 1] + 1)


def analyze_transitions(
    run_dir: Path,
    requests: list[dict[str, Any]],
    lookup: dict[tuple[str, int], Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exact_match_transitions = 0
    tie_only_transitions = 0
    non_tie_transitions = 0
    mismatch_entries = 0
    non_tie_mismatch_entries = 0
    max_replacements = 0
    max_boundary_regret = 0.0
    max_tie_block = 0
    new_token_selected = 0
    target_counts: defaultdict[int, int] = defaultdict(int)

    total_chunks = len(requests) * 48
    chunk_index = 0
    for request in requests:
        rid = request["rid"]
        for layer in range(48):
            chunk_index += 1
            chunk = load_chunk(lookup[(rid, layer)])
            scores = chunk["scores"]
            indices = chunk["indices"]
            lens = chunk["score_lens"]
            for step in range(31):
                common = min(int(lens[step]), int(lens[step + 1]))
                previous_scores = scores[step, :common]
                if not np.isfinite(previous_scores).all():
                    raise ValueError(f"non-finite score: {rid}, L{layer}, step {step}")
                order = np.argsort(-previous_scores, kind="stable")
                ranks = np.empty(common, dtype=np.int32)
                ranks[order] = np.arange(common, dtype=np.int32)

                next_indices = indices[step + 1]
                target = next_indices[(next_indices >= 0) & (next_indices < common)]
                target = np.unique(target.astype(np.int64))
                target_count = len(target)
                if target_count not in {TARGET_K - 1, TARGET_K}:
                    raise ValueError(
                        f"unexpected common target count {target_count}: "
                        f"{rid}, L{layer}, step {step}"
                    )
                selected_new = int(
                    np.count_nonzero((next_indices >= common) & (next_indices < lens[step + 1]))
                )
                if target_count + selected_new != TARGET_K:
                    raise ValueError("next top-k does not partition into common + new")
                new_token_selected += selected_new
                target_counts[target_count] += 1

                target_ranks = np.sort(ranks[target])
                mismatch, non_tie, tie_block, boundary_regret = tie_audit(
                    previous_scores, order, indices[step]
                )
                if non_tie:
                    non_tie_transitions += 1
                    non_tie_mismatch_entries += mismatch
                elif mismatch:
                    tie_only_transitions += 1
                else:
                    exact_match_transitions += 1
                mismatch_entries += mismatch
                replacements = mismatch // 2
                if replacements > 5:
                    raise ValueError(
                        "fast_topk differs from exact FP32 ranking by more than "
                        f"the kernel-test tolerance: {rid}, L{layer}, step {step}, "
                        f"replacements={replacements}"
                    )
                max_replacements = max(max_replacements, replacements)
                max_boundary_regret = max(max_boundary_regret, boundary_regret)
                max_tie_block = max(max_tie_block, tie_block)

                row: dict[str, Any] = {
                    "rid": rid,
                    "trajectory_id": request["trajectory_id"],
                    "category": request["category"],
                    "round_id": int(request["round_id"]),
                    "layer": layer,
                    "step": step,
                    "common_len": common,
                    "target_common_count": target_count,
                    "new_token_selected": bool(selected_new),
                }
                previous_coverage: list[float] = []
                previous_recency: list[float] = []
                for k in ABSOLUTE_K:
                    covered = int(np.searchsorted(target_ranks, k, side="left"))
                    coverage = covered / target_count
                    recent_covered = int(np.count_nonzero(target >= common - k))
                    recency_coverage = recent_covered / target_count
                    row[f"score_coverage_k{k}"] = coverage
                    row[f"recency_coverage_k{k}"] = recency_coverage
                    previous_coverage.append(coverage)
                    previous_recency.append(recency_coverage)
                    oracle = min(1.0, k / target_count)
                    if coverage > oracle + 1e-12:
                        raise AssertionError("coverage exceeds capacity oracle")
                if np.any(np.diff(previous_coverage) < -1e-12):
                    raise AssertionError("score coverage is not monotonic in K")
                if np.any(np.diff(previous_recency) < -1e-12):
                    raise AssertionError("recency coverage is not monotonic in K")

                for fraction in FRACTIONS:
                    k = max(1, min(common, int(math.floor(common * fraction))))
                    covered = int(np.searchsorted(target_ranks, k, side="left"))
                    recent_covered = int(np.count_nonzero(target >= common - k))
                    suffix = int(round(fraction * 100))
                    row[f"fraction_k_{suffix}"] = k
                    row[f"score_coverage_f{suffix}"] = covered / target_count
                    row[f"recency_coverage_f{suffix}"] = recent_covered / target_count

                for quantile in REQUIRED_Q:
                    suffix = int(round(quantile * 100))
                    row[f"required_k_{suffix}"] = required_k(
                        target_ranks, quantile
                    )
                if required_k(target_ranks, 1.0) > common:
                    raise AssertionError("full-context coverage must equal one")
                rows.append(row)

            if chunk_index % 96 == 0 or chunk_index == total_chunks:
                print(
                    f"[{chunk_index:04d}/{total_chunks}] analyzed {rid}, layer {layer}",
                    flush=True,
                )

    validation = {
        "requests": len(requests),
        "layers": 48,
        "steps_per_chunk": 32,
        "adjacent_transitions": len(rows),
        "expected_transitions": len(requests) * 48 * 31,
        "absolute_k": ABSOLUTE_K,
        "normalized_fractions": FRACTIONS,
        "exact_k2048_reconstruction_transitions": exact_match_transitions,
        "tie_only_reconstruction_transitions": tie_only_transitions,
        "non_tie_fast_topk_divergence_transitions": non_tie_transitions,
        "all_symmetric_difference_entries": mismatch_entries,
        "non_tie_symmetric_difference_entries": non_tie_mismatch_entries,
        "max_fast_topk_boundary_replacements": max_replacements,
        "max_fast_topk_boundary_score_regret": max_boundary_regret,
        "max_kth_tie_block": max_tie_block,
        "next_new_token_selected_count": new_token_selected,
        "target_common_count_histogram": {
            str(key): value for key, value in sorted(target_counts.items())
        },
        "checks": {
            "all_expected_transitions": len(rows) == len(requests) * 48 * 31,
            "score_coverage_monotonic": True,
            "recency_coverage_monotonic": True,
            "coverage_within_capacity_oracle": True,
            "full_context_coverage_one": True,
            "fast_topk_replacements_within_kernel_test_tolerance": (
                max_replacements <= 5
            ),
            "common_target_is_2047_or_2048": True,
        },
    }
    return pd.DataFrame(rows), validation


def bootstrap_ci(values: pd.Series, rng: np.random.Generator) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) < 2:
        return math.nan, math.nan
    samples = rng.choice(array, size=(BOOTSTRAP_SAMPLES, len(array)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def method_series(frame: pd.DataFrame, method: str, k: int) -> pd.Series:
    if method == "previous_score_rank":
        return frame[f"score_coverage_k{k}"]
    if method == "recency":
        return frame[f"recency_coverage_k{k}"]
    if method == "random_expectation":
        return k / frame["common_len"]
    if method == "current_score_oracle":
        return np.minimum(1.0, k / frame["target_common_count"])
    raise ValueError(method)


def summarize(
    transitions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    methods = list(METHOD_LABELS)
    overall_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    cdf_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []

    context_labels = ["3k-4k", "4k-5k", "5k-6k", "6k-7k", "7k-8k"]
    transitions = transitions.copy()
    transitions["context_bucket"] = pd.cut(
        transitions.common_len,
        bins=[3000, 4000, 5000, 6000, 7000, 8001],
        labels=context_labels,
        right=False,
    )

    for k in ABSOLUTE_K:
        for method in methods:
            value_column = f"_value_{method}_{k}"
            transitions[value_column] = method_series(transitions, method, k)
            trajectory = transitions.groupby("trajectory_id")[value_column].mean()
            diagnostics = transitions[
                ["trajectory_id", "target_common_count"]
            ].copy()
            diagnostics["intersection"] = (
                transitions[value_column] * transitions.target_common_count
            )
            diagnostics["miss_tokens"] = (
                diagnostics.target_common_count - diagnostics.intersection
            )
            diagnostics["candidate_precision"] = diagnostics.intersection / k
            diagnostics["oracle_efficiency"] = transitions[value_column] / np.minimum(
                1.0, k / transitions.target_common_count
            )
            trajectory_diagnostics = diagnostics.groupby("trajectory_id")[
                ["miss_tokens", "candidate_precision", "oracle_efficiency"]
            ].mean()
            ci_low, ci_high = bootstrap_ci(trajectory, rng)
            overall_rows.append(
                {
                    "method": method,
                    "k": k,
                    "candidate_amplification": k / TARGET_K,
                    "mean_coverage": float(trajectory.mean()),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p10_trajectory": float(trajectory.quantile(0.10)),
                    "p50_trajectory": float(trajectory.quantile(0.50)),
                    "p90_trajectory": float(trajectory.quantile(0.90)),
                    "mean_miss_tokens": float(
                        trajectory_diagnostics.miss_tokens.mean()
                    ),
                    "mean_candidate_precision": float(
                        trajectory_diagnostics.candidate_precision.mean()
                    ),
                    "mean_oracle_efficiency": float(
                        trajectory_diagnostics.oracle_efficiency.mean()
                    ),
                    "estimated_kv_mib_per_layer_tp2": k * 2048 / 2**20,
                }
            )

            unit = (
                transitions.groupby(["trajectory_id", "layer"])[value_column]
                .mean()
                .rename("coverage")
                .reset_index()
            )
            for layer, layer_values in unit.groupby("layer"):
                ci_low, ci_high = bootstrap_ci(layer_values.coverage, rng)
                layer_rows.append(
                    {
                        "method": method,
                        "k": k,
                        "layer": int(layer),
                        "mean_coverage": float(layer_values.coverage.mean()),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "p10": float(layer_values.coverage.quantile(0.10)),
                        "p50": float(layer_values.coverage.quantile(0.50)),
                        "p90": float(layer_values.coverage.quantile(0.90)),
                    }
                )
            for category, values in transitions.groupby("category"):
                per_trajectory = values.groupby("trajectory_id")[value_column].mean()
                category_rows.append(
                    {
                        "method": method,
                        "k": k,
                        "category": category,
                        "trajectory_count": int(len(per_trajectory)),
                        "mean_coverage": float(per_trajectory.mean()),
                        "p10": float(per_trajectory.quantile(0.10)),
                        "p50": float(per_trajectory.quantile(0.50)),
                    }
                )
            for bucket, values in transitions.groupby(
                "context_bucket", observed=True
            ):
                per_trajectory = values.groupby("trajectory_id")[value_column].mean()
                context_rows.append(
                    {
                        "method": method,
                        "k": k,
                        "context_bucket": str(bucket),
                        "transition_count": int(len(values)),
                        "trajectory_count": int(len(per_trajectory)),
                        "mean_coverage": float(per_trajectory.mean()),
                        "p10": float(per_trajectory.quantile(0.10)),
                        "p50": float(per_trajectory.quantile(0.50)),
                    }
                )

            if method == "previous_score_rank" and k in CDF_K:
                for row in unit.itertuples():
                    cdf_rows.append(
                        {
                            "cdf_type": "fixed_k_coverage",
                            "k": k,
                            "quantile_target": math.nan,
                            "unit_statistic": "mean",
                            "trajectory_id": row.trajectory_id,
                            "layer": int(row.layer),
                            "value": float(row.coverage),
                        }
                    )
            del transitions[value_column]

    for fraction in FRACTIONS:
        suffix = int(round(fraction * 100))
        for method in ["previous_score_rank", "recency", "random_expectation"]:
            if method == "previous_score_rank":
                values = transitions[f"score_coverage_f{suffix}"]
            elif method == "recency":
                values = transitions[f"recency_coverage_f{suffix}"]
            else:
                values = transitions[f"fraction_k_{suffix}"] / transitions.common_len
            working = transitions[["trajectory_id", "context_bucket"]].copy()
            working["coverage"] = values
            trajectory = working.groupby("trajectory_id").coverage.mean()
            ci_low, ci_high = bootstrap_ci(trajectory, rng)
            normalized_rows.append(
                {
                    "method": method,
                    "context_bucket": "all",
                    "candidate_fraction": fraction,
                    "mean_coverage": float(trajectory.mean()),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "trajectory_count": int(len(trajectory)),
                }
            )
            for bucket, bucket_values in working.groupby(
                "context_bucket", observed=True
            ):
                per_trajectory = bucket_values.groupby(
                    "trajectory_id"
                ).coverage.mean()
                normalized_rows.append(
                    {
                        "method": method,
                        "context_bucket": str(bucket),
                        "candidate_fraction": fraction,
                        "mean_coverage": float(per_trajectory.mean()),
                        "ci95_low": math.nan,
                        "ci95_high": math.nan,
                        "trajectory_count": int(len(per_trajectory)),
                    }
                )

    required_transition_columns = [
        "rid",
        "trajectory_id",
        "category",
        "round_id",
        "layer",
        "step",
        "common_len",
        *[f"required_k_{int(q * 100)}" for q in REQUIRED_Q],
    ]
    required_transition = transitions[required_transition_columns].copy()
    required_unit_rows: list[dict[str, Any]] = []
    for quantile in [0.90, 0.95]:
        suffix = int(quantile * 100)
        column = f"required_k_{suffix}"
        for (trajectory_id, layer), values in transitions.groupby(
            ["trajectory_id", "layer"]
        ):
            series = values[column]
            median = float(series.median())
            p90 = float(series.quantile(0.90))
            required_unit_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "layer": int(layer),
                    "coverage_target": quantile,
                    "median_required_k": median,
                    "p90_required_k": p90,
                }
            )
            for statistic, value in [("median", median), ("p90", p90)]:
                cdf_rows.append(
                    {
                        "cdf_type": "required_k",
                        "k": math.nan,
                        "quantile_target": quantile,
                        "unit_statistic": statistic,
                        "trajectory_id": trajectory_id,
                        "layer": int(layer),
                        "value": value,
                    }
                )

    required_unit = pd.DataFrame(required_unit_rows)
    required_layer_rows: list[dict[str, Any]] = []
    for (target, layer), values in required_unit.groupby(
        ["coverage_target", "layer"]
    ):
        ci_low, ci_high = bootstrap_ci(values.median_required_k, rng)
        required_layer_rows.append(
            {
                "coverage_target": float(target),
                "layer": int(layer),
                "mean_required_k": float(values.median_required_k.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "p50_required_k": float(values.median_required_k.median()),
                "trajectory_count": int(len(values)),
            }
        )

    return {
        "summary_by_k_overall": pd.DataFrame(overall_rows),
        "summary_by_k_layer": pd.DataFrame(layer_rows),
        "summary_by_k_category": pd.DataFrame(category_rows),
        "summary_by_context_bucket": pd.DataFrame(context_rows),
        "summary_by_normalized_k": pd.DataFrame(normalized_rows),
        "cdf_points": pd.DataFrame(cdf_rows),
        "required_k_by_transition": required_transition,
        "required_k_by_trajectory_layer": required_unit,
        "required_k_by_layer": pd.DataFrame(required_layer_rows),
    }


def save_figure(fig: matplotlib.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values.dropna().to_numpy(dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_results(tables: dict[str, pd.DataFrame], figure_dir: Path) -> None:
    overall = tables["summary_by_k_overall"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), layout="constrained")
    for method in METHOD_LABELS:
        values = overall[overall.method == method].sort_values("k")
        axes[0].plot(
            values.k,
            values.mean_coverage,
            marker="o",
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linestyle="--" if "oracle" in method or "random" in method else "-",
        )
        if method == "previous_score_rank":
            axes[0].fill_between(
                values.k,
                values.ci95_low,
                values.ci95_high,
                color=METHOD_COLORS[method],
                alpha=0.18,
                label="95% trajectory bootstrap CI",
            )
    axes[0].set(
        xlabel="Previous-step candidate K",
        ylabel="Next-step historical top-k coverage",
        ylim=(0, 1.01),
    )
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(alpha=0.25)
    amplification_axis = axes[0].secondary_xaxis(
        "top",
        functions=(
            lambda candidate_k: candidate_k / TARGET_K,
            lambda amplification: amplification * TARGET_K,
        ),
    )
    amplification_axis.set_xlabel("Candidate amplification K / 2048")

    score = overall[overall.method == "previous_score_rank"].sort_values("k")
    gain = np.diff(score.mean_coverage) / np.diff(score.k) * 256
    axes[1].plot(
        score.k.iloc[1:], gain, marker="o", color=METHOD_COLORS["previous_score_rank"]
    )
    axes[1].axhline(0, color="#4D4D4D", linestyle="--")
    axes[1].set(
        xlabel="Previous-step candidate K",
        ylabel="Coverage gain per +256 candidates",
    )
    axes[1].grid(alpha=0.25)
    save_figure(fig, figure_dir / "coverage_vs_k")

    cdf = tables["cdf_points"]
    fixed = cdf[cdf.cdf_type == "fixed_k_coverage"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8), layout="constrained")
    for color, k in zip(COLORS, CDF_K):
        x, y = ecdf(fixed[fixed.k == k].value)
        ax.step(x, y, where="post", color=color, label=f"K={k}")
    for threshold in [0.80, 0.90, 0.95]:
        ax.axvline(threshold, color="#9E9E9E", linestyle="--", linewidth=0.8)
    ax.set(
        xlabel="Mean historical top-k coverage per trajectory-layer",
        ylabel="Empirical CDF",
        xlim=(0, 1.0),
        ylim=(0, 1.0),
    )
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "coverage_ecdf")

    required = cdf[cdf.cdf_type == "required_k"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8), layout="constrained")
    styles = {"median": "-", "p90": "--"}
    for color, target in zip([COLORS[0], COLORS[1]], [0.90, 0.95]):
        for statistic in ["median", "p90"]:
            values = required[
                (required.quantile_target == target)
                & (required.unit_statistic == statistic)
            ].value
            x, y = ecdf(values)
            ax.step(
                x,
                y,
                where="post",
                color=color,
                linestyle=styles[statistic],
                label=f"{int(target * 100)}% target, {statistic}",
            )
    for k in [2048, 2560, 3072]:
        ax.axvline(k, color="#9E9E9E", linestyle=":", linewidth=0.9)
    ax.set(
        xlabel="Required previous-step candidate K",
        ylabel="Empirical CDF",
        ylim=(0, 1.0),
    )
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "required_k_ecdf")

    layer = tables["summary_by_k_layer"]
    score_layer = layer[layer.method == "previous_score_rank"]
    matrix = np.array(
        [
            [
                score_layer[
                    (score_layer.layer == layer_id) & (score_layer.k == k)
                ].mean_coverage.iloc[0]
                for k in ABSOLUTE_K
            ]
            for layer_id in range(48)
        ]
    )
    required_layer = tables["required_k_by_layer"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), layout="constrained")
    image = axes[0].imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(ABSOLUTE_K)), ABSOLUTE_K, rotation=45)
    axes[0].set(xlabel="Candidate K", ylabel="DSA layer")
    fig.colorbar(image, ax=axes[0], label="Mean coverage")
    for color, target in zip([COLORS[0], COLORS[1]], [0.90, 0.95]):
        values = required_layer[
            required_layer.coverage_target == target
        ].sort_values("layer")
        axes[1].plot(
            values.layer,
            values.mean_required_k,
            color=color,
            label=f"Required K@{int(target * 100)}",
        )
        axes[1].fill_between(
            values.layer,
            values.ci95_low,
            values.ci95_high,
            color=color,
            alpha=0.14,
        )
    axes[1].axhline(2048, color="#4D4D4D", linestyle="--", label="K=2048")
    axes[1].set(
        xlabel="DSA layer", ylabel="Mean required K across trajectories"
    )
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    save_figure(fig, figure_dir / "layer_k_heatmap_and_budget")

    context = tables["summary_by_context_bucket"]
    normalized = tables["summary_by_normalized_k"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), layout="constrained")
    buckets = ["3k-4k", "4k-5k", "5k-6k", "6k-7k", "7k-8k"]
    for color, bucket in zip(COLORS, buckets):
        values = context[
            (context.method == "previous_score_rank")
            & (context.context_bucket == bucket)
        ].sort_values("k")
        if not values.empty:
            axes[0].plot(values.k, values.mean_coverage, color=color, label=bucket)
        random_values = context[
            (context.method == "random_expectation")
            & (context.context_bucket == bucket)
        ].sort_values("k")
        if not random_values.empty:
            axes[0].plot(
                random_values.k,
                random_values.mean_coverage,
                color=color,
                linestyle="--",
                alpha=0.50,
            )
        frac = normalized[
            (normalized.method == "previous_score_rank")
            & (normalized.context_bucket == bucket)
        ].sort_values("candidate_fraction")
        if not frac.empty:
            axes[1].plot(
                frac.candidate_fraction,
                frac.mean_coverage,
                color=color,
                label=bucket,
            )
    axes[1].plot(
        FRACTIONS,
        FRACTIONS,
        color="#4D4D4D",
        linestyle="--",
        label="Random expectation",
    )
    axes[0].set(
        xlabel="Candidate K",
        ylabel="Mean historical top-k coverage",
        ylim=(0, 1.0),
    )
    axes[1].set(
        xlabel="Candidate fraction K / common context",
        ylabel="Mean historical top-k coverage",
        ylim=(0, 1.0),
    )
    context_legend = axes[0].legend(
        title="Context length", frameon=False, fontsize=7, loc="upper left"
    )
    axes[0].add_artist(context_legend)
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="#4D4D4D", label="Score rank"),
            Line2D(
                [0],
                [0],
                color="#4D4D4D",
                linestyle="--",
                label="Random expectation",
            ),
        ],
        frameon=False,
        fontsize=7,
        loc="lower right",
    )
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "context_length_sensitivity")


def build_summary(
    transitions: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    validation: dict[str, Any],
) -> dict[str, Any]:
    overall = tables["summary_by_k_overall"]
    score = overall[overall.method == "previous_score_rank"].set_index("k")
    recency = overall[overall.method == "recency"].set_index("k")
    required = tables["required_k_by_trajectory_layer"]
    result: dict[str, Any] = {
        "requests": validation["requests"],
        "adjacent_transitions": validation["adjacent_transitions"],
        "trajectory_clusters": int(transitions.trajectory_id.nunique()),
        "layers": 48,
        "target_topk": TARGET_K,
        "previous_score_rank": {},
        "recency": {},
        "required_k": {},
        "new_token_selected_rate": float(transitions.new_token_selected.mean()),
    }
    for k in ABSOLUTE_K:
        result["previous_score_rank"][str(k)] = {
            "mean_coverage": float(score.loc[k, "mean_coverage"]),
            "ci95": [
                float(score.loc[k, "ci95_low"]),
                float(score.loc[k, "ci95_high"]),
            ],
            "p10_trajectory": float(score.loc[k, "p10_trajectory"]),
            "mean_miss_tokens": float(score.loc[k, "mean_miss_tokens"]),
            "mean_candidate_precision": float(
                score.loc[k, "mean_candidate_precision"]
            ),
            "mean_oracle_efficiency": float(
                score.loc[k, "mean_oracle_efficiency"]
            ),
            "estimated_kv_mib_per_layer_tp2": float(
                score.loc[k, "estimated_kv_mib_per_layer_tp2"]
            ),
        }
        result["recency"][str(k)] = {
            "mean_coverage": float(recency.loc[k, "mean_coverage"])
        }
    for target in [0.90, 0.95]:
        values = required[required.coverage_target == target]
        result["required_k"][str(target)] = {
            "trajectory_layer_median_p50": float(
                values.median_required_k.median()
            ),
            "trajectory_layer_median_p90": float(
                values.median_required_k.quantile(0.90)
            ),
            "trajectory_layer_transition_p90_p50": float(
                values.p90_required_k.median()
            ),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-requests", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requests = read_jsonl(args.run_dir / "prepared_requests.jsonl")
    requests.sort(key=lambda row: (row["trajectory_id"], int(row["round_id"])))
    if args.limit_requests is not None:
        requests = requests[: args.limit_requests]
    lookup = build_lookup(args.run_dir, requests)
    output_dir = args.output_dir or args.run_dir / "analysis" / "k-sweep"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # The shared style currently has one matplotlib-version-incompatible
    # prop-cycle entry. Load every other setting quietly, then install the
    # equivalent colorblind-safe cycle explicitly below.
    with contextlib.redirect_stderr(io.StringIO()):
        plt.style.use(args.style)
    plt.rcParams["axes.prop_cycle"] = cycler(
        color=COLORS + ["#56B4E9", "#4D4D4D"]
    )

    transitions, validation = analyze_transitions(args.run_dir, requests, lookup)
    if not all(validation["checks"].values()):
        raise ValueError(f"validation failed: {validation}")
    transitions.to_parquet(
        table_dir / "adjacent_step_k_coverage.parquet", index=False
    )
    tables = summarize(transitions)
    for name, frame in tables.items():
        frame.to_parquet(table_dir / f"{name}.parquet", index=False)
    plot_results(tables, figure_dir)
    summary = build_summary(transitions, tables, validation)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    validation["validated_at"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n"
    )
    reproducibility = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_commit": git_revision(Path.cwd()),
        "run_dir": str(args.run_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "style": str(args.style.resolve()),
        "versions": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "target_topk": TARGET_K,
        "absolute_k": ABSOLUTE_K,
        "normalized_fractions": FRACTIONS,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "cluster": "trajectory_id",
        },
        "common_context_policy": "exclude positions not visible at previous step",
        "cdf_unit": "trajectory-layer",
        "figure_formats": ["PDF", "PNG 300 dpi"],
        "figure_inputs": {
            "coverage_vs_k": "tables/summary_by_k_overall.parquet",
            "coverage_ecdf": "tables/cdf_points.parquet; unit=trajectory-layer mean",
            "required_k_ecdf": "tables/cdf_points.parquet; unit=trajectory-layer median/p90",
            "layer_k_heatmap_and_budget": "tables/summary_by_k_layer.parquet + tables/required_k_by_layer.parquet",
            "context_length_sensitivity": "tables/summary_by_context_bucket.parquet + tables/summary_by_normalized_k.parquet",
        },
    }
    (output_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
