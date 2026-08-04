#!/usr/bin/env python3
"""Analyze cross-layer Keye indexer lookahead fidelity and calibration gates."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.stats import spearmanr

TARGET_LAYERS = list(range(1, 48))
K_VALUES = [2048, 2304, 2560, 3072]
SPEARMAN_STEPS = {0, 15, 31}
STYLE_PATH = Path("/home10T/bzx/.codex/skills/research-figure-style/assets/matplotlib_style.mplstyle")
PRIMARY = "#0072B2"
BASELINE = "#4D4D4D"
TEST = "#D55E00"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def topk_set(values: np.ndarray, valid_count: int, k: int | None = None) -> set[int]:
    if k is not None:
        values = values[:k]
    return {int(value) for value in values if 0 <= int(value) < valid_count}


def overlap(predicted: set[int], target: set[int]) -> tuple[float, float]:
    intersection = len(predicted & target)
    union = len(predicted | target)
    return intersection / len(target), intersection / union


def required_k(ranking: np.ndarray, target: set[int], recall: float) -> tuple[int, bool]:
    needed = math.ceil(recall * len(target))
    ranks = sorted(index + 1 for index, value in enumerate(ranking) if int(value) in target)
    if len(ranks) < needed:
        return len(ranking) + 1, True
    return ranks[needed - 1], False


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_inventory(run_dir: Path, actual_rids: set[str]) -> dict[tuple[str, int], Path]:
    event_dir = run_dir / "events"
    rows = [row for row in jsonl(event_dir / "manifest.jsonl") if row["request_id"] in actual_rids]
    lookup: dict[tuple[str, int], Path] = {}
    for row in rows:
        key = (row["request_id"], int(row["target_layer_id"]))
        if key in lookup:
            raise ValueError(f"duplicate event chunk: {key}")
        if int(row["num_steps"]) != 32:
            raise ValueError(f"{key}: expected 32 steps")
        lookup[key] = event_dir / row["file"]
    expected = len(actual_rids) * len(TARGET_LAYERS)
    if len(lookup) != expected:
        missing = [(rid, layer) for rid in actual_rids for layer in TARGET_LAYERS if (rid, layer) not in lookup]
        raise ValueError(f"expected {expected} chunks, found {len(lookup)}; missing={missing[:8]}")
    return lookup


def analyze(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selection = json.loads((run_dir / "selection.json").read_text())
    selected = {row["rid"]: row for row in selection["selected"]}
    requests = jsonl(run_dir / "requests.jsonl")
    request_meta = {row["rid"]: row for row in requests}
    if len(request_meta) != 24:
        raise ValueError(f"expected 24 executed requests, found {len(request_meta)}")
    prepared_to_actual = {row["prepared_rid"]: row["rid"] for row in requests}
    if set(prepared_to_actual) != set(selected):
        raise ValueError("executed requests do not match selection")
    lookup = load_inventory(run_dir, set(request_meta))

    metric_rows: list[dict[str, Any]] = []
    for prepared_rid, meta in selected.items():
        actual_rid = prepared_to_actual[prepared_rid]
        for target_layer in TARGET_LAYERS:
            record = torch.load(lookup[(actual_rid, target_layer)], weights_only=False)
            if record["schema_version"] != 1 or record["decode_step_ids"] != list(range(32)):
                raise ValueError(f"invalid schema or steps: {actual_rid}, layer {target_layer}")
            if not record["self_token_forced"] or not record["canonical_historical_k_cache"]:
                raise ValueError(f"invalid experimental variant: {actual_rid}, layer {target_layer}")
            exact_indices = record["exact_indices"].numpy(force=True)
            lookahead_indices = record["lookahead_indices"].numpy(force=True)
            direct_indices = record["direct_reuse_indices"].numpy(force=True)
            valid_counts = record["valid_counts"].numpy(force=True).astype(int)
            exact_scores = record["exact_scores"].numpy(force=True).astype(np.float64)
            lookahead_scores = record["lookahead_scores"].numpy(force=True).astype(np.float64)
            for step in range(32):
                valid_count = int(valid_counts[step])
                target = topk_set(exact_indices[step], valid_count)
                if len(target) != 2048:
                    raise ValueError(f"expected exact top-k size 2048, got {len(target)}")
                lookahead = lookahead_indices[step]
                if len(topk_set(lookahead, valid_count)) != min(3072, valid_count):
                    raise ValueError(f"duplicate or invalid lookahead indices: {actual_rid}, layer {target_layer}, step {step}")
                direct_recall, direct_jaccard = overlap(topk_set(direct_indices[step], valid_count), target)
                score_exact = exact_scores[step, :valid_count]
                score_approx = lookahead_scores[step, :valid_count]
                mae = float(np.mean(np.abs(score_approx - score_exact)))
                nmae = mae / (float(np.mean(np.abs(score_exact))) + 1e-12)
                denom = float(np.linalg.norm(score_exact) * np.linalg.norm(score_approx))
                cosine = float(np.dot(score_exact, score_approx) / denom) if denom else math.nan
                spearman = (
                    float(spearmanr(score_exact, score_approx).statistic)
                    if step in SPEARMAN_STEPS
                    else math.nan
                )
                row = {
                    "rid": actual_rid,
                    "prepared_rid": prepared_rid,
                    "trajectory_id": meta["trajectory_id"],
                    "category": meta["category"],
                    "split": meta["split"],
                    "prompt_len": int(meta["prompt_len"]),
                    "source_layer": target_layer - 1,
                    "target_layer": target_layer,
                    "step": step,
                    "valid_count": valid_count,
                    "direct_recall_at_2048": direct_recall,
                    "direct_jaccard_at_2048": direct_jaccard,
                    "score_mae": mae,
                    "score_nmae": nmae,
                    "score_cosine": cosine,
                    "score_spearman": spearman,
                }
                for k in K_VALUES:
                    recall, jaccard = overlap(topk_set(lookahead, valid_count, k), target)
                    row[f"lookahead_recall_at_{k}"] = recall
                    row[f"lookahead_jaccard_at_{k}"] = jaccard
                for threshold in [0.90, 0.95]:
                    value, censored = required_k(lookahead, target, threshold)
                    suffix = int(threshold * 100)
                    row[f"required_k_at_{suffix}"] = value
                    row[f"required_k_at_{suffix}_censored"] = censored
                metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    request_layer = (
        metrics.groupby(["split", "category", "trajectory_id", "source_layer", "target_layer"], as_index=False)
        .agg(
            lookahead_recall_at_2048=("lookahead_recall_at_2048", "mean"),
            lookahead_recall_at_2560=("lookahead_recall_at_2560", "mean"),
            direct_recall_at_2048=("direct_recall_at_2048", "mean"),
            required_k_at_90=("required_k_at_90", "mean"),
            score_nmae=("score_nmae", "mean"),
            score_cosine=("score_cosine", "mean"),
        )
    )
    layer_rows = []
    for (split, target_layer), group in request_layer.groupby(["split", "target_layer"]):
        layer_rows.append(
            {
                "split": split,
                "source_layer": int(target_layer) - 1,
                "target_layer": int(target_layer),
                "request_count": len(group),
                "lookahead_recall_at_2048_mean": group["lookahead_recall_at_2048"].mean(),
                "lookahead_recall_at_2048_request_p10": group["lookahead_recall_at_2048"].quantile(0.10),
                "lookahead_recall_at_2560_mean": group["lookahead_recall_at_2560"].mean(),
                "direct_recall_at_2048_mean": group["direct_recall_at_2048"].mean(),
                "recall_improvement_pp": 100 * (group["lookahead_recall_at_2048"] - group["direct_recall_at_2048"]).mean(),
                "required_k_at_90_mean": group["required_k_at_90"].mean(),
                "score_nmae_mean": group["score_nmae"].mean(),
                "score_cosine_mean": group["score_cosine"].mean(),
            }
        )
    layers = pd.DataFrame(layer_rows).sort_values(["split", "target_layer"])
    calibration = layers[layers["split"] == "calibration"].copy()
    calibration["gate_mean_recall_2048"] = calibration["lookahead_recall_at_2048_mean"] >= 0.90
    calibration["gate_request_p10"] = calibration["lookahead_recall_at_2048_request_p10"] >= 0.80
    calibration["gate_vs_direct"] = calibration["recall_improvement_pp"] >= 3.0
    calibration["gate_recall_2560"] = calibration["lookahead_recall_at_2560_mean"] >= 0.95
    calibration["eligible"] = calibration[["gate_mean_recall_2048", "gate_request_p10", "gate_vs_direct", "gate_recall_2560"]].all(axis=1)
    eligible_layers = calibration.loc[calibration["eligible"], "target_layer"].astype(int).tolist()
    proceed = len(eligible_layers) >= 24

    summary = {
        "schema_version": 1,
        "request_count": 24,
        "layer_pair_count": 47,
        "decode_steps": 32,
        "pair_step_count": len(metrics),
        "gate_policy": {
            "mean_recall_at_2048_min": 0.90,
            "request_p10_recall_at_2048_min": 0.80,
            "improvement_over_direct_min_pp": 3.0,
            "mean_recall_at_2560_min": 0.95,
            "minimum_eligible_layer_pairs": 24,
        },
        "eligible_layer_count": len(eligible_layers),
        "eligible_target_layers": eligible_layers,
        "phase_b_proceed": proceed,
        "overall": {},
    }
    for split, group in metrics.groupby("split"):
        summary["overall"][split] = {
            "lookahead_recall_at_2048_mean": safe_float(group["lookahead_recall_at_2048"].mean()),
            "lookahead_recall_at_2560_mean": safe_float(group["lookahead_recall_at_2560"].mean()),
            "direct_recall_at_2048_mean": safe_float(group["direct_recall_at_2048"].mean()),
            "score_nmae_mean": safe_float(group["score_nmae"].mean()),
            "score_cosine_mean": safe_float(group["score_cosine"].mean()),
            "required_k_at_90_median": safe_float(group["required_k_at_90"].median()),
            "required_k_at_90_censored_fraction": safe_float(group["required_k_at_90_censored"].mean()),
        }
    return metrics, layers, summary


def plot(metrics: pd.DataFrame, layers: pd.DataFrame, figure_dir: Path) -> None:
    plt.style.use(STYLE_PATH)
    figure_dir.mkdir(parents=True, exist_ok=True)

    calibration = layers[layers["split"] == "calibration"].set_index("target_layer")
    matrix = np.vstack(
        [
            calibration["lookahead_recall_at_2048_mean"].to_numpy(),
            calibration["lookahead_recall_at_2560_mean"].to_numpy(),
            calibration["direct_recall_at_2048_mean"].to_numpy(),
        ]
    )
    fig, ax = plt.subplots(figsize=(10.0, 2.8), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.70, vmax=1.0)
    ax.set_yticks(range(3), ["Lookahead K=2048", "Lookahead K=2560", "Direct reuse K=2048"])
    tick_positions = np.arange(0, 47, 4)
    ax.set_xticks(tick_positions, calibration.index.to_numpy()[tick_positions])
    ax.set_xlabel("Target layer")
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("Mean Recall@2048 (higher is better)")
    save_figure(fig, figure_dir / "calibration_layer_fidelity_heatmap")

    aggregated = (
        metrics.groupby("split", as_index=False)[[f"lookahead_recall_at_{k}" for k in K_VALUES]]
        .mean()
        .set_index("split")
    )
    direct = metrics.groupby("split")["direct_recall_at_2048"].mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for split, color, marker in [("calibration", PRIMARY, "o"), ("test", TEST, "s")]:
        ax.plot(K_VALUES, [aggregated.loc[split, f"lookahead_recall_at_{k}"] for k in K_VALUES], color=color, marker=marker, label=f"Lookahead ({split})")
        ax.axhline(direct.loc[split], color=color, linestyle=":", linewidth=1.2, alpha=0.8, label=f"Direct reuse ({split})")
    ax.axhline(0.90, color=BASELINE, linestyle="--", linewidth=1.2, label="Gate: 0.90")
    ax.set_xlabel("Predicted top-k budget K")
    ax.set_ylabel("Mean Recall@2048 (higher is better)")
    ax.set_ylim(0.75, 1.005)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, figure_dir / "recall_vs_prediction_budget")

    request_layer = (
        metrics.groupby(["split", "trajectory_id", "target_layer"], as_index=False)["lookahead_recall_at_2048"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for split, color, linestyle in [("calibration", PRIMARY, "-"), ("test", TEST, "--")]:
        values = np.sort(request_layer.loc[request_layer["split"] == split, "lookahead_recall_at_2048"].to_numpy())
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cdf, color=color, linestyle=linestyle, label=split)
    ax.axvline(0.80, color=BASELINE, linestyle=":", linewidth=1.2, label="Request gate: 0.80")
    ax.set_xlabel("Per-request, per-layer Recall@2048 (higher is better)")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(0.35, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.legend(frameon=False)
    save_figure(fig, figure_dir / "request_layer_recall_cdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics, layers, summary = analyze(run_dir)
    metrics.to_parquet(analysis_dir / "pair_step_metrics.parquet", index=False)
    layers.to_csv(analysis_dir / "layer_summary.csv", index=False)
    layers.to_parquet(analysis_dir / "layer_summary.parquet", index=False)
    calibration = layers[layers["split"] == "calibration"].copy()
    calibration["eligible"] = (
        (calibration["lookahead_recall_at_2048_mean"] >= 0.90)
        & (calibration["lookahead_recall_at_2048_request_p10"] >= 0.80)
        & (calibration["recall_improvement_pp"] >= 3.0)
        & (calibration["lookahead_recall_at_2560_mean"] >= 0.95)
    )
    write_json(analysis_dir / "gate_decisions.json", calibration.to_dict(orient="records"))
    write_json(analysis_dir / "summary.json", summary)
    write_json(
        analysis_dir / "reproducibility.json",
        {
            "script": str(Path(__file__).resolve()),
            "sglang_commit": git_revision(Path.cwd()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "input_run_dir": str(run_dir),
            "style_path": str(STYLE_PATH),
            "aggregation": "32-step mean per request/layer; calibration gate across 12 requests",
        },
    )
    plot(metrics, layers, analysis_dir / "figures")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
