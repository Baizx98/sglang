#!/usr/bin/env python3
"""Analyze layered Keye DSA score/top-k traces and render research figures."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.stats import spearmanr

LAYERS = [0, 1, 15, 16, 31, 32, 46, 47]
ADJACENT_LAYER_PAIRS = [(0, 1), (15, 16), (31, 32), (46, 47)]
STEP_DELTAS = [1, 2, 4, 8]
RID_PATTERN = re.compile(r"bfcl__(T[12])__round_(\d+)$")


@dataclass
class TraceItem:
    rid: str
    trajectory: str
    round_id: int
    layer: int
    step: int
    valid_len: int
    scores: np.ndarray
    indices: np.ndarray
    file: str


def parse_rid(rid: str) -> tuple[str, int]:
    match = RID_PATTERN.fullmatch(rid)
    if match is None:
        raise ValueError(f"Unexpected experiment rid: {rid}")
    return match.group(1), int(match.group(2))


def load_traces(
    run_dir: Path,
) -> tuple[dict[tuple[str, int, int], TraceItem], pd.DataFrame]:
    event_dir = run_dir / "events"
    manifest = [
        json.loads(line)
        for line in (event_dir / "manifest.jsonl").read_text().splitlines()
        if line
    ]
    experiment_rows = [
        row
        for row in manifest
        if row["request_ids"] and row["request_ids"][0].startswith("bfcl__")
    ]

    traces: dict[tuple[str, int, int], TraceItem] = {}
    inventory = []
    for row in experiment_rows:
        record = torch.load(event_dir / row["file"], weights_only=False)
        rid = record["request_ids"][0]
        trajectory, round_id = parse_rid(rid)
        layer = int(record["layer_id"])
        step = int(record["decode_step_ids"][0])
        valid_len = int(record["score_valid_counts"][0])
        scores = record["scores"][0, :valid_len].numpy(force=True).astype(np.float32)
        indices = record["indices"][0].numpy(force=True).astype(np.int32)
        item = TraceItem(
            rid=rid,
            trajectory=trajectory,
            round_id=round_id,
            layer=layer,
            step=step,
            valid_len=valid_len,
            scores=scores,
            indices=indices,
            file=row["file"],
        )
        key = (rid, layer, step)
        if key in traces:
            raise ValueError(f"Duplicate trace key: {key}")
        traces[key] = item
        inventory.append(
            {
                "rid": rid,
                "trajectory": trajectory,
                "round_id": round_id,
                "layer": layer,
                "step": step,
                "valid_len": valid_len,
                "topk_count": int(np.count_nonzero(indices >= 0)),
                "file": row["file"],
            }
        )
    return traces, pd.DataFrame(inventory)


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else math.nan


def topk_sets(item: TraceItem, common_len: int) -> set[int]:
    return {int(index) for index in item.indices if 0 <= int(index) < common_len}


def compare_items(left: TraceItem, right: TraceItem) -> dict[str, float | int]:
    common_len = min(left.valid_len, right.valid_len)
    left_topk = topk_sets(left, common_len)
    right_topk = topk_sets(right, common_len)
    union = left_topk | right_topk
    intersection = left_topk & right_topk
    left_scores = left.scores[:common_len]
    right_scores = right.scores[:common_len]
    delta = right_scores - left_scores
    return {
        "common_len": common_len,
        "topk_jaccard": len(intersection) / len(union) if union else math.nan,
        "topk_intersection": len(intersection),
        "topk_union": len(union),
        "topk_enters": len(right_topk - left_topk),
        "topk_exits": len(left_topk - right_topk),
        "score_spearman": finite_spearman(left_scores, right_scores),
        "score_mae": float(np.mean(np.abs(delta))),
        "score_rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "score_delta_mean": float(np.mean(delta)),
    }


def build_step_metrics(traces: dict[tuple[str, int, int], TraceItem]) -> pd.DataFrame:
    rows = []
    rids = sorted({key[0] for key in traces})
    for rid in rids:
        trajectory, round_id = parse_rid(rid)
        for layer in LAYERS:
            available_steps = sorted(
                step
                for item_rid, item_layer, step in traces
                if item_rid == rid and item_layer == layer
            )
            for delta_step in STEP_DELTAS:
                for step_a in available_steps:
                    step_b = step_a + delta_step
                    if (rid, layer, step_b) not in traces:
                        continue
                    comparison = compare_items(
                        traces[(rid, layer, step_a)], traces[(rid, layer, step_b)]
                    )
                    rows.append(
                        {
                            "rid": rid,
                            "trajectory": trajectory,
                            "round_id": round_id,
                            "layer": layer,
                            "step_a": step_a,
                            "step_b": step_b,
                            "step_delta": delta_step,
                            **comparison,
                        }
                    )
    return pd.DataFrame(rows)


def build_layer_metrics(traces: dict[tuple[str, int, int], TraceItem]) -> pd.DataFrame:
    rows = []
    rids = sorted({key[0] for key in traces})
    adjacent_pairs = set(ADJACENT_LAYER_PAIRS)
    for rid in rids:
        trajectory, round_id = parse_rid(rid)
        steps = sorted({key[2] for key in traces if key[0] == rid})
        for step in steps:
            for layer_a, layer_b in combinations(LAYERS, 2):
                comparison = compare_items(
                    traces[(rid, layer_a, step)], traces[(rid, layer_b, step)]
                )
                rows.append(
                    {
                        "rid": rid,
                        "trajectory": trajectory,
                        "round_id": round_id,
                        "step": step,
                        "layer_a": layer_a,
                        "layer_b": layer_b,
                        "adjacent_sample_pair": (layer_a, layer_b) in adjacent_pairs,
                        **comparison,
                    }
                )
    return pd.DataFrame(rows)


def build_round_metrics(traces: dict[tuple[str, int, int], TraceItem]) -> pd.DataFrame:
    rows = []
    rounds_by_trajectory: dict[str, list[int]] = {}
    for rid, _, _ in traces:
        trajectory, round_id = parse_rid(rid)
        rounds_by_trajectory.setdefault(trajectory, []).append(round_id)
    for trajectory, round_ids in rounds_by_trajectory.items():
        unique_rounds = sorted(set(round_ids))
        for round_a, round_b in zip(unique_rounds, unique_rounds[1:]):
            rid_a = f"bfcl__{trajectory}__round_{round_a:02d}"
            rid_b = f"bfcl__{trajectory}__round_{round_b:02d}"
            for layer in LAYERS:
                for step in range(32):
                    comparison = compare_items(
                        traces[(rid_a, layer, step)], traces[(rid_b, layer, step)]
                    )
                    rows.append(
                        {
                            "trajectory": trajectory,
                            "round_a": round_a,
                            "round_b": round_b,
                            "transition": f"{trajectory} {round_a}→{round_b}",
                            "layer": layer,
                            "step": step,
                            **comparison,
                        }
                    )
    return pd.DataFrame(rows)


def save_figure(fig: matplotlib.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_heatmap(
    ax: matplotlib.axes.Axes,
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> matplotlib.image.AxesImage:
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title)
    return image


def plot_step_stability(step_metrics: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(
        2, 4, figsize=(7.2, 4.8), sharex=True, sharey=True, layout="constrained"
    )
    for ax, layer in zip(axes.flat, LAYERS):
        layer_data = step_metrics[step_metrics.layer == layer]
        summary = layer_data.groupby("step_delta")[
            ["topk_jaccard", "score_spearman"]
        ].mean()
        ax.plot(
            summary.index,
            summary.topk_jaccard,
            color="#0072B2",
            marker="o",
            label="Top-k Jaccard",
        )
        ax.plot(
            summary.index,
            summary.score_spearman,
            color="#D55E00",
            marker="s",
            label="Score Spearman",
        )
        ax.set_title(f"Layer {layer}")
        ax.set_xticks(STEP_DELTAS)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    fig.supxlabel("Decode-step distance Δ")
    fig.supylabel("Similarity")
    axes.flat[0].legend(loc="lower left", fontsize=6.5, frameon=False)
    fig.suptitle("Step stability, computed independently within each layer")
    save_figure(fig, figure_dir / "step_stability_by_layer")


def layer_similarity_matrix(layer_metrics: pd.DataFrame, column: str) -> np.ndarray:
    matrix = np.eye(len(LAYERS), dtype=float)
    means = layer_metrics.groupby(["layer_a", "layer_b"])[column].mean()
    for i, layer_a in enumerate(LAYERS):
        for j, layer_b in enumerate(LAYERS):
            if i == j:
                continue
            key = tuple(sorted((layer_a, layer_b)))
            matrix[i, j] = means.loc[key]
    return matrix


def plot_layer_similarity(layer_metrics: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
    labels = [str(layer) for layer in LAYERS]
    for ax, column, title in zip(
        axes,
        ["topk_jaccard", "score_spearman"],
        ["Top-k Jaccard", "Score Spearman"],
    ):
        image = add_heatmap(
            ax,
            layer_similarity_matrix(layer_metrics, column),
            labels,
            labels,
            title,
            "viridis",
            0,
            1,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Cross-layer similarity at the same decode step")
    save_figure(fig, figure_dir / "cross_layer_similarity")


def plot_round_similarity(round_metrics: pd.DataFrame, figure_dir: Path) -> None:
    transitions = list(dict.fromkeys(round_metrics.transition))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
    for ax, column, title in zip(
        axes,
        ["topk_jaccard", "score_spearman"],
        ["Top-k Jaccard", "Score Spearman"],
    ):
        grouped = round_metrics.groupby(["layer", "transition"])[column].mean()
        matrix = np.array(
            [
                [grouped.loc[(layer, transition)] for transition in transitions]
                for layer in LAYERS
            ]
        )
        image = add_heatmap(
            ax,
            matrix,
            transitions,
            [str(layer) for layer in LAYERS],
            title,
            "viridis",
            0,
            1,
        )
        ax.set_xlabel("Consecutive round transition")
        ax.set_ylabel("Layer")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Same-step stability across consecutive rounds, by layer")
    save_figure(fig, figure_dir / "round_stability_by_layer")


def plot_adjacent_layer_scores(
    traces: dict[tuple[str, int, int], TraceItem], figure_dir: Path
) -> None:
    representative = {"T1": 4, "T2": 3}
    for trajectory, round_id in representative.items():
        rid = f"bfcl__{trajectory}__round_{round_id:02d}"
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout="constrained")
        for ax, (layer_a, layer_b) in zip(axes.flat, ADJACENT_LAYER_PAIRS):
            left = traces[(rid, layer_a, 0)]
            right = traces[(rid, layer_b, 0)]
            common_len = min(left.valid_len, right.valid_len)
            x = left.scores[:common_len]
            y = right.scores[:common_len]
            artist = ax.hexbin(x, y, gridsize=55, mincnt=1, cmap="viridis")
            lower = float(min(x.min(), y.min()))
            upper = float(max(x.max(), y.max()))
            ax.plot([lower, upper], [lower, upper], color="#D55E00", linestyle="--")
            rho = finite_spearman(x, y)
            mae = float(np.mean(np.abs(y - x)))
            ax.set_title(f"L{layer_a} → L{layer_b}: ρ={rho:.3f}, MAE={mae:.3g}")
            ax.set_xlabel(f"Layer {layer_a} score")
            ax.set_ylabel(f"Layer {layer_b} score")
            fig.colorbar(artist, ax=ax, label="Token count")
        fig.suptitle(
            f"Adjacent-layer same-token scores: {trajectory}, round {round_id}, step 0"
        )
        save_figure(fig, figure_dir / f"adjacent_layer_scores_{trajectory.lower()}")


def plot_adjacent_step_score_heatmaps(
    traces: dict[tuple[str, int, int], TraceItem], figure_dir: Path
) -> None:
    rids = sorted({key[0] for key in traces})
    output_dir = figure_dir / "adjacent_step_by_round"
    output_dir.mkdir(parents=True, exist_ok=True)
    for rid in rids:
        trajectory, round_id = parse_rid(rid)
        fig, axes = plt.subplots(
            2, 4, figsize=(12.0, 6.0), sharey=True, layout="constrained"
        )
        for ax, layer in zip(axes.flat, LAYERS):
            max_len = traces[(rid, layer, 31)].valid_len
            matrix = np.full((31, max_len), np.nan, dtype=np.float32)
            raw_delta_values = []
            for step in range(31):
                left = traces[(rid, layer, step)]
                right = traces[(rid, layer, step + 1)]
                common_len = min(left.valid_len, right.valid_len)
                delta = right.scores[:common_len] - left.scores[:common_len]
                matrix[step, :common_len] = delta
                raw_delta_values.append(delta)
            flat = np.concatenate(raw_delta_values)
            mean = float(flat.mean())
            std = float(flat.std())
            zscores = (matrix - mean) / max(std, np.finfo(np.float32).eps)
            image = ax.imshow(
                zscores,
                cmap="coolwarm",
                vmin=-3,
                vmax=3,
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            mae = float(np.mean(np.abs(flat)))
            rmse = float(np.sqrt(np.mean(np.square(flat))))
            ax.set_title(f"Layer {layer}\nMAE={mae:.3g}, RMSE={rmse:.3g}")
            ax.set_xlabel("Logical token position")
        axes[0, 0].set_ylabel("Step transition t→t+1")
        axes[1, 0].set_ylabel("Step transition t→t+1")
        fig.colorbar(image, ax=axes, label="Layer-specific z-score of score change")
        fig.suptitle(
            f"Adjacent-step same-token score change: {trajectory}, round {round_id}"
        )
        save_figure(
            fig,
            output_dir
            / f"adjacent_step_scores_{trajectory.lower()}_round_{round_id:02d}",
        )


def plot_request_performance(run_dir: Path, figure_dir: Path) -> pd.DataFrame:
    requests = pd.DataFrame(
        json.loads(line)
        for line in (run_dir / "requests.jsonl").read_text().splitlines()
        if line
    )
    requests["prompt_tokens"] = requests.response.map(
        lambda value: value["usage"]["prompt_tokens"]
    )
    requests["completion_tokens"] = requests.response.map(
        lambda value: value["usage"]["completion_tokens"]
    )
    requests["decode_throughput_tps"] = requests.completion_tokens / requests.latency_s
    labels = [f"{row.trajectory}-R{row.round_id}" for row in requests.itertuples()]
    x = np.arange(len(requests))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    axes[0].bar(x, requests.latency_s, color="#0072B2")
    axes[0].set_ylabel("End-to-end latency (s)")
    axes[1].bar(x, requests.decode_throughput_tps, color="#009E73")
    axes[1].set_ylabel("Completion throughput (token/s)")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Single-request TP2 replay performance")
    save_figure(fig, figure_dir / "request_performance")
    return requests.drop(columns=["payload", "response"])


def build_summary(
    run_dir: Path,
    inventory: pd.DataFrame,
    step_metrics: pd.DataFrame,
    layer_metrics: pd.DataFrame,
    round_metrics: pd.DataFrame,
    requests: pd.DataFrame,
) -> dict[str, Any]:
    step_one = step_metrics[step_metrics.step_delta == 1]
    adjacent_layers = layer_metrics[layer_metrics.adjacent_sample_pair]
    return {
        "run_dir": str(run_dir.resolve()),
        "trace_events": int(len(inventory)),
        "trace_bytes": sum(
            path.stat().st_size for path in (run_dir / "events").glob("*.pt")
        ),
        "request_count": int(len(requests)),
        "prompt_token_range": [
            int(requests.prompt_tokens.min()),
            int(requests.prompt_tokens.max()),
        ],
        "latency_s_mean": float(requests.latency_s.mean()),
        "latency_s_p95": float(requests.latency_s.quantile(0.95)),
        "completion_throughput_tps_mean": float(requests.decode_throughput_tps.mean()),
        "adjacent_step_by_layer": step_one.groupby("layer")[
            ["topk_jaccard", "score_spearman", "score_mae", "score_rmse"]
        ]
        .mean()
        .round(6)
        .to_dict(orient="index"),
        "adjacent_layer_pairs": adjacent_layers.groupby(["layer_a", "layer_b"])[
            ["topk_jaccard", "score_spearman", "score_mae", "score_rmse"]
        ]
        .mean()
        .round(6)
        .to_dict(orient="index"),
        "consecutive_round_by_layer": round_metrics.groupby("layer")[
            ["topk_jaccard", "score_spearman", "score_mae", "score_rmse"]
        ]
        .mean()
        .round(6)
        .to_dict(orient="index"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    args = parser.parse_args()

    plt.style.use(args.style)
    analysis_dir = args.run_dir / "analysis"
    table_dir = analysis_dir / "tables"
    figure_dir = analysis_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    traces, inventory = load_traces(args.run_dir)
    expected = 9 * len(LAYERS) * 32
    if len(inventory) != expected:
        raise ValueError(f"Expected {expected} events, found {len(inventory)}")
    group_sizes = inventory.groupby(["rid", "layer"]).size()
    if not (group_sizes == 32).all():
        raise ValueError("Every request/layer group must contain exactly 32 steps")

    step_metrics = build_step_metrics(traces)
    layer_metrics = build_layer_metrics(traces)
    round_metrics = build_round_metrics(traces)
    inventory.to_parquet(table_dir / "trace_inventory.parquet", index=False)
    step_metrics.to_parquet(table_dir / "step_metrics_by_layer.parquet", index=False)
    layer_metrics.to_parquet(table_dir / "layer_metrics.parquet", index=False)
    round_metrics.to_parquet(table_dir / "round_metrics_by_layer.parquet", index=False)

    plot_step_stability(step_metrics, figure_dir)
    plot_layer_similarity(layer_metrics, figure_dir)
    plot_round_similarity(round_metrics, figure_dir)
    plot_adjacent_layer_scores(traces, figure_dir)
    plot_adjacent_step_score_heatmaps(traces, figure_dir)
    requests = plot_request_performance(args.run_dir, figure_dir)
    requests.to_parquet(table_dir / "request_performance.parquet", index=False)

    summary = build_summary(
        args.run_dir,
        inventory,
        step_metrics,
        layer_metrics,
        round_metrics,
        requests,
    )
    # JSON does not support tuple keys.
    summary["adjacent_layer_pairs"] = {
        f"{key[0]}-{key[1]}": value
        for key, value in summary["adjacent_layer_pairs"].items()
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_run_dir": str(args.run_dir.resolve()),
        "script": str(Path(__file__).resolve()),
        "style": str(args.style.resolve()),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "figure_formats": ["pdf", "png@300dpi"],
        "step_grouping": "all step metrics are computed separately per layer",
    }
    (analysis_dir / "reproducibility.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
