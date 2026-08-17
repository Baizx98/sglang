#!/usr/bin/env python3
"""Measure whether adjacent-step DSA top-k changes follow positional shifts.

The main statistic deliberately conditions on positions that were *not* in the
previous top-k.  This avoids mistaking ordinary top-k overlap for evidence of
an ``i -> i + delta`` transition rule.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

TARGET_K = 2048
DELTAS = [delta for delta in range(-16, 17) if delta != 0]
RADII = [1, 2, 4, 8, 16, 32]
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_SAMPLES = 2000


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
    manifest = read_jsonl(run_dir / "events" / "manifest.jsonl")
    lookup: dict[tuple[str, int], Path] = {}
    for row in manifest:
        if row.get("request_id") not in expected_rids:
            continue
        key = (row["request_id"], int(row["layer_id"]))
        if key in lookup:
            raise ValueError(f"duplicate chunk: {key}")
        lookup[key] = run_dir / "events" / row["file"]
    expected = len(requests) * 48
    if len(lookup) != expected:
        raise ValueError(f"expected {expected} chunks, found {len(lookup)}")
    return lookup


def shifted_frontier(
    previous: np.ndarray, previous_mask: np.ndarray, delta: int
) -> np.ndarray:
    shifted = previous + delta
    shifted = shifted[(shifted >= 0) & (shifted < len(previous_mask))]
    if not len(shifted):
        return shifted
    # ``previous`` is unique and adding a constant preserves uniqueness.
    return shifted[~previous_mask[shifted]]


def neighborhood_frontier(
    previous: np.ndarray, previous_mask: np.ndarray, radius: int
) -> np.ndarray:
    parts = [
        shifted_frontier(previous, previous_mask, delta)
        for delta in range(-radius, radius + 1)
        if delta
    ]
    parts = [part for part in parts if len(part)]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def add_counts(target: dict[str, int], values: dict[str, int]) -> None:
    for key, value in values.items():
        target[key] += int(value)


def finalize_counts(row: dict[str, Any]) -> dict[str, Any]:
    entries = int(row["entry_count"])
    nonselected = int(row["nonselected_count"])
    candidates = int(row["candidate_count"])
    hits = int(row["hit_count"])
    target = int(row["target_count"])
    union_hits = int(row["union_hit_count"])
    score_hits = int(row["score_hit_count"])
    prevalence = ratio(entries, nonselected)
    precision = ratio(hits, candidates)
    row.update(
        {
            "entry_recall": ratio(hits, entries),
            "candidate_precision": precision,
            "entry_prevalence": prevalence,
            "precision_lift": (
                precision / prevalence
                if np.isfinite(precision) and prevalence > 0
                else float("nan")
            ),
            "union_recall": ratio(union_hits, target),
            "matched_score_recall": ratio(score_hits, target),
            "candidate_width_mean": ratio(
                int(row["candidate_width_sum"]), int(row["transitions"])
            ),
        }
    )
    return row


def analyze(
    run_dir: Path,
    requests: list[dict[str, Any]],
    lookup: dict[tuple[str, int], Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    delta_counts: defaultdict[tuple[Any, ...], defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    radius_counts: defaultdict[tuple[Any, ...], defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    transitions = 0
    empty_entry_transitions = 0
    new_token_selected = 0

    total_chunks = len(requests) * 48
    chunk_number = 0
    for request in requests:
        rid = request["rid"]
        trajectory = request["trajectory_id"]
        category = request["category"]
        for layer in range(48):
            chunk_number += 1
            chunk = load_chunk(lookup[(rid, layer)])
            for step in range(31):
                common = min(
                    int(chunk["score_lens"][step]),
                    int(chunk["score_lens"][step + 1]),
                )
                scores = chunk["scores"][step, :common]
                if not np.isfinite(scores).all():
                    raise ValueError(f"non-finite scores: {rid}, L{layer}, step {step}")

                previous = chunk["indices"][step]
                previous = np.unique(
                    previous[(previous >= 0) & (previous < common)].astype(np.int64)
                )
                current_raw = chunk["indices"][step + 1]
                current = np.unique(
                    current_raw[(current_raw >= 0) & (current_raw < common)].astype(
                        np.int64
                    )
                )
                appended = int(
                    np.count_nonzero(
                        (current_raw >= common)
                        & (current_raw < chunk["score_lens"][step + 1])
                    )
                )
                new_token_selected += appended
                if len(previous) != TARGET_K or len(current) not in {
                    TARGET_K - 1,
                    TARGET_K,
                }:
                    raise ValueError(
                        f"unexpected top-k width: {rid}, L{layer}, step {step}, "
                        f"previous={len(previous)}, current={len(current)}"
                    )

                previous_mask = np.zeros(common, dtype=bool)
                previous_mask[previous] = True
                current_mask = np.zeros(common, dtype=bool)
                current_mask[current] = True
                entries = current[~previous_mask[current]]
                overlap = len(current) - len(entries)
                if not len(entries):
                    empty_entry_transitions += 1
                nonselected = common - len(previous)
                order = np.argsort(-scores, kind="stable")
                score_hit_prefix = np.cumsum(current_mask[order], dtype=np.int32)

                def matched_score_hits(width: int) -> int:
                    return int(score_hit_prefix[width - 1]) if width else 0

                common_meta = {
                    "transitions": 1,
                    "entry_count": len(entries),
                    "nonselected_count": nonselected,
                    "target_count": len(current),
                }

                frontiers = {
                    delta: shifted_frontier(previous, previous_mask, delta)
                    for delta in range(-max(RADII), max(RADII) + 1)
                    if delta
                }
                for delta in DELTAS:
                    candidates = frontiers[delta]
                    hits = int(np.count_nonzero(current_mask[candidates]))
                    width = min(common, len(previous) + len(candidates))
                    score_hits = matched_score_hits(width)
                    key = (trajectory, category, layer, delta)
                    add_counts(
                        delta_counts[key],
                        {
                            **common_meta,
                            "candidate_count": len(candidates),
                            "hit_count": hits,
                            "union_hit_count": overlap + hits,
                            "score_hit_count": score_hits,
                            "candidate_width_sum": width,
                        },
                    )

                radius_set = set(RADII)
                neighborhood_mask = np.zeros(common, dtype=bool)
                for radius in range(1, max(RADII) + 1):
                    neighborhood_mask[frontiers[-radius]] = True
                    neighborhood_mask[frontiers[radius]] = True
                    if radius not in radius_set:
                        continue
                    candidate_count = int(np.count_nonzero(neighborhood_mask))
                    hits = int(np.count_nonzero(neighborhood_mask & current_mask))
                    width = min(common, len(previous) + candidate_count)
                    score_hits = matched_score_hits(width)
                    key = (trajectory, category, layer, radius)
                    add_counts(
                        radius_counts[key],
                        {
                            **common_meta,
                            "candidate_count": candidate_count,
                            "hit_count": hits,
                            "union_hit_count": overlap + hits,
                            "score_hit_count": score_hits,
                            "candidate_width_sum": width,
                        },
                    )
                transitions += 1

            if chunk_number % 96 == 0 or chunk_number == total_chunks:
                print(
                    f"[{chunk_number:04d}/{total_chunks}] analyzed {rid}, L{layer}",
                    flush=True,
                )

    def make_table(
        counts: dict[tuple[Any, ...], dict[str, int]], parameter: str
    ) -> pd.DataFrame:
        rows = []
        for (trajectory, category, layer, value), values in counts.items():
            row: dict[str, Any] = {
                "trajectory_id": trajectory,
                "category": category,
                "layer": layer,
                parameter: value,
                **dict(values),
            }
            rows.append(finalize_counts(row))
        return pd.DataFrame(rows).sort_values(
            [parameter, "layer", "trajectory_id"]
        )

    validation = {
        "requests": len(requests),
        "trajectories": len({row["trajectory_id"] for row in requests}),
        "layers": 48,
        "steps_per_chunk": 32,
        "adjacent_transitions": transitions,
        "expected_transitions": len(requests) * 48 * 31,
        "empty_entry_transitions": empty_entry_transitions,
        "next_new_token_selected": new_token_selected,
        "deltas": DELTAS,
        "radii": RADII,
    }
    if transitions != validation["expected_transitions"]:
        raise AssertionError("transition count mismatch")
    return (
        make_table(delta_counts, "delta"),
        make_table(radius_counts, "radius"),
        validation,
    )


def trajectory_summary(table: pd.DataFrame, parameter: str) -> pd.DataFrame:
    count_cols = [
        "transitions",
        "entry_count",
        "nonselected_count",
        "target_count",
        "candidate_count",
        "hit_count",
        "union_hit_count",
        "score_hit_count",
        "candidate_width_sum",
    ]
    grouped = table.groupby(["trajectory_id", parameter], as_index=False)[
        count_cols
    ].sum()
    return pd.DataFrame([finalize_counts(row) for row in grouped.to_dict("records")])


def bootstrap_mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    mean = float(array.mean())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(
        array, size=(BOOTSTRAP_SAMPLES, len(array)), replace=True
    ).mean(axis=1)
    return mean, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def aggregate_summary(table: pd.DataFrame, parameter: str) -> pd.DataFrame:
    trajectories = trajectory_summary(table, parameter)
    rows = []
    metrics = [
        "entry_recall",
        "candidate_precision",
        "entry_prevalence",
        "precision_lift",
        "union_recall",
        "matched_score_recall",
        "candidate_width_mean",
    ]
    for value, part in trajectories.groupby(parameter):
        row: dict[str, Any] = {parameter: int(value)}
        for metric in metrics:
            mean, low, high = bootstrap_mean_ci(part[metric])
            row[metric] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(parameter)


def set_style() -> None:
    # Keep the style local and explicit.  This also makes the analysis portable
    # when the shared research style package is not installed on another host.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta(summary: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    color = "#0072B2"
    axes[0].plot(summary.delta, summary.precision_lift, color=color, marker="o")
    axes[0].fill_between(
        summary.delta,
        summary.precision_lift_ci_low,
        summary.precision_lift_ci_high,
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    axes[0].axhline(1.0, color="#4D4D4D", linestyle="--", linewidth=1.2)
    axes[0].set(
        xlabel="Position shift delta (tokens)",
        ylabel="New-entry precision lift vs. chance (x)",
    )
    axes[1].plot(summary.delta, summary.entry_recall * 100, color=color, marker="o")
    axes[1].fill_between(
        summary.delta,
        summary.entry_recall_ci_low * 100,
        summary.entry_recall_ci_high * 100,
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    axes[1].set(
        xlabel="Position shift delta (tokens)",
        ylabel="Next-step new entries covered (%)",
    )
    for axis in axes:
        axis.axvline(0, color="#9E9E9E", linewidth=0.8)
    save_figure(fig, figure_dir / "topk_shift_delta")


def plot_radius(summary: pd.DataFrame, figure_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    axis.plot(
        summary.candidate_width_mean,
        summary.union_recall * 100,
        color="#D55E00",
        marker="o",
        label="Shift-neighborhood union",
    )
    axis.plot(
        summary.candidate_width_mean,
        summary.matched_score_recall * 100,
        color="#4D4D4D",
        marker="s",
        linestyle="--",
        label="Previous-score rank (same width)",
    )
    for row in summary.itertuples():
        offset = {
            16: (-24, 5),
            32: (8, 9),
        }.get(row.radius, (3, 4))
        axis.annotate(
            f"r={row.radius}",
            (row.candidate_width_mean, row.union_recall * 100),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    axis.set(
        xlabel="Mean candidate set size (tokens)",
        ylabel="Next-step top-2048 recall (%)",
    )
    axis.set_ylim(top=100.25)
    axis.legend(frameon=False)
    save_figure(fig, figure_dir / "topk_shift_candidate_tradeoff")


def plot_layer_heatmap(table: pd.DataFrame, figure_dir: Path) -> None:
    selected = table[table.delta.between(-8, 8)].copy()
    layer_means = (
        selected.groupby(["layer", "delta"], as_index=False)
        .precision_lift.mean()
        .pivot(index="layer", columns="delta", values="precision_lift")
    )
    fig, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    image = axis.imshow(layer_means.to_numpy(), aspect="auto", cmap="viridis")
    axis.set(
        xlabel="Position shift delta (tokens)",
        ylabel="DSA layer",
        xticks=np.arange(len(layer_means.columns)),
        xticklabels=layer_means.columns,
    )
    axis.set_yticks(np.arange(0, 48, 4), labels=np.arange(0, 48, 4))
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("New-entry precision lift vs. chance (x)")
    save_figure(fig, figure_dir / "topk_shift_layer_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from existing tables without rereading trace chunks.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else run_dir / "analysis" / "topk-shift-v01"
    )
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        delta_table = pd.read_parquet(
            table_dir / "shift_by_trajectory_layer.parquet"
        )
        radius_table = pd.read_parquet(
            table_dir / "shift_radius_by_trajectory_layer.parquet"
        )
        delta_summary = pd.read_csv(table_dir / "shift_delta_summary.csv")
        radius_summary = pd.read_csv(table_dir / "shift_radius_summary.csv")
        validation = json.loads((output_dir / "validation.json").read_text())
    else:
        requests = read_jsonl(run_dir / "prepared_requests.jsonl")
        lookup = build_lookup(run_dir, requests)
        delta_table, radius_table, validation = analyze(run_dir, requests, lookup)
        delta_summary = aggregate_summary(delta_table, "delta")
        radius_summary = aggregate_summary(radius_table, "radius")

        delta_table.to_parquet(
            table_dir / "shift_by_trajectory_layer.parquet", index=False
        )
        radius_table.to_parquet(
            table_dir / "shift_radius_by_trajectory_layer.parquet", index=False
        )
        delta_summary.to_csv(table_dir / "shift_delta_summary.csv", index=False)
        radius_summary.to_csv(table_dir / "shift_radius_summary.csv", index=False)

    set_style()
    plot_delta(delta_summary, figure_dir)
    plot_radius(radius_summary, figure_dir)
    plot_layer_heatmap(delta_table, figure_dir)

    best = delta_summary.loc[delta_summary.precision_lift.idxmax()]
    delta_one = delta_summary.loc[delta_summary.delta == 1].iloc[0]
    radius_one = radius_summary.loc[radius_summary.radius == 1].iloc[0]
    summary = {
        "metric_definition": {
            "new_entry": "token in next-step top-k but not previous-step top-k",
            "shift_candidate": "j not in previous top-k and j-delta in previous top-k",
            "precision_lift": "P(new-entry | shift-candidate) / P(new-entry | previous-nonselected)",
            "matched_score_recall": "previous-score top-K recall at the same candidate width as the shift union",
        },
        "delta_1": delta_one.to_dict(),
        "best_delta_by_precision_lift": best.to_dict(),
        "radius_1": radius_one.to_dict(),
        "validation": validation,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "reproducibility.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "script": str(Path(__file__).resolve()),
                "code_commit": git_revision(Path(__file__).resolve().parents[2]),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "aggregation": "trajectory-equal mean after summing transition counts",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
