#!/usr/bin/env python3
"""Render publication-ready Figure 3 heatmaps from analyzed data files only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


AGE_BINS = ["0", "1", "2", "3", "4-7", "8+"]
TYPE_ORDER = [
    "system_instruction",
    "tool_schema",
    "user_turn",
    "assistant_tool_call",
    "tool_result",
    "assistant_response",
]
TYPE_LABELS = {
    "system_instruction": "System",
    "tool_schema": "Tool schema",
    "user_turn": "User turn",
    "assistant_tool_call": "Asst. tool call",
    "tool_result": "Tool result",
    "assistant_response": "Asst. response",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#555555")
    ax.tick_params(direction="in", top=False, right=False)


def recall_matrix(data_dir: Path) -> tuple[np.ndarray, list[int], list[int]]:
    data = pd.read_csv(data_dir / "topk_recall_heatmap.csv")
    x_values = [0, 1, 2, 4, 8, 16]
    y_values = [8, 4, 2, 1, 0]
    matrix = np.full((len(y_values), len(x_values)), np.nan)
    for row_index, delta_layer in enumerate(y_values):
        for column_index, delta_step in enumerate(x_values):
            matched = data[
                (data.delta_layer == delta_layer) & (data.delta_step == delta_step)
            ]
            if not matched.empty:
                matrix[row_index, column_index] = float(matched.iloc[0].mean_recall)
    return matrix, x_values, y_values


def draw_recall(
    ax: plt.Axes,
    data_dir: Path,
    *,
    add_colorbar: bool = True,
    colorbar_label: str = "Top-k recall",
) -> None:
    matrix, x_values, y_values = recall_matrix(data_dir)
    cmap = mpl.colormaps["YlGnBu"].copy()
    cmap.set_bad("#f4f4f4")
    image = ax.imshow(matrix, cmap=cmap, vmin=0.54, vmax=0.82, aspect="auto")
    ax.set_xticks(range(len(x_values)), [str(value) for value in x_values])
    ax.set_yticks(range(len(y_values)), [str(value) for value in y_values])
    ax.set_xlabel(r"Past step distance  $\Delta t$")
    ax.set_ylabel(r"Past layer distance  $\Delta l$")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isnan(value):
                ax.text(column, row, "—", ha="center", va="center", color="#777777", fontsize=8)
                continue
            color = "white" if value > 0.70 else "#20303c"
            ax.text(
                column,
                row,
                f"{100 * value:.0f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7.2,
            )
    style_axes(ax)
    if add_colorbar:
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
        colorbar.set_label(colorbar_label)
        colorbar.ax.tick_params(direction="in", labelsize=7.5)
        colorbar.outline.set_linewidth(0.6)


def instance_matrix(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray, list[tuple[str, int, int]]]:
    compact_path = data_dir / "figure3b_instance_heatmap.csv"
    if compact_path.exists():
        regions = pd.read_csv(compact_path)
        matrix = regions[[f"age_{age}" for age in AGE_BINS]].to_numpy(dtype=float)
        groups = []
        for region_type in TYPE_ORDER:
            positions = np.flatnonzero(regions.region_type.to_numpy() == region_type)
            if positions.size:
                groups.append((region_type, int(positions.min()), int(positions.max()) + 1))
        return regions, matrix, groups
    activations = pd.read_parquet(data_dir / "region_activation.parquet")
    regions = pd.read_parquet(data_dir / "regions.parquet")
    regions["type_rank"] = regions.region_type.map(
        {name: index for index, name in enumerate(TYPE_ORDER)}
    )
    regions = regions.sort_values(
        ["type_rank", "session_id", "created_turn", "region_id"], kind="stable"
    ).reset_index(drop=True)
    grouped = (
        activations.groupby(["region_id", "turn_age_bin"], as_index=False)
        .activation_enrichment.mean()
    )
    lookup = {
        (row.region_id, row.turn_age_bin): float(row.activation_enrichment)
        for row in grouped.itertuples()
    }
    matrix = np.full((len(regions), len(AGE_BINS)), np.nan)
    for row_index, region in regions.iterrows():
        for column_index, age in enumerate(AGE_BINS):
            matrix[row_index, column_index] = lookup.get((region.region_id, age), np.nan)
    groups = []
    for region_type in TYPE_ORDER:
        positions = np.flatnonzero(regions.region_type.to_numpy() == region_type)
        if positions.size:
            groups.append((region_type, int(positions.min()), int(positions.max()) + 1))
    return regions, matrix, groups


def draw_instances(
    ax: plt.Axes,
    data_dir: Path,
    *,
    add_colorbar: bool = True,
    label_instances: bool = False,
    show_ylabel: bool = True,
) -> None:
    regions, matrix, groups = instance_matrix(data_dir)
    log_matrix = np.log2(matrix)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#f2f2f2")
    image = ax.imshow(
        log_matrix,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0),
        interpolation="nearest",
        aspect="auto",
        rasterized=True,
    )
    ax.set_xticks(range(len(AGE_BINS)), AGE_BINS)
    ax.set_xlabel("Region age (turns)")
    if label_instances:
        labels = [f"{row.session_id} · {row.region_label}" for row in regions.itertuples()]
        ax.set_yticks(range(len(regions)), labels)
        ax.tick_params(axis="y", labelsize=5.5)
    else:
        centers = [(start + end - 1) / 2 for _, start, end in groups]
        ax.set_yticks(centers, [TYPE_LABELS[name] for name, _, _ in groups])
    ax.set_ylabel("Region instances" if show_ylabel else "")
    for _, _, end in groups[:-1]:
        ax.axhline(end - 0.5, color="white", linewidth=1.0)
        ax.axhline(end - 0.5, color="#777777", linewidth=0.35)
    style_axes(ax)
    if add_colorbar:
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.032, pad=0.025)
        ticks = np.arange(-3, 4)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels(["0.125×", "0.25×", "0.5×", "1×", "2×", "4×", "8×"])
        colorbar.set_label("Activation enrichment")
        colorbar.ax.tick_params(direction="in", labelsize=7.2)
        colorbar.outline.set_linewidth(0.6)


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    fig_a, ax_a = plt.subplots(figsize=(3.25, 2.55))
    fig_a.subplots_adjust(left=0.18, right=0.89, bottom=0.22, top=0.97)
    draw_recall(ax_a, args.data_dir)
    save(fig_a, args.output_dir / "figure3a_recent_topk_recall")

    fig_b, ax_b = plt.subplots(figsize=(4.75, 3.10))
    fig_b.subplots_adjust(left=0.22, right=0.91, bottom=0.18, top=0.97)
    draw_instances(ax_b, args.data_dir)
    save(fig_b, args.output_dir / "figure3b_region_instance_activation")

    fig_full, ax_full = plt.subplots(figsize=(8.0, 15.0))
    fig_full.subplots_adjust(left=0.42, right=0.91, bottom=0.06, top=0.99)
    draw_instances(ax_full, args.data_dir, label_instances=True)
    save(fig_full, args.output_dir / "figure3b_region_instance_activation_labeled")

    combined = plt.figure(figsize=(7.15, 3.12))
    grid = combined.add_gridspec(
        1, 2, width_ratios=[0.83, 1.45], left=0.075, right=0.985, bottom=0.18, top=0.96, wspace=0.58
    )
    combined_a = combined.add_subplot(grid[0, 0])
    combined_b = combined.add_subplot(grid[0, 1])
    draw_recall(combined_a, args.data_dir, colorbar_label="Recall")
    draw_instances(combined_b, args.data_dir, show_ylabel=False)
    combined_a.text(-0.30, 1.04, "(a)", transform=combined_a.transAxes, fontsize=9)
    combined_b.text(-0.25, 1.04, "(b)", transform=combined_b.transAxes, fontsize=9)
    save(combined, args.output_dir / "figure3_tau3_topk_predictability")

    regions, matrix, _ = instance_matrix(args.data_dir)
    regions.assign(figure_row=np.arange(len(regions))).to_csv(
        args.output_dir / "figure3b_row_order.csv", index=False
    )
    matrix_rows = regions[
        ["session_id", "domain", "region_id", "region_type", "region_label", "created_turn"]
    ].copy()
    for column_index, age in enumerate(AGE_BINS):
        matrix_rows[f"age_{age}"] = matrix[:, column_index]
    matrix_rows.to_csv(args.output_dir / "figure3b_instance_heatmap.csv", index=False)
    activation_path = args.data_dir / "region_activation.parquet"
    if activation_path.exists():
        activations = pd.read_parquet(activation_path)
        type_age = (
            activations.groupby(["region_type", "turn_age_bin"])["activation_enrichment"]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
        )
        type_age.to_csv(args.output_dir / "figure3b_type_age_summary.csv", index=False)


if __name__ == "__main__":
    main()
