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
from mpl_toolkits.axes_grid1 import make_axes_locatable


AGE_BINS = ["0", "1", "2", "3", "4-7", "8+"]
AGE_TICK_LABELS = ["0", "1", "2", "3", "4", "8"]
AGE_PLUS_COLUMNS = (4, 5)
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
    "assistant_tool_call": "Tool call",
    "tool_result": "Tool result",
    "assistant_response": "Response",
}

# Standalone panels are designed in physical inches so their plot frames have
# exactly the same height.  Their widths intentionally differ: panel (b) is
# narrow to avoid stretching 205 instance rows into very long rectangles.
PANEL_HEIGHT = 2.50
PLOT_BOTTOM = 0.68
PLOT_HEIGHT = 1.50
COLORBAR_WIDTH = 0.06
COLORBAR_PAD = 0.04
A_WIDTH = 2.60
A_PLOT_LEFT = 0.45
A_PLOT_WIDTH = 1.45
B_WIDTH = 3.38
B_PLOT_LEFT = 1.33
B_PLOT_WIDTH = 1.25
COMBINED_GAP = 0.08


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            # The two standalone PDFs are expected to be scaled by about 0.56x
            # when placed side by side in a FAST single column.  These source
            # sizes therefore render at roughly 9 pt / 8 pt in the paper.
            "font.size": 16.0,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 0.0,
            "ytick.major.size": 0.0,
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
    # Retain tick labels, but remove tick marks from both heatmap frames.
    ax.tick_params(axis="both", which="both", length=0, top=False, right=False)


def style_colorbar(colorbar: mpl.colorbar.Colorbar, *, labelsize: float = 12.5) -> None:
    """Place colorbar ticks outside on the right without changing its height."""

    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.yaxis.set_label_position("right")
    colorbar.ax.tick_params(
        axis="y",
        which="both",
        direction="out",
        left=False,
        right=True,
        length=3.0,
        width=0.7,
        labelsize=labelsize,
        pad=2.0,
    )
    colorbar.ax.yaxis.label.set_size(13.0)
    colorbar.outline.set_linewidth(0.6)


def add_panel_axes(
    fig: plt.Figure,
    *,
    x_offset: float,
    plot_left: float,
    plot_width: float,
) -> tuple[plt.Axes, plt.Axes]:
    """Create a plot frame and an exactly height-aligned colorbar axis."""

    figure_width = fig.get_figwidth()
    figure_height = fig.get_figheight()
    plot_x = x_offset + plot_left
    colorbar_x = plot_x + plot_width + COLORBAR_PAD
    ax = fig.add_axes(
        [
            plot_x / figure_width,
            PLOT_BOTTOM / figure_height,
            plot_width / figure_width,
            PLOT_HEIGHT / figure_height,
        ]
    )
    colorbar_ax = fig.add_axes(
        [
            colorbar_x / figure_width,
            PLOT_BOTTOM / figure_height,
            COLORBAR_WIDTH / figure_width,
            PLOT_HEIGHT / figure_height,
        ]
    )
    return ax, colorbar_ax


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
    colorbar_ax: plt.Axes | None = None,
    colorbar_label: str = "Recall",
) -> None:
    matrix, x_values, y_values = recall_matrix(data_dir)
    cmap = mpl.colormaps["YlGnBu"].copy()
    cmap.set_bad("#f4f4f4")
    image = ax.imshow(matrix, cmap=cmap, vmin=0.54, vmax=0.82, aspect="auto")
    ax.set_xticks(range(len(x_values)), [str(value) for value in x_values])
    ax.set_yticks(range(len(y_values)), [str(value) for value in y_values])
    ax.set_xlabel(r"Step gap  $\Delta t$")
    ax.set_ylabel(r"Layer gap  $\Delta l$")
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
                fontsize=11.5,
            )
    style_axes(ax)
    if add_colorbar:
        if colorbar_ax is None:
            divider = make_axes_locatable(ax)
            colorbar_ax = divider.append_axes(
                "right", size=COLORBAR_WIDTH, pad=COLORBAR_PAD
            )
        colorbar = ax.figure.colorbar(image, cax=colorbar_ax)
        colorbar.set_label(colorbar_label)
        style_colorbar(colorbar)


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
    colorbar_ax: plt.Axes | None = None,
    label_instances: bool = False,
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
    ax.set_xticks(range(len(AGE_BINS)), AGE_TICK_LABELS)
    # Keep the base digits centered on their heatmap columns.  Draw the compact
    # plus signs separately so their width does not shift the 4/8 tick labels.
    for column in AGE_PLUS_COLUMNS:
        ax.annotate(
            "+",
            xy=(column, 0.0),
            xycoords=ax.get_xaxis_transform(),
            xytext=(5.0, -1.5),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9.0,
            annotation_clip=False,
        )
    ax.set_xlabel("Region age (turns)")
    if label_instances:
        labels = [f"{row.session_id} · {row.region_label}" for row in regions.itertuples()]
        ax.set_yticks(range(len(regions)), labels)
        ax.tick_params(axis="y", labelsize=5.5)
    else:
        centers = [(start + end - 1) / 2 for _, start, end in groups]
        ax.set_yticks(centers, [TYPE_LABELS[name] for name, _, _ in groups])
    # The grouped y tick labels already identify the instance rows; omitting a
    # redundant y-axis title preserves readable type labels at column scale.
    ax.set_ylabel("")
    for _, _, end in groups[:-1]:
        ax.axhline(end - 0.5, color="white", linewidth=1.0)
        ax.axhline(end - 0.5, color="#777777", linewidth=0.35)
    style_axes(ax)
    if add_colorbar:
        if colorbar_ax is None:
            divider = make_axes_locatable(ax)
            colorbar_ax = divider.append_axes(
                "right", size=COLORBAR_WIDTH, pad=COLORBAR_PAD
            )
        colorbar = ax.figure.colorbar(image, cax=colorbar_ax)
        ticks = np.arange(-3, 4)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels(["⅛×", "¼×", "½×", "1×", "2×", "4×", "8×"])
        colorbar.set_label("Enrichment")
        style_colorbar(colorbar, labelsize=12.0)


def save(fig: plt.Figure, stem: Path, *, tight: bool = False) -> None:
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    fig.savefig(stem.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(stem.with_suffix(".png"), dpi=300, **save_kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    fig_a = plt.figure(figsize=(A_WIDTH, PANEL_HEIGHT))
    ax_a, colorbar_a = add_panel_axes(
        fig_a,
        x_offset=0.0,
        plot_left=A_PLOT_LEFT,
        plot_width=A_PLOT_WIDTH,
    )
    draw_recall(ax_a, args.data_dir, colorbar_ax=colorbar_a)
    save(fig_a, args.output_dir / "figure3a_recent_topk_recall")

    fig_b = plt.figure(figsize=(B_WIDTH, PANEL_HEIGHT))
    ax_b, colorbar_b = add_panel_axes(
        fig_b,
        x_offset=0.0,
        plot_left=B_PLOT_LEFT,
        plot_width=B_PLOT_WIDTH,
    )
    draw_instances(ax_b, args.data_dir, colorbar_ax=colorbar_b)
    save(fig_b, args.output_dir / "figure3b_region_instance_activation")

    fig_full, ax_full = plt.subplots(figsize=(8.0, 15.0))
    fig_full.subplots_adjust(left=0.42, right=0.91, bottom=0.06, top=0.99)
    draw_instances(ax_full, args.data_dir, label_instances=True)
    save(fig_full, args.output_dir / "figure3b_region_instance_activation_labeled", tight=True)

    combined_width = A_WIDTH + COMBINED_GAP + B_WIDTH
    combined = plt.figure(figsize=(combined_width, PANEL_HEIGHT))
    combined_a, combined_colorbar_a = add_panel_axes(
        combined,
        x_offset=0.0,
        plot_left=A_PLOT_LEFT,
        plot_width=A_PLOT_WIDTH,
    )
    combined_b, combined_colorbar_b = add_panel_axes(
        combined,
        x_offset=A_WIDTH + COMBINED_GAP,
        plot_left=B_PLOT_LEFT,
        plot_width=B_PLOT_WIDTH,
    )
    draw_recall(
        combined_a,
        args.data_dir,
        colorbar_ax=combined_colorbar_a,
        colorbar_label="Recall",
    )
    draw_instances(
        combined_b,
        args.data_dir,
        colorbar_ax=combined_colorbar_b,
    )
    save(combined, args.output_dir / "figure3_tau3_topk_predictability")

    activation_path = args.data_dir / "region_activation.parquet"
    if activation_path.exists():
        # Only regenerate compact data artifacts from the authoritative raw
        # parquet.  A figure-only rerender from copied CSVs must not round or
        # drop provenance columns in those inputs.
        regions, matrix, _ = instance_matrix(args.data_dir)
        regions.assign(figure_row=np.arange(len(regions))).to_csv(
            args.output_dir / "figure3b_row_order.csv", index=False
        )
        matrix_rows = regions[
            [
                "session_id",
                "domain",
                "region_id",
                "region_type",
                "region_label",
                "created_turn",
            ]
        ].copy()
        for column_index, age in enumerate(AGE_BINS):
            matrix_rows[f"age_{age}"] = matrix[:, column_index]
        matrix_rows.to_csv(
            args.output_dir / "figure3b_instance_heatmap.csv", index=False
        )
        activations = pd.read_parquet(activation_path)
        type_age = (
            activations.groupby(["region_type", "turn_age_bin"])["activation_enrichment"]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
        )
        type_age.to_csv(args.output_dir / "figure3b_type_age_summary.csv", index=False)


if __name__ == "__main__":
    main()
