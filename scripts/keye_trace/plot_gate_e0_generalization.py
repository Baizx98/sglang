#!/usr/bin/env python3
"""Create publication-style Gate E0 generalization and calibration figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

STYLE_GUIDE = "/home10T/bzx/.codex/skills/research-figure-style/references/style-guide.md"
COLORS = {
    "LongBench-v2": "#0072B2",
    "RULER": "#D55E00",
    "InfiniteBench": "#009E73",
}
MARKERS = {"full48": "o", "seven_layer_nearest_weighted": "s"}
LABELS = {"full48": "48-layer measurement", "seven_layer_nearest_weighted": "7-layer estimate"}


def normalize_dataset(value: str) -> str:
    key = value.lower()
    if key == "longbench-v2":
        return "LongBench-v2"
    if key == "ruler":
        return "RULER"
    if key == "infinitebench":
        return "InfiniteBench"
    return value


def short_dataset(value: str) -> str:
    return {"LongBench-v2": "LB-v2", "InfiniteBench": "InfiniteBench"}.get(
        normalize_dataset(value), normalize_dataset(value)
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_layer_profiles(full48: pd.DataFrame, output_dir: Path) -> None:
    grouped = full48.groupby(["dataset", "length_config", "layer"], as_index=False)[
        ["previous_top4096_token_recall", "previous_top4096_page_recall"]
    ].mean()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75), sharex=True)
    metrics = [
        ("previous_top4096_token_recall", "Token recall (%)"),
        ("previous_top4096_page_recall", "Page recall (%)"),
    ]
    for axis, (metric, label) in zip(axes, metrics):
        for (dataset, length), part in grouped.groupby(["dataset", "length_config"]):
            display = normalize_dataset(str(dataset))
            linestyle = "-" if int(length) == 65536 else "--"
            axis.plot(
                part.layer,
                100 * part[metric],
                color=COLORS[display],
                linestyle=linestyle,
                linewidth=1.8,
                label=f"{display}, {int(length / 1024)}K",
            )
        axis.set_xlabel("DSA layer")
        axis.set_ylabel(label)
        axis.set_xlim(0, 47)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    save(fig, output_dir, "gate_e0_full48_layer_profiles")


def plot_calibration(calibration: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("previous_top4096_token_recall", "Token recall bias (pp)"),
        ("previous_top4096_page_recall", "Page recall bias (pp)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharex=True)
    order = sorted(
        calibration[["dataset", "length_config"]].drop_duplicates().itertuples(index=False, name=None)
    )
    labels = [f"{short_dataset(str(d))}\n{int(n / 1024)}K" for d, n in order]
    for axis, (metric, ylabel) in zip(axes, metrics):
        part = calibration[calibration.metric == metric].set_index(["dataset", "length_config"])
        means = np.array([part.loc[key, "bias_mean"] for key in order]) * 100
        lows = np.array([part.loc[key, "bias_ci95_low"] for key in order]) * 100
        highs = np.array([part.loc[key, "bias_ci95_high"] for key in order]) * 100
        colors = [COLORS[normalize_dataset(str(key[0]))] for key in order]
        x = np.arange(len(order))
        axis.axhline(0, color="#4D4D4D", linewidth=1.0, linestyle="--")
        for i in x:
            axis.errorbar(
                i,
                means[i],
                yerr=[[means[i] - lows[i]], [highs[i] - means[i]]],
                fmt="o",
                color=colors[i],
                capsize=3,
                linewidth=1.4,
                markersize=5,
            )
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    save(fig, output_dir, "gate_e0_seven_layer_calibration_bias")


def plot_generalization(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("previous_top4096_token_recall", "Token recall (%)"),
        ("previous_top4096_page_recall", "Page recall (%)"),
    ]
    groups = sorted(
        summary[["dataset", "length_config"]].drop_duplicates().itertuples(index=False, name=None)
    )
    labels = [f"{short_dataset(str(d))}\n{int(n / 1024)}K" for d, n in groups]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharex=True)
    offsets = {"full48": -0.10, "seven_layer_nearest_weighted": 0.10}
    for axis, (metric, ylabel) in zip(axes, metrics):
        part = summary[summary.metric == metric]
        for estimator in ["full48", "seven_layer_nearest_weighted"]:
            estimator_rows = part[part.estimator == estimator].set_index(["dataset", "length_config"])
            for i, key in enumerate(groups):
                if key not in estimator_rows.index:
                    continue
                row = estimator_rows.loc[key]
                mean = 100 * float(row["task_balanced_mean"])
                low = 100 * float(row["task_cluster_ci95_low"])
                high = 100 * float(row["task_cluster_ci95_high"])
                axis.errorbar(
                    i + offsets[estimator],
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt=MARKERS[estimator],
                    color=COLORS[normalize_dataset(str(key[0]))],
                    markerfacecolor="white" if estimator == "full48" else COLORS[normalize_dataset(str(key[0]))],
                    capsize=2.5,
                    linewidth=1.2,
                    markersize=5,
                )
        axis.set_xticks(np.arange(len(groups)), labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    legend = [
        Line2D([0], [0], marker=MARKERS[key], color="#4D4D4D", linestyle="none",
               markerfacecolor="white" if key == "full48" else "#4D4D4D", label=LABELS[key])
        for key in ["full48", "seven_layer_nearest_weighted"]
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
    save(fig, output_dir, "gate_e0_cross_dataset_k4096_recall")


def plot_prompt_length(requests: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharex=True)
    metrics = [
        ("previous_top4096_token_recall", "Token recall (%)"),
        ("previous_top4096_page_recall", "Page recall (%)"),
    ]
    for axis, (metric, ylabel) in zip(axes, metrics):
        for (dataset, estimator), part in requests.groupby(["dataset", "estimator"]):
            display = normalize_dataset(str(dataset))
            axis.scatter(
                part.prompt_len / 1024,
                100 * part[metric],
                color=COLORS[display],
                marker=MARKERS[estimator],
                facecolors="none" if estimator == "full48" else COLORS[display],
                s=24,
                alpha=0.8,
                linewidths=0.9,
            )
        axis.set_xlabel("Prompt length (K tokens)")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    legend = [
        Line2D([0], [0], marker="o", color=color, linestyle="none", label=name)
        for name, color in COLORS.items()
    ]
    legend.extend(
        [
            Line2D(
                [0],
                [0],
                marker=MARKERS[key],
                color="#4D4D4D",
                linestyle="none",
                markerfacecolor="white" if key == "full48" else "#4D4D4D",
                label=LABELS[key],
            )
            for key in ["full48", "seven_layer_nearest_weighted"]
        ]
    )
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=5, frameon=False)
    save(fig, output_dir, "gate_e0_prompt_length_vs_recall")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--full48-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # The bundled style's axes.prop_cycle is rejected by this matplotlib
    # version, so apply the same documented choices explicitly.
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables = args.analysis_dir / "tables"
    summary = pd.read_csv(tables / "dataset_length_summary.csv")
    calibration = pd.read_csv(tables / "layer_sampling_calibration_summary.csv")
    requests = pd.read_csv(tables / "request_metrics.csv")
    for frame in [summary, calibration, requests]:
        frame["dataset"] = frame.dataset.map(normalize_dataset)
    full48 = pd.read_parquet(args.full48_table)
    plot_layer_profiles(full48, args.output_dir)
    plot_calibration(calibration, args.output_dir)
    plot_generalization(summary, args.output_dir)
    plot_prompt_length(requests, args.output_dir)
    metadata = {
        "schema_version": 1,
        "input_analysis_dir": str(args.analysis_dir.resolve()),
        "input_full48_table": str(args.full48_table.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "style_guide": STYLE_GUIDE,
        "style_application": "explicit rcParams; bundled cycle incompatible with installed matplotlib",
        "formats": ["pdf", "png"],
        "png_dpi": 300,
        "aggregation": "task-balanced mean; 95% bootstrap CI over task means",
        "marker_semantics": LABELS,
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
