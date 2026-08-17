#!/usr/bin/env python3
"""Plot seven-layer error for Gate E1 policy deltas against full48 replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "LongBench-v2": "#0072B2",
    "RULER": "#D55E00",
}
STYLE_GUIDE = "/home10T/bzx/.codex/skills/research-figure-style/references/style-guide.md"


def normalize(value: str) -> str:
    return {"longbench-v2": "LongBench-v2", "ruler": "RULER"}.get(value.lower(), value)


def short(value: str) -> str:
    return {"LongBench-v2": "LB-v2"}.get(normalize(value), normalize(value))


def set_style() -> None:
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.analysis_dir / "sampling_error_summary.csv")
    request = pd.read_csv(args.analysis_dir / "sampling_error_by_request.csv")
    summary["dataset"] = summary.dataset.map(normalize)
    request["dataset"] = request.dataset.map(normalize)
    metrics = [
        ("hbm_page_recall_delta_pp", "Recall-delta sampling error (pp)"),
        (
            "correction_mib_per_token_per_gpu_delta",
            "Correction-delta sampling error\n(MiB/token/GPU)",
        ),
    ]
    groups = sorted(
        summary[["dataset", "context_config"]].drop_duplicates().itertuples(index=False, name=None)
    )
    labels = [f"{short(str(dataset))}\n{int(length / 1024)}K" for dataset, length in groups]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for axis, (metric, ylabel) in zip(axes, metrics):
        part = summary[summary.metric == metric].set_index(["dataset", "context_config"])
        for index, key in enumerate(groups):
            row = part.loc[key]
            mean = float(row.task_balanced_bias_mean)
            low = float(row.task_cluster_bias_ci95_low)
            high = float(row.task_cluster_bias_ci95_high)
            axis.errorbar(
                index,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                color=COLORS[str(key[0])],
                capsize=3,
                markersize=5,
                linewidth=1.3,
            )
        axis.axhline(0, color="#4D4D4D", linestyle="--", linewidth=1.0)
        axis.set_xticks(np.arange(len(groups)), labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    fig.subplots_adjust(bottom=0.24, wspace=0.42)
    save(fig, args.output_dir, "gate_e1_layer_sampling_bias")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    scatter_labels = [
        ("hbm_page_recall_delta_pp", "HBM page recall delta (pp)"),
        (
            "correction_mib_per_token_per_gpu_delta",
            "Correction traffic delta (MiB/token/GPU)",
        ),
    ]
    for axis, (metric, label) in zip(axes, scatter_labels):
        part = request[request.metric == metric]
        for dataset, values in part.groupby("dataset"):
            axis.scatter(
                values.full48,
                values.seven_layer_estimate,
                color=COLORS[dataset],
                s=23,
                alpha=0.85,
                label=dataset,
            )
        low = min(part.full48.min(), part.seven_layer_estimate.min())
        high = max(part.full48.max(), part.seven_layer_estimate.max())
        axis.plot([low, high], [low, high], color="#4D4D4D", linestyle="--", linewidth=1.0)
        axis.set_xlabel(f"48-layer measurement\n{label}")
        axis.set_ylabel(f"7-layer estimate\n{label}")
        axis.grid(color="#E6E6E6", linewidth=0.7)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.8, bottom=0.24, wspace=0.42)
    save(fig, args.output_dir, "gate_e1_layer_sampling_request_scatter")

    metadata = {
        "schema_version": 1,
        "input_analysis_dir": str(args.analysis_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "style_guide": STYLE_GUIDE,
        "formats": ["pdf", "png"],
        "png_dpi": 300,
        "interval": "95% task-cluster bootstrap CI",
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
