#!/usr/bin/env python3
"""Plot Gate E1 resident-page protection quality and traffic deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

COLORS = {
    "LongBench-v2": "#0072B2",
    "RULER": "#D55E00",
    "InfiniteBench": "#009E73",
}
STYLE_GUIDE = "/home10T/bzx/.codex/skills/research-figure-style/references/style-guide.md"


def normalize_dataset(value: str) -> str:
    mapping = {
        "longbench-v2": "LongBench-v2",
        "ruler": "RULER",
        "infinitebench": "InfiniteBench",
    }
    return mapping.get(value.lower(), value)


def short_dataset(value: str) -> str:
    return {"LongBench-v2": "LB-v2"}.get(normalize_dataset(value), normalize_dataset(value))


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


def plot_request_deltas(paired: pd.DataFrame, output_dir: Path) -> None:
    ordered = paired.sort_values("hbm_page_recall_delta_pp").reset_index(drop=True)
    x = np.arange(len(ordered))
    colors = [COLORS[value] for value in ordered.dataset]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharex=True)
    panels = [
        ("hbm_page_recall_delta_pp", "HBM page recall delta (pp)"),
        (
            "correction_mib_per_token_per_gpu_delta",
            "Correction traffic delta\n(MiB/token/GPU)",
        ),
    ]
    for axis, (metric, ylabel) in zip(axes, panels):
        axis.axhline(0, color="#4D4D4D", linewidth=1.0, linestyle="--")
        axis.scatter(x, ordered[metric], c=colors, marker="o", s=22, linewidths=0)
        axis.set_xlabel("Requests sorted by recall delta")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    fig.subplots_adjust(top=0.78, bottom=0.2, wspace=0.42)
    legend = [
        Line2D([0], [0], marker="o", color=color, linestyle="none", label=name)
        for name, color in COLORS.items()
        if name in set(ordered.dataset)
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(legend), frameon=False)
    save(fig, output_dir, "gate_e1_request_deltas")


def plot_group_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("hbm_page_recall_delta_pp", "HBM page recall delta (pp)"),
        (
            "correction_mib_per_token_per_gpu_delta",
            "Correction traffic delta\n(MiB/token/GPU)",
        ),
    ]
    groups = sorted(
        summary[["dataset", "context_config"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    labels = [f"{short_dataset(str(dataset))}\n{int(length / 1024)}K" for dataset, length in groups]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharex=True)
    for axis, (metric, ylabel) in zip(axes, metrics):
        part = summary[summary.metric == metric].set_index(["dataset", "context_config"])
        for index, key in enumerate(groups):
            row = part.loc[key]
            mean = float(row.task_balanced_mean)
            low = float(row.task_cluster_ci95_low)
            high = float(row.task_cluster_ci95_high)
            axis.errorbar(
                index,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                color=COLORS[str(key[0])],
                markersize=5,
                linewidth=1.3,
                capsize=3,
            )
        axis.axhline(0, color="#4D4D4D", linewidth=1.0, linestyle="--")
        axis.set_xticks(np.arange(len(groups)), labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    fig.subplots_adjust(bottom=0.24, wspace=0.42)
    save(fig, output_dir, "gate_e1_dataset_length_deltas")


def plot_traffic_tradeoff(paired: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(4.2, 2.7))
    for dataset, part in paired.groupby("dataset"):
        axis.scatter(
            part.total_pcie_mib_per_token_per_gpu_delta,
            part.correction_mib_per_token_per_gpu_delta,
            color=COLORS[dataset],
            label=dataset,
            s=25,
            alpha=0.85,
        )
    axis.axhline(0, color="#4D4D4D", linewidth=0.9, linestyle="--")
    axis.axvline(0, color="#4D4D4D", linewidth=0.9, linestyle="--")
    axis.set_xlabel("Total PCIe traffic delta (MiB/token/GPU)")
    axis.set_ylabel("Correction traffic delta\n(MiB/token/GPU)")
    axis.grid(color="#E6E6E6", linewidth=0.7)
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    fig.subplots_adjust(top=0.79, bottom=0.22, left=0.18, right=0.98)
    save(fig, output_dir, "gate_e1_traffic_tradeoff")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(args.analysis_dir / "tables/paired_by_request.csv")
    summary = pd.read_csv(args.analysis_dir / "tables/dataset_length_summary.csv")
    paired["dataset"] = paired.dataset.map(normalize_dataset)
    summary["dataset"] = summary.dataset.map(normalize_dataset)
    plot_request_deltas(paired, args.output_dir)
    plot_group_summary(summary, args.output_dir)
    plot_traffic_tradeoff(paired, args.output_dir)
    metadata = {
        "schema_version": 1,
        "input_analysis_dir": str(args.analysis_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "style_guide": STYLE_GUIDE,
        "formats": ["pdf", "png"],
        "png_dpi": 300,
        "group_interval": "95% bootstrap CI over task means",
        "scope": "shadow quality and transfer-volume deltas; no measured speedup",
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
