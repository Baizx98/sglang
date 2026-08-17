#!/usr/bin/env python3
"""Plot the full48 per-layer Gate E1 policy delta profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#4D4D4D"
STYLE_GUIDE = "/home10T/bzx/.codex/skills/research-figure-style/references/style-guide.md"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.analysis_dir / "layer_summary.csv")
    panels = [
        ("hbm_page_recall_delta_pp", "HBM page recall delta (pp)"),
        (
            "correction_mib_per_token_per_gpu_delta",
            "Correction traffic delta\n(MiB/token/GPU/layer)",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharex=True)
    for axis, (metric, ylabel) in zip(axes, panels):
        part = data[data.metric == metric].sort_values("layer")
        x = part.layer.to_numpy(int)
        mean = part["mean"].to_numpy(float)
        low = part.ci95_low.to_numpy(float)
        high = part.ci95_high.to_numpy(float)
        axis.fill_between(x, low, high, color=BLUE, alpha=0.17, linewidth=0)
        axis.plot(x, mean, color=BLUE, linewidth=1.4)
        harmful = mean < 0 if "recall" in metric else mean > 0
        axis.scatter(
            x[harmful],
            mean[harmful],
            color=ORANGE,
            s=20,
            zorder=3,
            label="Mean moves in harmful direction",
        )
        axis.axhline(0, color=GRAY, linestyle="--", linewidth=1.0)
        axis.set_xlabel("DSA layer")
        axis.set_ylabel(ylabel)
        axis.set_xticks(np.arange(0, 48, 8))
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            frameon=False,
        )
        fig.subplots_adjust(top=0.82, bottom=0.2, wspace=0.42)
    else:
        fig.subplots_adjust(bottom=0.2, wspace=0.42)
    stem = "gate_e1_full48_layer_profile"
    fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "schema_version": 1,
        "input_analysis_dir": str(args.analysis_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "style_guide": STYLE_GUIDE,
        "formats": ["pdf", "png"],
        "png_dpi": 300,
        "interval": "95% request bootstrap CI",
        "scope": "post-hoc layer diagnosis only; frozen policy unchanged; no speedup",
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
