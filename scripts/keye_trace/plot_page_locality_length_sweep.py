#!/usr/bin/env python3
"""Plot fixed-page locality trends across compact-trace context lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {4: "#0072B2", 16: "#D55E00", 64: "#009E73", 256: "#CC79A7"}
MARKERS = {4: "o", 16: "s", 64: "^", 256: "D"}


def load_run(path: Path) -> list[dict[str, Any]]:
    request = json.loads((path / "requests.jsonl").read_text().splitlines()[0])
    summary = json.loads(
        (path / "analysis" / "topk-page-locality-v01" / "summary.json").read_text()
    )
    return [
        {
            "run_dir": str(path.resolve()),
            "context_tokens": int(request["prompt_len"]),
            **row,
        }
        for row in summary["page_sizes"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for run_dir in args.run_dir for row in load_run(run_dir.resolve())]
    frame = pd.DataFrame(rows).sort_values(["page_size", "context_tokens"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "page_length_sweep.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
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
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), constrained_layout=True)
    for page_size, selected in frame.groupby("page_size"):
        color = COLORS[int(page_size)]
        marker = MARKERS[int(page_size)]
        label = f"{int(page_size)} tokens/page"
        x = selected.context_tokens / 1024
        axes[0].plot(
            x,
            selected.active_page_fraction * 100,
            color=color,
            marker=marker,
            label=label,
        )
        axes[1].plot(
            x,
            selected.read_amplification,
            color=color,
            marker=marker,
        )
        axes[2].plot(
            x,
            selected.page_reuse_recall * 100,
            color=color,
            marker=marker,
        )

    axes[0].set_ylabel("Context pages touched (%)")
    axes[1].set_ylabel("Read amplification (x)")
    axes[2].set_ylabel("Previous-step page coverage (%)")
    tick_values = sorted(frame.context_tokens.unique() / 1024)
    tick_labels = ["32K", "64K", "128K"]
    for axis in axes:
        axis.set_xlabel("Prompt length")
        axis.set_xticks(tick_values, tick_labels)
    axes[0].set_ylim(0, 100)
    axes[2].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=4,
        frameon=False,
    )
    base = args.output_dir / "page_locality_vs_context_length"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "schema_version": 1,
        "input_run_dirs": [str(path.resolve()) for path in args.run_dir],
        "output_table": str((args.output_dir / "page_length_sweep.csv").resolve()),
        "output_figure_pdf": str(base.with_suffix(".pdf").resolve()),
        "output_figure_png": str(base.with_suffix(".png").resolve()),
        "aggregation": "one RULER request, seven sampled layers, 32 decode steps per context length",
        "uncertainty": "none; pilot has one request at each context length",
    }
    (args.output_dir / "reproducibility.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
