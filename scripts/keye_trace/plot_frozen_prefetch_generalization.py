#!/usr/bin/env python3
"""Plot frozen-policy generalization across labeled evaluation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "Discovery": "#9E9E9E",
    "Calibration": "#D55E00",
    "Blind test": "#0072B2",
    "RULER blind": "#0072B2",
    "LongBench-v2": "#009E73",
}
MARKERS = {
    "Discovery": "o",
    "Calibration": "s",
    "Blind test": "D",
    "RULER blind": "D",
    "LongBench-v2": "^",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from exc
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluations", nargs="+", type=parse_labeled_path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = []
    for label, evaluation_dir in args.evaluations:
        path = evaluation_dir / "summary_overall.csv"
        frame = pd.read_csv(path)
        frame["split"] = label
        frames.append(frame)
        inputs.append(
            {"split": label, "path": str(path.resolve()), "sha256": sha256(path)}
        )
    data = pd.concat(frames, ignore_index=True)
    contexts = sorted(int(value) for value in data.context_config.unique())
    if contexts != [65536, 131072]:
        raise ValueError(f"expected contexts [65536, 131072], found {contexts}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), sharey=True)
    preferred_order = [
        "Discovery",
        "Calibration",
        "Blind test",
        "RULER blind",
        "LongBench-v2",
    ]
    split_order = [label for label in preferred_order if label in set(data.split)]
    split_order += sorted(set(data.split) - set(split_order))
    for axis, context in zip(axes, contexts, strict=True):
        context_data = data[data.context_config == context]
        for split in split_order:
            part = context_data[context_data.split == split].sort_values(
                "overlap_deadline_ms"
            )
            if part.empty:
                continue
            axis.plot(
                part.overlap_deadline_ms,
                part.reduction_vs_lru_mean * 100,
                label=split,
                color=COLORS.get(split, "#009E73"),
                marker=MARKERS.get(split, "^"),
                linewidth=1.9,
                markersize=5,
            )
        axis.axhline(0, color="#4D4D4D", linewidth=1.0, linestyle="--")
        axis.set_title(f"{context // 1024}K context")
        axis.set_xlabel("Overlap deadline (ms)")
        axis.set_xticks([1, 5, 10, 20])
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Modeled stall reduction vs. LRU (%)\nhigher is better")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.subplots_adjust(top=0.77, bottom=0.22, left=0.10, right=0.99, wspace=0.13)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    combined = output / "combined_summary.csv"
    pdf = output / "frozen_policy_generalization.pdf"
    png = output / "frozen_policy_generalization.png"
    data.sort_values(
        ["split", "context_config", "overlap_deadline_ms"]
    ).to_csv(combined, index=False)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "schema_version": 1,
        "metric": "100 * (1 - frozen-policy modeled mean stall / demand-only LRU modeled mean stall)",
        "aggregation": (
            "mean within each labeled evaluation; the current external test uses "
            "one request per RULER task or LongBench-v2 domain"
        ),
        "inputs": inputs,
        "outputs": {
            combined.name: sha256(combined),
            pdf.name: sha256(pdf),
            png.name: sha256(png),
        },
        "style": "research-figure-style defaults embedded in plotting script",
        "random_seed": None,
        "limitations": "Transfer time is modeled; the figure does not claim measured speedup.",
    }
    (output / "reproducibility.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
