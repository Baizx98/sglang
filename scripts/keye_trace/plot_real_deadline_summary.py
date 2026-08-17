#!/usr/bin/env python3
"""Combine 64K/128K real deadline traces into paper-style summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"64K": "#0072B2", "128K": "#D55E00"}
MARKERS = {"64K": "o", "128K": "^"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-64k", type=Path, required=True)
    parser.add_argument("--input-128k", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for label, path in [("64K", args.input_64k), ("128K", args.input_128k)]:
        frame = pd.read_csv(path)
        frame["context_label"] = label
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if not np.isfinite(combined.interval_ms).all():
        raise ValueError("all intervals must be finite")
    if not (combined.tp_rank_count == 2).all():
        raise ValueError("all intervals must contain both TP ranks")

    layer_rows = []
    for (context, layer), part in combined.groupby(
        ["context_label", "layer_id"], sort=True
    ):
        values = part.interval_ms.to_numpy()
        layer_rows.append(
            {
                "context_label": context,
                "layer_id": int(layer),
                "requests": int(part.request_id.nunique()),
                "intervals": len(values),
                "p10_ms": float(np.quantile(values, 0.10)),
                "p50_ms": float(np.quantile(values, 0.50)),
                "p90_ms": float(np.quantile(values, 0.90)),
                "min_ms": float(values.min()),
                "fraction_ge_20ms": float((values >= 20).mean()),
                "mean_tp_rank_skew_ms": float(part.tp_rank_skew_ms.mean()),
            }
        )
    per_layer = pd.DataFrame(layer_rows)

    request_rows = []
    for (context, request_id), part in combined.groupby(
        ["context_label", "request_id"], sort=True
    ):
        values = part.interval_ms.to_numpy()
        request_rows.append(
            {
                "context_label": context,
                "request_id": request_id,
                "context_tokens_median": float(part.context_tokens.median()),
                "intervals": len(values),
                "p10_ms": float(np.quantile(values, 0.10)),
                "p50_ms": float(np.quantile(values, 0.50)),
                "p90_ms": float(np.quantile(values, 0.90)),
                "fraction_ge_20ms": float((values >= 20).mean()),
            }
        )
    per_request = pd.DataFrame(request_rows)

    overall_rows = []
    for context, part in combined.groupby("context_label", sort=True):
        values = part.interval_ms.to_numpy()
        overall_rows.append(
            {
                "context_label": context,
                "requests": int(part.request_id.nunique()),
                "layers": int(part.layer_id.nunique()),
                "intervals": len(values),
                "p10_ms": float(np.quantile(values, 0.10)),
                "p50_ms": float(np.quantile(values, 0.50)),
                "p90_ms": float(np.quantile(values, 0.90)),
                "min_ms": float(values.min()),
                "max_ms": float(values.max()),
                "fraction_ge_1ms": float((values >= 1).mean()),
                "fraction_ge_5ms": float((values >= 5).mean()),
                "fraction_ge_10ms": float((values >= 10).mean()),
                "fraction_ge_20ms": float((values >= 20).mean()),
                "mean_tp_rank_skew_ms": float(part.tp_rank_skew_ms.mean()),
            }
        )
    overall = pd.DataFrame(overall_rows)

    combined.to_csv(args.output_dir / "combined_paired_intervals.csv", index=False)
    per_layer.to_csv(args.output_dir / "per_layer_summary.csv", index=False)
    per_request.to_csv(args.output_dir / "per_request_summary.csv", index=False)
    overall.to_csv(args.output_dir / "overall_summary.csv", index=False)

    configure_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(6.8, 2.7), constrained_layout=True
    )
    for context in ["64K", "128K"]:
        part = per_layer[per_layer.context_label == context].sort_values("layer_id")
        axes[0].plot(
            part.layer_id,
            part.p10_ms,
            color=COLORS[context],
            marker=MARKERS[context],
            markersize=4.5,
            label=context,
        )
    axes[0].axhspan(0, 20, color="#9E9E9E", alpha=0.15, label="Simulated range")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("p10 available window (ms, higher is better)")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)

    for context in ["64K", "128K"]:
        values = np.sort(
            combined.loc[combined.context_label == context, "interval_ms"].to_numpy()
        )
        cdf = np.arange(1, len(values) + 1) / len(values)
        axes[1].plot(
            values,
            cdf,
            color=COLORS[context],
            label=context,
        )
    axes[1].axvline(20, color="#4D4D4D", linestyle="--", linewidth=1.2)
    axes[1].text(22, 0.08, "20 ms", color="#4D4D4D", fontsize=8)
    axes[1].set_xlabel("Available window (ms)")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)

    output_stem = args.output_dir / "real_prefetch_deadline_distribution"
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    metadata = {
        "schema_version": 1,
        "inputs": {
            "64K": {
                "path": str(args.input_64k.resolve()),
                "sha256": sha256_file(args.input_64k),
            },
            "128K": {
                "path": str(args.input_128k.resolve()),
                "sha256": sha256_file(args.input_128k),
            },
        },
        "tp_aggregation": "minimum available window across two ranks",
        "speedup_measured": False,
        "outputs": {
            path.name: sha256_file(path)
            for path in [
                output_stem.with_suffix(".pdf"),
                output_stem.with_suffix(".png"),
                args.output_dir / "per_layer_summary.csv",
                args.output_dir / "per_request_summary.csv",
                args.output_dir / "overall_summary.csv",
            ]
        },
    }
    (args.output_dir / "reproducibility.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
