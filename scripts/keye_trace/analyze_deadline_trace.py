#!/usr/bin/env python3
"""Validate and summarize real CUDA-event windows for DSA prefetch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEADLINES_MS = (1.0, 5.0, 10.0, 20.0, 40.0, 50.0)
KEY_COLUMNS = [
    "kind",
    "request_id",
    "layer_id",
    "producer_decode_step",
    "consumer_decode_step",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_int_set(raw: str) -> set[int]:
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def parse_str_set(raw: str) -> set[str]:
    return {value.strip() for value in raw.split(",") if value.strip()}


def load_events(trace_dir: Path) -> tuple[pd.DataFrame, list[Path], list[Path]]:
    event_files = sorted(trace_dir.glob("deadline_events_rank_*.jsonl"))
    metadata_files = sorted(trace_dir.glob("deadline_metadata_rank_*.json"))
    if not event_files:
        raise ValueError(f"no deadline event JSONL files found in {trace_dir}")
    if len(event_files) != len(metadata_files):
        raise ValueError(
            f"event/metadata rank count mismatch: {len(event_files)} vs "
            f"{len(metadata_files)}"
        )

    rows: list[dict[str, Any]] = []
    for path in event_files:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line:
                continue
            row = json.loads(line)
            row["source_file"] = path.name
            row["source_line"] = line_number
            rows.append(row)
    frame = pd.DataFrame(rows)
    required = set(KEY_COLUMNS) | {
        "schema_version",
        "interval_ms",
        "context_tokens",
        "batch_size",
        "global_rank",
        "device_index",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"deadline trace is missing columns: {missing}")
    if not (frame.schema_version == 1).all():
        raise ValueError("only deadline trace schema_version=1 is supported")
    if not np.isfinite(frame.interval_ms).all() or not (frame.interval_ms >= 0).all():
        raise ValueError("interval_ms must be finite and non-negative")
    if frame.duplicated(KEY_COLUMNS + ["global_rank"]).any():
        duplicate = frame.loc[
            frame.duplicated(KEY_COLUMNS + ["global_rank"], keep=False),
            KEY_COLUMNS + ["global_rank"],
        ]
        raise ValueError(f"duplicate rank-local intervals:\n{duplicate.head(20)}")
    return frame, event_files, metadata_files


def pair_tp_ranks(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ranks = sorted(int(rank) for rank in frame.global_rank.unique())
    expected_rank_count = len(ranks)
    rank_counts = frame.groupby(KEY_COLUMNS, dropna=False).global_rank.nunique()
    incomplete = rank_counts[rank_counts != expected_rank_count]
    if len(incomplete):
        raise ValueError(
            f"{len(incomplete)} intervals are missing one or more TP ranks"
        )

    grouped = frame.groupby(KEY_COLUMNS, as_index=False, dropna=False)
    paired = grouped.agg(
        interval_ms=("interval_ms", "min"),
        interval_ms_max_rank=("interval_ms", "max"),
        context_tokens=("context_tokens", "max"),
        batch_size=("batch_size", "max"),
        tp_rank_count=("global_rank", "nunique"),
    )
    paired["tp_rank_skew_ms"] = (
        paired.interval_ms_max_rank - paired.interval_ms
    )
    return paired, {
        "ranks": ranks,
        "rank_count": expected_rank_count,
        "pairing_rule": "minimum interval across TP ranks",
        "all_intervals_have_all_ranks": True,
    }


def validate_coverage(
    frame: pd.DataFrame,
    *,
    expected_layers: set[int],
    expected_kinds: set[str],
    expected_intervals: int,
) -> dict[str, Any]:
    actual_layers = set(int(value) for value in frame.layer_id.unique())
    actual_kinds = set(str(value) for value in frame.kind.unique())
    if actual_layers != expected_layers:
        raise ValueError(
            f"layer mismatch: expected={sorted(expected_layers)}, "
            f"actual={sorted(actual_layers)}"
        )
    if actual_kinds != expected_kinds:
        raise ValueError(
            f"kind mismatch: expected={sorted(expected_kinds)}, "
            f"actual={sorted(actual_kinds)}"
        )

    counts = (
        frame.groupby(["global_rank", "kind", "request_id", "layer_id"])
        .size()
        .rename("intervals")
    )
    bad = counts[counts != expected_intervals]
    if len(bad):
        raise ValueError(
            f"{len(bad)} rank/request/kind/layer groups do not contain "
            f"exactly {expected_intervals} intervals:\n{bad.head(20)}"
        )
    return {
        "expected_layers": sorted(expected_layers),
        "expected_kinds": sorted(expected_kinds),
        "expected_intervals_per_request_layer_rank": expected_intervals,
        "rank_request_kind_layer_groups": int(len(counts)),
        "all_groups_complete": True,
    }


def summarize(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = ["kind", "layer_id"]
    rows: list[dict[str, Any]] = []
    for key, part in paired.groupby(group_columns, sort=True):
        values = part.interval_ms.to_numpy(dtype=np.float64)
        row: dict[str, Any] = {
            "kind": key[0],
            "layer_id": int(key[1]),
            "intervals": len(values),
            "requests": int(part.request_id.nunique()),
            "mean_ms": float(values.mean()),
            "p1_ms": float(np.quantile(values, 0.01)),
            "p10_ms": float(np.quantile(values, 0.10)),
            "p50_ms": float(np.quantile(values, 0.50)),
            "p90_ms": float(np.quantile(values, 0.90)),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
            "mean_tp_rank_skew_ms": float(part.tp_rank_skew_ms.mean()),
            "median_observed_batch_size": float(part.batch_size.median()),
            "maximum_observed_batch_size": int(part.batch_size.max()),
        }
        for deadline in DEADLINES_MS:
            row[f"fraction_ge_{deadline:g}ms"] = float((values >= deadline).mean())
        rows.append(row)
    per_layer = pd.DataFrame(rows)

    overall_rows: list[dict[str, Any]] = []
    for kind, part in paired.groupby("kind", sort=True):
        values = part.interval_ms.to_numpy(dtype=np.float64)
        row = {
            "kind": kind,
            "intervals": len(values),
            "requests": int(part.request_id.nunique()),
            "layers": int(part.layer_id.nunique()),
            "mean_ms": float(values.mean()),
            "p1_ms": float(np.quantile(values, 0.01)),
            "p10_ms": float(np.quantile(values, 0.10)),
            "p50_ms": float(np.quantile(values, 0.50)),
            "p90_ms": float(np.quantile(values, 0.90)),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
            "median_observed_batch_size": float(part.batch_size.median()),
            "maximum_observed_batch_size": int(part.batch_size.max()),
        }
        for deadline in DEADLINES_MS:
            row[f"fraction_ge_{deadline:g}ms"] = float((values >= deadline).mean())
        overall_rows.append(row)
    return per_layer, pd.DataFrame(overall_rows)


def apply_style() -> None:
    # The shared palette/typography contract is reproduced explicitly here.
    # This avoids depending on matplotlib's version-sensitive mplstyle cycler
    # parser while keeping the figure consistent with the project figures.
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


def plot_quantiles(per_layer: pd.DataFrame, output_dir: Path) -> None:
    apply_style()
    kinds = list(per_layer.kind.drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(kinds),
        figsize=(3.3 * len(kinds), 2.5),
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )
    labels = {
        "previous_step_same_layer": "Previous step, same layer",
        "cross_layer_candidate": "Cross-layer candidate",
    }
    colors = {"p10_ms": "#9E9E9E", "p50_ms": "#0072B2", "p90_ms": "#D55E00"}
    markers = {"p10_ms": "v", "p50_ms": "o", "p90_ms": "^"}
    for axis, kind in zip(axes[0], kinds):
        part = per_layer[per_layer.kind == kind].sort_values("layer_id")
        for metric, label in [
            ("p10_ms", "p10"),
            ("p50_ms", "p50"),
            ("p90_ms", "p90"),
        ]:
            axis.plot(
                part.layer_id,
                part[metric],
                color=colors[metric],
                marker=markers[metric],
                linewidth=1.8,
                markersize=4.5,
                label=label,
            )
        axis.set_xlabel("Layer")
        axis.set_ylabel("Available window (ms, higher is better)")
        axis.set_title(labels.get(kind, kind))
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    for suffix, kwargs in [("pdf", {}), ("png", {"dpi": 300})]:
        fig.savefig(
            output_dir / f"deadline_quantiles_by_layer.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def plot_attainment(per_layer: pd.DataFrame, output_dir: Path) -> None:
    apply_style()
    kinds = list(per_layer.kind.drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(kinds),
        figsize=(3.3 * len(kinds), 2.8),
        constrained_layout=True,
        squeeze=False,
    )
    labels = {
        "previous_step_same_layer": "Previous step, same layer",
        "cross_layer_candidate": "Cross-layer candidate",
    }
    image = None
    for axis, kind in zip(axes[0], kinds):
        part = per_layer[per_layer.kind == kind].sort_values("layer_id")
        matrix = part[
            [f"fraction_ge_{deadline:g}ms" for deadline in DEADLINES_MS]
        ].to_numpy()
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(
            range(len(DEADLINES_MS)),
            [f"{value:g}" for value in DEADLINES_MS],
        )
        axis.set_yticks(range(len(part)), [str(value) for value in part.layer_id])
        axis.set_xlabel("Required deadline (ms)")
        axis.set_ylabel("Layer")
        axis.set_title(labels.get(kind, kind))
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86)
    colorbar.set_label("Fraction with window ≥ deadline")
    for suffix, kwargs in [("pdf", {}), ("png", {"dpi": 300})]:
        fig.savefig(
            output_dir / f"deadline_attainment_by_layer.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-layers", required=True)
    parser.add_argument(
        "--expected-kinds",
        default="previous_step_same_layer",
    )
    parser.add_argument("--expected-intervals", type=int, default=32)
    parser.add_argument("--context-label", required=True)
    parser.add_argument("--requested-concurrency", type=int, default=0)
    parser.add_argument(
        "--rid-prefix",
        default="",
        help="analyze only request IDs with this prefix",
    )
    args = parser.parse_args()

    if args.expected_intervals <= 0:
        raise ValueError("--expected-intervals must be positive")
    if args.requested_concurrency < 0:
        raise ValueError("--requested-concurrency must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, event_files, metadata_files = load_events(args.trace_dir)
    if args.rid_prefix:
        frame = frame[frame.request_id.str.startswith(args.rid_prefix)].copy()
        if frame.empty:
            raise ValueError(
                f"no deadline rows match --rid-prefix={args.rid_prefix!r}"
            )
    coverage = validate_coverage(
        frame,
        expected_layers=parse_int_set(args.expected_layers),
        expected_kinds=parse_str_set(args.expected_kinds),
        expected_intervals=args.expected_intervals,
    )
    paired, rank_validation = pair_tp_ranks(frame)
    if args.requested_concurrency and not (
        paired.batch_size == args.requested_concurrency
    ).all():
        observed = {
            int(size): int(count)
            for size, count in paired.batch_size.value_counts().sort_index().items()
        }
        raise ValueError(
            "observed decode batch does not sustain requested concurrency: "
            f"requested={args.requested_concurrency}, observed={observed}"
        )
    paired["context_label"] = args.context_label
    per_layer, overall = summarize(paired)
    per_layer["context_label"] = args.context_label
    overall["context_label"] = args.context_label

    paired.to_csv(args.output_dir / "paired_intervals.csv", index=False)
    per_layer.to_csv(args.output_dir / "per_layer_summary.csv", index=False)
    overall.to_csv(args.output_dir / "overall_summary.csv", index=False)
    plot_quantiles(per_layer, args.output_dir)
    plot_attainment(per_layer, args.output_dir)

    inputs = {
        path.name: sha256_file(path) for path in event_files + metadata_files
    }
    validation = {
        "schema_version": 1,
        "trace_dir": str(args.trace_dir.resolve()),
        "context_label": args.context_label,
        "rid_prefix_filter": args.rid_prefix or None,
        "raw_rows": len(frame),
        "paired_intervals": len(paired),
        "coverage": coverage,
        "tp_pairing": rank_validation,
        "all_intervals_finite_nonnegative": bool(
            np.isfinite(paired.interval_ms).all() and (paired.interval_ms >= 0).all()
        ),
        "speedup_measured": False,
        "batching": {
            "requested_concurrency": args.requested_concurrency or None,
            "observed_batch_size_counts": {
                str(int(size)): int(count)
                for size, count in paired.batch_size.value_counts().sort_index().items()
            },
            "maximum_observed_batch_size": int(paired.batch_size.max()),
            "fraction_at_requested_concurrency": (
                float(
                    (paired.batch_size == args.requested_concurrency).mean()
                )
                if args.requested_concurrency
                else None
            ),
            "all_requests_reached_requested_concurrency": (
                bool(
                    (
                        paired.groupby("request_id").batch_size.max()
                        >= args.requested_concurrency
                    ).all()
                )
                if args.requested_concurrency
                else None
            ),
        },
        "input_sha256": inputs,
        "sglang_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    write_json(args.output_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
