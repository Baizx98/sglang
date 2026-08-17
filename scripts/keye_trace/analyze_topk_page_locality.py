#!/usr/bin/env python3
"""Measure how token-level DSA top-k expands under fixed-size KV pages.

This is a feasibility gate for page-based offload.  A high adjacent-step page
reuse is not useful by itself when each step already touches nearly every page,
so the analysis reports active-page fraction and read amplification together
with temporal reuse.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

TARGET_K = 2048
DEFAULT_LAYERS = [0, 7, 15, 23, 31, 39, 47]
DEFAULT_PAGE_SIZES = [4, 16, 64, 256]
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_SAMPLES = 2000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("expected a non-empty list of non-negative ints")
    return values


def select_requests(
    requests: list[dict[str, Any]], first_round_per_trajectory: bool
) -> list[dict[str, Any]]:
    if not first_round_per_trajectory:
        return requests
    selected: dict[str, dict[str, Any]] = {}
    for row in requests:
        trajectory = str(row.get("trajectory_id", row["rid"]))
        round_id = int(row.get("round_id", 0))
        current = selected.get(trajectory)
        current_round = int(current.get("round_id", 0)) if current else None
        if current is None or round_id < current_round:
            selected[trajectory] = row
    return sorted(
        selected.values(), key=lambda row: str(row.get("trajectory_id", row["rid"]))
    )


def request_metadata(request: dict[str, Any]) -> tuple[str, str, int]:
    """Normalize BFCL multi-round and RULER single-request metadata."""
    return (
        str(request.get("trajectory_id", request["rid"])),
        str(request.get("category", request.get("task", request.get("dataset", "unknown")))),
        int(request.get("round_id", 0)),
    )


def build_lookup(run_dir: Path) -> dict[tuple[str, int], Path]:
    lookup: dict[tuple[str, int], Path] = {}
    for row in read_jsonl(run_dir / "events" / "manifest.jsonl"):
        key = (row["request_id"], int(row["layer_id"]))
        if key in lookup:
            raise ValueError(f"duplicate trace chunk: {key}")
        lookup[key] = run_dir / "events" / row["file"]
    return lookup


def load_chunk(path: Path) -> tuple[np.ndarray, np.ndarray]:
    record = torch.load(path, weights_only=False)
    if int(record["schema_version"]) not in {4, 5}:
        raise ValueError(f"{path}: expected schema v4 or v5")
    if record["decode_step_ids"] != list(range(32)):
        raise ValueError(f"{path}: expected decode steps 0..31")
    return (
        record["indices"].numpy(force=True).astype(np.int32),
        record["score_valid_counts"].numpy(force=True).astype(np.int32),
    )


def page_slots(page_ids: np.ndarray, page_size: int, valid_tokens: int) -> int:
    starts = page_ids.astype(np.int64) * page_size
    return int(np.minimum(page_size, valid_tokens - starts).sum())


def analyze(
    run_dir: Path,
    requests: list[dict[str, Any]],
    lookup: dict[tuple[str, int], Path],
    layers: list[int],
    page_sizes: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    step_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    chunks = 0
    expected_chunks = len(requests) * len(layers)

    for request in requests:
        rid = request["rid"]
        trajectory_id, category, round_id = request_metadata(request)
        for layer in layers:
            path = lookup.get((rid, layer))
            if path is None:
                raise ValueError(f"missing trace chunk: {(rid, layer)}")
            indices, valid_counts = load_chunk(path)
            chunks += 1

            for page_size in page_sizes:
                previous_pages: np.ndarray | None = None
                previous_page_fraction = float("nan")
                for step, (raw_topk, valid_tokens_raw) in enumerate(
                    zip(indices, valid_counts, strict=True)
                ):
                    valid_tokens = int(valid_tokens_raw)
                    selected = np.unique(
                        raw_topk[(raw_topk >= 0) & (raw_topk < valid_tokens)]
                    )
                    if len(selected) != TARGET_K:
                        raise ValueError(
                            f"unexpected top-k width: {rid}, L{layer}, step {step}, "
                            f"got {len(selected)}"
                        )
                    pages = np.unique(selected // page_size)
                    total_pages = (valid_tokens + page_size - 1) // page_size
                    transferred_slots = page_slots(pages, page_size, valid_tokens)
                    active_fraction = len(pages) / total_pages
                    step_rows.append(
                        {
                            "trajectory_id": trajectory_id,
                            "request_id": rid,
                            "category": category,
                            "round_id": round_id,
                            "layer": layer,
                            "step": step,
                            "page_size": page_size,
                            "valid_tokens": valid_tokens,
                            "selected_tokens": len(selected),
                            "total_pages": total_pages,
                            "active_pages": len(pages),
                            "active_page_fraction": active_fraction,
                            "transferred_slots": transferred_slots,
                            "read_amplification": transferred_slots / len(selected),
                        }
                    )

                    if previous_pages is not None:
                        common = np.intersect1d(
                            previous_pages, pages, assume_unique=True
                        ).size
                        union = np.union1d(previous_pages, pages).size
                        reuse = common / len(pages)
                        reuse_lift = (
                            reuse / previous_page_fraction
                            if previous_page_fraction > 0
                            else float("nan")
                        )
                        transition_rows.append(
                            {
                                "trajectory_id": trajectory_id,
                                "request_id": rid,
                                "category": category,
                                "round_id": round_id,
                                "layer": layer,
                                "step": step,
                                "page_size": page_size,
                                "previous_active_page_fraction": previous_page_fraction,
                                "page_reuse_recall": reuse,
                                "page_reuse_lift": reuse_lift,
                                "page_jaccard": common / union,
                                "new_page_fraction": (len(pages) - common) / len(pages),
                            }
                        )
                    previous_pages = pages
                    previous_page_fraction = active_fraction

            if chunks % 24 == 0 or chunks == expected_chunks:
                print(f"[{chunks:03d}/{expected_chunks}] analyzed {rid}, L{layer}")

    validation = {
        "requests": len(requests),
        "trajectories": len(
            {request_metadata(row)[0] for row in requests}
        ),
        "layers": layers,
        "page_sizes": page_sizes,
        "chunks": chunks,
        "expected_chunks": expected_chunks,
        "steps": len(step_rows),
        "expected_steps": expected_chunks * 32 * len(page_sizes),
        "transitions": len(transition_rows),
        "expected_transitions": expected_chunks * 31 * len(page_sizes),
    }
    if chunks != expected_chunks:
        raise AssertionError("chunk count mismatch")
    if len(step_rows) != validation["expected_steps"]:
        raise AssertionError("step count mismatch")
    if len(transition_rows) != validation["expected_transitions"]:
        raise AssertionError("transition count mismatch")
    return pd.DataFrame(step_rows), pd.DataFrame(transition_rows), validation


def bootstrap_mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    mean = float(array.mean())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(
        array, size=(BOOTSTRAP_SAMPLES, len(array)), replace=True
    ).mean(axis=1)
    return mean, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def aggregate_summary(
    steps: pd.DataFrame, transitions: pd.DataFrame
) -> pd.DataFrame:
    step_metrics = ["active_page_fraction", "read_amplification"]
    transition_metrics = [
        "page_reuse_recall",
        "page_reuse_lift",
        "page_jaccard",
        "new_page_fraction",
    ]
    trajectory_steps = (
        steps.groupby(["trajectory_id", "page_size"], as_index=False)[step_metrics]
        .mean()
    )
    trajectory_transitions = (
        transitions.groupby(["trajectory_id", "page_size"], as_index=False)[
            transition_metrics
        ].mean()
    )
    trajectory = trajectory_steps.merge(
        trajectory_transitions, on=["trajectory_id", "page_size"], validate="1:1"
    )

    rows: list[dict[str, Any]] = []
    for page_size, part in trajectory.groupby("page_size"):
        row: dict[str, Any] = {"page_size": int(page_size)}
        for metric in step_metrics + transition_metrics:
            mean, low, high = bootstrap_mean_ci(part[metric])
            row[metric] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        raw = steps[steps.page_size == page_size]
        row["read_amplification_p50"] = float(raw.read_amplification.quantile(0.5))
        row["read_amplification_p90"] = float(raw.read_amplification.quantile(0.9))
        row["read_amplification_p99"] = float(raw.read_amplification.quantile(0.99))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("page_size")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
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


def save_figure(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    x = summary.page_size
    axes[0].plot(
        x,
        summary.active_page_fraction * 100,
        color="#0072B2",
        marker="o",
        label="Context pages touched",
    )
    axes[0].set(
        xscale="log",
        xlabel="KV page size (tokens)",
        ylabel="Context pages touched (%)",
        xticks=x,
        xticklabels=x,
        ylim=(0, 105),
    )
    axis_amp = axes[0].twinx()
    axis_amp.spines["right"].set_visible(True)
    axis_amp.grid(False)
    axis_amp.plot(
        x,
        summary.read_amplification,
        color="#D55E00",
        marker="s",
        linestyle="--",
        label="Read amplification",
    )
    axis_amp.set_ylabel("Read amplification (x)")
    handles_a, labels_a = axes[0].get_legend_handles_labels()
    handles_b, labels_b = axis_amp.get_legend_handles_labels()
    axes[0].legend(handles_a + handles_b, labels_a + labels_b, frameon=False)

    axes[1].plot(
        x,
        summary.page_reuse_recall * 100,
        color="#009E73",
        marker="o",
        label="Previous-step page coverage",
    )
    axes[1].plot(
        x,
        summary.active_page_fraction * 100,
        color="#4D4D4D",
        marker="s",
        linestyle="--",
        label="Active-page fraction",
    )
    axes[1].set(
        xscale="log",
        xlabel="KV page size (tokens)",
        ylabel="Fraction of pages (%)",
        xticks=x,
        xticklabels=x,
        ylim=(0, 105),
    )
    axes[1].legend(frameon=False)
    save_figure(fig, figure_dir / "topk_page_granularity_gate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--layers", type=parse_int_list, default=DEFAULT_LAYERS, help="comma list"
    )
    parser.add_argument(
        "--page-sizes",
        type=parse_int_list,
        default=DEFAULT_PAGE_SIZES,
        help="comma list of token counts",
    )
    parser.add_argument(
        "--all-rounds",
        action="store_true",
        help="Analyze every round instead of the first round of each trajectory.",
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else run_dir / "analysis" / "topk-page-locality-v01"
    )
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        steps = pd.read_parquet(table_dir / "page_metrics_by_step.parquet")
        transitions = pd.read_parquet(
            table_dir / "page_metrics_by_transition.parquet"
        )
        summary = pd.read_csv(table_dir / "page_size_summary.csv")
        validation = json.loads((output_dir / "validation.json").read_text())
    else:
        requests = select_requests(
            read_jsonl(run_dir / "prepared_requests.jsonl"),
            first_round_per_trajectory=not args.all_rounds,
        )
        lookup = build_lookup(run_dir)
        steps, transitions, validation = analyze(
            run_dir, requests, lookup, args.layers, args.page_sizes
        )
        summary = aggregate_summary(steps, transitions)
        steps.to_parquet(table_dir / "page_metrics_by_step.parquet", index=False)
        transitions.to_parquet(
            table_dir / "page_metrics_by_transition.parquet", index=False
        )
        summary.to_csv(table_dir / "page_size_summary.csv", index=False)

    set_style()
    plot_summary(summary, figure_dir)
    result = {
        "metric_definition": {
            "active_page_fraction": "pages containing at least one top-k token / all valid context pages",
            "read_amplification": "valid token slots in touched pages / selected top-k tokens",
            "page_reuse_recall": "current active pages also active at the previous decode step / current active pages",
            "page_reuse_lift": "page reuse recall / previous active-page fraction; values near 1 indicate reuse explained by broad coverage",
        },
        "page_sizes": summary.to_dict("records"),
        "validation": validation,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "reproducibility.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "script": str(Path(__file__).resolve()),
                "code_commit": git_revision(Path(__file__).resolve().parents[2]),
                "request_scope": "all rounds" if args.all_rounds else "first round per trajectory",
                "layers": args.layers,
                "page_sizes": args.page_sizes,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "aggregation": "trajectory-equal mean",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
