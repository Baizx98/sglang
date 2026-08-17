#!/usr/bin/env python3
"""Compare previous-score KV prefetch across multiple traced long-context tasks.

This analysis uses one relative cache setting at a time so requests with
different realized prompt lengths remain comparable.  Transfer time is a
parameterized model, not measured serving latency or speedup.
"""

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

from simulate_multitier_prefetch import (
    DEFAULT_LAYERS,
    configure_style,
    load_run,
    parse_float_list,
    parse_int_list,
    run_policy,
    simulate,
    transfer_ms,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def add_time_model(
    detail: pd.DataFrame,
    num_model_layers: int,
    sampled_layers: int,
    tp: int,
    pcie_gbps: float,
    ssd_gbps: float,
    pcie_latency_us: float,
    ssd_latency_us: float,
    overlap_windows_ms: list[float],
) -> pd.DataFrame:
    groups = [
        "request_id",
        "task",
        "source_index",
        "context_config",
        "prompt_tokens",
        "page_size",
        "hbm_fraction",
        "dram_fraction",
        "method",
        "candidate_k",
        "step",
    ]
    columns = [
        "target_tokens",
        "hbm_hit_tokens",
        "target_page_bytes",
        "prefetch_dram_bytes",
        "prefetch_ssd_bytes",
        "critical_dram_bytes",
        "critical_ssd_bytes",
        "total_pcie_bytes",
        "total_ssd_bytes",
    ]
    per_step = detail.groupby(groups, as_index=False, dropna=False)[columns].sum()
    scale = num_model_layers / sampled_layers
    byte_columns = [column for column in columns if column.endswith("_bytes")]
    per_step[byte_columns] = per_step[byte_columns] * scale
    prefetch_pcie = per_step.prefetch_dram_bytes + per_step.prefetch_ssd_bytes
    correction_pcie = per_step.critical_dram_bytes + per_step.critical_ssd_bytes
    per_step["prefetch_ms"] = transfer_ms(
        prefetch_pcie,
        per_step.prefetch_ssd_bytes,
        tp,
        pcie_gbps,
        ssd_gbps,
        pcie_latency_us,
        ssd_latency_us,
    )
    per_step["correction_ms"] = transfer_ms(
        correction_pcie,
        per_step.critical_ssd_bytes,
        tp,
        pcie_gbps,
        ssd_gbps,
        pcie_latency_us,
        ssd_latency_us,
    )
    for window in overlap_windows_ms:
        per_step[f"stall_{window:g}ms"] = per_step.correction_ms + np.maximum(
            0.0, per_step.prefetch_ms - window
        )
    return per_step


def simulate_fixed_capacity(
    run_dirs: list[Path],
    layers: list[int],
    page_size: int,
    candidate_k: list[int],
    hbm_logical_gib: float,
    dram_logical_gib: float,
    num_model_layers: int,
    kv_bytes_per_token_layer: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bytes_per_page_model = (
        page_size * kv_bytes_per_token_layer * num_model_layers
    )
    hbm_pages = max(1, int(hbm_logical_gib * 2**30 // bytes_per_page_model))
    dram_pages = max(0, int(dram_logical_gib * 2**30 // bytes_per_page_model))
    rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    chunk_count = 0
    for run_dir in run_dirs:
        chunks = load_run(
            run_dir,
            layers,
            max(4096, max(candidate_k, default=2048)),
        )
        chunk_count += len(chunks)
        for chunk in chunks:
            rows.extend(
                run_policy(
                    chunk,
                    page_size,
                    -1.0,
                    -1.0,
                    "lru-demand",
                    0,
                    kv_bytes_per_token_layer,
                    hbm_pages,
                    dram_pages,
                )
            )
            for k in candidate_k:
                rows.extend(
                    run_policy(
                        chunk,
                        page_size,
                        -1.0,
                        -1.0,
                        "previous-score",
                        k,
                        kv_bytes_per_token_layer,
                        hbm_pages,
                        dram_pages,
                    )
                )
            rows.extend(
                run_policy(
                    chunk,
                    page_size,
                    -1.0,
                    -1.0,
                    "oracle@k3072-pages",
                    3072,
                    kv_bytes_per_token_layer,
                    hbm_pages,
                    dram_pages,
                )
            )
        for request_id in sorted({chunk.request_id for chunk in chunks}):
            request_chunks = [
                chunk for chunk in chunks if chunk.request_id == request_id
            ]
            requests.append(
                {
                    "run_dir": str(run_dir.resolve()),
                    "request_id": request_id,
                    "task": request_chunks[0].task,
                    "prompt_tokens": request_chunks[0].prompt_tokens,
                    "layers": [chunk.layer for chunk in request_chunks],
                }
            )
    detail = pd.DataFrame(rows)
    expected = chunk_count * (len(candidate_k) + 2) * 31
    if len(detail) != expected:
        raise AssertionError(f"row mismatch: {len(detail)} != {expected}")
    if not np.all(
        detail.hbm_hit_pages + detail.critical_miss_pages == detail.target_pages
    ):
        raise AssertionError("target page accounting mismatch")
    actual_hbm_gib = hbm_pages * bytes_per_page_model / 2**30
    actual_dram_gib = dram_pages * bytes_per_page_model / 2**30
    validation = {
        "runs": requests,
        "expected_rows": expected,
        "actual_rows": len(detail),
        "contexts": sorted(int(value) for value in detail.context_config.unique()),
        "layers": layers,
        "transitions_per_layer": 31,
        "capacity_mode": "fixed logical bytes across the full model",
        "hbm_capacity_pages_per_layer": hbm_pages,
        "dram_capacity_pages_per_layer": dram_pages,
        "requested_hbm_logical_gib": hbm_logical_gib,
        "actual_hbm_logical_gib": actual_hbm_gib,
        "requested_dram_logical_gib": dram_logical_gib,
        "actual_dram_logical_gib": actual_dram_gib,
        "new_decode_position_excluded": True,
        "exact_topk_prefix_checked": True,
        "page_accounting_checked": True,
        "exclusive_tier_invariant_checked_each_transition": True,
    }
    return detail, validation


def aggregate(
    per_step: pd.DataFrame, overlap_windows_ms: list[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [
        "task",
        "context_config",
        "page_size",
        "hbm_fraction",
        "dram_fraction",
        "method",
        "candidate_k",
    ]
    rows: list[dict[str, Any]] = []
    for key, part in per_step.groupby(groups, sort=True):
        row = dict(zip(groups, key, strict=True))
        row.update(
            {
                "requests": int(part.request_id.nunique()),
                "request_steps": len(part),
                "mean_prompt_tokens": float(
                    part[["request_id", "prompt_tokens"]]
                    .drop_duplicates()
                    .prompt_tokens.mean()
                ),
                "hbm_token_recall": float(
                    part.hbm_hit_tokens.sum() / part.target_tokens.sum()
                ),
                "transfer_read_amplification": float(
                    part.total_pcie_bytes.sum() / part.target_page_bytes.sum()
                ),
                "prefetch_ms_mean": float(part.prefetch_ms.mean()),
                "correction_ms_mean": float(part.correction_ms.mean()),
                "correction_ms_p95": float(part.correction_ms.quantile(0.95)),
            }
        )
        for window in overlap_windows_ms:
            metric = f"stall_{window:g}ms"
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_p95"] = float(part[metric].quantile(0.95))
        rows.append(row)
    by_task = pd.DataFrame(rows).sort_values(groups)

    baseline_keys = [
        "task",
        "context_config",
        "page_size",
        "hbm_fraction",
        "dram_fraction",
    ]
    compare_metrics = [
        "correction_ms_mean",
        *[f"stall_{window:g}ms_mean" for window in overlap_windows_ms],
    ]
    baseline = by_task[by_task.method == "lru-demand"][
        baseline_keys + compare_metrics
    ].rename(columns={metric: f"lru_{metric}" for metric in compare_metrics})
    by_task = by_task.merge(baseline, on=baseline_keys, validate="many_to_one")
    for metric in compare_metrics:
        by_task[f"{metric}_reduction_vs_lru"] = 1 - (
            by_task[metric] / by_task[f"lru_{metric}"].replace(0, np.nan)
        )

    numeric = [
        "hbm_token_recall",
        "transfer_read_amplification",
        "prefetch_ms_mean",
        "correction_ms_mean",
        "correction_ms_p95",
        *[f"stall_{window:g}ms_mean" for window in overlap_windows_ms],
        *[f"stall_{window:g}ms_p95" for window in overlap_windows_ms],
    ]
    overall_groups = [
        "context_config",
        "page_size",
        "hbm_fraction",
        "dram_fraction",
        "method",
        "candidate_k",
    ]
    overall = by_task.groupby(overall_groups, as_index=False)[numeric].mean()
    overall["tasks"] = by_task.task.nunique()
    baseline = overall[overall.method == "lru-demand"][
        overall_groups[:-2] + compare_metrics
    ].rename(columns={metric: f"lru_{metric}" for metric in compare_metrics})
    overall = overall.merge(
        baseline, on=overall_groups[:-2], validate="many_to_one"
    )
    for metric in compare_metrics:
        overall[f"{metric}_reduction_vs_lru"] = 1 - (
            overall[metric] / overall[f"lru_{metric}"].replace(0, np.nan)
        )
    return by_task, overall


def best_k_table(
    by_task: pd.DataFrame, overlap_windows_ms: list[float]
) -> pd.DataFrame:
    candidates = by_task[by_task.method == "previous-score"]
    rows: list[dict[str, Any]] = []
    for window in overlap_windows_ms:
        metric = f"stall_{window:g}ms_mean"
        for task, part in candidates.groupby("task"):
            best = part.loc[part[metric].idxmin()]
            rows.append(
                {
                    "task": task,
                    "overlap_window_ms": window,
                    "best_candidate_k": int(best.candidate_k),
                    "modeled_stall_ms_mean": float(best[metric]),
                    "reduction_vs_lru": float(
                        best[f"{metric}_reduction_vs_lru"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["overlap_window_ms", "task"])


def save_figure(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_task_tradeoff(
    by_task: pd.DataFrame, output: Path, overlap_window_ms: float
) -> None:
    data = by_task[by_task.method == "previous-score"]
    tasks = sorted(data.task.unique())
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    markers = ["o", "s", "^", "D", "P", "X"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.85))
    reduction = f"stall_{overlap_window_ms:g}ms_mean_reduction_vs_lru"
    for task, color, marker in zip(tasks, colors, markers, strict=False):
        part = data[data.task == task].sort_values("candidate_k")
        axes[0].plot(
            part.candidate_k,
            part.hbm_token_recall * 100,
            color=color,
            marker=marker,
            label=task,
        )
        axes[1].plot(
            part.candidate_k,
            part[reduction] * 100,
            color=color,
            marker=marker,
            label=task,
        )
    axes[1].axhline(0, color="#4D4D4D", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Next-step top-2048 in HBM (%)\nhigher is better")
    axes[1].set_ylabel("Modeled stall reduction vs. LRU (%)\nhigher is better")
    axes[0].set_title("(a) HBM coverage")
    axes[1].set_title(f"(b) {overlap_window_ms:g} ms overlap model")
    for axis in axes:
        axis.set_xlabel("Previous-score prefetch candidate K")
        axis.set_xticks(sorted(data.candidate_k.unique()))
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.70, bottom=0.20, left=0.10, right=0.99, wspace=0.46)
    save_figure(fig, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=parse_int_list, default=DEFAULT_LAYERS)
    parser.add_argument("--candidate-k", type=parse_int_list, default=[2048, 2560, 3072, 4096])
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--hbm-fraction", type=float, default=0.20)
    parser.add_argument("--dram-fraction", type=float, default=0.25)
    parser.add_argument("--hbm-logical-gib", type=float)
    parser.add_argument("--dram-logical-gib", type=float)
    parser.add_argument(
        "--overlap-windows-ms",
        type=parse_float_list,
        default=[1.0, 5.0, 10.0, 20.0],
    )
    parser.add_argument("--num-model-layers", type=int, default=48)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--kv-bytes-per-token-layer", type=int, default=2048)
    parser.add_argument("--pcie-gbps", type=float, default=25.0)
    parser.add_argument("--ssd-gbps", type=float, default=7.0)
    parser.add_argument("--pcie-latency-us", type=float, default=10.0)
    parser.add_argument("--ssd-latency-us", type=float, default=100.0)
    parser.add_argument("--figure-window-ms", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    fixed_capacity = args.hbm_logical_gib is not None or args.dram_logical_gib is not None
    if fixed_capacity:
        if args.hbm_logical_gib is None or args.dram_logical_gib is None:
            raise ValueError("fixed capacity requires both --hbm-logical-gib and --dram-logical-gib")
        detail, validation = simulate_fixed_capacity(
            [path.resolve() for path in args.run_dirs],
            args.layers,
            args.page_size,
            args.candidate_k,
            args.hbm_logical_gib,
            args.dram_logical_gib,
            args.num_model_layers,
            args.kv_bytes_per_token_layer,
        )
    else:
        detail, validation = simulate(
            [path.resolve() for path in args.run_dirs],
            args.layers,
            [args.page_size],
            args.candidate_k,
            [args.hbm_fraction],
            [args.dram_fraction],
            args.kv_bytes_per_token_layer,
        )
    per_step = add_time_model(
        detail,
        args.num_model_layers,
        len(args.layers),
        args.tp,
        args.pcie_gbps,
        args.ssd_gbps,
        args.pcie_latency_us,
        args.ssd_latency_us,
        args.overlap_windows_ms,
    )
    by_task, overall = aggregate(per_step, args.overlap_windows_ms)
    best = best_k_table(by_task, args.overlap_windows_ms)
    detail.to_parquet(tables / "by_transition.parquet", index=False)
    per_step.to_parquet(tables / "by_request_step.parquet", index=False)
    by_task.to_csv(tables / "summary_by_task.csv", index=False)
    overall.to_csv(tables / "summary_overall.csv", index=False)
    best.to_csv(tables / "best_k_by_task_deadline.csv", index=False)
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n"
    )
    configure_style(None)
    plot_task_tradeoff(
        by_task, figures / "candidate_k_cross_task_tradeoff", args.figure_window_ms
    )

    artifact_paths = [
        tables / "by_transition.parquet",
        tables / "by_request_step.parquet",
        tables / "summary_by_task.csv",
        tables / "summary_overall.csv",
        tables / "best_k_by_task_deadline.csv",
        output / "validation.json",
        figures / "candidate_k_cross_task_tradeoff.pdf",
        figures / "candidate_k_cross_task_tradeoff.png",
    ]
    result = {
        "schema_version": 1,
        "analysis_kind": "trace-driven descriptive simulation; transfer time is not measured speed",
        "requests": len(validation["runs"]),
        "tasks": sorted(by_task.task.unique()),
        "setting": {
            "page_size_tokens": args.page_size,
            "capacity_mode": validation.get("capacity_mode", "fraction of each request KV"),
            "hbm_fraction_of_request_kv": None if fixed_capacity else args.hbm_fraction,
            "dram_fraction_of_request_kv": None if fixed_capacity else args.dram_fraction,
            "hbm_logical_gib": validation.get("actual_hbm_logical_gib"),
            "dram_logical_gib": validation.get("actual_dram_logical_gib"),
            "sampled_layers": args.layers,
            "scaled_model_layers": args.num_model_layers,
            "tensor_parallel_size": args.tp,
            "overlap_windows_ms": args.overlap_windows_ms,
        },
        "overall_previous_score": overall[
            overall.method == "previous-score"
        ].to_dict("records"),
        "best_k_by_task_deadline": best.to_dict("records"),
        "limitations": [
            "this pilot has one request per task and context bucket",
            (
                "fixed byte budgets are equal across requests"
                if fixed_capacity
                else "HBM and DRAM are fixed fractions of each request KV, not equal absolute byte budgets"
            ),
            "seven sampled layers are scaled to 48 layers",
            "transfer time is parameterized and does not establish real speedup",
        ],
        "artifacts": {
            str(path.relative_to(output)): sha256(path) for path in artifact_paths
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    reproducibility = {
        "script": str(Path(__file__).resolve()),
        "git_commit": git_revision(Path.cwd()),
        "inputs": [str(path.resolve()) for path in args.run_dirs],
        "output_dir": str(output),
        "arguments": vars(args) | {"run_dirs": [str(path) for path in args.run_dirs], "output_dir": str(args.output_dir)},
    }
    (output / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
