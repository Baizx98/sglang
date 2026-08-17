#!/usr/bin/env python3
"""Evaluate top-4096/6144/8192 previous-step candidates without speed claims.

All reported quantities are trace-derived set coverage or logical transfer
bytes from the deterministic tier simulator.  The script deliberately does
not convert bytes into latency or throughput.
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
import torch

from simulate_multitier_prefetch import (
    Chunk,
    DEFAULT_LAYERS,
    candidate_pages,
    configure_style,
    load_run,
    parse_int_list,
    run_policy,
    target_for_transition,
)

TARGET_K = 2048
DEFAULT_CANDIDATE_K = [4096, 6144, 8192]
COLORS = {
    "RULER 64K": "#0072B2",
    "RULER 128K": "#56B4E9",
    "LongBench-v2 64K": "#D55E00",
    "LongBench-v2 128K": "#E69F00",
}
MARKERS = {
    "RULER 64K": "o",
    "RULER 128K": "s",
    "LongBench-v2 64K": "^",
    "LongBench-v2 128K": "D",
}


def parse_layer_list(value: str) -> list[int]:
    layers = sorted({int(item) for item in value.split(",") if item.strip()})
    if not layers or any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError(
            "expected non-negative comma-separated layer IDs"
        )
    return layers


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


def series_label(dataset: str, context_config: int) -> str:
    short_dataset = "RULER" if dataset == "RULER" else "LongBench-v2"
    return f"{short_dataset} {context_config // 1024}K"


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe records with missing numeric values encoded as null."""
    return json.loads(frame.to_json(orient="records"))


def trace_candidate_rows(
    chunk: Chunk, page_size: int, candidate_ks: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in range(31):
        target_tokens, target_page_order, common = target_for_transition(
            chunk, step, page_size
        )
        target_token_set = set(int(token) for token in target_tokens)
        target_page_set = set(target_page_order)
        for candidate_k in candidate_ks:
            candidate_tokens = chunk.ranked[step, :candidate_k]
            candidate_tokens = candidate_tokens[
                (candidate_tokens >= 0) & (candidate_tokens < common)
            ]
            candidate_token_set = set(int(token) for token in candidate_tokens)
            candidate_page_set = set(
                candidate_pages(chunk, step, page_size, candidate_k, common)
            )
            token_hits = len(target_token_set & candidate_token_set)
            page_hits = len(target_page_set & candidate_page_set)
            rows.append(
                {
                    "run_dir": str(chunk.source_file.parents[1]),
                    "request_id": chunk.request_id,
                    "dataset": chunk.dataset,
                    "task": chunk.task,
                    "source_index": chunk.source_index,
                    "context_config": chunk.context_config,
                    "prompt_tokens": chunk.prompt_tokens,
                    "layer": chunk.layer,
                    "step": step,
                    "candidate_k": candidate_k,
                    "common_tokens": common,
                    "target_tokens": len(target_token_set),
                    "candidate_tokens": len(candidate_token_set),
                    "candidate_token_hits": token_hits,
                    "target_pages": len(target_page_set),
                    "candidate_pages": len(candidate_page_set),
                    "candidate_page_hits": page_hits,
                    "source_file": chunk.source_file.name,
                }
            )
    return rows


def simulate_gate(
    run_dirs: list[Path],
    layers: list[int],
    candidate_ks: list[int],
    page_size: int,
    hbm_logical_gib: float,
    dram_logical_gib: float,
    num_model_layers: int,
    kv_bytes_per_token_layer: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bytes_per_page_model = (
        page_size * kv_bytes_per_token_layer * num_model_layers
    )
    hbm_pages = max(1, int(hbm_logical_gib * 2**30 // bytes_per_page_model))
    dram_pages = max(0, int(dram_logical_gib * 2**30 // bytes_per_page_model))
    candidate_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    request_records: list[dict[str, Any]] = []
    chunk_count = 0

    for run_dir in run_dirs:
        chunks = load_run(run_dir, layers, max(candidate_ks))
        chunk_count += len(chunks)
        for chunk in chunks:
            candidate_rows.extend(
                trace_candidate_rows(chunk, page_size, candidate_ks)
            )
            cache_rows.extend(
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
                    extended_metrics=True,
                )
            )
            for candidate_k in candidate_ks:
                cache_rows.extend(
                    run_policy(
                        chunk,
                        page_size,
                        -1.0,
                        -1.0,
                        "previous-score",
                        candidate_k,
                        kv_bytes_per_token_layer,
                        hbm_pages,
                        dram_pages,
                        extended_metrics=True,
                    )
                )
            cache_rows.extend(
                run_policy(
                    chunk,
                    page_size,
                    -1.0,
                    -1.0,
                    "oracle-pages",
                    max(candidate_ks),
                    kv_bytes_per_token_layer,
                    hbm_pages,
                    dram_pages,
                    extended_metrics=True,
                )
            )
        for request_id in sorted({chunk.request_id for chunk in chunks}):
            request_chunks = [
                chunk for chunk in chunks if chunk.request_id == request_id
            ]
            request_records.append(
                {
                    "run_dir": str(run_dir),
                    "request_id": request_id,
                    "dataset": request_chunks[0].dataset,
                    "task": request_chunks[0].task,
                    "context_config": request_chunks[0].context_config,
                    "prompt_tokens": request_chunks[0].prompt_tokens,
                    "layers": [chunk.layer for chunk in request_chunks],
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    cache = pd.DataFrame(cache_rows)
    # RULER uses integer source IDs while LongBench-v2 uses hexadecimal IDs.
    # Normalize the provenance field before writing one cross-dataset Parquet.
    candidates["source_index"] = candidates.source_index.astype(str)
    cache["source_index"] = cache.source_index.astype(str)
    expected_candidate_rows = chunk_count * 31 * len(candidate_ks)
    expected_cache_rows = chunk_count * 31 * (len(candidate_ks) + 2)
    if len(candidates) != expected_candidate_rows:
        raise AssertionError(
            f"candidate row mismatch: {len(candidates)} != {expected_candidate_rows}"
        )
    if len(cache) != expected_cache_rows:
        raise AssertionError(
            f"cache row mismatch: {len(cache)} != {expected_cache_rows}"
        )
    if not np.all(
        cache.prefetch_admit_pages
        == cache.useful_prefetch_pages + cache.unused_prefetch_pages
    ):
        raise AssertionError("prefetch useful/unused accounting mismatch")
    if not np.all(cache.pollution_miss_pages <= cache.critical_miss_pages):
        raise AssertionError("pollution miss is not a subset of critical misses")
    if not np.all(
        cache.hbm_hit_pages + cache.critical_miss_pages == cache.target_pages
    ):
        raise AssertionError("target page accounting mismatch")

    validation = {
        "schema_version": 1,
        "analysis_kind": "trace-derived coverage and deterministic byte accounting; no speed measurement",
        "runs": request_records,
        "requests": len(request_records),
        "chunks": chunk_count,
        "candidate_rows_expected": expected_candidate_rows,
        "candidate_rows_actual": len(candidates),
        "cache_rows_expected": expected_cache_rows,
        "cache_rows_actual": len(cache),
        "candidate_k": candidate_ks,
        "layers": layers,
        "transitions_per_layer": 31,
        "page_size_tokens": page_size,
        "requested_hbm_logical_gib": hbm_logical_gib,
        "actual_hbm_logical_gib": hbm_pages * bytes_per_page_model / 2**30,
        "requested_dram_logical_gib": dram_logical_gib,
        "actual_dram_logical_gib": dram_pages * bytes_per_page_model / 2**30,
        "hbm_capacity_pages_per_layer": hbm_pages,
        "dram_capacity_pages_per_layer": dram_pages,
        "checks": {
            "exact_top2048_is_candidate_prefix": True,
            "request_layer_coverage": True,
            "prefetch_useful_plus_unused": True,
            "pollution_subset_of_correction": True,
            "hbm_hit_plus_correction": True,
        },
    }
    return candidates, cache, validation


def aggregate_candidates(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    request_groups = [
        "dataset",
        "task",
        "request_id",
        "context_config",
        "candidate_k",
    ]
    counts = [
        "target_tokens",
        "candidate_tokens",
        "candidate_token_hits",
        "target_pages",
        "candidate_pages",
        "candidate_page_hits",
    ]
    by_request = detail.groupby(request_groups, as_index=False)[counts].sum()
    by_request["candidate_token_recall"] = (
        by_request.candidate_token_hits / by_request.target_tokens
    )
    by_request["candidate_token_precision"] = (
        by_request.candidate_token_hits / by_request.candidate_tokens
    )
    by_request["candidate_page_recall"] = (
        by_request.candidate_page_hits / by_request.target_pages
    )
    by_request["candidate_page_precision"] = (
        by_request.candidate_page_hits / by_request.candidate_pages
    )

    metric_columns = [
        "candidate_token_recall",
        "candidate_token_precision",
        "candidate_page_recall",
        "candidate_page_precision",
    ]
    summary = by_request.groupby(
        ["dataset", "context_config", "candidate_k"], as_index=False
    )[metric_columns].mean()
    request_counts = by_request.groupby(
        ["dataset", "context_config", "candidate_k"], as_index=False
    ).request_id.nunique()
    summary = summary.merge(
        request_counts.rename(columns={"request_id": "requests"}),
        on=["dataset", "context_config", "candidate_k"],
        validate="1:1",
    )

    layer_counts = detail.groupby(
        [
            "dataset",
            "task",
            "request_id",
            "context_config",
            "layer",
            "candidate_k",
        ],
        as_index=False,
    )[counts].sum()
    layer_counts["candidate_token_recall"] = (
        layer_counts.candidate_token_hits / layer_counts.target_tokens
    )
    by_layer = layer_counts.groupby(
        ["dataset", "context_config", "layer", "candidate_k"],
        as_index=False,
    ).candidate_token_recall.mean()
    return by_request, summary, by_layer


def aggregate_cache(
    detail: pd.DataFrame, num_model_layers: int, sampled_layers: int, tp: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_groups = [
        "dataset",
        "task",
        "request_id",
        "context_config",
        "method",
        "candidate_k",
        "step",
    ]
    count_columns = [
        "target_tokens",
        "target_pages",
        "hbm_hit_tokens",
        "hbm_hit_pages",
        "prefetch_admit_pages",
        "useful_prefetch_pages",
        "critical_miss_pages",
        "target_page_bytes",
        "prefetch_dram_bytes",
        "prefetch_ssd_bytes",
        "critical_miss_bytes",
        "total_pcie_bytes",
        "unused_prefetch_bytes",
        "prefetch_self_evicted_target_bytes",
        "pollution_miss_bytes",
    ]
    per_step = detail.groupby(step_groups, as_index=False)[count_columns].sum()
    scale = num_model_layers / sampled_layers
    byte_columns = [column for column in count_columns if column.endswith("_bytes")]
    per_step[byte_columns] = per_step[byte_columns] * scale

    request_groups = step_groups[:-1]
    by_request_counts = per_step.groupby(request_groups, as_index=False)[
        count_columns
    ].sum()
    steps = per_step.groupby(request_groups).size().rename("decode_steps")
    by_request = by_request_counts.merge(
        steps.reset_index(), on=request_groups, validate="1:1"
    )
    by_request["hbm_token_recall"] = (
        by_request.hbm_hit_tokens / by_request.target_tokens
    )
    by_request["prefetch_precision"] = (
        by_request.useful_prefetch_pages
        / by_request.prefetch_admit_pages.replace(0, np.nan)
    )
    for source, label in [
        ("critical_miss_bytes", "correction"),
        ("prefetch_dram_bytes", "prefetch_dram"),
        ("prefetch_ssd_bytes", "prefetch_ssd"),
        ("total_pcie_bytes", "total_pcie"),
        ("unused_prefetch_bytes", "unused_prefetch"),
        (
            "prefetch_self_evicted_target_bytes",
            "self_evicted_target",
        ),
        ("pollution_miss_bytes", "pollution_correction"),
    ]:
        by_request[f"{label}_mib_per_token_per_gpu"] = (
            by_request[source] / by_request.decode_steps / tp / 2**20
        )

    metric_columns = [
        "hbm_token_recall",
        "prefetch_precision",
        "correction_mib_per_token_per_gpu",
        "prefetch_dram_mib_per_token_per_gpu",
        "prefetch_ssd_mib_per_token_per_gpu",
        "total_pcie_mib_per_token_per_gpu",
        "unused_prefetch_mib_per_token_per_gpu",
        "self_evicted_target_mib_per_token_per_gpu",
        "pollution_correction_mib_per_token_per_gpu",
    ]
    summary = by_request.groupby(
        ["dataset", "context_config", "method", "candidate_k"],
        as_index=False,
    )[metric_columns].mean()
    request_counts = by_request.groupby(
        ["dataset", "context_config", "method", "candidate_k"],
        as_index=False,
    ).request_id.nunique()
    summary = summary.merge(
        request_counts.rename(columns={"request_id": "requests"}),
        on=["dataset", "context_config", "method", "candidate_k"],
        validate="1:1",
    )

    previous = summary[summary.method == "previous-score"].copy()
    baseline = previous[previous.candidate_k == 4096][
        [
            "dataset",
            "context_config",
            "correction_mib_per_token_per_gpu",
            "total_pcie_mib_per_token_per_gpu",
        ]
    ].rename(
        columns={
            "correction_mib_per_token_per_gpu": "k4096_correction_mib_per_token_per_gpu",
            "total_pcie_mib_per_token_per_gpu": "k4096_total_pcie_mib_per_token_per_gpu",
        }
    )
    previous = previous.merge(
        baseline,
        on=["dataset", "context_config"],
        validate="many_to_one",
    )
    previous["correction_reduction_vs_k4096"] = 1 - (
        previous.correction_mib_per_token_per_gpu
        / previous.k4096_correction_mib_per_token_per_gpu.replace(0, np.nan)
    )
    previous["total_pcie_change_vs_k4096"] = (
        previous.total_pcie_mib_per_token_per_gpu
        / previous.k4096_total_pcie_mib_per_token_per_gpu.replace(0, np.nan)
        - 1
    )
    summary = summary.merge(
        previous[
            [
                "dataset",
                "context_config",
                "method",
                "candidate_k",
                "correction_reduction_vs_k4096",
                "total_pcie_change_vs_k4096",
            ]
        ],
        on=["dataset", "context_config", "method", "candidate_k"],
        how="left",
        validate="1:1",
    )
    return by_request, summary


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_quality_tradeoff(
    candidate_summary: pd.DataFrame,
    cache_summary: pd.DataFrame,
    output: Path,
) -> None:
    cache = cache_summary[cache_summary.method == "previous-score"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    for (dataset, context), candidate_part in candidate_summary.groupby(
        ["dataset", "context_config"], sort=True
    ):
        label = series_label(str(dataset), int(context))
        candidate_part = candidate_part.sort_values("candidate_k")
        cache_part = cache[
            (cache.dataset == dataset) & (cache.context_config == context)
        ].sort_values("candidate_k")
        style = {
            "color": COLORS[label],
            "marker": MARKERS[label],
            "label": label,
        }
        axes[0].plot(
            candidate_part.candidate_k,
            candidate_part.candidate_token_recall * 100,
            **style,
        )
        axes[1].plot(
            cache_part.candidate_k,
            cache_part.hbm_token_recall * 100,
            **style,
        )
        axes[2].plot(
            cache_part.candidate_k,
            cache_part.correction_mib_per_token_per_gpu,
            **style,
        )
    axes[0].set_ylabel("Next top-2048 covered (%)")
    axes[1].set_ylabel("Next top-2048 in HBM (%)")
    axes[2].set_ylabel("Correction (MiB/token/GPU)")
    titles = [
        "(a) Candidate coverage",
        "(b) HBM coverage",
        "(c) Exact correction bytes",
    ]
    for axis, title in zip(axes, titles, strict=True):
        axis.set_title(title)
        axis.set_xlabel("Previous-step candidate K")
        axis.set_xticks(sorted(candidate_summary.candidate_k.unique()))
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.72, bottom=0.20, left=0.08, right=0.99, wspace=0.46)
    save_figure(fig, output)


def plot_waste_tradeoff(cache_summary: pd.DataFrame, output: Path) -> None:
    cache = cache_summary[cache_summary.method == "previous-score"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65))
    for (dataset, context), part in cache.groupby(
        ["dataset", "context_config"], sort=True
    ):
        label = series_label(str(dataset), int(context))
        part = part.sort_values("candidate_k")
        style = {
            "color": COLORS[label],
            "marker": MARKERS[label],
            "label": label,
        }
        axes[0].plot(
            part.candidate_k,
            part.unused_prefetch_mib_per_token_per_gpu,
            **style,
        )
        axes[1].plot(
            part.candidate_k,
            part.pollution_correction_mib_per_token_per_gpu,
            **style,
        )
    axes[0].set_ylabel("Unused prefetch (MiB/token/GPU)")
    axes[1].set_ylabel("Pollution correction (MiB/token/GPU)")
    axes[0].set_title("(a) Transferred but unused next step")
    axes[1].set_title("(b) Useful resident pages evicted")
    for axis in axes:
        axis.set_xlabel("Previous-step candidate K")
        axis.set_xticks(sorted(cache.candidate_k.unique()))
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.70, bottom=0.20, left=0.10, right=0.99, wspace=0.38)
    save_figure(fig, output)


def self_test() -> None:
    generator = np.random.default_rng(20260810)
    ranked = np.stack([generator.permutation(8192) for _ in range(32)])
    chunk = Chunk(
        request_id="synthetic",
        dataset="synthetic",
        task="synthetic",
        source_index=0,
        context_config=8192,
        prompt_tokens=8192,
        layer=0,
        topk=ranked[:, :TARGET_K].copy(),
        ranked=ranked,
        valid_counts=np.full(32, 8192, dtype=np.int64),
        source_file=Path("synthetic/events/chunk.pt"),
    )
    rows = pd.DataFrame(
        run_policy(
            chunk,
            4,
            -1.0,
            -1.0,
            "previous-score",
            4096,
            2048,
            1800,
            1024,
            extended_metrics=True,
        )
    )
    assert np.all(
        rows.prefetch_admit_pages
        == rows.useful_prefetch_pages + rows.unused_prefetch_pages
    )
    assert np.all(rows.pollution_miss_pages <= rows.critical_miss_pages)
    assert rows.unused_prefetch_pages.sum() > 0
    print(
        json.dumps(
            {
                "passed": True,
                "rows": len(rows),
                "unused_prefetch_pages": int(rows.unused_prefetch_pages.sum()),
                "pollution_miss_pages": int(rows.pollution_miss_pages.sum()),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layers", type=parse_layer_list, default=DEFAULT_LAYERS)
    parser.add_argument(
        "--candidate-k", type=parse_int_list, default=DEFAULT_CANDIDATE_K
    )
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--hbm-logical-gib", type=float, default=1.2)
    parser.add_argument("--dram-logical-gib", type=float, default=3.0)
    parser.add_argument("--num-model-layers", type=int, default=48)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--kv-bytes-per-token-layer", type=int, default=2048)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.run_dirs or args.output_dir is None:
        raise SystemExit("--run-dirs and --output-dir are required")
    if min(args.candidate_k) < TARGET_K:
        raise ValueError(f"candidate K must be at least {TARGET_K}")

    output = args.output_dir.resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    candidates, cache, validation = simulate_gate(
        [path.resolve() for path in args.run_dirs],
        args.layers,
        args.candidate_k,
        args.page_size,
        args.hbm_logical_gib,
        args.dram_logical_gib,
        args.num_model_layers,
        args.kv_bytes_per_token_layer,
    )
    candidate_by_request, candidate_summary, candidate_by_layer = (
        aggregate_candidates(candidates)
    )
    cache_by_request, cache_summary = aggregate_cache(
        cache, args.num_model_layers, len(args.layers), args.tp
    )

    artifacts = [
        tables / "candidate_by_transition.parquet",
        tables / "cache_by_transition.parquet",
        tables / "candidate_by_request.csv",
        tables / "candidate_summary.csv",
        tables / "candidate_by_layer.csv",
        tables / "cache_by_request.csv",
        tables / "cache_summary.csv",
        output / "validation.json",
        figures / "large_candidate_quality_tradeoff.pdf",
        figures / "large_candidate_quality_tradeoff.png",
        figures / "large_candidate_waste_tradeoff.pdf",
        figures / "large_candidate_waste_tradeoff.png",
    ]
    candidates.to_parquet(artifacts[0], index=False)
    cache.to_parquet(artifacts[1], index=False)
    candidate_by_request.to_csv(artifacts[2], index=False)
    candidate_summary.to_csv(artifacts[3], index=False)
    candidate_by_layer.to_csv(artifacts[4], index=False)
    cache_by_request.to_csv(artifacts[5], index=False)
    cache_summary.to_csv(artifacts[6], index=False)
    artifacts[7].write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    configure_style(None)
    plot_quality_tradeoff(candidate_summary, cache_summary, artifacts[8])
    plot_waste_tradeoff(cache_summary, artifacts[10])

    result = {
        "schema_version": 1,
        "analysis_kind": "trace-derived coverage and deterministic byte accounting; no speed measurement",
        "git_commit": git_revision(Path.cwd()),
        "inputs": [str(path.resolve()) for path in args.run_dirs],
        "setting": {
            "candidate_k": args.candidate_k,
            "target_k": TARGET_K,
            "sampled_layers": args.layers,
            "page_size_tokens": args.page_size,
            "hbm_logical_gib": validation["actual_hbm_logical_gib"],
            "dram_logical_gib": validation["actual_dram_logical_gib"],
            "num_model_layers": args.num_model_layers,
            "tensor_parallel_size": args.tp,
            "task_weighting": "equal weight per request/task within each dataset-context cell",
        },
        "candidate_summary": json_records(candidate_summary),
        "cache_summary": json_records(cache_summary),
        "limitations": [
            "bytes are deterministic logical simulator counts, not measured latency or speedup",
            "seven sampled layers are scaled linearly to 48 layers",
            "one fixed 1.2 GiB HBM and 3.0 GiB DRAM capacity point is evaluated",
            "each dataset-context cell currently has one request per selected task",
        ],
        "artifacts": {
            str(path.relative_to(output)): sha256(path) for path in artifacts
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
