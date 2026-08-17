#!/usr/bin/env python3
"""Gate C0: rank top-8192 pages under the top-4096 transfer budget.

The experiment is deliberately trace driven.  It keeps exact top-2048 demand,
HBM/DRAM capacities, and the number of pages transferred by the top-4096
baseline unchanged.  Wider candidates may only change which pages are kept or
admitted; they cannot buy a larger transfer budget.  No latency or speedup is
reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulate_multitier_prefetch import (
    DEFAULT_LAYERS,
    TARGET_K,
    Chunk,
    TierCache,
    TransferCounts,
    candidate_pages,
    configure_style,
    load_run,
    page_slots,
    run_policy,
    target_for_transition,
    warm_cache,
)

BASELINE_K = 4096
POOL_K = 8192
DEPLOYABLE_METHODS = [
    "wide-rank",
    "page-count",
    "page-exp-mass",
    "page-excess-mass",
    "page-rrf",
    "recent-exact-count",
]
METHOD_LABELS = {
    "baseline-rank4096": "K4096 rank prefix",
    "wide-rank": "K8192 token rank",
    "page-count": "Page candidate count",
    "page-exp-mass": "Page exp-score mass",
    "page-excess-mass": "Page excess-score mass",
    "page-rrf": "Page reciprocal-rank mass",
    "recent-exact-count": "Recent exact use + score mass",
    "oracle-equal-budget": "Equal-budget oracle",
}
COLORS = {
    "baseline-rank4096": "#4D4D4D",
    "wide-rank": "#56B4E9",
    "page-count": "#0072B2",
    "page-exp-mass": "#009E73",
    "page-excess-mass": "#E69F00",
    "page-rrf": "#CC79A7",
    "recent-exact-count": "#D55E00",
    "oracle-equal-budget": "#000000",
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


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def page_feature_order(
    chunk: Chunk,
    step: int,
    common: int,
    page_size: int,
    method: str,
    recent_exact: Counter[int],
    target_pages: Iterable[int] | None = None,
) -> list[int]:
    if chunk.ranked_scores is None:
        raise ValueError("compact candidate scores are required for Gate C0")
    tokens = chunk.ranked[step, :POOL_K]
    scores = chunk.ranked_scores[step, :POOL_K]
    valid = (tokens >= 0) & (tokens < common) & np.isfinite(scores)
    tokens = tokens[valid]
    scores = scores[valid].astype(np.float64)
    if not len(tokens):
        return []

    pages = tokens.astype(np.int64) // page_size
    unique_pages, inverse = np.unique(pages, return_inverse=True)
    best_rank = np.full(len(unique_pages), len(tokens), dtype=np.int64)
    np.minimum.at(best_rank, inverse, np.arange(len(tokens), dtype=np.int64))
    count = np.bincount(inverse, minlength=len(unique_pages))
    maximum = float(scores[0])
    threshold = float(scores[-1])
    exp_mass = np.bincount(
        inverse,
        weights=np.exp(scores - maximum),
        minlength=len(unique_pages),
    )
    excess_mass = np.bincount(
        inverse,
        weights=np.maximum(0.0, scores - threshold),
        minlength=len(unique_pages),
    )
    reciprocal_mass = np.bincount(
        inverse,
        weights=1.0 / (np.arange(len(tokens), dtype=np.float64) + 64.0),
        minlength=len(unique_pages),
    )
    if method == "wide-rank":
        order = np.lexsort((unique_pages, best_rank))
    elif method == "page-count":
        order = np.lexsort((unique_pages, best_rank, -count))
    elif method == "page-exp-mass":
        order = np.lexsort((unique_pages, best_rank, -exp_mass))
    elif method == "page-excess-mass":
        order = np.lexsort((unique_pages, best_rank, -excess_mass))
    elif method == "page-rrf":
        order = np.lexsort((unique_pages, best_rank, -reciprocal_mass))
    elif method == "recent-exact-count":
        recent = np.fromiter(
            (recent_exact[int(page)] for page in unique_pages),
            dtype=np.int64,
            count=len(unique_pages),
        )
        order = np.lexsort(
            (unique_pages, best_rank, -exp_mass, -recent)
        )
    elif method == "oracle-equal-budget":
        if target_pages is None:
            raise ValueError("oracle requires next-step target pages")
        target = set(target_pages)
        outside_target = np.fromiter(
            (int(int(page) not in target) for page in unique_pages),
            dtype=np.int8,
            count=len(unique_pages),
        )
        order = np.lexsort((unique_pages, best_rank, outside_target))
    else:
        raise ValueError(f"unknown admission method {method}")
    return [int(page) for page in unique_pages[order]]


def choose_equal_page_budget(
    order: list[int], cache: TierCache, transfer_page_budget: int
) -> tuple[list[int], list[int]]:
    """Choose exactly the baseline number of initially nonresident pages.

    Current HBM pages do not consume transfer budget.  They are retained by
    value, while the highest-value nonresident pages consume the fixed budget.
    The final selected set never exceeds HBM capacity.
    """
    nonresident = [page for page in order if cache.tier(page) != "hbm"]
    admitted = nonresident[:transfer_page_budget]
    if len(admitted) != transfer_page_budget:
        raise AssertionError(
            f"insufficient nonresident candidates: {len(admitted)} < "
            f"{transfer_page_budget}"
        )
    admitted_set = set(admitted)
    resident_slots = max(0, cache.hbm_capacity - len(admitted))
    retained = [
        page
        for page in order
        if page in cache.hbm and page not in admitted_set
    ][:resident_slots]
    if len(retained) < resident_slots:
        # Pages outside the wide pool remain eligible only as deterministic
        # filler; this never displaces a ranked resident candidate.
        for page in reversed(cache.hbm):
            if page not in retained and page not in admitted_set:
                retained.append(page)
                if len(retained) == resident_slots:
                    break
    return admitted, retained


def simulate_admission_policy(
    chunk: Chunk,
    page_size: int,
    method: str,
    kv_bytes_per_token_layer: int,
    hbm_pages: int,
    dram_pages: int,
    baseline_rows: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    total_tokens = int(chunk.valid_counts.max())
    cache = warm_cache(chunk, page_size, hbm_pages, dram_pages)
    recent_windows: deque[set[int]] = deque(maxlen=4)
    rows: list[dict[str, Any]] = []

    for step in range(31):
        target_tokens, target_order, common = target_for_transition(
            chunk, step, page_size
        )
        target_set = set(target_order)
        current_exact = set(
            candidate_pages(chunk, step, page_size, TARGET_K, common)
        )
        if not recent_windows or recent_windows[-1] != current_exact:
            recent_windows.append(current_exact)
        recent_exact: Counter[int] = Counter()
        for age, pages in enumerate(reversed(recent_windows), start=1):
            for page in pages:
                recent_exact[page] += len(recent_windows) - age + 1

        order = page_feature_order(
            chunk,
            step,
            common,
            page_size,
            method,
            recent_exact,
            target_order,
        )
        baseline = baseline_rows[step]
        page_budget = int(baseline["prefetch_admit_pages"])
        baseline_prefetch_bytes = int(baseline["prefetch_bytes"])
        hbm_before = set(cache.hbm)
        admitted, retained = choose_equal_page_budget(
            order, cache, page_budget
        )

        prefetch = TransferCounts()
        # Protect the selected resident working set before admitting new pages.
        # Otherwise a selected LRU resident can be evicted and silently
        # re-promoted below, which would violate transfer accounting.
        for page in reversed(retained):
            if cache.promote(page) != "hbm":
                raise AssertionError("retained page unexpectedly nonresident")
        for page in reversed(admitted):
            source = cache.promote(page)
            if source == "hbm":
                raise AssertionError("admitted page unexpectedly resident")
            prefetch.add(
                source,
                page_slots(page, page_size, total_tokens)
                * kv_bytes_per_token_layer,
            )

        # Re-touch the complete selected set in reverse value order.  This is
        # metadata-only and leaves the most valuable page most recent.
        selected = set(admitted) | set(retained)
        if not selected.issubset(cache.hbm):
            raise AssertionError("selected page evicted during fixed-budget admit")
        for page in reversed([page for page in order if page in selected]):
            if cache.promote(page) != "hbm":
                raise AssertionError("selected page unexpectedly requires transfer")

        hbm_ready = set(cache.hbm)
        prefetched_pages = set(admitted)
        unused_prefetch = prefetched_pages - target_set
        pollution_miss = (hbm_before & target_set) - hbm_ready
        hit_pages = target_set & hbm_ready
        hit_tokens = sum(
            (int(token) // page_size) in hbm_ready for token in target_tokens
        )

        critical = TransferCounts()
        for page in target_order:
            source = cache.tier(page)
            if source != "hbm":
                critical.add(
                    source,
                    page_slots(page, page_size, total_tokens)
                    * kv_bytes_per_token_layer,
                )
        for page in reversed(target_order):
            cache.promote(page)
        next_valid = int(chunk.valid_counts[step + 1])
        if next_valid > common:
            cache.create_in_hbm((next_valid - 1) // page_size)
        cache.check()
        recent_windows.append(target_set)

        prefetch_bytes = prefetch.dram_bytes + prefetch.ssd_bytes
        critical_bytes = critical.dram_bytes + critical.ssd_bytes
        rows.append(
            {
                "run_dir": str(chunk.source_file.parents[1]),
                "request_id": chunk.request_id,
                "dataset": chunk.dataset,
                "task": chunk.task,
                "source_index": str(chunk.source_index),
                "context_config": chunk.context_config,
                "prompt_tokens": chunk.prompt_tokens,
                "layer": chunk.layer,
                "step": step,
                "page_size": page_size,
                "method": method,
                "candidate_k": POOL_K,
                "common_tokens": common,
                "hbm_capacity_pages": cache.hbm_capacity,
                "dram_capacity_pages": cache.dram_capacity,
                "target_tokens": len(target_tokens),
                "target_pages": len(target_set),
                "hbm_hit_tokens": hit_tokens,
                "hbm_hit_pages": len(hit_pages),
                "prefetch_admit_pages": len(admitted),
                "baseline_prefetch_admit_pages": page_budget,
                "prefetch_dram_pages": prefetch.dram_pages,
                "prefetch_dram_bytes": prefetch.dram_bytes,
                "prefetch_ssd_pages": prefetch.ssd_pages,
                "prefetch_ssd_bytes": prefetch.ssd_bytes,
                "prefetch_bytes": prefetch_bytes,
                "baseline_prefetch_bytes": baseline_prefetch_bytes,
                "prefetch_byte_delta_vs_baseline": (
                    prefetch_bytes - baseline_prefetch_bytes
                ),
                "useful_prefetch_pages": len(prefetched_pages & target_set),
                "unused_prefetch_pages": len(unused_prefetch),
                "pollution_miss_pages": len(pollution_miss),
                "critical_dram_pages": critical.dram_pages,
                "critical_dram_bytes": critical.dram_bytes,
                "critical_ssd_pages": critical.ssd_pages,
                "critical_ssd_bytes": critical.ssd_bytes,
                "critical_miss_pages": critical.dram_pages + critical.ssd_pages,
                "critical_miss_bytes": critical_bytes,
                "total_pcie_bytes": prefetch_bytes + critical_bytes,
                "source_file": chunk.source_file.name,
            }
        )
    return rows


def collect_detail(
    run_dirs: list[Path],
    layers: list[int],
    page_size: int,
    hbm_logical_gib: float,
    dram_logical_gib: float,
    num_model_layers: int,
    kv_bytes_per_token_layer: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bytes_per_model_page = (
        page_size * kv_bytes_per_token_layer * num_model_layers
    )
    hbm_pages = max(1, int(hbm_logical_gib * 2**30 // bytes_per_model_page))
    dram_pages = max(0, int(dram_logical_gib * 2**30 // bytes_per_model_page))
    methods = DEPLOYABLE_METHODS + ["oracle-equal-budget"]
    rows: list[dict[str, Any]] = []
    request_records: list[dict[str, Any]] = []
    chunk_count = 0

    for run_dir in run_dirs:
        chunks = load_run(run_dir, layers, POOL_K)
        chunk_count += len(chunks)
        for chunk in chunks:
            baseline = run_policy(
                chunk,
                page_size,
                -1.0,
                -1.0,
                "previous-score",
                BASELINE_K,
                kv_bytes_per_token_layer,
                hbm_pages,
                dram_pages,
                extended_metrics=True,
            )
            baseline_by_step: dict[int, dict[str, Any]] = {}
            for row in baseline:
                row["method"] = "baseline-rank4096"
                row["prefetch_bytes"] = (
                    row["prefetch_dram_bytes"] + row["prefetch_ssd_bytes"]
                )
                row["baseline_prefetch_admit_pages"] = row[
                    "prefetch_admit_pages"
                ]
                row["baseline_prefetch_bytes"] = row["prefetch_bytes"]
                row["prefetch_byte_delta_vs_baseline"] = 0
                row["source_index"] = str(row["source_index"])
                baseline_by_step[int(row["step"])] = row
                rows.append(row)
            for method in methods:
                rows.extend(
                    simulate_admission_policy(
                        chunk,
                        page_size,
                        method,
                        kv_bytes_per_token_layer,
                        hbm_pages,
                        dram_pages,
                        baseline_by_step,
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
                    "layers": [chunk.layer for chunk in request_chunks],
                }
            )

    detail = pd.DataFrame(rows)
    expected_rows = chunk_count * 31 * (len(methods) + 1)
    if len(detail) != expected_rows:
        raise AssertionError(f"row mismatch: {len(detail)} != {expected_rows}")
    proposed = detail[detail.method != "baseline-rank4096"]
    if not np.all(
        proposed.prefetch_admit_pages
        == proposed.baseline_prefetch_admit_pages
    ):
        raise AssertionError("equal page-transfer budget violated")
    validation = {
        "schema_version": 1,
        "analysis_kind": "equal-page-budget deterministic cache replay; no speed measurement",
        "requests": len(request_records),
        "chunks": chunk_count,
        "rows_expected": expected_rows,
        "rows_actual": len(detail),
        "layers": layers,
        "methods": ["baseline-rank4096"] + methods,
        "baseline_candidate_k": BASELINE_K,
        "candidate_pool_k": POOL_K,
        "page_size_tokens": page_size,
        "hbm_capacity_pages_per_layer": hbm_pages,
        "dram_capacity_pages_per_layer": dram_pages,
        "requested_hbm_logical_gib": hbm_logical_gib,
        "requested_dram_logical_gib": dram_logical_gib,
        "runs": request_records,
        "checks": {
            "exact_top2048_demand_preserved": True,
            "equal_prefetch_page_count_per_transition": True,
            "hbm_capacity_preserved": True,
            "exact_correction_materialized_before_next_step": True,
        },
    }
    return detail, validation


def aggregate(
    detail: pd.DataFrame,
    num_model_layers: int,
    sampled_layer_count: int,
    tp: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    request_keys = [
        "dataset",
        "task",
        "request_id",
        "context_config",
        "method",
    ]
    sums = (
        detail.groupby(request_keys, as_index=False)
        .agg(
            transitions=("step", "size"),
            target_tokens=("target_tokens", "sum"),
            target_pages=("target_pages", "sum"),
            hbm_hit_tokens=("hbm_hit_tokens", "sum"),
            hbm_hit_pages=("hbm_hit_pages", "sum"),
            prefetch_admit_pages=("prefetch_admit_pages", "sum"),
            prefetch_bytes=("prefetch_bytes", "sum"),
            baseline_prefetch_bytes=("baseline_prefetch_bytes", "sum"),
            critical_miss_bytes=("critical_miss_bytes", "sum"),
            total_pcie_bytes=("total_pcie_bytes", "sum"),
            unused_prefetch_pages=("unused_prefetch_pages", "sum"),
            pollution_miss_pages=("pollution_miss_pages", "sum"),
            max_abs_step_prefetch_byte_delta=(
                "prefetch_byte_delta_vs_baseline",
                lambda values: int(np.max(np.abs(values))),
            ),
        )
    )
    request = sums.copy()
    request["hbm_token_recall"] = request.hbm_hit_tokens / request.target_tokens
    request["hbm_page_recall"] = request.hbm_hit_pages / request.target_pages
    request["prefetch_precision"] = (
        (request.prefetch_admit_pages - request.unused_prefetch_pages)
        / request.prefetch_admit_pages
    )
    scale = num_model_layers / sampled_layer_count / tp / 2**20
    denom = request.transitions / sampled_layer_count
    for source, target in [
        ("prefetch_bytes", "prefetch_mib_per_token_per_gpu"),
        ("critical_miss_bytes", "correction_mib_per_token_per_gpu"),
        ("total_pcie_bytes", "total_pcie_mib_per_token_per_gpu"),
    ]:
        request[target] = request[source] * scale / denom

    baseline = request[request.method == "baseline-rank4096"][
        [
            "request_id",
            "hbm_token_recall",
            "correction_mib_per_token_per_gpu",
            "total_pcie_mib_per_token_per_gpu",
        ]
    ].rename(
        columns={
            "hbm_token_recall": "baseline_hbm_token_recall",
            "correction_mib_per_token_per_gpu": "baseline_correction_mib",
            "total_pcie_mib_per_token_per_gpu": "baseline_total_pcie_mib",
        }
    )
    request = request.merge(baseline, on="request_id", how="left", validate="many_to_one")
    request["hbm_recall_delta_vs_baseline"] = (
        request.hbm_token_recall - request.baseline_hbm_token_recall
    )
    request["correction_reduction_vs_baseline"] = 1.0 - (
        request.correction_mib_per_token_per_gpu
        / request.baseline_correction_mib
    )
    request["total_pcie_change_vs_baseline"] = (
        request.total_pcie_mib_per_token_per_gpu
        / request.baseline_total_pcie_mib
        - 1.0
    )

    cell = (
        request.groupby(["dataset", "context_config", "method"], as_index=False)
        .agg(
            requests=("request_id", "nunique"),
            hbm_token_recall=("hbm_token_recall", "mean"),
            hbm_page_recall=("hbm_page_recall", "mean"),
            prefetch_precision=("prefetch_precision", "mean"),
            prefetch_mib_per_token_per_gpu=(
                "prefetch_mib_per_token_per_gpu",
                "mean",
            ),
            correction_mib_per_token_per_gpu=(
                "correction_mib_per_token_per_gpu",
                "mean",
            ),
            total_pcie_mib_per_token_per_gpu=(
                "total_pcie_mib_per_token_per_gpu",
                "mean",
            ),
            hbm_recall_delta_vs_baseline=(
                "hbm_recall_delta_vs_baseline",
                "mean",
            ),
            correction_reduction_vs_baseline=(
                "correction_reduction_vs_baseline",
                "mean",
            ),
            total_pcie_change_vs_baseline=(
                "total_pcie_change_vs_baseline",
                "mean",
            ),
            no_worse_hbm_requests=(
                "hbm_recall_delta_vs_baseline",
                lambda values: int(np.sum(values >= -1e-12)),
            ),
            lower_correction_requests=(
                "correction_reduction_vs_baseline",
                lambda values: int(np.sum(values > 1e-12)),
            ),
            max_abs_step_prefetch_byte_delta=(
                "max_abs_step_prefetch_byte_delta",
                "max",
            ),
        )
    )
    return request, cell


def choose_frozen_policy(request: pd.DataFrame) -> dict[str, Any]:
    calibration = request[
        (request.dataset == "RULER")
        & request.method.isin(DEPLOYABLE_METHODS)
    ]
    ranking = (
        calibration.groupby("method", as_index=False)
        .agg(
            correction_mib=("correction_mib_per_token_per_gpu", "mean"),
            hbm_recall=("hbm_token_recall", "mean"),
            total_pcie_mib=("total_pcie_mib_per_token_per_gpu", "mean"),
        )
        .sort_values(
            ["correction_mib", "total_pcie_mib", "method"],
            ascending=[True, True, True],
        )
    )
    selected = str(ranking.iloc[0].method)
    test = request[
        (request.dataset == "LongBench-v2") & (request.method == selected)
    ]
    all_selected = request[request.method == selected]
    return {
        "selection_rule": "minimum mean correction bytes on RULER only; ties by total PCIe then method name",
        "selected_method": selected,
        "calibration_ranking": json_records(ranking),
        "longbench_requests": int(test.request_id.nunique()),
        "longbench_no_worse_hbm_requests": int(
            np.sum(test.hbm_recall_delta_vs_baseline >= -1e-12)
        ),
        "longbench_mean_hbm_recall_delta": float(
            test.hbm_recall_delta_vs_baseline.mean()
        ),
        "longbench_mean_correction_reduction": float(
            test.correction_reduction_vs_baseline.mean()
        ),
        "longbench_mean_total_pcie_change": float(
            test.total_pcie_change_vs_baseline.mean()
        ),
        "all_requests": int(all_selected.request_id.nunique()),
        "all_no_worse_hbm_requests": int(
            np.sum(all_selected.hbm_recall_delta_vs_baseline >= -1e-12)
        ),
        "gate_pass": bool(
            len(all_selected) == 24
            and np.all(all_selected.hbm_recall_delta_vs_baseline >= -1e-12)
            and all_selected.correction_mib_per_token_per_gpu.mean()
            < all_selected.baseline_correction_mib.mean()
            and all_selected.total_pcie_mib_per_token_per_gpu.mean()
            <= all_selected.baseline_total_pcie_mib.mean()
        ),
    }


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_summary(cell: pd.DataFrame, selected: str, output: Path) -> None:
    # The zero line is the K4096 baseline; only the two non-zero comparisons
    # need bars and legend entries.
    methods = [selected, "oracle-equal-budget"]
    view = cell[cell.method.isin(methods)].copy()
    view["series"] = view.dataset.str.replace("LongBench-v2", "LB-v2") + " " + (
        view.context_config // 1024
    ).astype(str) + "K"
    series = ["RULER 64K", "RULER 128K", "LB-v2 64K", "LB-v2 128K"]
    x = np.arange(len(series))
    width = 0.30
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    for index, method in enumerate(methods):
        part = view[view.method == method].set_index("series").reindex(series)
        offset = (index - 0.5) * width
        axes[0].bar(
            x + offset,
            part.hbm_recall_delta_vs_baseline * 100,
            width,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
        axes[1].bar(
            x + offset,
            part.total_pcie_change_vs_baseline * 100,
            width,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0].set_ylabel("HBM recall change (pp)")
    axes[1].set_ylabel("Total PCIe change (%)")
    axes[0].set_title("(a) Change vs. K4096")
    axes[1].set_title("(b) Change vs. K4096")
    for axis in axes:
        axis.set_xticks(x, series, rotation=20, ha="right")
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        axis.axhline(0.0, color="#333333", linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(top=0.73, bottom=0.25, left=0.10, right=0.99, wspace=0.36)
    save_figure(fig, output)


def self_test() -> None:
    generator = np.random.default_rng(20260810)
    ranked = np.stack([generator.permutation(8192) for _ in range(32)])
    scores = np.sort(generator.normal(size=(32, 8192)), axis=1)[:, ::-1].copy()
    chunk = Chunk(
        request_id="synthetic",
        dataset="RULER",
        task="synthetic",
        source_index=0,
        context_config=8192,
        prompt_tokens=8192,
        layer=0,
        topk=ranked[:, :TARGET_K].copy(),
        ranked=ranked,
        valid_counts=np.full(32, 8192, dtype=np.int64),
        source_file=Path("synthetic/events/chunk.pt"),
        ranked_scores=scores.astype(np.float32),
    )
    baseline = run_policy(
        chunk, 4, -1.0, -1.0, "previous-score", BASELINE_K, 2048, 1000, 512,
        extended_metrics=True,
    )
    baseline_by_step = {}
    for row in baseline:
        row["prefetch_bytes"] = row["prefetch_dram_bytes"] + row["prefetch_ssd_bytes"]
        baseline_by_step[int(row["step"])] = row
    rows = simulate_admission_policy(
        chunk, 4, "page-count", 2048, 1000, 512, baseline_by_step
    )
    assert len(rows) == 31
    assert all(
        row["prefetch_admit_pages"] == row["baseline_prefetch_admit_pages"]
        for row in rows
    )
    print(json.dumps({"passed": True, "rows": len(rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layers", type=parse_layer_list, default=DEFAULT_LAYERS)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--hbm-logical-gib", type=float, default=1.2)
    parser.add_argument("--dram-logical-gib", type=float, default=3.0)
    parser.add_argument("--num-model-layers", type=int, default=48)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--kv-bytes-per-token-layer", type=int, default=2048)
    parser.add_argument("--style", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.run_dirs or args.output_dir is None:
        raise SystemExit("--run-dirs and --output-dir are required")
    if args.page_size != 4:
        raise ValueError("Gate C0 is pre-registered for 4-token pages")
    if args.tp <= 0 or args.num_model_layers <= 0:
        raise ValueError("TP and model layer counts must be positive")

    configure_style(args.style)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = args.output_dir / "tables"
    figures = args.output_dir / "figures"
    tables.mkdir(exist_ok=True)
    detail, validation = collect_detail(
        args.run_dirs,
        list(args.layers),
        args.page_size,
        args.hbm_logical_gib,
        args.dram_logical_gib,
        args.num_model_layers,
        args.kv_bytes_per_token_layer,
    )
    request, cell = aggregate(
        detail, args.num_model_layers, len(args.layers), args.tp
    )
    frozen = choose_frozen_policy(request)

    detail_path = tables / "transition_detail.parquet"
    request_path = tables / "request_summary.csv"
    cell_path = tables / "cell_summary.csv"
    detail.to_parquet(detail_path, index=False)
    request.to_csv(request_path, index=False)
    cell.to_csv(cell_path, index=False)
    plot_summary(
        cell,
        frozen["selected_method"],
        figures / "equal_budget_admission_summary",
    )

    validation.update(
        {
            "repository_revision": git_revision(Path(__file__).resolve().parents[2]),
            "transition_detail_sha256": sha256(detail_path),
            "request_summary_sha256": sha256(request_path),
            "cell_summary_sha256": sha256(cell_path),
            "max_abs_step_prefetch_byte_delta": int(
                request.max_abs_step_prefetch_byte_delta.max()
            ),
            "frozen_policy": frozen,
        }
    )
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "schema_version": 1,
        "scope": "Gate C0 equal-page-budget shadow admission; no speed measurement",
        "frozen_policy": frozen,
        "cell_summary": json_records(cell),
        "artifacts": {
            "transition_detail": str(detail_path),
            "request_summary": str(request_path),
            "cell_summary": str(cell_path),
            "figure_pdf": str(
                figures / "equal_budget_admission_summary.pdf"
            ),
            "figure_png": str(
                figures / "equal_budget_admission_summary.png"
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
