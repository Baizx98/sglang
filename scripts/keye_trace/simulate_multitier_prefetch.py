#!/usr/bin/env python3
"""Trace-driven HBM--DRAM--SSD simulator for DSA decode KV prefetch.

The simulator evaluates a deployable same-layer predictor: the score-ranked
prefix from decode step ``t`` predicts the historical KV pages needed at step
``t + 1``.  It intentionally does not report speedup.  Trace-derived page and
byte counts are separated from a parameterized transfer-time model.

Cache semantics are exclusive and deterministic.  A page resides in exactly
one of HBM, DRAM, or SSD.  HBM and DRAM use LRU replacement.  Newly generated
KV is inserted into HBM without transfer cost.  A prefetch promotes the
highest-ranked unique candidate pages that fit the HBM page capacity.  Demand
misses then correct the prediction, preserving exact top-2048 attention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import torch

TARGET_K = 2048
DEFAULT_REQUIRED_COMPACT_K = 4096
DEFAULT_LAYERS = [0, 7, 15, 23, 31, 39, 47]
DEFAULT_PAGE_SIZES = [4, 16, 64]
DEFAULT_CANDIDATE_K = [2048, 2560, 3072, 4096]
DEFAULT_HBM_FRACTIONS = [0.05, 0.10, 0.20, 0.40]
DEFAULT_DRAM_FRACTIONS = [0.10, 0.25, 0.50, 1.00]
DEFAULT_OVERLAP_WINDOWS_MS = [1.0, 5.0, 10.0, 20.0]
FIXED_HBM_LOGICAL_GIB = [0.6, 1.2]
FIXED_DRAM_LOGICAL_GIB = 3.0
DEFAULT_STYLE = Path(
    "/home10T/bzx/.codex/skills/research-figure-style/assets/"
    "matplotlib_style.mplstyle"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated ints")
    return values


def parse_float_list(value: str) -> list[float]:
    values = sorted({float(item) for item in value.split(",") if item.strip()})
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated floats")
    return values


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


def unique_page_order(tokens: Iterable[int], page_size: int) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for token in tokens:
        token = int(token)
        if token < 0:
            continue
        page = token // page_size
        if page not in seen:
            seen.add(page)
            result.append(page)
    return result


def page_slots(page: int, page_size: int, valid_tokens: int) -> int:
    start = page * page_size
    return max(0, min(page_size, valid_tokens - start))


def pages_bytes(
    pages: Iterable[int],
    page_size: int,
    valid_tokens: int,
    kv_bytes_per_token_layer: int,
) -> int:
    return sum(
        page_slots(page, page_size, valid_tokens) * kv_bytes_per_token_layer
        for page in pages
    )


@dataclass
class TransferCounts:
    dram_pages: int = 0
    dram_bytes: int = 0
    ssd_pages: int = 0
    ssd_bytes: int = 0

    def add(self, source: str, byte_count: int) -> None:
        if source == "dram":
            self.dram_pages += 1
            self.dram_bytes += byte_count
        elif source == "ssd":
            self.ssd_pages += 1
            self.ssd_bytes += byte_count
        elif source != "hbm":
            raise ValueError(f"unknown source {source}")


class TierCache:
    """Exclusive HBM/DRAM LRU state; all absent pages are on SSD."""

    def __init__(
        self,
        total_pages: int,
        hbm_capacity_pages: int,
        dram_capacity_pages: int,
    ) -> None:
        if total_pages <= 0 or hbm_capacity_pages <= 0:
            raise ValueError("total and HBM page capacities must be positive")
        if dram_capacity_pages < 0:
            raise ValueError("DRAM page capacity cannot be negative")
        self.total_pages = total_pages
        self.hbm_capacity = min(hbm_capacity_pages, total_pages)
        self.dram_capacity = min(dram_capacity_pages, total_pages)
        self.hbm: OrderedDict[int, None] = OrderedDict()
        self.dram: OrderedDict[int, None] = OrderedDict()

    def initialize_recent_dram(self, valid_pages: int) -> None:
        begin = max(0, valid_pages - self.dram_capacity)
        for page in range(begin, valid_pages):
            self.dram[page] = None

    def tier(self, page: int) -> str:
        if page in self.hbm:
            return "hbm"
        if page in self.dram:
            return "dram"
        return "ssd"

    def _insert_dram(self, page: int) -> None:
        if self.dram_capacity == 0:
            return
        self.dram.pop(page, None)
        self.dram[page] = None
        while len(self.dram) > self.dram_capacity:
            self.dram.popitem(last=False)

    def _insert_hbm(self, page: int) -> None:
        self.dram.pop(page, None)
        self.hbm.pop(page, None)
        self.hbm[page] = None
        while len(self.hbm) > self.hbm_capacity:
            evicted, _ = self.hbm.popitem(last=False)
            self._insert_dram(evicted)

    def promote(self, page: int) -> str:
        source = self.tier(page)
        self._insert_hbm(page)
        return source

    def create_in_hbm(self, page: int) -> None:
        self._insert_hbm(page)

    def check(self) -> None:
        if set(self.hbm) & set(self.dram):
            raise AssertionError("exclusive-tier invariant violated")
        if len(self.hbm) > self.hbm_capacity:
            raise AssertionError("HBM capacity exceeded")
        if len(self.dram) > self.dram_capacity:
            raise AssertionError("DRAM capacity exceeded")
        if any(page < 0 or page >= self.total_pages for page in self.hbm):
            raise AssertionError("invalid HBM page")
        if any(page < 0 or page >= self.total_pages for page in self.dram):
            raise AssertionError("invalid DRAM page")


@dataclass
class Chunk:
    request_id: str
    dataset: str
    task: str
    source_index: Any
    context_config: int
    prompt_tokens: int
    layer: int
    topk: np.ndarray
    ranked: np.ndarray
    valid_counts: np.ndarray
    source_file: Path
    ranked_scores: np.ndarray | None = None


def load_run(
    run_dir: Path,
    layers: list[int],
    required_compact_k: int = DEFAULT_REQUIRED_COMPACT_K,
) -> list[Chunk]:
    if required_compact_k < TARGET_K:
        raise ValueError(
            f"required compact K must be at least canonical K={TARGET_K}"
        )
    prepared_requests = read_jsonl(run_dir / "prepared_requests.jsonl")
    if not prepared_requests:
        raise ValueError(f"{run_dir}: no prepared requests")
    prepared_by_id = {
        str(request["rid"]): request for request in prepared_requests
    }
    if len(prepared_by_id) != len(prepared_requests):
        raise ValueError(f"{run_dir}: duplicate prepared request id")
    completed_path = run_dir / "requests.jsonl"
    if completed_path.exists():
        completed_rows = read_jsonl(completed_path)
        completed_ids = [str(row["rid"]) for row in completed_rows]
        if len(completed_ids) != len(set(completed_ids)):
            raise ValueError(f"{run_dir}: duplicate completed request id")
        unknown = set(completed_ids) - set(prepared_by_id)
        if unknown:
            raise ValueError(
                f"{run_dir}: completed requests missing from preparation: "
                f"{sorted(unknown)}"
            )
        requests_by_id = {
            request_id: prepared_by_id[request_id]
            for request_id in completed_ids
        }
    else:
        requests_by_id = prepared_by_id
    if not requests_by_id:
        raise ValueError(f"{run_dir}: no completed requests to load")
    wanted = set(layers)
    chunks: list[Chunk] = []
    for row in read_jsonl(run_dir / "events" / "manifest.jsonl"):
        layer = int(row["layer_id"])
        request_id = str(row["request_id"])
        if layer not in wanted or request_id not in requests_by_id:
            continue
        request = requests_by_id[request_id]
        path = run_dir / "events" / row["file"]
        record = torch.load(path, map_location="cpu", weights_only=False)
        if int(record["schema_version"]) != 5:
            raise ValueError(f"{path}: compact trace schema v5 required")
        if record.get("topk_backend") != "torch_exact":
            raise ValueError(f"{path}: torch_exact trace required")
        if int(record["compact_k"]) < required_compact_k:
            raise ValueError(
                f"{path}: top-{required_compact_k} prefix required, "
                f"found top-{int(record['compact_k'])}"
            )
        if record["decode_step_ids"] != list(range(32)):
            raise ValueError(f"{path}: expected decode steps 0..31")
        topk = record["indices"].numpy(force=True).astype(np.int64)
        ranked = record["candidate_indices"].numpy(force=True).astype(np.int64)
        ranked_scores = (
            record["candidate_scores"].numpy(force=True).astype(np.float32)
        )
        valid = record["score_valid_counts"].numpy(force=True).astype(np.int64)
        if (
            topk.shape != (32, TARGET_K)
            or ranked.shape[0] != 32
            or ranked_scores.shape != ranked.shape
        ):
            raise ValueError(f"{path}: unexpected compact trace shapes")
        for step in range(32):
            canonical = set(int(x) for x in topk[step] if int(x) >= 0)
            prefix = set(int(x) for x in ranked[step, :TARGET_K] if int(x) >= 0)
            if canonical != prefix:
                raise ValueError(f"{path}: exact prefix mismatch at step {step}")
        chunks.append(
            Chunk(
                request_id=request_id,
                dataset=str(request.get("dataset", "unknown")),
                task=str(request.get("task", "unknown")),
                source_index=request.get("source_index"),
                context_config=int(request["length_config"]),
                prompt_tokens=int(request["prompt_len"]),
                layer=layer,
                topk=topk,
                ranked=ranked,
                valid_counts=valid,
                source_file=path,
                ranked_scores=ranked_scores,
            )
        )
    chunks_by_request: dict[str, list[Chunk]] = {
        request_id: [] for request_id in requests_by_id
    }
    for chunk in chunks:
        chunks_by_request[chunk.request_id].append(chunk)
    for request_id, request_chunks in chunks_by_request.items():
        present = [chunk.layer for chunk in request_chunks]
        if len(present) != len(set(present)):
            raise ValueError(f"{run_dir}: duplicate layer for {request_id}")
        missing = wanted - set(present)
        if missing:
            raise ValueError(
                f"{run_dir}: request {request_id} missing layers {sorted(missing)}"
            )
    return sorted(chunks, key=lambda chunk: (chunk.request_id, chunk.layer))


def target_for_transition(
    chunk: Chunk, step: int, page_size: int
) -> tuple[np.ndarray, list[int], int]:
    common = min(int(chunk.valid_counts[step]), int(chunk.valid_counts[step + 1]))
    target_ranked = chunk.ranked[step + 1, :TARGET_K]
    target_tokens = target_ranked[
        (target_ranked >= 0) & (target_ranked < common)
    ]
    target_pages = unique_page_order(target_tokens, page_size)
    if len(set(int(token) for token in target_tokens)) != len(target_tokens):
        raise AssertionError("target prefix contains duplicate tokens")
    return target_tokens, target_pages, common


def candidate_pages(
    chunk: Chunk, step: int, page_size: int, candidate_k: int, common: int
) -> list[int]:
    ranked = chunk.ranked[step, :candidate_k]
    ranked = ranked[(ranked >= 0) & (ranked < common)]
    return unique_page_order(ranked, page_size)


def warm_cache(
    chunk: Chunk,
    page_size: int,
    hbm_pages: int,
    dram_pages: int,
) -> TierCache:
    total_tokens = int(chunk.valid_counts.max())
    total_pages = math.ceil(total_tokens / page_size)
    cache = TierCache(total_pages, hbm_pages, dram_pages)
    initial_tokens = int(chunk.valid_counts[0])
    initial_pages = math.ceil(initial_tokens / page_size)
    cache.initialize_recent_dram(initial_pages)
    # The current query KV is created in HBM and requires no transfer.
    cache.create_in_hbm((initial_tokens - 1) // page_size)
    initial_rank = chunk.ranked[0, :TARGET_K]
    initial_rank = initial_rank[(initial_rank >= 0) & (initial_rank < initial_tokens)]
    for page in reversed(unique_page_order(initial_rank, page_size)):
        cache.promote(page)
    cache.check()
    return cache


def run_policy(
    chunk: Chunk,
    page_size: int,
    hbm_fraction: float,
    dram_fraction: float,
    method: str,
    candidate_k: int,
    kv_bytes_per_token_layer: int,
    hbm_capacity_pages: int | None = None,
    dram_capacity_pages: int | None = None,
    extended_metrics: bool = False,
) -> list[dict[str, Any]]:
    total_tokens = int(chunk.valid_counts.max())
    total_pages = math.ceil(total_tokens / page_size)
    hbm_pages = (
        max(1, math.ceil(total_pages * hbm_fraction))
        if hbm_capacity_pages is None
        else max(1, min(total_pages, hbm_capacity_pages))
    )
    dram_pages = (
        min(total_pages, math.ceil(total_pages * dram_fraction))
        if dram_capacity_pages is None
        else max(0, min(total_pages, dram_capacity_pages))
    )
    cache = warm_cache(chunk, page_size, hbm_pages, dram_pages)
    rows: list[dict[str, Any]] = []

    for step in range(31):
        target_tokens, target_order, common = target_for_transition(
            chunk, step, page_size
        )
        target_set = set(target_order)
        score_order = candidate_pages(
            chunk, step, page_size, max(candidate_k, TARGET_K), common
        )
        if method == "lru-demand":
            prefetch_order: list[int] = []
        elif method == "previous-score":
            prefetch_order = score_order[:hbm_pages]
        elif method == "oracle@k3072-pages":
            budget = min(len(score_order), hbm_pages)
            remainder = [page for page in score_order if page not in target_set]
            prefetch_order = (target_order + remainder)[:budget]
        elif method == "oracle-pages":
            # Capacity-matched ceiling: an oracle knows only the next demand
            # set and never transfers a page that the next attention will not
            # use.  This method is descriptive and not deployable.
            prefetch_order = target_order[:hbm_pages]
        else:
            raise ValueError(f"unknown method {method}")

        hbm_before = set(cache.hbm)
        prefetch = TransferCounts()
        prefetched_pages: set[int] = set()
        for page in reversed(prefetch_order):
            if page >= total_pages:
                continue
            source = cache.tier(page)
            if source != "hbm":
                prefetched_pages.add(page)
                prefetch.add(
                    source,
                    page_slots(page, page_size, total_tokens)
                    * kv_bytes_per_token_layer,
                )
            cache.promote(page)

        hbm_ready = set(cache.hbm)
        unused_prefetch_pages = prefetched_pages - target_set
        self_evicted_target_pages = (
            prefetched_pages & target_set
        ) - hbm_ready
        pollution_miss_pages = (hbm_before & target_set) - hbm_ready
        hit_pages = target_set & hbm_ready
        hit_tokens = sum(
            (int(token) // page_size) in hbm_ready for token in target_tokens
        )
        critical = TransferCounts()
        critical_sources: dict[int, str] = {}
        for page in target_order:
            source = cache.tier(page)
            critical_sources[page] = source
            if source != "hbm":
                critical.add(
                    source,
                    page_slots(page, page_size, total_tokens)
                    * kv_bytes_per_token_layer,
                )

        # Materialize exact demand in reverse score order so higher-score pages
        # remain more recent for the next transition.
        for page in reversed(target_order):
            cache.promote(page)

        # The new decode position is generated locally after this attention and
        # is therefore inserted without host/storage traffic.
        next_valid = int(chunk.valid_counts[step + 1])
        if next_valid > common:
            cache.create_in_hbm((next_valid - 1) // page_size)
        cache.check()

        target_bytes = sum(
            page_slots(page, page_size, total_tokens) * kv_bytes_per_token_layer
            for page in target_set
        )
        hbm_bytes = sum(
            page_slots(page, page_size, total_tokens) * kv_bytes_per_token_layer
            for page in hit_pages
        )
        critical_bytes = critical.dram_bytes + critical.ssd_bytes
        prefetch_bytes = prefetch.dram_bytes + prefetch.ssd_bytes
        row = {
                "run_dir": str(chunk.source_file.parents[1]),
                "request_id": chunk.request_id,
                "dataset": chunk.dataset,
                "task": chunk.task,
                "source_index": chunk.source_index,
                "context_config": chunk.context_config,
                "prompt_tokens": chunk.prompt_tokens,
                "layer": chunk.layer,
                "step": step,
                "page_size": page_size,
                "hbm_fraction": hbm_fraction,
                "dram_fraction": dram_fraction,
                "method": method,
                "candidate_k": candidate_k,
                "common_tokens": common,
                "total_pages": total_pages,
                "hbm_capacity_pages": hbm_pages,
                "dram_capacity_pages": dram_pages,
                "target_tokens": len(target_tokens),
                "target_pages": len(target_set),
                "target_page_bytes": target_bytes,
                "hbm_hit_tokens": hit_tokens,
                "hbm_hit_pages": len(hit_pages),
                "hbm_hit_page_bytes": hbm_bytes,
                "capacity_feasible": hbm_pages >= len(target_set),
                "candidate_unique_pages": len(score_order),
                "prefetch_admit_pages": len(prefetched_pages),
                "useful_prefetch_pages": len(prefetched_pages & target_set),
                "prefetch_dram_pages": prefetch.dram_pages,
                "prefetch_dram_bytes": prefetch.dram_bytes,
                "prefetch_ssd_pages": prefetch.ssd_pages,
                "prefetch_ssd_bytes": prefetch.ssd_bytes,
                "critical_dram_pages": critical.dram_pages,
                "critical_dram_bytes": critical.dram_bytes,
                "critical_ssd_pages": critical.ssd_pages,
                "critical_ssd_bytes": critical.ssd_bytes,
                "critical_miss_pages": critical.dram_pages + critical.ssd_pages,
                "critical_miss_bytes": critical_bytes,
                "total_pcie_bytes": prefetch_bytes + critical_bytes,
                "total_ssd_bytes": prefetch.ssd_bytes + critical.ssd_bytes,
                "hbm_pages_before_prefetch": len(hbm_before),
                "source_file": chunk.source_file.name,
        }
        if extended_metrics:
            row.update(
                {
                    "unused_prefetch_pages": len(unused_prefetch_pages),
                    "unused_prefetch_bytes": pages_bytes(
                        unused_prefetch_pages,
                        page_size,
                        total_tokens,
                        kv_bytes_per_token_layer,
                    ),
                    "prefetch_self_evicted_target_pages": len(
                        self_evicted_target_pages
                    ),
                    "prefetch_self_evicted_target_bytes": pages_bytes(
                        self_evicted_target_pages,
                        page_size,
                        total_tokens,
                        kv_bytes_per_token_layer,
                    ),
                    "pollution_miss_pages": len(pollution_miss_pages),
                    "pollution_miss_bytes": pages_bytes(
                        pollution_miss_pages,
                        page_size,
                        total_tokens,
                        kv_bytes_per_token_layer,
                    ),
                }
            )
        rows.append(row)
    return rows


def simulate(
    run_dirs: list[Path],
    layers: list[int],
    page_sizes: list[int],
    candidate_ks: list[int],
    hbm_fractions: list[float],
    dram_fractions: list[float],
    kv_bytes_per_token_layer: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validation_runs: list[dict[str, Any]] = []
    chunk_count = 0
    for run_dir in run_dirs:
        chunks = load_run(
            run_dir,
            layers,
            max(DEFAULT_REQUIRED_COMPACT_K, max(candidate_ks, default=TARGET_K)),
        )
        chunk_count += len(chunks)
        for chunk in chunks:
            for page_size in page_sizes:
                for hbm_fraction in hbm_fractions:
                    for dram_fraction in dram_fractions:
                        rows.extend(
                            run_policy(
                                chunk,
                                page_size,
                                hbm_fraction,
                                dram_fraction,
                                "lru-demand",
                                0,
                                kv_bytes_per_token_layer,
                            )
                        )
                        for candidate_k in candidate_ks:
                            rows.extend(
                                run_policy(
                                    chunk,
                                    page_size,
                                    hbm_fraction,
                                    dram_fraction,
                                    "previous-score",
                                    candidate_k,
                                    kv_bytes_per_token_layer,
                                )
                            )
                        rows.extend(
                            run_policy(
                                chunk,
                                page_size,
                                hbm_fraction,
                                dram_fraction,
                                "oracle@k3072-pages",
                                3072,
                                kv_bytes_per_token_layer,
                            )
                        )
        for request_id in sorted({chunk.request_id for chunk in chunks}):
            request_chunks = [
                chunk for chunk in chunks if chunk.request_id == request_id
            ]
            validation_runs.append(
                {
                    "run_dir": str(run_dir),
                    "context_config": request_chunks[0].context_config,
                    "request_id": request_id,
                    "task": request_chunks[0].task,
                    "source_index": request_chunks[0].source_index,
                    "prompt_tokens": request_chunks[0].prompt_tokens,
                    "layers": [chunk.layer for chunk in request_chunks],
                    "source_files": [
                        str(chunk.source_file) for chunk in request_chunks
                    ],
                }
            )
    detail = pd.DataFrame(rows)
    expected_methods = 2 + len(candidate_ks)
    expected_rows = (
        chunk_count
        * len(page_sizes)
        * len(hbm_fractions)
        * len(dram_fractions)
        * expected_methods
        * 31
    )
    if len(detail) != expected_rows:
        raise AssertionError(f"row mismatch: {len(detail)} != {expected_rows}")
    if not np.all(
        detail.hbm_hit_pages + detail.critical_miss_pages == detail.target_pages
    ):
        raise AssertionError("target page accounting mismatch")
    if not np.all(detail.hbm_hit_tokens <= detail.target_tokens):
        raise AssertionError("target token accounting mismatch")
    validation = {
        "runs": validation_runs,
        "expected_rows": expected_rows,
        "actual_rows": len(detail),
        "contexts": sorted(int(value) for value in detail.context_config.unique()),
        "layers": layers,
        "transitions_per_layer": 31,
        "new_decode_position_excluded": True,
        "exact_topk_prefix_checked": True,
        "page_accounting_checked": True,
        "exclusive_tier_invariant_checked_each_transition": True,
    }
    return detail, validation


def aggregate_counts(detail: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "context_config",
        "prompt_tokens",
        "page_size",
        "hbm_fraction",
        "dram_fraction",
        "method",
        "candidate_k",
    ]
    count_columns = [
        "target_tokens",
        "target_pages",
        "target_page_bytes",
        "hbm_hit_tokens",
        "hbm_hit_pages",
        "hbm_hit_page_bytes",
        "candidate_unique_pages",
        "prefetch_admit_pages",
        "useful_prefetch_pages",
        "prefetch_dram_pages",
        "prefetch_dram_bytes",
        "prefetch_ssd_pages",
        "prefetch_ssd_bytes",
        "critical_dram_pages",
        "critical_dram_bytes",
        "critical_ssd_pages",
        "critical_ssd_bytes",
        "critical_miss_pages",
        "critical_miss_bytes",
        "total_pcie_bytes",
        "total_ssd_bytes",
    ]
    result = detail.groupby(groups, as_index=False)[count_columns].sum()
    means = (
        detail.groupby(groups, as_index=False)[
            [
                "total_pages",
                "hbm_capacity_pages",
                "dram_capacity_pages",
                "capacity_feasible",
            ]
        ]
        .mean()
        .rename(columns={"capacity_feasible": "capacity_feasible_rate"})
    )
    result = result.merge(means, on=groups, validate="1:1")
    result["hbm_token_recall"] = result.hbm_hit_tokens / result.target_tokens
    result["hbm_page_recall"] = result.hbm_hit_pages / result.target_pages
    result["hbm_byte_recall"] = (
        result.hbm_hit_page_bytes / result.target_page_bytes
    )
    result["prefetch_precision"] = (
        result.useful_prefetch_pages
        / result.prefetch_admit_pages.replace(0, np.nan)
    )
    result["critical_dram_share"] = (
        result.critical_dram_bytes
        / result.critical_miss_bytes.replace(0, np.nan)
    )
    result["critical_ssd_share"] = (
        result.critical_ssd_bytes
        / result.critical_miss_bytes.replace(0, np.nan)
    )
    result["transfer_read_amplification"] = (
        result.total_pcie_bytes / result.target_page_bytes
    )
    result["ssd_read_amplification"] = (
        result.total_ssd_bytes / result.target_page_bytes
    )
    baseline = result[result.method == "lru-demand"][[
        "context_config",
        "prompt_tokens",
        "page_size",
        "hbm_fraction",
        "dram_fraction",
        "critical_miss_bytes",
    ]].rename(columns={"critical_miss_bytes": "lru_critical_miss_bytes"})
    result = result.merge(
        baseline,
        on=[
            "context_config",
            "prompt_tokens",
            "page_size",
            "hbm_fraction",
            "dram_fraction",
        ],
        validate="many_to_one",
    )
    result["critical_miss_byte_reduction_vs_lru"] = 1 - (
        result.critical_miss_bytes
        / result.lru_critical_miss_bytes.replace(0, np.nan)
    )
    return result.sort_values(groups)


def transfer_ms(
    pcie_bytes: pd.Series,
    ssd_bytes: pd.Series,
    tp: int,
    pcie_gbps: float,
    ssd_gbps: float,
    pcie_latency_us: float,
    ssd_latency_us: float,
) -> pd.Series:
    per_gpu_pcie = pcie_bytes / tp
    per_gpu_ssd = ssd_bytes / tp
    result = per_gpu_pcie / (pcie_gbps * 1e9) * 1e3
    result += per_gpu_ssd / (ssd_gbps * 1e9) * 1e3
    result += (per_gpu_pcie > 0) * (pcie_latency_us / 1e3)
    result += (per_gpu_ssd > 0) * (ssd_latency_us / 1e3)
    return result


def aggregate_time_model(
    detail: pd.DataFrame,
    num_model_layers: int,
    sampled_layer_count: int,
    tp: int,
    pcie_gbps: float,
    ssd_gbps: float,
    pcie_latency_us: float,
    ssd_latency_us: float,
    overlap_windows_ms: list[float],
    kv_bytes_per_token_layer: int,
) -> pd.DataFrame:
    groups = [
        "context_config",
        "prompt_tokens",
        "page_size",
        "hbm_fraction",
        "dram_fraction",
        "method",
        "candidate_k",
        "step",
    ]
    bytes_columns = [
        "prefetch_dram_bytes",
        "prefetch_ssd_bytes",
        "critical_dram_bytes",
        "critical_ssd_bytes",
        "target_page_bytes",
    ]
    per_step = detail.groupby(groups, as_index=False)[bytes_columns].sum()
    scale = num_model_layers / sampled_layer_count
    for column in bytes_columns:
        per_step[column] *= scale
    prefetch_pcie = per_step.prefetch_dram_bytes + per_step.prefetch_ssd_bytes
    critical_pcie = per_step.critical_dram_bytes + per_step.critical_ssd_bytes
    per_step["prefetch_required_overlap_ms"] = transfer_ms(
        prefetch_pcie,
        per_step.prefetch_ssd_bytes,
        tp,
        pcie_gbps,
        ssd_gbps,
        pcie_latency_us,
        ssd_latency_us,
    )
    per_step["critical_correction_ms"] = transfer_ms(
        critical_pcie,
        per_step.critical_ssd_bytes,
        tp,
        pcie_gbps,
        ssd_gbps,
        pcie_latency_us,
        ssd_latency_us,
    )
    for window in overlap_windows_ms:
        label = f"estimated_stall_at_{window:g}ms_window_ms"
        per_step[label] = per_step.critical_correction_ms + np.maximum(
            0.0, per_step.prefetch_required_overlap_ms - window
        )

    summary_groups = groups[:-1]
    metric_columns = [
        "prefetch_required_overlap_ms",
        "critical_correction_ms",
        *[
            f"estimated_stall_at_{window:g}ms_window_ms"
            for window in overlap_windows_ms
        ],
    ]
    rows: list[dict[str, Any]] = []
    for key, part in per_step.groupby(summary_groups, sort=True):
        row = dict(zip(summary_groups, key, strict=True))
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_p95"] = float(part[metric].quantile(0.95))
        total_tokens = int(part.prompt_tokens.iloc[0]) + 31
        logical_full_bytes = (
            total_tokens * kv_bytes_per_token_layer * num_model_layers
        )
        row["full_kv_logical_gib"] = logical_full_bytes / 2**30
        row["full_kv_per_gpu_gib"] = logical_full_bytes / tp / 2**30
        row["hbm_budget_logical_gib"] = (
            logical_full_bytes * float(row["hbm_fraction"]) / 2**30
        )
        row["hbm_budget_per_gpu_gib"] = (
            logical_full_bytes * float(row["hbm_fraction"]) / tp / 2**30
        )
        row["dram_budget_logical_gib"] = (
            logical_full_bytes * float(row["dram_fraction"]) / 2**30
        )
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(summary_groups)
    baseline_keys = [
        "context_config",
        "prompt_tokens",
        "page_size",
        "hbm_fraction",
        "dram_fraction",
    ]
    baseline_metrics = [
        "critical_correction_ms_mean",
        *[
            f"estimated_stall_at_{window:g}ms_window_ms_mean"
            for window in overlap_windows_ms
        ],
    ]
    baseline = result[result.method == "lru-demand"][
        baseline_keys + baseline_metrics
    ].rename(columns={metric: f"lru_{metric}" for metric in baseline_metrics})
    result = result.merge(baseline, on=baseline_keys, validate="many_to_one")
    for metric in baseline_metrics:
        result[f"{metric}_reduction_vs_lru"] = 1 - (
            result[metric] / result[f"lru_{metric}"].replace(0, np.nan)
        )
    return result.sort_values(summary_groups)


def configure_style(style: Path | None) -> None:
    # The shared style's categorical cycler contains hexadecimal colors, which
    # some matplotlib releases parse as comments in ``.mplstyle`` files.  The
    # figures below set method colors explicitly, so mirror the remaining
    # shared defaults locally when that bundled file is selected.
    if style and style.exists() and style.resolve() != DEFAULT_STYLE.resolve():
        plt.style.use(style)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fixed_capacity_slice(
    summary: pd.DataFrame,
    time_summary: pd.DataFrame,
    hbm_targets_gib: list[float] = FIXED_HBM_LOGICAL_GIB,
    dram_target_gib: float = FIXED_DRAM_LOGICAL_GIB,
) -> pd.DataFrame:
    """Select the closest common absolute capacities from the fraction grid."""
    selected: list[pd.DataFrame] = []
    for context, context_rows in time_summary.groupby("context_config"):
        capacity = context_rows[
            [
                "hbm_fraction",
                "dram_fraction",
                "hbm_budget_logical_gib",
                "hbm_budget_per_gpu_gib",
                "dram_budget_logical_gib",
            ]
        ].drop_duplicates()
        hbm_options = capacity[
            ["hbm_fraction", "hbm_budget_logical_gib", "hbm_budget_per_gpu_gib"]
        ].drop_duplicates()
        dram_options = capacity[
            ["dram_fraction", "dram_budget_logical_gib"]
        ].drop_duplicates()
        dram_choice = dram_options.iloc[
            (dram_options.dram_budget_logical_gib - dram_target_gib).abs().argmin()
        ]
        for hbm_target in hbm_targets_gib:
            hbm_choice = hbm_options.iloc[
                (hbm_options.hbm_budget_logical_gib - hbm_target).abs().argmin()
            ]
            part = context_rows[
                np.isclose(context_rows.hbm_fraction, hbm_choice.hbm_fraction)
                & np.isclose(context_rows.dram_fraction, dram_choice.dram_fraction)
            ].copy()
            part["target_hbm_logical_gib"] = hbm_target
            part["target_dram_logical_gib"] = dram_target_gib
            selected.append(part)
    fixed = pd.concat(selected, ignore_index=True)
    observed_columns = [
        "context_config",
        "prompt_tokens",
        "page_size",
        "hbm_fraction",
        "dram_fraction",
        "method",
        "candidate_k",
        "hbm_token_recall",
        "hbm_page_recall",
        "critical_miss_byte_reduction_vs_lru",
        "transfer_read_amplification",
        "ssd_read_amplification",
        "prefetch_precision",
        "capacity_feasible_rate",
    ]
    fixed = fixed.merge(
        summary[observed_columns],
        on=[
            "context_config",
            "prompt_tokens",
            "page_size",
            "hbm_fraction",
            "dram_fraction",
            "method",
            "candidate_k",
        ],
        validate="1:1",
    )
    fixed["hbm_capacity_relative_error"] = (
        fixed.hbm_budget_logical_gib - fixed.target_hbm_logical_gib
    ) / fixed.target_hbm_logical_gib
    fixed["dram_capacity_relative_error"] = (
        fixed.dram_budget_logical_gib - fixed.target_dram_logical_gib
    ) / fixed.target_dram_logical_gib
    if fixed.hbm_capacity_relative_error.abs().max() > 0.02:
        raise AssertionError("fraction grid is not close enough to fixed HBM target")
    if fixed.dram_capacity_relative_error.abs().max() > 0.02:
        raise AssertionError("fraction grid is not close enough to fixed DRAM target")
    return fixed.sort_values(
        [
            "target_hbm_logical_gib",
            "context_config",
            "page_size",
            "method",
            "candidate_k",
        ]
    )


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_tradeoff(summary: pd.DataFrame, figure_dir: Path) -> None:
    data = summary[
        (summary.method == "previous-score")
        & np.isclose(summary.hbm_fraction, 0.20)
        & np.isclose(summary.dram_fraction, 0.25)
    ]
    contexts = sorted(data.context_config.unique())
    fig, axes = plt.subplots(
        1, len(contexts), figsize=(6.8, 2.55), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    colors = {4: "#0072B2", 16: "#D55E00", 64: "#009E73"}
    markers = {4: "o", 16: "s", 64: "^"}
    for axis, context in zip(axes, contexts, strict=True):
        part = data[data.context_config == context]
        for page_size in sorted(part.page_size.unique()):
            line = part[part.page_size == page_size].sort_values("candidate_k")
            axis.plot(
                line.candidate_k,
                line.hbm_token_recall * 100,
                color=colors[int(page_size)],
                marker=markers[int(page_size)],
                label=f"{int(page_size)} tokens/page",
            )
        axis.set_title(f"{int(context // 1024)}K context")
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axes[0].set_ylabel("Top-2048 tokens in HBM (%)\nhigher is better")
    fig.supxlabel("Previous-score candidate K", y=0.01)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.21, wspace=0.12)
    save_figure(fig, figure_dir / "candidate_k_page_tradeoff")


def plot_capacity_heatmap(time_summary: pd.DataFrame, figure_dir: Path) -> None:
    metric = "estimated_stall_at_5ms_window_ms_mean_reduction_vs_lru"
    data = time_summary[
        (time_summary.method == "previous-score")
        & (time_summary.candidate_k == 3072)
        & (time_summary.page_size == 4)
    ]
    contexts = sorted(data.context_config.unique())
    fig, axes = plt.subplots(
        1, len(contexts), figsize=(6.8, 2.75), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    images = []
    values = data[metric] * 100
    lower = min(-1.0, float(values.min()))
    upper = max(1.0, float(values.max()))
    norm = TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
    for axis, context in zip(axes, contexts, strict=True):
        part = data[data.context_config == context]
        matrix = part.pivot(
            index="hbm_fraction",
            columns="dram_fraction",
            values=metric,
        ).sort_index().sort_index(axis=1)
        image = axis.imshow(
            matrix.to_numpy() * 100,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            norm=norm,
        )
        images.append(image)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column] * 100
                axis.text(
                    column,
                    row,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        axis.set_title(f"{int(context // 1024)}K context")
        axis.set_xticks(
            range(matrix.shape[1]),
            [f"{100 * value:g}%" for value in matrix.columns],
        )
        axis.set_yticks(
            range(matrix.shape[0]),
            [f"{100 * value:g}%" for value in matrix.index],
        )
    axes[0].set_ylabel("HBM capacity / full KV")
    fig.supxlabel("DRAM capacity / full KV", y=0.01)
    colorbar = fig.colorbar(images[-1], ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("Modeled stall reduced vs. LRU (%)\n5 ms overlap window")
    fig.subplots_adjust(bottom=0.21, wspace=0.12, right=0.86)
    save_figure(fig, figure_dir / "k3072_p4_capacity_heatmap")


def plot_page_time_tradeoff(
    summary: pd.DataFrame, time_summary: pd.DataFrame, figure_dir: Path
) -> None:
    filters = (
        (summary.method == "previous-score")
        & (summary.candidate_k == 3072)
        & np.isclose(summary.hbm_fraction, 0.20)
        & np.isclose(summary.dram_fraction, 0.25)
    )
    data = summary[filters].merge(
        time_summary[
            (time_summary.method == "previous-score")
            & (time_summary.candidate_k == 3072)
            & np.isclose(time_summary.hbm_fraction, 0.20)
            & np.isclose(time_summary.dram_fraction, 0.25)
        ],
        on=[
            "context_config",
            "prompt_tokens",
            "page_size",
            "hbm_fraction",
            "dram_fraction",
            "method",
            "candidate_k",
        ],
        validate="1:1",
        suffixes=("", "_time"),
    )
    contexts = sorted(data.context_config.unique())
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75))
    colors = {32768: "#0072B2", 65536: "#D55E00", 131072: "#009E73"}
    markers = {32768: "o", 65536: "s", 131072: "^"}
    for context in contexts:
        part = data[data.context_config == context].sort_values("page_size")
        label = f"{int(context // 1024)}K"
        axes[0].plot(
            part.page_size,
            part.transfer_read_amplification,
            color=colors[int(context)],
            marker=markers[int(context)],
            label=label,
        )
        axes[1].plot(
            part.page_size,
            part.critical_correction_ms_mean,
            color=colors[int(context)],
            marker=markers[int(context)],
            label=label,
        )
    axes[0].set_xlabel("KV page size (tokens)")
    axes[0].set_ylabel("PCIe bytes / target-page bytes (x)\nlower is better")
    axes[1].set_xlabel("KV page size (tokens)")
    axes[1].set_ylabel("Estimated correction time (ms)\nlower is better")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks([4, 16, 64], ["4", "16", "64"])
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.80, wspace=0.38)
    save_figure(fig, figure_dir / "k3072_page_time_tradeoff")


def plot_fixed_capacity_deadline(fixed: pd.DataFrame, figure_dir: Path) -> None:
    data = fixed[
        np.isclose(fixed.target_hbm_logical_gib, 1.2)
        & np.isclose(fixed.target_dram_logical_gib, 3.0)
        & (fixed.page_size == 4)
        & (fixed.method == "previous-score")
        & fixed.candidate_k.isin([2560, 3072, 4096])
    ]
    contexts = sorted(data.context_config.unique())
    windows = [5.0, 10.0, 20.0]
    colors = {5.0: "#D55E00", 10.0: "#0072B2", 20.0: "#009E73"}
    markers = {5.0: "s", 10.0: "o", 20.0: "^"}
    fig, axes = plt.subplots(
        1, len(contexts), figsize=(6.8, 2.65), sharex=True, sharey=False
    )
    axes = np.atleast_1d(axes)
    for axis, context in zip(axes, contexts, strict=True):
        part = data[data.context_config == context].sort_values("candidate_k")
        for window in windows:
            metric = f"estimated_stall_at_{window:g}ms_window_ms_mean"
            axis.plot(
                part.candidate_k,
                part[metric],
                color=colors[window],
                marker=markers[window],
                label=f"{window:g} ms window",
            )
        baseline = float(part.lru_critical_correction_ms_mean.iloc[0])
        axis.axhline(
            baseline,
            color="#4D4D4D",
            linestyle="--",
            linewidth=1.4,
            label="Demand-only LRU",
        )
        axis.set_title(f"{int(context // 1024)}K context")
        axis.set_xticks([2560, 3072, 4096], ["2560", "3072", "4096"])
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axes[0].set_ylabel("Modeled stall per token (ms)\nlower is better")
    fig.supxlabel("Previous-score candidate K", y=0.01)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.21, wspace=0.28)
    save_figure(fig, figure_dir / "fixed_capacity_k_deadline_tradeoff")


def self_test() -> None:
    cache = TierCache(total_pages=8, hbm_capacity_pages=2, dram_capacity_pages=3)
    cache.initialize_recent_dram(8)
    assert list(cache.dram) == [5, 6, 7]
    assert cache.promote(6) == "dram"
    assert list(cache.hbm) == [6]
    assert cache.promote(1) == "ssd"
    assert list(cache.hbm) == [6, 1]
    assert cache.promote(2) == "ssd"
    assert list(cache.hbm) == [1, 2]
    assert 6 in cache.dram
    cache.create_in_hbm(3)
    cache.check()
    assert unique_page_order([0, 1, 4, 5, 2], 2) == [0, 2, 1]
    assert page_slots(2, 4, 10) == 2
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layers", type=parse_int_list, default=DEFAULT_LAYERS)
    parser.add_argument(
        "--page-sizes", type=parse_int_list, default=DEFAULT_PAGE_SIZES
    )
    parser.add_argument(
        "--candidate-k", type=parse_int_list, default=DEFAULT_CANDIDATE_K
    )
    parser.add_argument(
        "--hbm-fractions", type=parse_float_list, default=DEFAULT_HBM_FRACTIONS
    )
    parser.add_argument(
        "--dram-fractions", type=parse_float_list, default=DEFAULT_DRAM_FRACTIONS
    )
    parser.add_argument(
        "--overlap-windows-ms",
        type=parse_float_list,
        default=DEFAULT_OVERLAP_WINDOWS_MS,
    )
    parser.add_argument("--num-model-layers", type=int, default=48)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--kv-bytes-per-token-layer", type=int, default=2048)
    parser.add_argument("--pcie-gbps", type=float, default=25.0)
    parser.add_argument("--ssd-gbps", type=float, default=7.0)
    parser.add_argument("--pcie-latency-us", type=float, default=10.0)
    parser.add_argument("--ssd-latency-us", type=float, default=100.0)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.run_dirs or args.output_dir is None:
        raise SystemExit("--run-dirs and --output-dir are required")
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        detail = pd.read_parquet(table_dir / "multitier_by_transition.parquet")
        summary = pd.read_csv(table_dir / "multitier_summary.csv")
        time_summary = aggregate_time_model(
            detail,
            args.num_model_layers,
            len(args.layers),
            args.tp,
            args.pcie_gbps,
            args.ssd_gbps,
            args.pcie_latency_us,
            args.ssd_latency_us,
            args.overlap_windows_ms,
            args.kv_bytes_per_token_layer,
        )
        time_summary.to_csv(table_dir / "time_model_summary.csv", index=False)
        validation = json.loads((output_dir / "validation.json").read_text())
    else:
        detail, validation = simulate(
            [path.resolve() for path in args.run_dirs],
            args.layers,
            args.page_sizes,
            args.candidate_k,
            args.hbm_fractions,
            args.dram_fractions,
            args.kv_bytes_per_token_layer,
        )
        summary = aggregate_counts(detail)
        time_summary = aggregate_time_model(
            detail,
            args.num_model_layers,
            len(args.layers),
            args.tp,
            args.pcie_gbps,
            args.ssd_gbps,
            args.pcie_latency_us,
            args.ssd_latency_us,
            args.overlap_windows_ms,
            args.kv_bytes_per_token_layer,
        )
        detail.to_parquet(table_dir / "multitier_by_transition.parquet", index=False)
        summary.to_csv(table_dir / "multitier_summary.csv", index=False)
        time_summary.to_csv(table_dir / "time_model_summary.csv", index=False)
        (output_dir / "validation.json").write_text(
            json.dumps(validation, indent=2, ensure_ascii=False) + "\n"
        )

    fixed = fixed_capacity_slice(summary, time_summary)
    fixed.to_csv(table_dir / "fixed_capacity_summary.csv", index=False)
    configure_style(args.style)
    plot_candidate_tradeoff(summary, figure_dir)
    plot_capacity_heatmap(time_summary, figure_dir)
    plot_page_time_tradeoff(summary, time_summary, figure_dir)
    plot_fixed_capacity_deadline(fixed, figure_dir)

    main_filter = (
        (summary.method == "previous-score")
        & (summary.candidate_k == 3072)
        & (summary.page_size == 4)
        & np.isclose(summary.hbm_fraction, 0.20)
        & np.isclose(summary.dram_fraction, 0.25)
    )
    main_rows = summary[main_filter].sort_values("context_config")
    main_times = time_summary[
        (time_summary.method == "previous-score")
        & (time_summary.candidate_k == 3072)
        & (time_summary.page_size == 4)
        & np.isclose(time_summary.hbm_fraction, 0.20)
        & np.isclose(time_summary.dram_fraction, 0.25)
    ].sort_values("context_config")
    result = {
        "schema_version": 1,
        "analysis_kind": "trace-driven descriptive simulator; not measured speed",
        "trace_evidence": {
            "contexts": validation["contexts"],
            "requests": len(validation["runs"]),
            "sampled_layers": args.layers,
            "decode_transitions_per_layer": 31,
            "topk_backend": "torch_exact",
        },
        "cache_model": {
            "tiers": "exclusive HBM/DRAM/SSD with LRU replacement",
            "initial_dram_placement": "most recent pages",
            "new_decode_kv": "inserted into HBM without transfer cost",
            "correction": "all predicted misses are demand-loaded; exact semantics retained",
            "candidate_signal": "same-layer previous-step exact score ranking",
            "oracle": "target pages first at K3072 predictor page budget",
        },
        "time_model": {
            "not_measured": True,
            "num_model_layers": args.num_model_layers,
            "sampled_layer_scale": args.num_model_layers / len(args.layers),
            "tensor_parallel_size": args.tp,
            "logical_kv_bytes_per_token_layer": args.kv_bytes_per_token_layer,
            "per_gpu_kv_bytes_per_token_layer": (
                args.kv_bytes_per_token_layer / args.tp
            ),
            "pcie_effective_gbps": args.pcie_gbps,
            "ssd_effective_gbps": args.ssd_gbps,
            "pcie_batch_latency_us": args.pcie_latency_us,
            "ssd_batch_latency_us": args.ssd_latency_us,
            "transfer_composition": "serial SSD read plus PCIe copy; one optimistic batch per source per step",
            "overlap_windows_ms": args.overlap_windows_ms,
        },
        "main_setting": {
            "method": "previous-score K3072",
            "page_size_tokens": 4,
            "hbm_fraction_of_full_kv": 0.20,
            "dram_fraction_of_full_kv": 0.25,
            "observed": main_rows.to_dict("records"),
            "modeled_time": main_times.to_dict("records"),
        },
        "fixed_capacity_slice": {
            "target_hbm_logical_gib": FIXED_HBM_LOGICAL_GIB,
            "target_dram_logical_gib": FIXED_DRAM_LOGICAL_GIB,
            "maximum_capacity_relative_error": float(
                max(
                    fixed.hbm_capacity_relative_error.abs().max(),
                    fixed.dram_capacity_relative_error.abs().max(),
                )
            ),
            "primary_rows": fixed[
                np.isclose(fixed.target_hbm_logical_gib, 1.2)
                & (fixed.page_size == 4)
                & (fixed.method == "previous-score")
                & fixed.candidate_k.isin([2560, 3072, 4096])
            ].to_dict("records"),
        },
        "limitations": [
            "one RULER niah_single_1 request per context length",
            "seven sampled layers are scaled to 48 layers",
            "no concurrency, DMA contention, kernel overlap, NUMA, or SSD random-I/O measurement",
            "bandwidth-derived time is a parameterized estimate and cannot support a speedup claim",
            "same-layer previous-step prediction is evaluated here; cross-layer h^l lookahead quality is a separate trace",
        ],
        "artifacts": {},
    }
    artifact_paths = [
        table_dir / "multitier_by_transition.parquet",
        table_dir / "multitier_summary.csv",
        table_dir / "time_model_summary.csv",
        table_dir / "fixed_capacity_summary.csv",
        output_dir / "validation.json",
        figure_dir / "candidate_k_page_tradeoff.pdf",
        figure_dir / "candidate_k_page_tradeoff.png",
        figure_dir / "k3072_p4_capacity_heatmap.pdf",
        figure_dir / "k3072_p4_capacity_heatmap.png",
        figure_dir / "k3072_page_time_tradeoff.pdf",
        figure_dir / "k3072_page_time_tradeoff.png",
        figure_dir / "fixed_capacity_k_deadline_tradeoff.pdf",
        figure_dir / "fixed_capacity_k_deadline_tradeoff.png",
    ]
    for path in artifact_paths:
        result["artifacts"][str(path.relative_to(output_dir))] = sha256(path)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    reproducibility = {
        "script": str(Path(__file__).resolve()),
        "git_commit": git_revision(Path.cwd()),
        "inputs": [str(path.resolve()) for path in args.run_dirs],
        "output_dir": str(output_dir),
        "parameters": {
            "layers": args.layers,
            "page_sizes": args.page_sizes,
            "candidate_k": args.candidate_k,
            "hbm_fractions": args.hbm_fractions,
            "dram_fractions": args.dram_fractions,
            "num_model_layers": args.num_model_layers,
            "tp": args.tp,
            "kv_bytes_per_token_layer": args.kv_bytes_per_token_layer,
            "pcie_gbps": args.pcie_gbps,
            "ssd_gbps": args.ssd_gbps,
            "pcie_latency_us": args.pcie_latency_us,
            "ssd_latency_us": args.ssd_latency_us,
            "overlap_windows_ms": args.overlap_windows_ms,
            "style": str(args.style),
        },
    }
    (output_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result["main_setting"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
