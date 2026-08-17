#!/usr/bin/env python3
"""Gate page co-activation as an increment over previous-score prediction.

The predictor is online and leakage-free: when predicting step t+1 it may use
the current page set, earlier transitions in the same request, and earlier
rounds of the same trajectory.  A small history-neighbor model represents the
co-activation signal without building a quadratic page graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

TARGET_K = 2048
CANDIDATE_K = 3072
COMPACT_K = 4096
DEFAULT_LAYERS = [0, 7, 15, 23, 31, 39, 47]
DEFAULT_PAGE_SIZES = [4, 16, 64]
BETAS = [0.05, 0.1, 0.25, 0.5, 1.0]
KNN = 4
HISTORY_LIMIT = 64
MIN_HISTORY = 4
KV_BYTES_PER_TOKEN_LAYER = 2048
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_SAMPLES = 2000


@dataclass
class Chunk:
    topk: np.ndarray
    ranked: np.ndarray
    valid_counts: np.ndarray
    topk_source: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated ints")
    return values


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def set_mask(values: Iterable[int]) -> int:
    mask = 0
    for value in values:
        mask |= 1 << int(value)
    return mask


def jaccard_mask(left: int, right: int) -> float:
    union = (left | right).bit_count()
    return (left & right).bit_count() / union if union else 1.0


def unique_page_order(tokens: np.ndarray, page_size: int) -> list[int]:
    seen: set[int] = set()
    pages: list[int] = []
    for token in tokens:
        page = int(token) // page_size
        if page in seen:
            continue
        seen.add(page)
        pages.append(page)
    return pages


def exact_rank_prefix(scores: np.ndarray, width: int) -> np.ndarray:
    width = min(width, len(scores))
    if width == len(scores):
        return np.argsort(-scores, kind="stable")
    candidate = np.argpartition(-scores, width - 1)[:width]
    return candidate[np.argsort(-scores[candidate], kind="stable")]


def load_chunk(path: Path) -> Chunk:
    record = torch.load(path, weights_only=False)
    schema = int(record["schema_version"])
    if schema not in {4, 5}:
        raise ValueError(f"{path}: expected schema v4/v5")
    if record["decode_step_ids"] != list(range(32)):
        raise ValueError(f"{path}: expected decode steps 0..31")
    valid = record["score_valid_counts"].numpy(force=True).astype(np.int64)
    if schema == 5:
        if record.get("topk_backend") != "torch_exact":
            raise ValueError(f"{path}: schema v5 must use torch_exact")
        topk = record["indices"].numpy(force=True).astype(np.int64)
        ranked = record["candidate_indices"].numpy(force=True).astype(np.int64)
        source = "trace_torch_exact"
    else:
        scores = record["scores"].numpy(force=True).astype(np.float32)
        ranked = np.full((32, COMPACT_K), -1, dtype=np.int64)
        for step, count in enumerate(valid):
            current = scores[step, : int(count)]
            if not np.isfinite(current).all():
                raise ValueError(f"{path}: non-finite score at step {step}")
            prefix = exact_rank_prefix(current, COMPACT_K)
            ranked[step, : len(prefix)] = prefix
        topk = ranked[:, :TARGET_K].copy()
        source = "reconstructed_torch_exact_from_full_scores"
    if ranked.shape[1] < COMPACT_K:
        raise ValueError(f"{path}: fewer than {COMPACT_K} ranked candidates")
    for step in range(32):
        if len(np.unique(topk[step])) != TARGET_K:
            raise ValueError(f"{path}: duplicate top-k token at step {step}")
        if set(topk[step]) != set(ranked[step, :TARGET_K]):
            raise ValueError(f"{path}: top-k prefix mismatch at step {step}")
    return Chunk(topk=topk, ranked=ranked, valid_counts=valid, topk_source=source)


def request_metadata(row: dict[str, Any]) -> tuple[str, str, int, int]:
    trajectory = str(row.get("trajectory_id", row["rid"]))
    category = str(row.get("category", row.get("task", "unknown")))
    round_id = int(row.get("round_id", 0))
    context = int(row.get("length_config", row.get("prompt_len", 0)))
    return trajectory, category, round_id, context


def build_split_map(requests: list[dict[str, Any]]) -> dict[str, str]:
    by_category: defaultdict[str, set[str]] = defaultdict(set)
    for row in requests:
        trajectory, category, _, _ = request_metadata(row)
        by_category[category].add(trajectory)
    result: dict[str, str] = {}
    for trajectories in by_category.values():
        ordered = sorted(trajectories)
        midpoint = (len(ordered) + 1) // 2
        for index, trajectory in enumerate(ordered):
            result[trajectory] = "calibration" if index < midpoint else "test"
    if len(result) == 1:
        result[next(iter(result))] = "pilot"
    return result


def rank_by_counter(
    counter: Counter[int], total_pages: int
) -> list[int]:
    return sorted(
        (page for page in counter if 0 <= page < total_pages),
        key=lambda page: (-counter[page], page),
    )


def fill_to_budget(
    preferred: Iterable[int], fallback: Iterable[int], budget: int
) -> set[int] | None:
    selected: set[int] = set()
    for source in (preferred, fallback):
        for page in source:
            selected.add(int(page))
            if len(selected) == budget:
                return selected
    return None


def association_scores(
    current_mask: int,
    history: list[tuple[int, tuple[int, ...]]],
) -> dict[int, float]:
    scored = [
        (jaccard_mask(current_mask, source), order, target)
        for order, (source, target) in enumerate(history[-HISTORY_LIMIT:])
    ]
    neighbors = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[
        :KNN
    ]
    denominator = sum(similarity for similarity, _, _ in neighbors)
    if denominator <= 0:
        return {}
    scores: defaultdict[int, float] = defaultdict(float)
    for similarity, _, target in neighbors:
        for page in target:
            scores[page] += similarity / denominator
    return dict(scores)


def hybrid_rank(
    score_pages: list[int], association: dict[int, float], beta: float
) -> list[int]:
    score_signal = {
        page: 1.0 - rank / max(1, len(score_pages) - 1)
        for rank, page in enumerate(score_pages)
    }
    universe = set(score_signal) | set(association)
    return sorted(
        universe,
        key=lambda page: (
            -(score_signal.get(page, 0.0) + beta * association.get(page, 0.0)),
            -score_signal.get(page, 0.0),
            page,
        ),
    )


def metric_row(
    selected: set[int],
    target_pages: set[int],
    target_tokens: np.ndarray,
    page_size: int,
) -> dict[str, float | int]:
    hits = len(selected & target_pages)
    token_hits = sum((int(token) // page_size) in selected for token in target_tokens)
    missing = len(target_pages) - hits
    return {
        "budget_pages": len(selected),
        "target_pages": len(target_pages),
        "page_hits": hits,
        "page_recall": hits / len(target_pages),
        "page_precision": hits / len(selected),
        "target_tokens": len(target_tokens),
        "token_hits": token_hits,
        "topk_token_recall": token_hits / len(target_tokens),
        "critical_miss_pages": missing,
        "critical_miss_bytes": missing * page_size * KV_BYTES_PER_TOKEN_LAYER,
    }


def build_lookup(run_dir: Path) -> dict[tuple[str, int], Path]:
    lookup: dict[tuple[str, int], Path] = {}
    for row in read_jsonl(run_dir / "events" / "manifest.jsonl"):
        key = (str(row["request_id"]), int(row["layer_id"]))
        if key in lookup:
            raise ValueError(f"{run_dir}: duplicate chunk {key}")
        lookup[key] = run_dir / "events" / row["file"]
    return lookup


def analyze_run(
    run_dir: Path, layers: list[int], page_sizes: list[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requests = read_jsonl(run_dir / "prepared_requests.jsonl")
    lookup = build_lookup(run_dir)
    split_map = build_split_map(requests)
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        trajectory, _, _, _ = request_metadata(request)
        grouped[trajectory].append(request)
    for rows in grouped.values():
        rows.sort(key=lambda row: request_metadata(row)[2])

    output: list[dict[str, Any]] = []
    chunks = 0
    topk_sources: set[str] = set()
    for trajectory, trajectory_requests in sorted(grouped.items()):
        for layer in layers:
            chunks_by_request: list[tuple[dict[str, Any], Chunk]] = []
            for request in trajectory_requests:
                path = lookup.get((request["rid"], layer))
                if path is None:
                    raise ValueError(f"{run_dir}: missing {(request['rid'], layer)}")
                chunk = load_chunk(path)
                chunks_by_request.append((request, chunk))
                chunks += 1
                topk_sources.add(chunk.topk_source)

            for page_size in page_sizes:
                history: list[tuple[int, tuple[int, ...]]] = []
                observed_frequency: Counter[int] = Counter()
                for request, chunk in chunks_by_request:
                    _, category, round_id, context = request_metadata(request)
                    step_pages: list[set[int]] = []
                    step_tokens: list[np.ndarray] = []
                    for step in range(32):
                        valid = int(chunk.valid_counts[step])
                        tokens = np.unique(
                            chunk.topk[step][
                                (chunk.topk[step] >= 0) & (chunk.topk[step] < valid)
                            ]
                        )
                        if len(tokens) != TARGET_K:
                            raise ValueError(
                                f"{request['rid']} L{layer} step{step}: top-k width"
                            )
                        step_tokens.append(tokens)
                        step_pages.append({int(token) // page_size for token in tokens})

                    for step in range(31):
                        current_pages = step_pages[step]
                        if step == 0:
                            observed_frequency.update(current_pages)
                        else:
                            history.append(
                                (
                                    set_mask(step_pages[step - 1]),
                                    tuple(sorted(current_pages)),
                                )
                            )
                            observed_frequency.update(current_pages)

                        if len(history) < MIN_HISTORY:
                            continue
                        common = min(
                            int(chunk.valid_counts[step]),
                            int(chunk.valid_counts[step + 1]),
                        )
                        target_tokens = np.unique(
                            chunk.topk[step + 1][
                                (chunk.topk[step + 1] >= 0)
                                & (chunk.topk[step + 1] < common)
                            ]
                        )
                        target_pages = {
                            int(token) // page_size for token in target_tokens
                        }
                        ranked = chunk.ranked[step]
                        ranked = ranked[(ranked >= 0) & (ranked < common)]
                        score_candidate_pages = set(
                            int(token) // page_size for token in ranked[:CANDIDATE_K]
                        )
                        budget = len(score_candidate_pages)
                        score_page_order = unique_page_order(ranked, page_size)
                        if set(score_page_order[:budget]) != score_candidate_pages:
                            raise AssertionError("score page budget mismatch")
                        total_pages = (common + page_size - 1) // page_size
                        frequency_order = rank_by_counter(
                            observed_frequency, total_pages
                        )
                        if len(frequency_order) < budget:
                            continue
                        association = association_scores(set_mask(current_pages), history)
                        association_order = sorted(
                            (page for page in association if page < total_pages),
                            key=lambda page: (-association[page], page),
                        )

                        methods: dict[str, set[int] | None] = {
                            "previous-score-k3072": score_candidate_pages,
                            "frequency": fill_to_budget(
                                frequency_order, (), budget
                            ),
                            "previous-topk+frequency": fill_to_budget(
                                sorted(current_pages), frequency_order, budget
                            ),
                            "association+frequency": fill_to_budget(
                                association_order, frequency_order, budget
                            ),
                        }
                        for beta in BETAS:
                            methods[f"score+association-beta{beta:g}"] = fill_to_budget(
                                hybrid_rank(score_page_order, association, beta),
                                (),
                                budget,
                            )
                        candidate_universe = set(score_page_order) | set(association)
                        oracle_first = sorted(candidate_universe & target_pages)
                        methods["candidate-oracle"] = fill_to_budget(
                            oracle_first, score_page_order, budget
                        )

                        common_row = {
                            "run_dir": str(run_dir),
                            "dataset": str(request.get("dataset", "BFCL")),
                            "trajectory_id": trajectory,
                            "split": split_map[trajectory],
                            "category": category,
                            "request_id": request["rid"],
                            "round_id": round_id,
                            "context_config": context,
                            "prompt_tokens": int(request.get("prompt_len", common)),
                            "layer": layer,
                            "step": step,
                            "page_size": page_size,
                            "history_transitions": len(history),
                            "association_candidates": len(association),
                            "total_pages": total_pages,
                        }
                        for method, selected in methods.items():
                            if selected is None or len(selected) != budget:
                                raise AssertionError(
                                    f"could not fill budget for {method}: "
                                    f"{request['rid']} L{layer} step{step}"
                                )
                            output.append(
                                {
                                    **common_row,
                                    "method": method,
                                    **metric_row(
                                        selected,
                                        target_pages,
                                        target_tokens,
                                        page_size,
                                    ),
                                }
                            )

                    # Make all transitions from the completed round available
                    # to its successor, including the final step 30 -> 31.
                    history.append(
                        (
                            set_mask(step_pages[30]),
                            tuple(sorted(step_pages[31])),
                        )
                    )
                    observed_frequency.update(step_pages[31])

        print(
            f"[{len(output):07d} rows] {trajectory}: {len(trajectory_requests)} rounds",
            flush=True,
        )

    validation = {
        "run_dir": str(run_dir),
        "requests": len(requests),
        "trajectories": len(grouped),
        "layers": layers,
        "page_sizes": page_sizes,
        "chunks": chunks,
        "expected_chunks": len(requests) * len(layers),
        "rows": len(output),
        "topk_sources": sorted(topk_sources),
        "methods": sorted({row["method"] for row in output}),
        "splits": dict(Counter(split_map.values())),
    }
    if chunks != validation["expected_chunks"]:
        raise AssertionError("chunk count mismatch")
    return output, validation


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    groups = ["dataset", "split", "page_size", "method"]
    count_columns = [
        "budget_pages",
        "target_pages",
        "page_hits",
        "target_tokens",
        "token_hits",
        "critical_miss_pages",
        "critical_miss_bytes",
    ]
    per_trajectory = rows.groupby(groups + ["trajectory_id"], as_index=False)[
        count_columns
    ].sum()
    per_trajectory["page_recall"] = (
        per_trajectory.page_hits / per_trajectory.target_pages
    )
    per_trajectory["topk_token_recall"] = (
        per_trajectory.token_hits / per_trajectory.target_tokens
    )
    per_trajectory["page_precision"] = (
        per_trajectory.page_hits / per_trajectory.budget_pages
    )

    records: list[dict[str, Any]] = []
    metrics = [
        "page_recall",
        "topk_token_recall",
        "page_precision",
        "critical_miss_bytes",
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for keys, part in per_trajectory.groupby(groups):
        row = dict(zip(groups, keys, strict=True))
        row["trajectories"] = len(part)
        for metric in metrics:
            values = part[metric].to_numpy(dtype=np.float64)
            row[metric] = float(values.mean())
            if len(values) >= 2:
                samples = rng.choice(
                    values,
                    size=(BOOTSTRAP_SAMPLES, len(values)),
                    replace=True,
                ).mean(axis=1)
                row[f"{metric}_ci_low"] = float(np.quantile(samples, 0.025))
                row[f"{metric}_ci_high"] = float(np.quantile(samples, 0.975))
            else:
                row[f"{metric}_ci_low"] = float("nan")
                row[f"{metric}_ci_high"] = float("nan")
        row["budget_pages_mean"] = float(
            rows[
                (rows.dataset == row["dataset"])
                & (rows.split == row["split"])
                & (rows.page_size == row["page_size"])
                & (rows.method == row["method"])
            ].budget_pages.mean()
        )
        records.append(row)
    return pd.DataFrame(records).sort_values(groups)


def choose_betas(summary: pd.DataFrame) -> dict[int, float]:
    calibration = summary[
        (summary.dataset == "BFCL")
        & (summary.split == "calibration")
        & summary.method.str.startswith("score+association-beta")
    ]
    chosen: dict[int, float] = {}
    for page_size, part in calibration.groupby("page_size"):
        best = part.sort_values(
            ["topk_token_recall", "page_recall", "method"], ascending=[False, False, True]
        ).iloc[0]
        chosen[int(page_size)] = float(str(best.method).split("beta", 1)[1])
    return chosen


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(
        values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True
    ).mean(axis=1)
    return mean, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def gate_table(
    detail: pd.DataFrame, summary: pd.DataFrame, chosen: dict[int, float]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in sorted(summary.dataset.unique()):
        splits = ["test"] if dataset == "BFCL" else ["pilot"]
        for split in splits:
            for page_size, beta in chosen.items():
                part = summary[
                    (summary.dataset == dataset)
                    & (summary.split == split)
                    & (summary.page_size == page_size)
                ]
                if part.empty:
                    continue
                baseline = part[part.method == "previous-score-k3072"]
                hybrid = part[part.method == f"score+association-beta{beta:g}"]
                previous_frequency = part[
                    part.method == "previous-topk+frequency"
                ]
                if baseline.empty or hybrid.empty or previous_frequency.empty:
                    continue
                base = baseline.iloc[0]
                candidate = hybrid.iloc[0]
                prev_freq = previous_frequency.iloc[0]

                selected = detail[
                    (detail.dataset == dataset)
                    & (detail.split == split)
                    & (detail.page_size == page_size)
                    & detail.method.isin(
                        [
                            "previous-score-k3072",
                            "previous-topk+frequency",
                            f"score+association-beta{beta:g}",
                        ]
                    )
                ]
                counts = selected.groupby(
                    ["trajectory_id", "method"], as_index=False
                )[
                    [
                        "target_pages",
                        "page_hits",
                        "target_tokens",
                        "token_hits",
                        "critical_miss_bytes",
                    ]
                ].sum()
                counts["page_recall"] = counts.page_hits / counts.target_pages
                counts["token_recall"] = counts.token_hits / counts.target_tokens
                page_pivot = counts.pivot(
                    index="trajectory_id", columns="method", values="page_recall"
                )
                token_pivot = counts.pivot(
                    index="trajectory_id", columns="method", values="token_recall"
                )
                miss_pivot = counts.pivot(
                    index="trajectory_id",
                    columns="method",
                    values="critical_miss_bytes",
                )
                hybrid_name = f"score+association-beta{beta:g}"
                page_delta = 100 * (
                    page_pivot[hybrid_name] - page_pivot["previous-score-k3072"]
                ).to_numpy()
                token_delta = 100 * (
                    token_pivot[hybrid_name] - token_pivot["previous-score-k3072"]
                ).to_numpy()
                miss_reduction = (
                    1
                    - miss_pivot[hybrid_name]
                    / miss_pivot["previous-score-k3072"]
                ).to_numpy()
                page_mean, page_low, page_high = mean_ci(page_delta)
                token_mean, token_low, token_high = mean_ci(token_delta)
                miss_mean, miss_low, miss_high = mean_ci(miss_reduction)
                beats_previous_frequency = bool(
                    candidate.page_recall >= prev_freq.page_recall
                    and candidate.topk_token_recall >= prev_freq.topk_token_recall
                )
                passes_numeric_gate = bool(
                    page_mean >= 2.0 or token_mean >= 2.0 or miss_mean >= 0.10
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "page_size": page_size,
                        "chosen_beta": beta,
                        "score_page_recall": base.page_recall,
                        "hybrid_page_recall": candidate.page_recall,
                        "page_recall_delta_pp": page_mean,
                        "page_recall_delta_ci_low_pp": page_low,
                        "page_recall_delta_ci_high_pp": page_high,
                        "score_token_recall": base.topk_token_recall,
                        "hybrid_token_recall": candidate.topk_token_recall,
                        "token_recall_delta_pp": token_mean,
                        "token_recall_delta_ci_low_pp": token_low,
                        "token_recall_delta_ci_high_pp": token_high,
                        "critical_miss_bytes_reduction": miss_mean,
                        "critical_miss_bytes_reduction_ci_low": miss_low,
                        "critical_miss_bytes_reduction_ci_high": miss_high,
                        "previous_topk_frequency_page_recall": prev_freq.page_recall,
                        "previous_topk_frequency_token_recall": prev_freq.topk_token_recall,
                        "hybrid_beats_previous_topk_frequency": beats_previous_frequency,
                        "trajectories": len(page_pivot),
                        "trajectory_page_recall_wins_vs_score": int(
                            np.count_nonzero(page_delta > 0)
                        ),
                        "passes_numeric_gate": passes_numeric_gate,
                        "passes_gate_and_strong_baselines": bool(
                            passes_numeric_gate and beats_previous_frequency
                        ),
                    }
                )
    return pd.DataFrame(rows)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
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


def plot_gate(gate: pd.DataFrame, output_dir: Path) -> None:
    bfcl = gate[(gate.dataset == "BFCL") & (gate.split == "test")]
    if bfcl.empty:
        return
    x = np.arange(len(bfcl))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), constrained_layout=True)
    axes[0].bar(
        x,
        bfcl.page_recall_delta_pp,
        yerr=np.vstack(
            [
                bfcl.page_recall_delta_pp - bfcl.page_recall_delta_ci_low_pp,
                bfcl.page_recall_delta_ci_high_pp - bfcl.page_recall_delta_pp,
            ]
        ),
        color="#0072B2",
        edgecolor="white",
        linewidth=0.7,
        capsize=3,
    )
    axes[0].axhline(2.0, color="#9E9E9E", linestyle="--", linewidth=1.2)
    axes[0].set(
        xticks=x,
        xticklabels=bfcl.page_size,
        xlabel="KV page size (tokens)",
        ylabel="Page-recall gain (pp)",
    )
    axes[1].bar(
        x,
        bfcl.critical_miss_bytes_reduction * 100,
        yerr=np.vstack(
            [
                (
                    bfcl.critical_miss_bytes_reduction
                    - bfcl.critical_miss_bytes_reduction_ci_low
                )
                * 100,
                (
                    bfcl.critical_miss_bytes_reduction_ci_high
                    - bfcl.critical_miss_bytes_reduction
                )
                * 100,
            ]
        ),
        color="#D55E00",
        edgecolor="white",
        linewidth=0.7,
        capsize=3,
    )
    axes[1].axhline(10.0, color="#9E9E9E", linestyle="--", linewidth=1.2)
    axes[1].set(
        xticks=x,
        xticklabels=bfcl.page_size,
        xlabel="KV page size (tokens)",
        ylabel="Miss-byte reduction (%)",
    )
    for suffix, kwargs in [("pdf", {}), ("png", {"dpi": 300})]:
        fig.savefig(
            output_dir / f"coactivation_increment_gate.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def run_self_test() -> None:
    history = [
        (set_mask({0, 1}), (2, 3)),
        (set_mask({4, 5}), (6,)),
        (set_mask({0, 1, 4}), (2, 7)),
        (set_mask({8}), (9,)),
    ]
    scores = association_scores(set_mask({0, 1}), history)
    assert scores[2] > scores.get(6, 0)
    assert fill_to_budget([2], [3, 4], 3) == {2, 3, 4}
    assert unique_page_order(np.array([0, 1, 4, 5, 2]), 2) == [0, 2, 1]
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layers", type=parse_int_list, default=DEFAULT_LAYERS)
    parser.add_argument(
        "--page-sizes", type=parse_int_list, default=DEFAULT_PAGE_SIZES
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reaggregate existing transition parquet without rereading traces.",
    )
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.output_dir is None or (not args.plot_only and not args.run_dirs):
        parser.error(
            "--output-dir is required; --run-dirs is also required unless --plot-only"
        )

    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        detail = pd.read_parquet(table_dir / "coactivation_by_transition.parquet")
        validations = json.loads((output_dir / "validation.json").read_text())
    else:
        records: list[dict[str, Any]] = []
        validations = []
        for run_dir in args.run_dirs:
            rows, validation = analyze_run(
                run_dir.resolve(), args.layers, args.page_sizes
            )
            records.extend(rows)
            validations.append(validation)
        detail = pd.DataFrame(records)
    summary = aggregate(detail)
    chosen = choose_betas(summary)
    gate = gate_table(detail, summary, chosen)
    detail.to_parquet(table_dir / "coactivation_by_transition.parquet", index=False)
    summary.to_csv(table_dir / "coactivation_summary.csv", index=False)
    gate.to_csv(table_dir / "coactivation_gate.csv", index=False)
    set_style()
    plot_gate(gate, figure_dir)

    gate_result = {
        "chosen_beta_by_page_size": chosen,
        "gate_rows": gate.to_dict("records"),
        "keep_threshold": {
            "page_or_token_recall_delta_pp": 2.0,
            "or_critical_miss_bytes_reduction": 0.10,
        },
        "validation": validations,
        "limitations": [
            "BFCL schema-v4 top-k is reconstructed exactly from retained full scores",
            "RULER has one request per context length and is descriptive only",
            (
                "history-neighbor is an online co-activation proxy, not a "
                "reproduction of Swarm clustering"
            ),
            "critical miss bytes assume every unpredicted page blocks and do not model overlap",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(gate_result, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validations, indent=2, ensure_ascii=False) + "\n"
    )
    reproducibility = {
        "run_dirs": (
            [str(path.resolve()) for path in args.run_dirs]
            if args.run_dirs
            else [row["run_dir"] for row in validations]
        ),
        "output_dir": str(output_dir),
        "script": str(Path(__file__).resolve()),
        "code_commit": git_revision(Path(__file__).resolve().parents[2]),
        "target_k": TARGET_K,
        "candidate_k": CANDIDATE_K,
        "layers": args.layers,
        "page_sizes_tokens": args.page_sizes,
        "betas": BETAS,
        "knn": KNN,
        "history_limit": HISTORY_LIMIT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "output_sha256": {},
    }
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "reproducibility.json":
            reproducibility["output_sha256"][str(path.relative_to(output_dir))] = sha256(path)
    (output_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(gate_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
