#!/usr/bin/env python3
"""Analyze full-layer Keye DSA traces with semantic prompt segments."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from cycler import cycler
from scipy.stats import spearmanr

LAYERS = list(range(48))
STEP_GAPS = [1, 2, 4, 8, 16]
LAYER_ANCHOR_STEPS = [0, 16, 31]
SEGMENT_ORDER = [
    "system_instruction",
    "tool_schema",
    "initial_state",
    "long_context_distractor",
    "user_turn",
    "assistant_tool_call",
    "assistant_response",
    "tool_result",
]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return math.nan
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else math.nan


def sampled_percentile_drift(left: np.ndarray, right: np.ndarray) -> float:
    """Mean absolute percentile-rank drift within a paired token sample."""
    count = len(left)
    if count < 2:
        return math.nan
    left_rank = np.empty(count, dtype=np.int32)
    right_rank = np.empty(count, dtype=np.int32)
    left_rank[np.argsort(left, kind="stable")] = np.arange(count)
    right_rank[np.argsort(right, kind="stable")] = np.arange(count)
    return float(np.mean(np.abs(left_rank - right_rank)) / (count - 1))


def topk_set(indices: np.ndarray, common_len: int) -> set[int]:
    return {int(value) for value in indices if 0 <= int(value) < common_len}


def set_metrics(left: set[int], right: set[int]) -> dict[str, float]:
    intersection = len(left & right)
    union = len(left | right)
    return {
        "topk_recall": intersection / len(right) if right else math.nan,
        "topk_jaccard": intersection / union if union else math.nan,
        "topk_churn": len(right - left) / len(right) if right else math.nan,
    }


def load_chunk(path: Path) -> dict[str, Any]:
    record = torch.load(path, weights_only=False)
    if record["schema_version"] != 4:
        raise ValueError(f"{path}: expected schema v4")
    if record["decode_step_ids"] != list(range(32)):
        raise ValueError(f"{path}: expected continuous decode steps 0..31")
    return {
        "scores": record["scores"].numpy(force=True).astype(np.float32),
        "indices": record["indices"].numpy(force=True).astype(np.int32),
        "score_lens": record["score_valid_counts"].numpy(force=True).astype(int),
        "valid_counts": record["valid_counts"].numpy(force=True).astype(int),
    }


def write_table(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)
    return frame


def save_figure(fig: matplotlib.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_inventory(
    run_dir: Path, requests: list[dict[str, Any]]
) -> tuple[pd.DataFrame, dict[tuple[str, int], Path]]:
    event_dir = run_dir / "events"
    expected_rids = {row["rid"] for row in requests}
    rows = [
        row
        for row in jsonl(event_dir / "manifest.jsonl")
        if row.get("request_id") in expected_rids
    ]
    lookup: dict[tuple[str, int], Path] = {}
    inventory = []
    for row in rows:
        key = (row["request_id"], int(row["layer_id"]))
        if key in lookup:
            raise ValueError(f"duplicate request/layer chunk: {key}")
        lookup[key] = event_dir / row["file"]
        inventory.append(
            {
                "rid": key[0],
                "layer": key[1],
                "steps": int(row["num_steps"]),
                "score_width": int(row["score_width"]),
                "bytes": int(row["bytes"]),
                "file": row["file"],
            }
        )
    expected = len(requests) * len(LAYERS)
    if len(lookup) != expected:
        missing = [
            (request["rid"], layer)
            for request in requests
            for layer in LAYERS
            if (request["rid"], layer) not in lookup
        ]
        raise ValueError(f"expected {expected} chunks, found {len(lookup)}; missing={missing[:5]}")
    return pd.DataFrame(inventory), lookup


def analyze(run_dir: Path) -> dict[str, pd.DataFrame]:
    prepared = jsonl(run_dir / "prepared_requests.jsonl")
    request_meta = {row["rid"]: row for row in prepared}
    request_order = sorted(
        prepared, key=lambda row: (row["trajectory_id"], int(row["round_id"]))
    )
    segments_by_rid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jsonl(run_dir / "segments.jsonl"):
        if row["segment_type"] != "chat_template":
            segments_by_rid[row["rid"]].append(row)

    inventory, lookup = build_inventory(run_dir, prepared)
    step_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    stripe_rows: list[dict[str, Any]] = []
    storage_rows: list[dict[str, Any]] = []
    previous_segments: dict[tuple[str, int, str], dict[str, Any]] = {}
    last_round = {
        trajectory: max(
            int(row["round_id"])
            for row in prepared
            if row["trajectory_id"] == trajectory
        )
        for trajectory in {row["trajectory_id"] for row in prepared}
    }

    for request_index, request in enumerate(request_order, 1):
        rid = request["rid"]
        prompt_len = int(request["prompt_len"])
        chunks = {layer: load_chunk(lookup[(rid, layer)]) for layer in LAYERS}
        category = request["category"]
        trajectory = request["trajectory_id"]
        round_id = int(request["round_id"])

        for layer, chunk in chunks.items():
            scores = chunk["scores"]
            indices = chunk["indices"]
            lens = chunk["score_lens"]
            sets = [topk_set(indices[step], int(lens[step])) for step in range(32)]
            all_membership = np.zeros((32, scores.shape[1]), dtype=np.uint8)
            for step, selected in enumerate(sets):
                all_membership[step, list(selected)] = 1
            step_iqr = []
            for step in range(32):
                valid = int(lens[step])
                scale_sample = np.linspace(
                    0, valid - 1, min(512, valid), dtype=int
                )
                q25, q75 = np.quantile(scores[step, scale_sample], [0.25, 0.75])
                step_iqr.append(float(q75 - q25))
            for gap in STEP_GAPS:
                for step_a in range(32 - gap):
                    step_b = step_a + gap
                    common = min(int(lens[step_a]), int(lens[step_b]))
                    left_mask = all_membership[step_a, :common]
                    right_mask = all_membership[step_b, :common]
                    left_count = int(left_mask.sum())
                    right_count = int(right_mask.sum())
                    intersection = int(np.count_nonzero(left_mask & right_mask))
                    union = left_count + right_count - intersection
                    delta = scores[step_b, :common] - scores[step_a, :common]
                    sample = np.linspace(0, common - 1, min(512, common), dtype=int)
                    scale = step_iqr[step_a]
                    sampled_correlation = step_a in {0, 8, 16, 24}
                    step_rows.append(
                        {
                            "rid": rid,
                            "trajectory_id": trajectory,
                            "category": category,
                            "round_id": round_id,
                            "layer": layer,
                            "step_a": step_a,
                            "step_b": step_b,
                            "step_gap": gap,
                            "topk_recall": intersection / right_count,
                            "topk_jaccard": intersection / union,
                            "topk_churn": (right_count - intersection)
                            / right_count,
                            "score_mae": float(np.mean(np.abs(delta))),
                            "score_rmse": float(np.sqrt(np.mean(delta * delta))),
                            "score_nmae_iqr_sampled": float(np.mean(np.abs(delta)))
                            / max(scale, np.finfo(np.float32).eps),
                            "score_spearman_sampled": (
                                safe_spearman(
                                    scores[step_a, sample], scores[step_b, sample]
                                )
                                if sampled_correlation
                                else math.nan
                            ),
                            "score_sample_count": (
                                len(sample) if sampled_correlation else 0
                            ),
                            "score_percentile_drift_sampled": (
                                sampled_percentile_drift(
                                    scores[step_a, sample],
                                    scores[step_b, sample],
                                )
                                if sampled_correlation
                                else math.nan
                            ),
                        }
                    )

            prior_semantic_candidates: set[int] = set()
            for segment in segments_by_rid[rid]:
                prior = previous_segments.get(
                    (trajectory, layer, segment["segment_id"])
                )
                if (
                    prior is None
                    or prior["round_id"] != round_id - 1
                    or prior["token_ids_sha256"] != segment["token_ids_sha256"]
                ):
                    continue
                start, end = int(segment["token_start"]), int(segment["token_end"])
                common = min(end - start, len(prior["frequency"]))
                stable_offsets = np.flatnonzero(
                    prior["frequency"][:common] >= 0.5
                )
                prior_semantic_candidates.update((start + stable_offsets).tolist())

            # Candidate-set simulations use no future information. The 2,048
            # bytes/token estimate aggregates K+V over both TP ranks from the
            # server's measured 0.38 GiB/rank, 8,192-token, 48-layer pool.
            for step in (
                range(1, 32) if round_id == last_round[trajectory] else []
            ):
                valid = int(lens[step - 1])
                target = topk_set(indices[step], valid)
                previous = topk_set(indices[step - 1], valid)
                hybrid = previous | prior_semantic_candidates
                for policy, candidate in [
                    ("previous_topk", previous),
                    ("hybrid_previous_plus_round_core", hybrid),
                ]:
                    storage_rows.append(
                        {
                            "rid": rid,
                            "layer": layer,
                            "step": step,
                            "policy": policy,
                            "candidate_tokens": len(candidate),
                            "budget": len(candidate),
                            "topk_recall": len(candidate & target) / len(target),
                            "estimated_kv_bytes_tp2": len(candidate) * 2048,
                        }
                    )
                score_order = np.argsort(scores[step - 1, :valid])
                for budget in [2048, 2304, 2560, 3072]:
                    take = min(budget, valid)
                    candidate = set(score_order[-take:].tolist())
                    storage_rows.append(
                        {
                            "rid": rid,
                            "layer": layer,
                            "step": step,
                            "policy": "previous_score_rank",
                            "candidate_tokens": len(candidate),
                            "budget": budget,
                            "topk_recall": len(candidate & target) / len(target),
                            "estimated_kv_bytes_tp2": len(candidate) * 2048,
                        }
                    )
                recency = set(range(max(0, valid - 2048), valid))
                storage_rows.append(
                    {
                        "rid": rid,
                        "layer": layer,
                        "step": step,
                        "policy": "recency_2048",
                        "candidate_tokens": len(recency),
                        "budget": 2048,
                        "topk_recall": len(recency & target) / len(target),
                        "estimated_kv_bytes_tp2": len(recency) * 2048,
                    }
                )

            membership = all_membership[:, :prompt_len]
            frequency = membership.mean(axis=0)
            mean_scores = scores[:, :prompt_len].mean(axis=0)
            for segment in segments_by_rid[rid]:
                start, end = int(segment["token_start"]), int(segment["token_end"])
                if end <= start:
                    continue
                segment_freq = frequency[start:end]
                selected_per_step = membership[:, start:end].sum(axis=1)
                expected = (end - start) * 2048 / prompt_len
                row = {
                    "rid": rid,
                    "trajectory_id": trajectory,
                    "category": category,
                    "round_id": round_id,
                    "layer": layer,
                    "segment_id": segment["segment_id"],
                    "segment_type": segment["segment_type"],
                    "source_round": segment["source_round"],
                    "round_age": segment["round_age"],
                    "tool_name": segment["tool_name"],
                    "token_start": start,
                    "token_end": end,
                    "token_count": end - start,
                    "mean_topk_frequency": float(segment_freq.mean()),
                    "segment_coverage": float(segment_freq.mean()),
                    "mean_selected_tokens": float(selected_per_step.mean()),
                    "topk_share": float(selected_per_step.mean() / 2048),
                    "selection_lift": float(selected_per_step.mean() / max(expected, 1e-12)),
                    "stable_core_50": int(np.count_nonzero(segment_freq >= 0.5)),
                    "stable_core_80": int(np.count_nonzero(segment_freq >= 0.8)),
                    "mean_score": float(mean_scores[start:end].mean()),
                }
                segment_rows.append(row)
                key = (trajectory, layer, segment["segment_id"])
                prior = previous_segments.get(key)
                if (
                    prior is not None
                    and prior["round_id"] == round_id - 1
                    and prior["token_ids_sha256"] == segment["token_ids_sha256"]
                ):
                    common = min(len(prior["frequency"]), len(segment_freq))
                    old_freq = prior["frequency"][:common]
                    new_freq = segment_freq[:common]
                    for threshold in [0.5, 0.8]:
                        old_core = set(np.flatnonzero(old_freq >= threshold).tolist())
                        new_core = set(np.flatnonzero(new_freq >= threshold).tolist())
                        round_rows.append(
                            {
                                "trajectory_id": trajectory,
                                "category": category,
                                "round_a": round_id - 1,
                                "round_b": round_id,
                                "layer": layer,
                                "segment_id": segment["segment_id"],
                                "segment_type": segment["segment_type"],
                                "source_round": segment["source_round"],
                                "round_age": segment["round_age"],
                                "threshold": threshold,
                                "common_tokens": common,
                                "frequency_spearman": safe_spearman(
                                    old_freq, new_freq
                                ),
                                "core_jaccard": set_metrics(
                                    old_core, new_core
                                )["topk_jaccard"],
                                "persistent_recall": set_metrics(
                                    new_core, old_core
                                )["topk_recall"],
                                "lift_drift": row["selection_lift"]
                                - prior["selection_lift"],
                            }
                        )
                previous_segments[key] = {
                    "round_id": round_id,
                    "frequency": segment_freq.copy(),
                    "selection_lift": row["selection_lift"],
                    "token_ids_sha256": segment["token_ids_sha256"],
                }

            if round_id == last_round[trajectory]:
                delta = np.diff(scores[:, :prompt_len], axis=0)
                centered = delta - delta.mean(axis=1, keepdims=True)
                covariance = centered @ centered.T
                eigenvalues = np.linalg.eigvalsh(covariance)
                eigenvalues = np.maximum(eigenvalues, 0)
                total = float(eigenvalues.sum())
                raw_energy = float(np.square(delta).sum())
                common_mode = float(
                    np.square(delta.mean(axis=1, keepdims=True)).sum() * prompt_len
                )
                thresholds = []
                concentrations = []
                enrichments = []
                overall_churn = []
                for step in range(31):
                    valid = min(int(lens[step]), prompt_len)
                    kth = np.partition(scores[step, :valid], -2048)[-2048]
                    distances = np.abs(scores[step, :valid] - kth)
                    changed = membership[step, :valid] != membership[step + 1, :valid]
                    order = np.argsort(distances)
                    changed_total = max(int(changed.sum()), 1)
                    overall_rate = float(changed.mean())
                    band_rates = []
                    band_concentrations = []
                    band_enrichments = []
                    for frac in [0.01, 0.05, 0.10]:
                        band = order[: max(1, int(valid * frac))]
                        band_changed = int(changed[band].sum())
                        rate = float(changed[band].mean())
                        band_rates.append(rate)
                        band_concentrations.append(band_changed / changed_total)
                        band_enrichments.append(rate / max(overall_rate, 1e-12))
                    thresholds.append(band_rates)
                    concentrations.append(band_concentrations)
                    enrichments.append(band_enrichments)
                    overall_churn.append(overall_rate)
                boundary = np.asarray(thresholds)
                concentration = np.asarray(concentrations)
                enrichment = np.asarray(enrichments)
                stripe_rows.append(
                    {
                        "trajectory_id": trajectory,
                        "category": category,
                        "round_id": round_id,
                        "layer": layer,
                        "raw_common_mode_energy_ratio": common_mode
                        / max(raw_energy, 1e-12),
                        "centered_top1_evr": float(eigenvalues[-1] / total),
                        "centered_top4_evr": float(eigenvalues[-4:].sum() / total),
                        "centered_top8_evr": float(eigenvalues[-8:].sum() / total),
                        "boundary_churn_1pct": float(boundary[:, 0].mean()),
                        "boundary_churn_5pct": float(boundary[:, 1].mean()),
                        "boundary_churn_10pct": float(boundary[:, 2].mean()),
                        "overall_membership_churn": float(
                            np.mean(overall_churn)
                        ),
                        "boundary_concentration_1pct": float(
                            concentration[:, 0].mean()
                        ),
                        "boundary_concentration_5pct": float(
                            concentration[:, 1].mean()
                        ),
                        "boundary_concentration_10pct": float(
                            concentration[:, 2].mean()
                        ),
                        "boundary_enrichment_1pct": float(
                            enrichment[:, 0].mean()
                        ),
                        "boundary_enrichment_5pct": float(
                            enrichment[:, 1].mean()
                        ),
                        "boundary_enrichment_10pct": float(
                            enrichment[:, 2].mean()
                        ),
                    }
                )

        # Full 48x48 layer matrices are evaluated at fixed steps on one final
        # request per trajectory, avoiding pseudo-replication across rounds.
        if round_id == last_round[trajectory]:
            for step in LAYER_ANCHOR_STEPS:
                layer_sets = {
                    layer: topk_set(
                        chunks[layer]["indices"][step],
                        int(chunks[layer]["score_lens"][step]),
                    )
                    for layer in LAYERS
                }
                common = min(
                    int(chunks[layer]["score_lens"][step]) for layer in LAYERS
                )
                sample = np.linspace(0, common - 1, min(512, common), dtype=int)
                for layer_a in LAYERS:
                    for layer_b in range(layer_a + 1, 48):
                        layer_rows.append(
                            {
                                "trajectory_id": trajectory,
                                "category": category,
                                "round_id": round_id,
                                "step": step,
                                "layer_a": layer_a,
                                "layer_b": layer_b,
                                **set_metrics(
                                    layer_sets[layer_a], layer_sets[layer_b]
                                ),
                                "score_spearman_sampled": safe_spearman(
                                    chunks[layer_a]["scores"][step, sample],
                                    chunks[layer_b]["scores"][step, sample],
                                ),
                                "score_sample_count": len(sample),
                            }
                        )
        print(
            f"[{request_index:03d}/{len(request_order)}] analyzed {rid}",
            flush=True,
        )

    request_perf = []
    for row in jsonl(run_dir / "requests.jsonl"):
        usage = row["response"].get("meta_info", row["response"].get("usage", {}))
        completion = int(usage.get("completion_tokens", 64))
        request_perf.append(
            {
                "rid": row["rid"],
                "trajectory_id": row["trajectory_id"],
                "category": row["category"],
                "round_id": row["round_id"],
                "prompt_tokens": int(
                    usage.get("prompt_tokens", row["prompt_len"])
                ),
                "completion_tokens": completion,
                "latency_s": row["latency_s"],
                "throughput_tps": completion / row["latency_s"],
            }
        )
    return {
        "trace_inventory": inventory,
        "step_metrics_by_layer": pd.DataFrame(step_rows),
        "layer_metrics": pd.DataFrame(layer_rows),
        "segment_metrics": pd.DataFrame(segment_rows),
        "cross_round_segment_metrics": pd.DataFrame(round_rows),
        "stripe_decomposition": pd.DataFrame(stripe_rows),
        "storage_simulation": pd.DataFrame(storage_rows),
        "request_performance": pd.DataFrame(request_perf),
    }


def plot_results(tables: dict[str, pd.DataFrame], figure_dir: Path) -> None:
    step = tables["step_metrics_by_layer"]
    segment = tables["segment_metrics"]
    cross_round = tables["cross_round_segment_metrics"]
    layer = tables["layer_metrics"]
    stripe = tables["stripe_decomposition"]
    storage = tables["storage_simulation"]

    adjacent = step[step.step_gap == 1].groupby("layer").topk_recall.mean()
    matrix = np.full((48, 31), np.nan)
    heat = step[step.step_gap == 1].groupby(["layer", "step_a"]).topk_recall.mean()
    for (layer_id, step_id), value in heat.items():
        matrix[layer_id, step_id] = value
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set(xlabel="Decode transition t→t+1", ylabel="DSA layer")
    fig.colorbar(image, ax=ax, label="Top-k recall")
    save_figure(fig, figure_dir / "adjacent_step_reuse_48l_heatmap")

    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    for color, gap in zip(COLORS, STEP_GAPS):
        values = step[step.step_gap == gap].groupby("layer").topk_recall.mean()
        ax.plot(values.index, values, label=f"Δ={gap}", color=color)
    ax.set(xlabel="DSA layer", ylabel="Top-k recall", ylim=(0, 1))
    ax.legend(ncol=5, frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "step_reuse_vs_layer")

    grouped = layer.groupby(["layer_a", "layer_b"]).topk_jaccard.mean()
    layer_matrix = np.eye(48)
    for (left, right), value in grouped.items():
        layer_matrix[left, right] = layer_matrix[right, left] = value
    fig, ax = plt.subplots(figsize=(5.4, 4.6), layout="constrained")
    image = ax.imshow(layer_matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set(xlabel="DSA layer", ylabel="DSA layer")
    fig.colorbar(image, ax=ax, label="Top-k Jaccard")
    save_figure(fig, figure_dir / "cross_layer_topk_similarity_48l")

    usable = segment[segment.segment_type.isin(SEGMENT_ORDER)]
    seg_group = usable.groupby(["segment_type", "layer"]).selection_lift.mean()
    seg_matrix = np.array(
        [
            [seg_group.get((kind, layer_id), np.nan) for layer_id in LAYERS]
            for kind in SEGMENT_ORDER
        ]
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    image = ax.imshow(seg_matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(SEGMENT_ORDER)), SEGMENT_ORDER)
    ax.set(xlabel="DSA layer", ylabel="Semantic segment")
    fig.colorbar(image, ax=ax, label="Selection lift over length baseline")
    save_figure(fig, figure_dir / "semantic_segment_lift_48l")

    stable = cross_round[
        (cross_round.threshold == 0.5)
        & cross_round.segment_type.isin(SEGMENT_ORDER)
    ]
    stable_group = stable.groupby(["segment_type", "layer"]).core_jaccard.mean()
    stable_matrix = np.array(
        [
            [stable_group.get((kind, layer_id), np.nan) for layer_id in LAYERS]
            for kind in SEGMENT_ORDER
        ]
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    image = ax.imshow(
        stable_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1
    )
    ax.set_yticks(range(len(SEGMENT_ORDER)), SEGMENT_ORDER)
    ax.set(xlabel="DSA layer", ylabel="Semantic segment")
    fig.colorbar(image, ax=ax, label="Cross-round stable-core Jaccard")
    save_figure(fig, figure_dir / "cross_round_semantic_core_48l")

    age = stable.dropna(subset=["round_age"])
    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
    for kind, values in age.groupby("segment_type"):
        curve = values.groupby("round_age").core_jaccard.mean()
        ax.plot(curve.index, curve, marker="o", label=kind)
    ax.set(xlabel="Segment age (rounds)", ylabel="Stable-core Jaccard", ylim=(0, 1))
    ax.legend(ncol=3, fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "cross_round_stability_by_age")

    stripe_mean = stripe.groupby("layer").mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained")
    for color, column in zip(
        COLORS,
        ["raw_common_mode_energy_ratio", "centered_top1_evr", "centered_top4_evr"],
    ):
        axes[0].plot(
            stripe_mean.index, stripe_mean[column], label=column, color=color
        )
    axes[0].set(xlabel="DSA layer", ylabel="Energy ratio", ylim=(0, 1))
    axes[0].legend(frameon=False, fontsize=6.5)
    for color, fraction in zip(COLORS, [1, 5, 10]):
        axes[1].plot(
            stripe_mean.index,
            stripe_mean[f"boundary_enrichment_{fraction}pct"],
            label=f"closest {fraction}%",
            color=color,
        )
    axes[1].axhline(1, color="#4D4D4D", linestyle="--")
    axes[1].set(xlabel="DSA layer", ylabel="Churn enrichment at top-k boundary")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "stripe_low_rank_decomposition")

    fig, ax = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    ranked = (
        storage[storage.policy == "previous_score_rank"]
        .groupby("budget")
        .agg(
            candidate_tokens=("candidate_tokens", "mean"),
            topk_recall=("topk_recall", "mean"),
        )
    )
    ax.plot(
        ranked.candidate_tokens,
        ranked.topk_recall,
        marker="o",
        label="previous score rank",
        color=COLORS[0],
    )
    for color, policy in zip(
        COLORS[1:],
        ["previous_topk", "recency_2048", "hybrid_previous_plus_round_core"],
    ):
        values = storage[storage.policy == policy]
        ax.scatter(
            values.candidate_tokens.mean(),
            values.topk_recall.mean(),
            label=policy,
            color=color,
        )
    ax.set(xlabel="Candidate tokens / layer", ylabel="Next-step top-k recall", ylim=(0, 1))
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "storage_candidate_recall_tradeoff")

    # Keep this scalar in scope so static linters catch accidental empty input.
    if adjacent.empty:
        raise ValueError("no adjacent-step metrics")


def plot_same_token_diagnostics(
    tables: dict[str, pd.DataFrame], run_dir: Path, figure_dir: Path
) -> None:
    performance = tables["request_performance"]
    final_rounds = performance.groupby("trajectory_id").round_id.max()
    trajectory = sorted(final_rounds.index)[0]
    round_id = int(final_rounds.loc[trajectory])
    rid = performance[
        (performance.trajectory_id == trajectory)
        & (performance.round_id == round_id)
    ].rid.iloc[0]
    inventory = tables["trace_inventory"]
    file_by_layer = {
        int(row.layer): run_dir / "events" / row.file
        for row in inventory[inventory.rid == rid].itertuples()
    }
    sampled_layers = [0, 7, 15, 23, 31, 39, 45, 47]
    chunks = {layer: load_chunk(file_by_layer[layer]) for layer in sampled_layers}

    fig, axes = plt.subplots(
        2, 4, figsize=(10.4, 5.2), sharey=True, layout="constrained"
    )
    for ax, layer in zip(axes.flat, sampled_layers):
        chunk = chunks[layer]
        prompt_len = int(chunk["score_lens"][0]) - 1
        delta = np.diff(chunk["scores"][:, :prompt_len], axis=0)
        centered = delta - delta.mean()
        scale = max(float(centered.std()), np.finfo(np.float32).eps)
        image = ax.imshow(
            centered / scale,
            aspect="auto",
            cmap="coolwarm",
            vmin=-3,
            vmax=3,
            interpolation="nearest",
            rasterized=True,
        )
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Logical token position")
    axes[0, 0].set_ylabel("Transition t→t+1")
    axes[1, 0].set_ylabel("Transition t→t+1")
    fig.colorbar(image, ax=axes, label="Score-delta z-score")
    fig.suptitle(f"Adjacent-step same-token score change: {trajectory}, round {round_id}")
    save_figure(fig, figure_dir / "adjacent_step_same_token_score_heatmap")

    adjacent_pairs = [(0, 1), (7, 8), (15, 16), (23, 24), (31, 32), (39, 40), (45, 46), (46, 47)]
    needed = sorted({layer for pair in adjacent_pairs for layer in pair})
    pair_chunks = {layer: load_chunk(file_by_layer[layer]) for layer in needed}
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.2), layout="constrained")
    for ax, (left, right) in zip(axes.flat, adjacent_pairs):
        common = min(
            int(pair_chunks[left]["score_lens"][0]),
            int(pair_chunks[right]["score_lens"][0]),
        )
        x = pair_chunks[left]["scores"][0, :common]
        y = pair_chunks[right]["scores"][0, :common]
        artist = ax.hexbin(x, y, gridsize=45, mincnt=1, cmap="viridis")
        ax.set_title(f"L{left}→L{right}, ρ={safe_spearman(x, y):.2f}")
        ax.set_xlabel(f"L{left} score")
        ax.set_ylabel(f"L{right} score")
    fig.colorbar(artist, ax=axes, label="Token count")
    fig.suptitle(f"Adjacent-layer same-token scores: {trajectory}, round {round_id}, step 0")
    save_figure(fig, figure_dir / "adjacent_layer_same_token_scores")


def bootstrap_mean(
    frame: pd.DataFrame, value: str, cluster: str = "trajectory_id"
) -> dict[str, float]:
    grouped = frame.groupby(cluster)[value].mean().dropna()
    rng = np.random.default_rng(20260731)
    samples = np.array(
        [
            rng.choice(grouped.to_numpy(), len(grouped), replace=True).mean()
            for _ in range(2000)
        ]
    )
    return {
        "mean": float(grouped.mean()),
        "cluster_bootstrap_ci95_low": float(np.quantile(samples, 0.025)),
        "cluster_bootstrap_ci95_high": float(np.quantile(samples, 0.975)),
        "trajectory_clusters": int(len(grouped)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    args = parser.parse_args()
    plt.style.use(args.style)
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLORS + ["#56B4E9", "#4D4D4D"])
    analysis_dir = args.run_dir / "analysis"
    table_dir = analysis_dir / "tables"
    figure_dir = analysis_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    tables = analyze(args.run_dir)
    for name, frame in tables.items():
        frame.to_parquet(table_dir / f"{name}.parquet", index=False)
    plot_results(tables, figure_dir)
    plot_same_token_diagnostics(tables, args.run_dir, figure_dir)

    adjacent = tables["step_metrics_by_layer"]
    adjacent = adjacent[adjacent.step_gap == 1]
    stable = tables["cross_round_segment_metrics"]
    stable = stable[stable.threshold == 0.5]
    summary = {
        "request_count": int(len(tables["request_performance"])),
        "trace_chunk_count": int(len(tables["trace_inventory"])),
        "trace_bytes": int(tables["trace_inventory"].bytes.sum()),
        "adjacent_step_topk_recall": bootstrap_mean(adjacent, "topk_recall"),
        "adjacent_step_score_nmae_iqr_sampled": bootstrap_mean(
            adjacent, "score_nmae_iqr_sampled"
        ),
        "cross_round_stable_core_jaccard": bootstrap_mean(
            stable, "core_jaccard"
        ),
        "request_latency_s": {
            "mean": float(tables["request_performance"].latency_s.mean()),
            "p95": float(tables["request_performance"].latency_s.quantile(0.95)),
        },
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    reproducibility = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "run_dir": str(args.run_dir.resolve()),
        "style": str(args.style.resolve()),
        "versions": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "seeds": {"selection_and_bootstrap": 20260731},
        "score_correlation_sampling": "512 evenly spaced token positions",
        "score_iqr_sampling": "512 evenly spaced reference token positions",
        "layer_matrix_sampling": "final request per trajectory; steps 0,16,31",
        "figure_formats": ["PDF", "PNG 300 dpi"],
    }
    (analysis_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
