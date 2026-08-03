#!/usr/bin/env python3
"""Length-aware semantic-segment analysis for Keye DSA traces."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import re
import subprocess
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

LAYERS = list(range(48))
STEPS = 32
WINDOW_SIZES = [16, 32, 64, 128]
BUDGETS = [256, 512, 1024, 1536, 2048]
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_SAMPLES = 2000
KV_BYTES_PER_TOKEN_LAYER_TP2 = 2048
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
METHOD_COLORS = {
    "random": "#9E9E9E",
    "recency": "#4D4D4D",
    "type_only": "#E69F00",
    "relevance_oracle": "#D55E00",
    "previous_frequency": "#009E73",
    "hybrid": "#0072B2",
    "current_oracle": "#CC79A7",
}
METHOD_LABELS = {
    "random": "Random expectation",
    "recency": "Recency",
    "type_only": "Type + age",
    "relevance_oracle": "GT relevance oracle",
    "previous_frequency": "Previous-round frequency",
    "hybrid": "Frequency + type/age",
    "current_oracle": "Current-round oracle",
}
ACTIONABLE_TYPES = [
    "system_instruction",
    "tool_schema",
    "initial_state",
    "long_context_distractor",
    "user_turn",
    "assistant_tool_call",
    "tool_result",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def git_revision(workdir: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workdir, text=True
    ).strip()


def load_chunk(path: Path) -> dict[str, np.ndarray]:
    record = torch.load(path, weights_only=True)
    if record["schema_version"] != 4:
        raise ValueError(f"{path}: expected schema v4")
    if record["decode_step_ids"] != list(range(STEPS)):
        raise ValueError(f"{path}: expected continuous steps 0..31")
    return {
        "indices": record["indices"].numpy(force=True).astype(np.int32),
        "score_lens": record["score_valid_counts"].numpy(force=True).astype(int),
        "valid_counts": record["valid_counts"].numpy(force=True).astype(int),
    }


def build_lookup(
    run_dir: Path, requests: list[dict[str, Any]]
) -> dict[tuple[str, int], Path]:
    expected_rids = {row["rid"] for row in requests}
    lookup: dict[tuple[str, int], Path] = {}
    for row in read_jsonl(run_dir / "events" / "manifest.jsonl"):
        rid = row.get("request_id")
        if rid not in expected_rids:
            continue
        key = (rid, int(row["layer_id"]))
        if key in lookup:
            raise ValueError(f"duplicate chunk {key}")
        lookup[key] = run_dir / "events" / row["file"]
    expected = len(requests) * len(LAYERS)
    if len(lookup) != expected:
        raise ValueError(f"expected {expected} chunks, found {len(lookup)}")
    return lookup


def call_name(call: str) -> str | None:
    match = re.match(r"\s*([A-Za-z_]\w*)\s*\(", str(call))
    return match.group(1) if match else None


def normalize_scalar(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if len(normalized) >= 4 and normalized not in {
            "true",
            "false",
            "none",
            "null",
            "status",
            "success",
            "failed",
        }:
            return normalized
    return None


def collect_scalars(value: Any) -> set[str]:
    output: set[str] = set()
    normalized = normalize_scalar(value)
    if normalized is not None:
        output.add(normalized)
    if isinstance(value, dict):
        for child in value.values():
            output.update(collect_scalars(child))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            output.update(collect_scalars(child))
    return output


def call_scalars(call: str) -> set[str]:
    try:
        tree = ast.parse(str(call), mode="eval")
    except SyntaxError:
        return set()
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            values.update(collect_scalars(node.value))
    return values


def result_scalars(result: str) -> set[str]:
    try:
        return collect_scalars(json.loads(str(result)))
    except (json.JSONDecodeError, TypeError):
        values = set()
        for quoted in re.findall(r"['\"]([^'\"]+)['\"]", str(result)):
            values.update(collect_scalars(quoted))
        for number in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", str(result)):
            values.update(collect_scalars(float(number) if "." in number else int(number)))
        return values


def age_bucket(age: Any) -> str:
    if age is None or (isinstance(age, float) and math.isnan(age)):
        return "persistent"
    value = int(age)
    return str(value) if value <= 3 else "4+"


def length_bin(length: int) -> tuple[int, str]:
    bounds = [(16, "1-16"), (32, "17-32"), (64, "33-64"), (128, "65-128"),
              (256, "129-256"), (512, "257-512")]
    lower = 1
    for index, (upper, label) in enumerate(bounds):
        if lower <= length <= upper:
            return index, label
        lower = upper + 1
    return len(bounds), "513+"


def build_contexts(requests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_trajectory: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        by_trajectory[request["trajectory_id"]].append(request)
    contexts: dict[str, dict[str, Any]] = {}
    for trajectory, rows in by_trajectory.items():
        rows.sort(key=lambda row: int(row["round_id"]))
        names_by_round = [
            {name for call in row["ground_truth_calls"] if (name := call_name(call))}
            for row in rows
        ]
        values_by_round = [
            set().union(*(call_scalars(call) for call in row["ground_truth_calls"]))
            for row in rows
        ]
        call_values_by_round = [
            set().union(*(call_scalars(call) for call in row["ground_truth_calls"]))
            for row in rows
        ]
        result_values_by_round = [
            [result_scalars(result) for result in row["ground_truth_results"]]
            for row in rows
        ]
        for index, row in enumerate(rows):
            contexts[row["rid"]] = {
                "trajectory_id": trajectory,
                "round_id": int(row["round_id"]),
                "current_names": names_by_round[index],
                "prior_names": set().union(*names_by_round[:index]) if index else set(),
                "future_names": set().union(*names_by_round[index + 1 :])
                if index + 1 < len(rows)
                else set(),
                "all_names": set().union(*names_by_round),
                "current_values": values_by_round[index],
                "source_names": names_by_round,
                "source_call_values": call_values_by_round,
                "source_result_values": result_values_by_round,
                "has_current_call": bool(names_by_round[index]),
            }
    return contexts


def classify_segment(segment: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    base = segment["segment_type"]
    age = segment.get("round_age")
    dependency = "not_applicable"
    if base == "tool_schema":
        name = segment.get("tool_name")
        if name in context["current_names"]:
            subtype = "tool_schema::current_target"
        elif name in context["prior_names"]:
            subtype = "tool_schema::prior_used"
        elif name in context["future_names"]:
            subtype = "tool_schema::future_only"
        else:
            subtype = "tool_schema::never_used"
    elif base == "user_turn":
        subtype = "user_turn::current" if int(age) == 0 else "user_turn::history"
    elif base in {"assistant_tool_call", "tool_result"}:
        source_round = int(segment["source_round"])
        source_names = context["source_names"][source_round]
        source_values = set(context["source_call_values"][source_round])
        if base == "tool_result":
            match = re.search(r"::(\d+)$", segment["segment_id"])
            result_index = int(match.group(1)) if match else -1
            result_rows = context["source_result_values"][source_round]
            if 0 <= result_index < len(result_rows):
                source_values.update(result_rows[result_index])
        if context["current_values"] & source_values:
            dependency = "value_dependency"
        elif context["current_names"] & source_names:
            dependency = "same_tool"
        else:
            dependency = "no_detected_dependency"
        subtype = f"{base}::{dependency}"
    else:
        subtype = base
    return {
        "base_type": base,
        "semantic_subtype": subtype,
        "age_bucket": age_bucket(age),
        "dependency_label": dependency,
    }


def segment_windows(start: int, end: int, width: int) -> list[tuple[int, int]]:
    length = end - start
    if length < width:
        return []
    available = max(1, length // width)
    count = min(4, available)
    starts = np.linspace(start, end - width, count, dtype=int)
    return [(int(value), int(value + width)) for value in np.unique(starts)]


def hypergeom_z(selected: np.ndarray, segment_len: int, history_k: np.ndarray, n: int) -> float:
    expected = history_k * segment_len / n
    variance = (
        history_k
        * (segment_len / n)
        * (1.0 - segment_len / n)
        * ((n - history_k) / max(n - 1, 1))
    )
    valid = variance > 1e-12
    if not np.any(valid):
        return math.nan
    return float(np.mean((selected[valid] - expected[valid]) / np.sqrt(variance[valid])))


def metric_block(
    membership: np.ndarray,
    frequency: np.ndarray,
    start: int,
    end: int,
    history_k: np.ndarray,
    prompt_len: int,
) -> dict[str, float]:
    length = end - start
    selected = membership[:, start:end].sum(axis=1).astype(float)
    coverage_step = selected / length
    baseline = history_k / prompt_len
    frequency_slice = frequency[start:end]
    return {
        "selection_coverage": float(coverage_step.mean()),
        "topk_share": float(np.mean(selected / history_k)),
        "mean_selected_tokens": float(selected.mean()),
        "selection_lift": float(np.mean(coverage_step / baseline)),
        "hypergeom_z": hypergeom_z(selected, length, history_k, prompt_len),
        "stable_core_fraction_50": float(np.mean(frequency_slice >= 0.5)),
        "stable_core_fraction_80": float(np.mean(frequency_slice >= 0.8)),
    }


def validate_segments(
    requests: list[dict[str, Any]], segments_by_rid: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    ranges = 0
    for request in requests:
        rid = request["rid"]
        rows = sorted(segments_by_rid[rid], key=lambda row: int(row["token_start"]))
        cursor = 0
        for row in rows:
            if int(row["token_start"]) != cursor:
                raise ValueError(f"segment gap/overlap: {rid}, cursor={cursor}, row={row}")
            cursor = int(row["token_end"])
            ranges += 1
        if cursor != int(request["prompt_len"]):
            raise ValueError(f"segment coverage mismatch: {rid}, {cursor}")
    return {"segment_ranges": ranges, "prompts_with_complete_coverage": len(requests)}


def analyze_segments(
    run_dir: Path,
    requests: list[dict[str, Any]],
    segments_by_rid: dict[str, list[dict[str, Any]]],
    lookup: dict[tuple[str, int], Path],
    contexts: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], dict[str, np.ndarray]], dict[str, Any]]:
    natural_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    history_partition_checks = 0
    semantic_segments_by_rid = {
        rid: [row for row in rows if row["segment_type"] != "chat_template"]
        for rid, rows in segments_by_rid.items()
    }
    prompt_lengths = np.asarray([int(row["prompt_len"]) for row in requests])
    quartiles = np.quantile(prompt_lengths, [0.25, 0.5, 0.75])
    total = len(requests) * len(LAYERS)
    processed = 0
    for request in requests:
        rid = request["rid"]
        prompt_len = int(request["prompt_len"])
        prompt_quartile = int(np.searchsorted(quartiles, prompt_len, side="right"))
        for layer in LAYERS:
            processed += 1
            chunk = load_chunk(lookup[(rid, layer)])
            indices = chunk["indices"]
            membership = np.zeros((STEPS, prompt_len), dtype=np.uint8)
            history_k = np.zeros(STEPS, dtype=np.int32)
            for step in range(STEPS):
                row = indices[step]
                history = row[(row >= 0) & (row < prompt_len)]
                decode = row[(row >= prompt_len) & (row < chunk["score_lens"][step])]
                if len(history) + len(decode) != int(chunk["valid_counts"][step]):
                    raise ValueError(f"top-k partition mismatch: {rid}, L{layer}, S{step}")
                membership[step, history] = 1
                history_k[step] = len(history)
                history_partition_checks += 1
            if np.any(history_k <= 0):
                raise ValueError(f"empty prompt-history top-k: {rid}, L{layer}")
            frequency = membership.mean(axis=0).astype(np.float32)
            cache[(rid, layer)] = {
                "frequency": frequency,
                "step0": membership[0].copy(),
                "history_k": history_k,
            }
            for segment in semantic_segments_by_rid[rid]:
                start, end = int(segment["token_start"]), int(segment["token_end"])
                count = end - start
                bin_id, bin_label = length_bin(count)
                label = classify_segment(segment, contexts[rid])
                common = {
                    "rid": rid,
                    "trajectory_id": request["trajectory_id"],
                    "category": request["category"],
                    "round_id": int(request["round_id"]),
                    "layer": layer,
                    "segment_id": segment["segment_id"],
                    "token_start": start,
                    "token_end": end,
                    "token_count": count,
                    "length_bin_id": bin_id,
                    "length_bin": bin_label,
                    "position_ratio": (start + end) / (2 * prompt_len),
                    "position_decile": min(9, int((start + end) / (2 * prompt_len) * 10)),
                    "prompt_len": prompt_len,
                    "prompt_quartile": prompt_quartile,
                    "source_round": segment.get("source_round"),
                    "round_age": segment.get("round_age"),
                    "tool_name": segment.get("tool_name"),
                    "token_ids_sha256": segment["token_ids_sha256"],
                    **label,
                }
                natural_rows.append(
                    {
                        **common,
                        **metric_block(
                            membership, frequency, start, end, history_k, prompt_len
                        ),
                        "mean_history_k": float(history_k.mean()),
                    }
                )
                for width in WINDOW_SIZES:
                    windows = segment_windows(start, end, width)
                    if not windows:
                        continue
                    metrics = [
                        metric_block(membership, frequency, left, right, history_k, prompt_len)
                        for left, right in windows
                    ]
                    window_rows.append(
                        {
                            **common,
                            "window_size": width,
                            "window_count": len(windows),
                            "window_position_ratio": float(
                                np.mean([(left + right) / (2 * prompt_len) for left, right in windows])
                            ),
                            "window_position_decile": min(
                                9,
                                int(
                                    np.mean(
                                        [(left + right) / (2 * prompt_len) for left, right in windows]
                                    )
                                    * 10
                                ),
                            ),
                            **{
                                key: float(np.mean([row[key] for row in metrics]))
                                for key in metrics[0]
                            },
                        }
                    )
            if processed % 96 == 0 or processed == total:
                print(f"[{processed:04d}/{total}] segment metrics {rid}, L{layer}", flush=True)
    validation = {
        "request_layer_chunks": processed,
        "expected_request_layer_chunks": total,
        "history_partition_checks": history_partition_checks,
        "expected_history_partition_checks": total * STEPS,
        "natural_rows": len(natural_rows),
        "fixed_window_rows": len(window_rows),
    }
    return pd.DataFrame(natural_rows), pd.DataFrame(window_rows), cache, validation


def matched_window_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (_, _, width), group in frame.groupby(["rid", "layer", "window_size"], sort=False):
        records = list(group.itertuples(index=False))
        for target in records:
            if target.base_type not in ACTIONABLE_TYPES:
                continue
            candidates = [
                control
                for control in records
                if control.base_type != target.base_type
                and abs(control.window_position_decile - target.window_position_decile) <= 1
            ]
            if target.age_bucket not in {"persistent", "0"}:
                same_age = [c for c in candidates if c.age_bucket == target.age_bucket]
                if len(same_age) >= 3:
                    candidates = same_age
            candidates.sort(
                key=lambda control: (
                    abs(control.window_position_ratio - target.window_position_ratio),
                    control.segment_id,
                )
            )
            controls = candidates[:10]
            if len(controls) < 3:
                continue
            rows.append(
                {
                    "rid": target.rid,
                    "trajectory_id": target.trajectory_id,
                    "category": target.category,
                    "round_id": target.round_id,
                    "layer": target.layer,
                    "segment_id": target.segment_id,
                    "base_type": target.base_type,
                    "semantic_subtype": target.semantic_subtype,
                    "age_bucket": target.age_bucket,
                    "window_size": target.window_size,
                    "control_count": len(controls),
                    "position_distance_mean": float(
                        np.mean(
                            [
                                abs(c.window_position_ratio - target.window_position_ratio)
                                for c in controls
                            ]
                        )
                    ),
                    "target_coverage": target.selection_coverage,
                    "control_coverage": float(np.mean([c.selection_coverage for c in controls])),
                    "excess_coverage": target.selection_coverage
                    - float(np.mean([c.selection_coverage for c in controls])),
                    "target_lift": target.selection_lift,
                    "control_lift": float(np.mean([c.selection_lift for c in controls])),
                    "excess_lift": target.selection_lift
                    - float(np.mean([c.selection_lift for c in controls])),
                }
            )
    return pd.DataFrame(rows)


def tool_relevance_matches(natural: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tool = natural[natural.base_type == "tool_schema"]
    for (_, _), group in tool.groupby(["rid", "layer"], sort=False):
        targets = group[group.semantic_subtype == "tool_schema::current_target"]
        controls = group[group.semantic_subtype == "tool_schema::never_used"]
        for target in targets.itertuples(index=False):
            candidates = controls[
                (controls.length_bin_id.sub(target.length_bin_id).abs() <= 1)
                & (controls.position_decile.sub(target.position_decile).abs() <= 1)
            ].copy()
            if len(candidates) < 3:
                candidates = controls[
                    controls.length_bin_id.sub(target.length_bin_id).abs() <= 1
                ].copy()
            if len(candidates) < 3:
                continue
            candidates["_distance"] = (
                np.abs(np.log(candidates.token_count / target.token_count))
                + np.abs(candidates.position_ratio - target.position_ratio)
            )
            chosen = candidates.nsmallest(10, "_distance")
            rows.append(
                {
                    "rid": target.rid,
                    "trajectory_id": target.trajectory_id,
                    "category": target.category,
                    "round_id": target.round_id,
                    "layer": target.layer,
                    "segment_id": target.segment_id,
                    "tool_name": target.tool_name,
                    "control_count": len(chosen),
                    "target_token_count": target.token_count,
                    "target_coverage": target.selection_coverage,
                    "control_coverage": float(chosen.selection_coverage.mean()),
                    "excess_coverage": target.selection_coverage
                    - float(chosen.selection_coverage.mean()),
                    "target_lift": target.selection_lift,
                    "control_lift": float(chosen.selection_lift.mean()),
                    "excess_lift": target.selection_lift - float(chosen.selection_lift.mean()),
                }
            )
    return pd.DataFrame(rows)


def add_loo_scores(natural: pd.DataFrame) -> pd.DataFrame:
    frame = natural.copy()
    frame["type_age_group"] = frame.base_type + "::" + frame.age_bucket.astype(str)
    frame["relevance_group"] = frame.semantic_subtype + "::" + frame.age_bucket.astype(str)
    for label, group_column in [
        ("loo_type_score", "type_age_group"),
        ("loo_relevance_score", "relevance_group"),
    ]:
        global_keys = ["layer", group_column]
        trajectory_keys = ["trajectory_id", *global_keys]
        global_sum = frame.groupby(global_keys).selection_coverage.transform("sum")
        global_count = frame.groupby(global_keys).selection_coverage.transform("count")
        trajectory_sum = frame.groupby(trajectory_keys).selection_coverage.transform("sum")
        trajectory_count = frame.groupby(trajectory_keys).selection_coverage.transform("count")
        denominator = global_count - trajectory_count
        loo = (global_sum - trajectory_sum) / denominator.where(denominator > 0)
        fallback_sum = frame.groupby("layer").selection_coverage.transform("sum")
        fallback_count = frame.groupby("layer").selection_coverage.transform("count")
        trajectory_layer_sum = frame.groupby(
            ["trajectory_id", "layer"]
        ).selection_coverage.transform("sum")
        trajectory_layer_count = frame.groupby(
            ["trajectory_id", "layer"]
        ).selection_coverage.transform("count")
        fallback = (fallback_sum - trajectory_layer_sum) / (
            fallback_count - trajectory_layer_count
        )
        frame[label] = loo.fillna(fallback).clip(0.0, 1.0)
    return frame


def select_top(scores: np.ndarray, positions: np.ndarray, budget: int) -> np.ndarray:
    take = min(budget, len(scores))
    order = np.lexsort((positions, scores))
    return order[-take:]


def simulate_candidates(
    *,
    metadata: pd.DataFrame,
    current_frequency: np.ndarray,
    current_step0: np.ndarray,
    positions: np.ndarray,
    previous_frequency: np.ndarray | None,
    phase: str,
    common: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_total = float(current_frequency.sum())
    target_step0 = float(current_step0.sum())
    if target_total <= 0 or len(positions) == 0:
        return rows
    type_scores = metadata.loo_type_score.to_numpy(dtype=float)
    relevance_scores = metadata.loo_relevance_score.to_numpy(dtype=float)
    score_map: dict[str, np.ndarray] = {
        "recency": positions.astype(float),
        "type_only": type_scores,
        "relevance_oracle": relevance_scores,
        "current_oracle": current_frequency,
    }
    if previous_frequency is not None:
        score_map["previous_frequency"] = previous_frequency
        # The deployable hybrid must not use the ground-truth relevance label of
        # the next request. Type/age is available from the rendered prompt,
        # while previous frequency comes only from the completed prior request.
        score_map["hybrid"] = 0.75 * previous_frequency + 0.25 * type_scores
    for budget in BUDGETS:
        take = min(budget, len(positions))
        fraction = take / len(positions)
        random_hits = target_total * fraction
        bytes_value = take * KV_BYTES_PER_TOKEN_LAYER_TP2
        rows.append(
            {
                **common,
                "phase": phase,
                "method": "random",
                "budget": budget,
                "candidate_tokens": take,
                "eligible_tokens": len(positions),
                "mean_topk_recall": fraction,
                "first_step_topk_recall": fraction if target_step0 > 0 else math.nan,
                "candidate_precision": random_hits / take,
                "mean_fallback_tokens": target_total - random_hits,
                "stable_hits_per_mib": random_hits / (bytes_value / 2**20),
                "estimated_kv_bytes_tp2": bytes_value,
            }
        )
        for method, scores in score_map.items():
            chosen = select_top(scores, positions, take)
            hits = float(current_frequency[chosen].sum())
            first_hits = float(current_step0[chosen].sum())
            rows.append(
                {
                    **common,
                    "phase": phase,
                    "method": method,
                    "budget": budget,
                    "candidate_tokens": take,
                    "eligible_tokens": len(positions),
                    "mean_topk_recall": hits / target_total,
                    "first_step_topk_recall": first_hits / target_step0
                    if target_step0 > 0
                    else math.nan,
                    "candidate_precision": hits / take,
                    "mean_fallback_tokens": target_total - hits,
                    "stable_hits_per_mib": hits / (bytes_value / 2**20),
                    "estimated_kv_bytes_tp2": bytes_value,
                }
            )
    return rows


def placement_simulation(
    requests: list[dict[str, Any]],
    segments_by_rid: dict[str, list[dict[str, Any]]],
    natural: pd.DataFrame,
    cache: dict[tuple[str, int], dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    request_by_traj: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        request_by_traj[request["trajectory_id"]].append(request)
    natural_lookup = {
        (row.rid, int(row.layer), row.segment_id): row
        for row in natural.itertuples(index=False)
    }
    semantic_by_rid = {
        rid: [row for row in segments if row["segment_type"] != "chat_template"]
        for rid, segments in segments_by_rid.items()
    }
    for trajectory, trajectory_requests in request_by_traj.items():
        trajectory_requests.sort(key=lambda request: int(request["round_id"]))
        for request_index, request in enumerate(trajectory_requests):
            rid = request["rid"]
            previous_request = trajectory_requests[request_index - 1] if request_index else None
            previous_segments = (
                {row["segment_id"]: row for row in semantic_by_rid[previous_request["rid"]]}
                if previous_request is not None
                else {}
            )
            for layer in LAYERS:
                current_cache = cache[(rid, layer)]
                position_parts: list[np.ndarray] = []
                current_frequency_parts: list[np.ndarray] = []
                current_step0_parts: list[np.ndarray] = []
                previous_frequency_parts: list[np.ndarray] = []
                metadata_rows: list[Any] = []
                for segment in semantic_by_rid[rid]:
                    natural_row = natural_lookup[(rid, layer, segment["segment_id"])]
                    start, end = int(segment["token_start"]), int(segment["token_end"])
                    previous_segment = previous_segments.get(segment["segment_id"])
                    if previous_request is not None:
                        if (
                            previous_segment is None
                            or previous_segment["token_ids_sha256"] != segment["token_ids_sha256"]
                            or int(previous_segment["token_count"]) != int(segment["token_count"])
                        ):
                            continue
                        previous_start = int(previous_segment["token_start"])
                        previous_end = int(previous_segment["token_end"])
                        previous_slice = cache[(previous_request["rid"], layer)]["frequency"][
                            previous_start:previous_end
                        ]
                        previous_frequency_parts.append(previous_slice)
                    position_parts.append(np.arange(start, end, dtype=np.int32))
                    current_frequency_parts.append(current_cache["frequency"][start:end])
                    current_step0_parts.append(current_cache["step0"][start:end])
                    metadata_rows.extend([natural_row] * (end - start))
                if not position_parts:
                    continue
                positions = np.concatenate(position_parts)
                current_frequency = np.concatenate(current_frequency_parts)
                current_step0 = np.concatenate(current_step0_parts)
                metadata = pd.DataFrame(metadata_rows)
                previous_frequency = (
                    np.concatenate(previous_frequency_parts)
                    if previous_request is not None
                    else None
                )
                rows.extend(
                    simulate_candidates(
                        metadata=metadata,
                        current_frequency=current_frequency,
                        current_step0=current_step0,
                        positions=positions,
                        previous_frequency=previous_frequency,
                        phase="continuation" if previous_request is not None else "cold_start",
                        common={
                            "rid": rid,
                            "trajectory_id": trajectory,
                            "category": request["category"],
                            "round_id": int(request["round_id"]),
                            "layer": layer,
                        },
                    )
                )
        print(f"placement simulation {trajectory}", flush=True)
    return pd.DataFrame(rows)


def bootstrap_ci(values: pd.Series, rng: np.random.Generator) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) < 2:
        return math.nan, math.nan
    samples = rng.choice(array, size=(BOOTSTRAP_SAMPLES, len(array)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_metric(
    frame: pd.DataFrame,
    groups: list[str],
    metric: str,
    *,
    minimum_trajectories: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    unit = frame.groupby(["trajectory_id", *groups], dropna=False)[metric].mean().reset_index()
    for keys, values in unit.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        trajectories = values.trajectory_id.nunique()
        ci_low, ci_high = (
            bootstrap_ci(values[metric], rng)
            if trajectories >= max(2, minimum_trajectories)
            else (math.nan, math.nan)
        )
        rows.append(
            {
                **dict(zip(groups, keys)),
                "metric": metric,
                "trajectory_count": int(trajectories),
                "mean": float(values[metric].mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "p10": float(values[metric].quantile(0.10)),
                "p50": float(values[metric].quantile(0.50)),
                "p90": float(values[metric].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def build_summaries(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    natural = tables["segment_selection_length_aware"]
    fixed = tables["fixed_window_selection"]
    matched = tables["matched_window_effects"]
    placement = tables["placement_simulation"]
    length_parts = []
    for metric in ["selection_coverage", "topk_share", "selection_lift"]:
        length_parts.append(
            summarize_metric(
                natural,
                ["base_type", "length_bin_id", "length_bin"],
                metric,
                minimum_trajectories=4,
            )
        )
    fixed_parts = []
    for metric in ["selection_coverage", "selection_lift"]:
        fixed_parts.append(summarize_metric(fixed, ["base_type", "window_size"], metric))
    matched_layer = summarize_metric(
        matched, ["base_type", "layer", "window_size"], "excess_coverage"
    )
    matched_overall = summarize_metric(
        matched, ["base_type", "window_size"], "excess_coverage"
    )
    tool_summary = summarize_metric(
        natural[natural.base_type == "tool_schema"],
        ["semantic_subtype", "layer"],
        "selection_lift",
    )
    history = natural[natural.base_type.isin(["user_turn", "assistant_tool_call", "tool_result"])]
    history_summary = summarize_metric(
        history, ["base_type", "dependency_label", "age_bucket"], "selection_coverage"
    )
    placement_parts = []
    for metric in [
        "mean_topk_recall",
        "first_step_topk_recall",
        "candidate_precision",
        "mean_fallback_tokens",
        "stable_hits_per_mib",
    ]:
        placement_parts.append(
            summarize_metric(placement, ["phase", "method", "budget"], metric)
        )
    return {
        "summary_by_length": pd.concat(length_parts, ignore_index=True),
        "summary_fixed_windows": pd.concat(fixed_parts, ignore_index=True),
        "summary_matched_by_layer": matched_layer,
        "summary_matched_overall": matched_overall,
        "summary_tool_relevance_by_layer": tool_summary,
        "summary_history_age_dependency": history_summary,
        "summary_placement": pd.concat(placement_parts, ignore_index=True),
    }


def save_figure(fig: matplotlib.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_results(summaries: dict[str, pd.DataFrame], figure_dir: Path) -> None:
    length = summaries["summary_by_length"]
    fixed = summaries["summary_fixed_windows"]
    selected_types = ["tool_schema", "user_turn", "assistant_tool_call", "tool_result", "initial_state"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    for color, base_type in zip(COLORS, selected_types):
        values = length[
            (length.base_type == base_type) & (length.metric == "selection_coverage")
        ].sort_values("length_bin_id")
        axes[0].plot(
            values.length_bin_id,
            values["mean"],
            marker="o",
            color=color,
            label=base_type.replace("_", " "),
        )
        axes[0].fill_between(
            values.length_bin_id,
            values.ci95_low,
            values.ci95_high,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        fvalues = fixed[
            (fixed.base_type == base_type) & (fixed.metric == "selection_lift")
        ].sort_values("window_size")
        axes[1].plot(
            fvalues.window_size,
            fvalues["mean"],
            marker="o",
            color=color,
            label=base_type.replace("_", " "),
        )
        axes[1].fill_between(
            fvalues.window_size,
            fvalues.ci95_low,
            fvalues.ci95_high,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    labels = ["1-16", "17-32", "33-64", "65-128", "129-256", "257-512", "513+"]
    axes[0].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[0].set(xlabel="Natural segment length (tokens)", ylabel="Per-token top-k coverage")
    axes[1].axhline(1.0, color="#9E9E9E", linestyle="--", label="Uniform-token baseline")
    axes[1].set(xlabel="Fixed window size (tokens)", ylabel="Selection lift vs. uniform")
    axes[0].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "length_conditioned_selection")

    matched = summaries["summary_matched_by_layer"]
    width = 32
    matrix = np.full((48, len(ACTIONABLE_TYPES)), np.nan)
    for column, base_type in enumerate(ACTIONABLE_TYPES):
        values = matched[(matched.base_type == base_type) & (matched.window_size == width)]
        for row in values.itertuples():
            matrix[int(row.layer), column] = row.mean
    limit = np.nanquantile(np.abs(matrix), 0.98) if np.isfinite(matrix).any() else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), layout="constrained")
    image = axes[0].imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[0].set_xticks(
        range(len(ACTIONABLE_TYPES)),
        ["System", "Schema", "Initial", "Distractor", "User", "Call", "Result"],
        fontsize=7,
    )
    axes[0].set(xlabel="Semantic type (32-token windows)", ylabel="DSA layer")
    fig.colorbar(image, ax=axes[0], label="Matched excess coverage")
    overall = summaries["summary_matched_overall"]
    overall = overall[overall.window_size == width].set_index("base_type").reindex(ACTIONABLE_TYPES)
    y = np.arange(len(ACTIONABLE_TYPES))
    axes[1].errorbar(
        overall["mean"],
        y,
        xerr=np.vstack([overall["mean"] - overall.ci95_low, overall.ci95_high - overall["mean"]]),
        fmt="o",
        color=COLORS[0],
    )
    axes[1].axvline(0, color="#4D4D4D", linestyle="--")
    axes[1].set_yticks(y, [x.replace("_", " ") for x in ACTIONABLE_TYPES])
    axes[1].set(xlabel="Matched excess coverage (95% trajectory CI)")
    axes[1].grid(alpha=0.25)
    save_figure(fig, figure_dir / "matched_semantic_effect_by_layer")

    tool = summaries["summary_tool_relevance_by_layer"]
    tool_order = [
        "tool_schema::current_target",
        "tool_schema::prior_used",
        "tool_schema::future_only",
        "tool_schema::never_used",
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.5), layout="constrained")
    for color, subtype in zip(COLORS, tool_order):
        values = tool[tool.semantic_subtype == subtype].sort_values("layer")
        ax.plot(values.layer, values["mean"], color=color, label=subtype.split("::", 1)[1])
    ax.axhline(1.0, color="#9E9E9E", linestyle="--", label="Uniform-token baseline")
    ax.set(xlabel="DSA layer", ylabel="Tool-schema selection lift")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "tool_schema_relevance")

    history = summaries["summary_history_age_dependency"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
    for ax, base_type in zip(axes, ["assistant_tool_call", "tool_result"]):
        for color, dependency in zip(
            COLORS,
            ["value_dependency", "same_tool", "no_detected_dependency"],
        ):
            values = history[
                (history.base_type == base_type) & (history.dependency_label == dependency)
            ].copy()
            order = {"1": 1, "2": 2, "3": 3, "4+": 4}
            values["age_order"] = values.age_bucket.map(order)
            values = values.sort_values("age_order")
            if not values.empty:
                ax.plot(
                    values.age_order,
                    values["mean"],
                    marker="o",
                    color=color,
                    label=dependency.replace("_", " "),
                )
        ax.set(
            xlabel="Round age (4 means 4+)",
            ylabel="Per-token top-k coverage",
            title=base_type.replace("_", " "),
        )
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    save_figure(fig, figure_dir / "history_age_dependency")

    placement = summaries["summary_placement"]
    placement = placement[
        (placement.phase == "continuation") & (placement.metric.isin(["mean_topk_recall", "stable_hits_per_mib"]))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
    for method in METHOD_LABELS:
        for ax, metric in zip(axes, ["mean_topk_recall", "stable_hits_per_mib"]):
            values = placement[(placement.method == method) & (placement.metric == metric)].sort_values("budget")
            if values.empty:
                continue
            ax.plot(
                values.budget * KV_BYTES_PER_TOKEN_LAYER_TP2 / 2**20,
                values["mean"],
                marker="o",
                color=METHOD_COLORS[method],
                linestyle="--" if "oracle" in method or method == "random" else "-",
                label=METHOD_LABELS[method],
            )
    axes[0].set(xlabel="Candidate KV (MiB/layer, TP2 estimate)", ylabel="Next-round top-k recall")
    axes[1].set(xlabel="Candidate KV (MiB/layer, TP2 estimate)", ylabel="Stable selected tokens per MiB")
    axes[0].legend(frameon=False, fontsize=6.5)
    for ax in axes:
        ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "placement_tradeoff")


def evaluate_gate(tables: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]) -> dict[str, Any]:
    matched = summaries["summary_matched_overall"]
    matched32 = matched[matched.window_size == 32]
    actionable = matched32[(matched32.ci95_low > 0) & (matched32.trajectory_count >= 4)]
    layer_summary = summaries["summary_matched_by_layer"]
    consistent: dict[str, int] = {}
    for base_type in ACTIONABLE_TYPES:
        values = layer_summary[
            (layer_summary.base_type == base_type) & (layer_summary.window_size == 32)
        ]
        consistent[base_type] = int((values["mean"] > 0).sum())
    tool_matches = tables["tool_relevance_matches"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    tool_unit = tool_matches.groupby("trajectory_id").excess_coverage.mean()
    tool_ci = bootstrap_ci(tool_unit, rng)
    placement = summaries["summary_placement"]
    p = placement[
        (placement.phase == "continuation")
        & (placement.metric == "mean_topk_recall")
        & (placement.budget == 1024)
    ].set_index("method")
    recency = float(p.loc["recency", "mean"]) if "recency" in p.index else math.nan
    candidates = {
        method: float(p.loc[method, "mean"])
        for method in ["type_only", "relevance_oracle", "hybrid"]
        if method in p.index
    }
    best_method = max(candidates, key=candidates.get) if candidates else None
    best_gain = candidates[best_method] - recency if best_method else math.nan
    previous_recall = (
        float(p.loc["previous_frequency", "mean"])
        if "previous_frequency" in p.index
        else math.nan
    )
    hybrid_recall = float(p.loc["hybrid", "mean"]) if "hybrid" in p.index else math.nan
    hybrid_gain_vs_previous = hybrid_recall - previous_recall
    deployable = tables["placement_simulation"]
    deployable = deployable[
        (deployable.phase == "continuation")
        & (deployable.budget == 1024)
        & deployable.method.isin(["recency", "hybrid"])
    ]
    category_means = deployable.groupby(["category", "method"]).mean_topk_recall.mean().unstack()
    category_gains = (
        category_means["hybrid"] - category_means["recency"]
        if {"hybrid", "recency"}.issubset(category_means.columns)
        else pd.Series(dtype=float)
    )
    condition1_types = [
        base_type
        for base_type in actionable.base_type.tolist()
        if consistent.get(base_type, 0) >= 36
    ]
    checks = {
        "matched_actionable_type": bool(condition1_types),
        "current_target_tool_positive": bool(tool_ci[0] > 0),
        "placement_gain_at_least_5pp_vs_recency": bool(best_gain >= 0.05),
        "placement_gain_positive_in_at_least_3_categories": bool((category_gains > 0).sum() >= 3),
        "hybrid_beats_previous_frequency": bool(hybrid_gain_vs_previous > 0),
    }
    return {
        "checks": checks,
        "phase_b_recommended": checks["current_target_tool_positive"],
        "online_policy_supported": all(checks.values()),
        "matched_actionable_types": condition1_types,
        "positive_layers_by_type": consistent,
        "tool_target_excess_coverage_mean": float(tool_unit.mean()),
        "tool_target_excess_coverage_ci95": list(tool_ci),
        "placement_budget": 1024,
        "recency_recall": recency,
        "best_semantic_method": best_method,
        "best_semantic_recall": candidates.get(best_method) if best_method else math.nan,
        "best_gain_vs_recency": best_gain,
        "previous_frequency_recall": previous_recall,
        "hybrid_gain_vs_previous_frequency": hybrid_gain_vs_previous,
        "hybrid_gain_vs_recency_by_category": category_gains.to_dict(),
        "oracle_relevance_is_deployable": False,
        "policy_class": "strong" if all(checks.values()) else "weak_hint",
    }


def dependency_audit_rows(
    requests: list[dict[str, Any]],
    segments_by_rid: dict[str, list[dict[str, Any]]],
    contexts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    request_lookup = {row["rid"]: row for row in requests}
    candidates = []
    for rid, segments in segments_by_rid.items():
        for segment in segments:
            if segment["segment_type"] not in {"assistant_tool_call", "tool_result"}:
                continue
            label = classify_segment(segment, contexts[rid])
            source_round = int(segment["source_round"])
            source_request = next(
                row
                for row in requests
                if row["trajectory_id"] == request_lookup[rid]["trajectory_id"]
                and int(row["round_id"]) == source_round
            )
            candidates.append(
                {
                    "rid": rid,
                    "segment_id": segment["segment_id"],
                    "segment_type": segment["segment_type"],
                    "round_age": segment.get("round_age"),
                    "dependency_label": label["dependency_label"],
                    "current_calls": request_lookup[rid]["ground_truth_calls"],
                    "source_calls": source_request["ground_truth_calls"],
                    "source_results": source_request["ground_truth_results"],
                }
            )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    if len(candidates) <= 30:
        return candidates
    indices = rng.choice(len(candidates), size=30, replace=False)
    return [candidates[int(index)] for index in sorted(indices)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-requests", type=int)
    parser.add_argument("--reuse-base-tables", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requests = read_jsonl(args.run_dir / "prepared_requests.jsonl")
    requests.sort(key=lambda row: (row["trajectory_id"], int(row["round_id"])))
    if args.limit_requests is not None:
        requests = requests[: args.limit_requests]
    requested_rids = {row["rid"] for row in requests}
    segments_by_rid: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.run_dir / "segments.jsonl"):
        if row["rid"] in requested_rids:
            segments_by_rid[row["rid"]].append(row)
    output_dir = args.output_dir or args.run_dir / "analysis" / "segment-selection-v02"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stderr(io.StringIO()):
        plt.style.use(args.style)
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLORS + ["#4D4D4D", "#9E9E9E"])

    contexts = build_contexts(requests)
    if args.reuse_base_tables:
        required = [
            "segment_selection_length_aware",
            "fixed_window_selection",
            "matched_window_effects",
            "tool_relevance_matches",
            "placement_simulation",
        ]
        loaded = {
            name: pd.read_parquet(table_dir / f"{name}.parquet")
            for name in required
        }
        natural = loaded["segment_selection_length_aware"]
        fixed = loaded["fixed_window_selection"]
        matched = loaded["matched_window_effects"]
        tool_matches = loaded["tool_relevance_matches"]
        placement = loaded["placement_simulation"]
        validation = json.loads((output_dir / "validation.json").read_text())
        validation["base_tables_reused"] = True
    else:
        validation = validate_segments(requests, segments_by_rid)
        lookup = build_lookup(args.run_dir, requests)
        natural, fixed, cache, trace_validation = analyze_segments(
            args.run_dir, requests, segments_by_rid, lookup, contexts
        )
        validation.update(trace_validation)
        natural = add_loo_scores(natural)
        matched = matched_window_effects(fixed)
        tool_matches = tool_relevance_matches(natural)
        placement = placement_simulation(requests, segments_by_rid, natural, cache)
    tables = {
        "segment_selection_length_aware": natural,
        "fixed_window_selection": fixed,
        "matched_window_effects": matched,
        "tool_relevance_matches": tool_matches,
        "history_dependency_metrics": natural[
            natural.base_type.isin(["user_turn", "assistant_tool_call", "tool_result"])
        ].copy(),
        "placement_simulation": placement,
    }
    summaries = build_summaries(tables)
    tables.update(summaries)
    for name, frame in tables.items():
        frame.to_parquet(table_dir / f"{name}.parquet", index=False)
    plot_results(summaries, figure_dir)
    gate = evaluate_gate(tables, summaries)
    audit = dependency_audit_rows(requests, segments_by_rid, contexts)
    with (output_dir / "dependency_audit_sample.jsonl").open("w", encoding="utf-8") as file:
        for row in audit:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    validation["checks"] = {
        "complete_segment_coverage": validation["prompts_with_complete_coverage"] == len(requests),
        "all_request_layer_chunks": validation["request_layer_chunks"] == len(requests) * 48,
        "all_topk_partitions": validation["history_partition_checks"] == len(requests) * 48 * STEPS,
        "fixed_windows_nonempty": len(fixed) > 0,
        "matched_controls_nonempty": len(matched) > 0,
        "tool_relevance_matches_nonempty": len(tool_matches) > 0,
        "placement_rows_nonempty": len(placement) > 0,
    }
    validation["validated_at"] = datetime.now(timezone.utc).isoformat()
    if not all(validation["checks"].values()):
        raise ValueError(f"validation failed: {validation}")
    summary = {
        "requests": len(requests),
        "trajectories": len({row["trajectory_id"] for row in requests}),
        "layers": 48,
        "steps": STEPS,
        "natural_segment_layer_rows": len(natural),
        "fixed_window_rows": len(fixed),
        "matched_window_rows": len(matched),
        "tool_relevance_match_rows": len(tool_matches),
        "placement_rows": len(placement),
        "phase_a_gate": gate,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n"
    )
    reproducibility = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_commit": git_revision(Path.cwd()),
        "run_dir": str(args.run_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "style": str(args.style.resolve()),
        "versions": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES, "cluster": "trajectory_id"},
        "window_sizes": WINDOW_SIZES,
        "placement_budgets": BUDGETS,
        "kv_bytes_per_token_layer_tp2": KV_BYTES_PER_TOKEN_LAYER_TP2,
        "matching": "same request/layer/window size; position decile +/-1; up to 10 controls",
        "figure_formats": ["PDF", "PNG 300 dpi"],
        "base_tables_reused": args.reuse_base_tables,
    }
    (output_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
