#!/usr/bin/env python3
"""Quantify how accurately seven sampled DSA layers represent all 48 layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

DEFAULT_SAMPLED_LAYERS = [0, 7, 15, 23, 31, 39, 47]
METRICS = [
    "previous_top2048_token_recall",
    "previous_top4096_token_recall",
    "previous_top2048_page_recall",
    "previous_top4096_page_recall",
    "top4096_page_amplification",
]
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_SAMPLES = 2000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def request_metadata(run_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for prepared in sorted(run_dir.glob("*/prepared_requests.jsonl")):
        group = prepared.parent.name
        dataset, length = group.rsplit("-", 1)
        for row in read_jsonl(prepared):
            rid = str(row["rid"])
            if rid in result:
                raise ValueError(f"duplicate request ID across prepared files: {rid}")
            result[rid] = {
                "dataset": dataset,
                "length_config": int(length),
                "task": str(row["task"]),
                "prompt_len": int(row["prompt_len"]),
            }
    if not result:
        raise ValueError(f"no prepared requests below {run_dir}")
    return result


def pages(indices: np.ndarray, page_size: int) -> np.ndarray:
    indices = indices[indices >= 0]
    return np.unique(indices // page_size)


def ratio_intersection(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.intersect1d(left, right, assume_unique=True).size / right.size)


def analyze_chunk(path: Path, page_size: int) -> list[dict[str, Any]]:
    record = torch.load(path, weights_only=False)
    if int(record["schema_version"]) != 5 or record["trace_mode"] != "compact":
        raise ValueError(f"{path}: expected compact schema v5")
    if int(record["compact_k"]) != 4096:
        raise ValueError(f"{path}: expected compact K=4096")
    steps = [int(value) for value in record["decode_step_ids"]]
    if steps != list(range(32)):
        raise ValueError(f"{path}: expected decode steps 0..31")
    canonical = record["indices"].numpy(force=True).astype(np.int32)
    candidates = record["candidate_indices"].numpy(force=True).astype(np.int32)
    valid_counts = record["score_valid_counts"].numpy(force=True).astype(np.int32)
    rows: list[dict[str, Any]] = []
    for step in range(1, 32):
        common = min(int(valid_counts[step - 1]), int(valid_counts[step]))
        target = np.unique(canonical[step][(canonical[step] >= 0) & (canonical[step] < common)])
        previous_2048 = np.unique(canonical[step - 1][(canonical[step - 1] >= 0) & (canonical[step - 1] < common)])
        previous_4096 = np.unique(candidates[step - 1][(candidates[step - 1] >= 0) & (candidates[step - 1] < common)])
        target_pages = pages(target, page_size)
        pages_2048 = pages(previous_2048, page_size)
        pages_4096 = pages(previous_4096, page_size)
        rows.append(
            {
                "step": step,
                "previous_top2048_token_recall": ratio_intersection(previous_2048, target),
                "previous_top4096_token_recall": ratio_intersection(previous_4096, target),
                "previous_top2048_page_recall": ratio_intersection(pages_2048, target_pages),
                "previous_top4096_page_recall": ratio_intersection(pages_4096, target_pages),
                "top4096_page_amplification": float(len(pages_4096) / len(target_pages)),
            }
        )
    return rows


def nearest_sample_weights(sampled_layers: list[int]) -> dict[int, int]:
    assignment = {
        layer: min(sampled_layers, key=lambda sample: (abs(sample - layer), sample))
        for layer in range(48)
    }
    return {sample: list(assignment.values()).count(sample) for sample in sampled_layers}


def summarize_request(
    layer_rows: pd.DataFrame, sampled_layers: list[int]
) -> pd.DataFrame:
    layer_means = layer_rows.groupby(
        ["rid", "dataset", "length_config", "task", "prompt_len", "layer"],
        as_index=False,
    )[METRICS].mean()
    weights = nearest_sample_weights(sampled_layers)
    rows: list[dict[str, Any]] = []
    keys = ["rid", "dataset", "length_config", "task", "prompt_len"]
    for values, part in layer_means.groupby(keys, sort=True):
        if sorted(part.layer.tolist()) != list(range(48)):
            raise ValueError(f"{values[0]}: incomplete 48-layer coverage")
        sampled = part[part.layer.isin(sampled_layers)].copy()
        sampled["nearest_weight"] = sampled.layer.map(weights)
        for metric in METRICS:
            true = float(part[metric].mean())
            naive = float(sampled[metric].mean())
            weighted = float(np.average(sampled[metric], weights=sampled.nearest_weight))
            base = dict(zip(keys, values))
            rows.append({**base, "metric": metric, "estimator": "seven_layer_unweighted", "full48": true, "estimate": naive, "error": naive - true, "absolute_error": abs(naive - true)})
            rows.append({**base, "metric": metric, "estimator": "seven_layer_nearest_weighted", "full48": true, "estimate": weighted, "error": weighted - true, "absolute_error": abs(weighted - true)})
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def aggregate(request_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in request_rows.groupby(["dataset", "length_config", "metric", "estimator"], sort=True):
        low, high = bootstrap_ci(part.error.to_numpy(float))
        rows.append(
            {
                "dataset": keys[0],
                "length_config": keys[1],
                "metric": keys[2],
                "estimator": keys[3],
                "requests": len(part),
                "full48_mean": float(part.full48.mean()),
                "estimate_mean": float(part.estimate.mean()),
                "bias_mean": float(part.error.mean()),
                "bias_ci95_low": low,
                "bias_ci95_high": high,
                "request_mae": float(part.absolute_error.mean()),
                "request_abs_error_p95": float(part.absolute_error.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument(
        "--sampled-layers",
        default=",".join(map(str, DEFAULT_SAMPLED_LAYERS)),
        help="comma-separated sampled layers whose estimator is evaluated",
    )
    args = parser.parse_args()
    sampled_layers = sorted(
        {int(value.strip()) for value in args.sampled_layers.split(",") if value.strip()}
    )
    if not sampled_layers or sampled_layers[0] < 0 or sampled_layers[-1] >= 48:
        raise ValueError("--sampled-layers must contain unique layers in [0, 47]")
    metadata = request_metadata(args.run_dir)
    manifest = read_jsonl(args.run_dir / "events" / "manifest.jsonl")
    lookup: dict[tuple[str, int], Path] = {}
    for row in manifest:
        rid = str(row["request_id"])
        if rid not in metadata:
            raise ValueError(f"trace request absent from prepared inputs: {rid}")
        key = (rid, int(row["layer_id"]))
        if key in lookup:
            raise ValueError(f"duplicate chunk: {key}")
        lookup[key] = args.run_dir / "events" / row["file"]
    expected = {(rid, layer) for rid in metadata for layer in range(48)}
    if set(lookup) != expected:
        raise ValueError(f"request-layer coverage mismatch: missing={len(expected-set(lookup))}, extra={len(set(lookup)-expected)}")

    rows: list[dict[str, Any]] = []
    for chunk_index, ((rid, layer), path) in enumerate(sorted(lookup.items()), start=1):
        for row in analyze_chunk(path, args.page_size):
            rows.append({"rid": rid, "layer": layer, **metadata[rid], **row})
        if chunk_index % 96 == 0:
            print(f"[{chunk_index}/{len(lookup)}] chunks", flush=True)
    layer_rows = pd.DataFrame(rows)
    request_rows = summarize_request(layer_rows, sampled_layers)
    summary_rows = aggregate(request_rows)
    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    layer_rows.to_parquet(table_dir / "by_request_layer_step.parquet", index=False)
    request_rows.to_csv(table_dir / "sampling_error_by_request.csv", index=False)
    summary_rows.to_csv(table_dir / "sampling_error_summary.csv", index=False)
    write_json(
        args.output_dir / "summary.json",
        {
            "schema_version": 1,
            "requests": len(metadata),
            "chunks": len(lookup),
            "transitions": len(layer_rows),
            "page_size": args.page_size,
            "sampled_layers": sampled_layers,
            "nearest_layer_weights": nearest_sample_weights(sampled_layers),
            "metrics": METRICS,
            "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES, "cluster": "request"},
            "all_contracts_passed": len(lookup) == len(metadata) * 48 and len(layer_rows) == len(metadata) * 48 * 31,
        },
    )


if __name__ == "__main__":
    main()
