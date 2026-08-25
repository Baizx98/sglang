#!/usr/bin/env python3
"""Analyze exact Keye Top-k traces for the two Figure 3 heatmaps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch


DELTA_STEPS = (0, 1, 2, 4, 8, 16)
DELTA_LAYERS = (0, 1, 2, 4, 8)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_invocation(
    event_dir: Path, manifest_rows: list[dict[str, Any]], request_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted(
        [row for row in manifest_rows if row["request_id"] == request_id],
        key=lambda row: int(row["layer_id"]),
    )
    if [int(row["layer_id"]) for row in rows] != list(range(48)):
        raise ValueError(f"{request_id}: expected exactly layers 0..47")
    indices, valid_counts, score_valid_counts = [], [], []
    for row in rows:
        if row["topk_backend"] != "torch_exact" or int(row["topk_width"]) != 2048:
            raise ValueError(f"{request_id}: non-exact or non-2048 trace: {row}")
        payload = torch.load(event_dir / row["file"], map_location="cpu", weights_only=False)
        if list(payload["decode_step_ids"]) != list(range(20)):
            raise ValueError(f"{request_id} layer {row['layer_id']}: incomplete steps")
        indices.append(payload["indices"].numpy().astype(np.int32, copy=False))
        valid_counts.append(payload["valid_counts"].numpy().astype(np.int32, copy=False))
        score_valid_counts.append(
            payload["score_valid_counts"].numpy().astype(np.int32, copy=False)
        )
    # [step, layer, topk]
    return (
        np.stack(indices, axis=1),
        np.stack(valid_counts, axis=1),
        np.stack(score_valid_counts, axis=1),
    )


def build_masks(indices: np.ndarray, valid_counts: np.ndarray, width: int) -> np.ndarray:
    steps, layers, _ = indices.shape
    masks = np.zeros((steps, layers, width), dtype=np.bool_)
    for step in range(steps):
        for layer in range(layers):
            count = int(valid_counts[step, layer])
            chosen = indices[step, layer, :count]
            if np.any(chosen < 0) or np.any(chosen >= width):
                raise ValueError("Top-k index is outside score-valid range")
            if np.unique(chosen).size != count:
                raise ValueError("Top-k indices are not unique")
            masks[step, layer, chosen] = True
    return masks


def arrow_writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(path, schema, compression="zstd")


def write_batch(writer: pq.ParquetWriter, schema: pa.Schema, rows: list[dict[str, Any]]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def age_bin(age: int) -> str:
    if age <= 3:
        return str(age)
    if age <= 7:
        return "4-7"
    return "8+"


def analyze(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    event_dir = run_dir / "events"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = read_jsonl(run_dir / "prepared_requests.jsonl")
    span_rows = read_jsonl(run_dir / "region_spans.jsonl")
    manifest_rows = read_jsonl(event_dir / "manifest.jsonl")
    responses = read_jsonl(run_dir / "replay_responses.jsonl")
    if len(requests) != len(responses):
        raise ValueError("request/response count mismatch")
    request_ids = {row["rid"] for row in requests}
    manifest_ids = {row["request_id"] for row in manifest_rows}
    if request_ids != manifest_ids:
        raise ValueError("request/manifest ID coverage mismatch")
    if len(manifest_rows) != len(requests) * 48:
        raise ValueError("manifest does not contain one chunk per request/layer")

    request_by_id = {row["rid"]: row for row in requests}
    spans_by_invocation: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in span_rows:
        spans_by_invocation[(row["session_id"], int(row["invocation_id"]))].append(row)

    trace_schema = pa.schema(
        [
            ("session_id", pa.string()),
            ("domain", pa.string()),
            ("turn_id", pa.int16()),
            ("invocation_id", pa.int16()),
            ("step_id", pa.int16()),
            ("layer_id", pa.int16()),
            ("score_valid_tokens", pa.int32()),
            ("topk_token_ids", pa.list_(pa.int32())),
        ]
    )
    pair_schema = pa.schema(
        [
            ("session_id", pa.string()),
            ("domain", pa.string()),
            ("turn_id", pa.int16()),
            ("invocation_id", pa.int16()),
            ("step_id", pa.int16()),
            ("layer_id", pa.int16()),
            ("delta_step", pa.int16()),
            ("delta_layer", pa.int16()),
            ("overlap_count", pa.int16()),
            ("current_topk_count", pa.int16()),
            ("recall", pa.float32()),
        ]
    )
    trace_writer = arrow_writer(output_dir / "topk_trace.parquet", trace_schema)
    pair_writer = arrow_writer(output_dir / "topk_recall_pairs.parquet", pair_schema)
    invocation_recall_rows: list[dict[str, Any]] = []
    region_activation_rows: list[dict[str, Any]] = []
    global_recall: dict[tuple[int, int], list[float]] = defaultdict(list)

    try:
        for request_index, request in enumerate(requests):
            rid = request["rid"]
            indices, valid_counts, score_valid_counts = load_invocation(
                event_dir, manifest_rows, rid
            )
            if indices.shape != (20, 48, 2048):
                raise ValueError(f"{rid}: unexpected trace shape {indices.shape}")
            max_width = int(score_valid_counts.max())
            masks = build_masks(indices, valid_counts, max_width)
            trace_rows = []
            for step in range(20):
                for layer in range(48):
                    count = int(valid_counts[step, layer])
                    trace_rows.append(
                        {
                            "session_id": request["session_id"],
                            "domain": request["domain"],
                            "turn_id": int(request["turn_id"]),
                            "invocation_id": int(request["invocation_id"]),
                            "step_id": step,
                            "layer_id": layer,
                            "score_valid_tokens": int(score_valid_counts[step, layer]),
                            "topk_token_ids": indices[step, layer, :count].tolist(),
                        }
                    )
            write_batch(trace_writer, trace_schema, trace_rows)

            pair_rows = []
            for delta_layer in DELTA_LAYERS:
                for delta_step in DELTA_STEPS:
                    if delta_layer == 0 and delta_step == 0:
                        continue
                    recalls = []
                    for step in range(delta_step, 20):
                        for layer in range(delta_layer, 48):
                            overlap = int(
                                np.logical_and(
                                    masks[step, layer],
                                    masks[step - delta_step, layer - delta_layer],
                                ).sum()
                            )
                            denominator = int(valid_counts[step, layer])
                            recall = overlap / denominator
                            recalls.append(recall)
                            pair_rows.append(
                                {
                                    "session_id": request["session_id"],
                                    "domain": request["domain"],
                                    "turn_id": int(request["turn_id"]),
                                    "invocation_id": int(request["invocation_id"]),
                                    "step_id": step,
                                    "layer_id": layer,
                                    "delta_step": delta_step,
                                    "delta_layer": delta_layer,
                                    "overlap_count": overlap,
                                    "current_topk_count": denominator,
                                    "recall": recall,
                                }
                            )
                    invocation_mean = float(np.mean(recalls))
                    global_recall[(delta_layer, delta_step)].extend(recalls)
                    invocation_recall_rows.append(
                        {
                            "session_id": request["session_id"],
                            "domain": request["domain"],
                            "turn_id": int(request["turn_id"]),
                            "invocation_id": int(request["invocation_id"]),
                            "delta_step": delta_step,
                            "delta_layer": delta_layer,
                            "num_pairs": len(recalls),
                            "mean_recall": invocation_mean,
                        }
                    )
            write_batch(pair_writer, pair_schema, pair_rows)

            invocation_spans = spans_by_invocation[
                (request["session_id"], int(request["invocation_id"]))
            ]
            for span in invocation_spans:
                start, end = int(span["token_start"]), int(span["token_end"])
                num_tokens = int(span["num_tokens"])
                hits = masks[:, :, start:end].sum(axis=2).astype(np.float64)
                baseline = valid_counts.astype(np.float64) / score_valid_counts
                enrichment = (hits / num_tokens) / baseline
                age = int(span["turn_age"])
                region_activation_rows.append(
                    {
                        "session_id": request["session_id"],
                        "domain": request["domain"],
                        "region_id": span["region_id"],
                        "region_type": span["region_type"],
                        "region_label": span["region_label"],
                        "created_turn": int(span["created_turn"]),
                        "current_turn": int(request["turn_id"]),
                        "turn_age": age,
                        "turn_age_bin": age_bin(age),
                        "invocation_id": int(request["invocation_id"]),
                        "num_tokens": num_tokens,
                        "raw_topk_hits": int(hits.sum()),
                        "mean_hits_per_step_layer": float(hits.mean()),
                        "activation_enrichment": float(enrichment.mean()),
                        "num_step_layer_observations": int(hits.size),
                    }
                )
            print(f"[{request_index + 1}/{len(requests)}] analyzed {rid}", flush=True)
    finally:
        trace_writer.close()
        pair_writer.close()

    invocation_df = pd.DataFrame(invocation_recall_rows)
    invocation_df.to_parquet(
        output_dir / "topk_recall_invocation.parquet", index=False, compression="zstd"
    )
    heatmap_rows = []
    for delta_layer in DELTA_LAYERS:
        for delta_step in DELTA_STEPS:
            values = global_recall.get((delta_layer, delta_step), [])
            heatmap_rows.append(
                {
                    "delta_step": delta_step,
                    "delta_layer": delta_layer,
                    "mean_recall": float(np.mean(values)) if values else np.nan,
                    "median_recall": float(np.median(values)) if values else np.nan,
                    "num_pairs": len(values),
                }
            )
    pd.DataFrame(heatmap_rows).to_csv(output_dir / "topk_recall_heatmap.csv", index=False)

    region_activation_df = pd.DataFrame(region_activation_rows)
    region_activation_df.to_parquet(
        output_dir / "region_activation.parquet", index=False, compression="zstd"
    )
    canonical_rows = []
    for region_id, group in pd.DataFrame(span_rows).groupby("region_id", sort=False):
        first = group.sort_values("invocation_id").iloc[0]
        canonical_rows.append(
            {
                "session_id": first["session_id"],
                "domain": first["domain"],
                "region_id": region_id,
                "region_type": first["region_type"],
                "region_label": first["region_label"],
                "created_turn": int(first["created_turn"]),
                "token_start": int(first["token_start"]),
                "token_end": int(first["token_end"]),
                "num_tokens": int(first["num_tokens"]),
                "token_ids_sha256": first["token_ids_sha256"],
            }
        )
    pd.DataFrame(canonical_rows).to_parquet(
        output_dir / "regions.parquet", index=False, compression="zstd"
    )
    summary = {
        "schema_version": 1,
        "request_count": len(requests),
        "session_count": len({row["session_id"] for row in requests}),
        "trace_rows": len(requests) * 20 * 48,
        "manifest_chunks": len(manifest_rows),
        "layers": 48,
        "decode_steps": 20,
        "topk": 2048,
        "topk_backend": "torch_exact",
        "region_instance_count": len(canonical_rows),
        "region_activation_rows": len(region_activation_rows),
        "delta_steps": list(DELTA_STEPS),
        "delta_layers": list(DELTA_LAYERS),
        "prompt_tokens_min": min(int(row["prompt_len"]) for row in requests),
        "prompt_tokens_max": max(int(row["prompt_len"]) for row in requests),
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
