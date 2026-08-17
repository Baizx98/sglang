#!/usr/bin/env python3
"""Validate compact Keye score-trace schema v5 and full-score shadow rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated ints")
    return values


def validate_chunk(path: Path) -> dict[str, Any]:
    record = torch.load(path, weights_only=False)
    if int(record["schema_version"]) != 5 or record["trace_mode"] != "compact":
        raise ValueError(f"{path}: expected compact schema v5")

    canonical = record["indices"]
    candidate_indices = record["candidate_indices"]
    candidate_scores = record["candidate_scores"]
    score_thresholds = record["score_thresholds"]
    threshold_ranks = record["score_threshold_ranks"].long()
    score_valid_counts = record["score_valid_counts"]
    score_block_counts = record["score_block_counts"]
    block_valid_counts = record["block_valid_counts"]
    block_size = int(record["score_block_size"])
    full_scores = record["scores"]
    topk_backend = str(record.get("topk_backend", "unknown"))

    steps = len(record["decode_step_ids"])
    compact_k = int(record["compact_k"])
    if canonical.shape != (steps, 2048):
        raise ValueError(f"{path}: invalid canonical shape {canonical.shape}")
    if candidate_indices.shape != (steps, compact_k):
        raise ValueError(f"{path}: invalid candidate shape {candidate_indices.shape}")
    if candidate_scores.shape != (steps, compact_k):
        raise ValueError(f"{path}: invalid candidate score shape")
    if score_thresholds.shape != (steps, len(threshold_ranks)):
        raise ValueError(f"{path}: invalid threshold shape")

    max_candidate_score_abs = 0.0
    max_block_mean_abs = 0.0
    max_block_std_abs = 0.0
    canonical_exact_overlaps: list[float] = []
    canonical_candidate_overlaps: list[float] = []
    for row in range(steps):
        valid_tokens = int(score_valid_counts[row])
        actual_k = min(compact_k, valid_tokens)
        valid_indices = candidate_indices[row, :actual_k]
        valid_candidate_scores = candidate_scores[row, :actual_k]
        if len(torch.unique(valid_indices)) != actual_k:
            raise ValueError(f"{path}: duplicate candidate index at row {row}")
        if not bool(
            torch.all((valid_indices >= 0) & (valid_indices < valid_tokens))
        ):
            raise ValueError(f"{path}: candidate index out of range at row {row}")
        if actual_k < compact_k:
            if not bool(torch.all(candidate_indices[row, actual_k:] == -1)):
                raise ValueError(f"{path}: invalid candidate padding at row {row}")
            if not bool(torch.isneginf(candidate_scores[row, actual_k:]).all()):
                raise ValueError(f"{path}: invalid score padding at row {row}")
        if not bool(
            torch.all(valid_candidate_scores[:-1] >= valid_candidate_scores[1:])
        ):
            raise ValueError(f"{path}: candidate scores not sorted at row {row}")
        canonical_set = set(canonical[row][canonical[row] >= 0].tolist())
        exact_set = set(valid_indices[: min(2048, actual_k)].tolist())
        candidate_set = set(valid_indices.tolist())
        canonical_exact_overlaps.append(
            len(canonical_set & exact_set) / len(canonical_set)
        )
        canonical_candidate_overlaps.append(
            len(canonical_set & candidate_set) / len(canonical_set)
        )
        if topk_backend == "torch_exact" and canonical_set != exact_set:
            raise ValueError(
                f"{path}: exact canonical/candidate top-k mismatch at row {row}"
            )

        for threshold_index, rank_tensor in enumerate(threshold_ranks):
            rank = int(rank_tensor)
            threshold = score_thresholds[row, threshold_index]
            if rank <= actual_k:
                if not torch.equal(threshold, valid_candidate_scores[rank - 1]):
                    raise ValueError(f"{path}: threshold mismatch at row {row}")
            elif not bool(torch.isnan(threshold)):
                raise ValueError(f"{path}: missing-rank threshold must be NaN")

        blocks = int(score_block_counts[row])
        if int(block_valid_counts[row, :blocks].sum()) != valid_tokens:
            raise ValueError(f"{path}: block valid counts do not cover score row")
        if blocks != (valid_tokens + block_size - 1) // block_size:
            raise ValueError(f"{path}: incorrect number of score blocks")

        if full_scores is None:
            continue
        scores = full_scores[row, :valid_tokens]
        gathered = scores[valid_indices.long()]
        max_candidate_score_abs = max(
            max_candidate_score_abs,
            float((gathered - valid_candidate_scores).abs().max()),
        )
        for block in range(blocks):
            part = scores[block * block_size : min((block + 1) * block_size, valid_tokens)]
            max_block_mean_abs = max(
                max_block_mean_abs,
                float((record["block_score_mean"][row, block] - part.mean()).abs()),
            )
            if not bool(
                torch.isclose(
                    record["block_score_mean"][row, block],
                    part.mean(),
                    rtol=1e-6,
                    atol=1e-6,
                )
            ):
                raise ValueError(f"{path}: block mean mismatch")
            max_block_std_abs = max(
                max_block_std_abs,
                float(
                    (
                        record["block_score_std"][row, block]
                        - part.std(unbiased=False)
                    ).abs()
                ),
            )
            if not bool(
                torch.isclose(
                    record["block_score_std"][row, block],
                    part.std(unbiased=False),
                    rtol=1e-6,
                    atol=1e-6,
                )
            ):
                raise ValueError(f"{path}: block std mismatch")
            if not torch.equal(record["block_score_min"][row, block], part.min()):
                raise ValueError(f"{path}: block min mismatch")
            if not torch.equal(record["block_score_max"][row, block], part.max()):
                raise ValueError(f"{path}: block max mismatch")
            for threshold_index, threshold in enumerate(score_thresholds[row]):
                expected = int((part >= threshold).sum()) if torch.isfinite(threshold) else 0
                actual = int(
                    record["block_threshold_counts"][row, block, threshold_index]
                )
                if actual != expected:
                    raise ValueError(f"{path}: block threshold count mismatch")

    if full_scores is not None and max_candidate_score_abs != 0.0:
        raise ValueError(f"{path}: full-score shadow mismatch")
    return {
        "file": path.name,
        "layer": int(record["layer_id"]),
        "steps": steps,
        "decode_step_ids": [int(step) for step in record["decode_step_ids"]],
        "compact_k": compact_k,
        "score_threshold_ranks": [int(rank) for rank in threshold_ranks],
        "score_valid_min": int(score_valid_counts.min()),
        "score_valid_max": int(score_valid_counts.max()),
        "bytes": path.stat().st_size,
        "full_scores_retained": full_scores is not None,
        "topk_backend": topk_backend,
        "canonical_exact_overlap_mean": sum(canonical_exact_overlaps)
        / len(canonical_exact_overlaps),
        "canonical_exact_overlap_min": min(canonical_exact_overlaps),
        "canonical_compact_candidate_overlap_min": min(
            canonical_candidate_overlaps
        ),
        "max_candidate_score_abs": max_candidate_score_abs,
        "max_block_mean_abs": max_block_mean_abs,
        "max_block_std_abs": max_block_std_abs,
        "block_reduction_tolerance": {"rtol": 1e-6, "atol": 1e-6},
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-compact-k", type=int)
    parser.add_argument("--expected-threshold-ranks", type=parse_int_list)
    parser.add_argument("--expected-layers", type=parse_int_list)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--expected-requests", type=int)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    events_dir = run_dir / "events"
    manifest = read_jsonl(events_dir / "manifest.jsonl")
    chunks = [validate_chunk(events_dir / row["file"]) for row in manifest]
    failures: list[str] = []
    if args.expected_compact_k is not None:
        bad = [row["file"] for row in chunks if row["compact_k"] != args.expected_compact_k]
        if bad:
            failures.append(
                f"compact_k != {args.expected_compact_k}: {bad[:3]}"
            )
    if args.expected_threshold_ranks is not None:
        bad = [
            row["file"]
            for row in chunks
            if row["score_threshold_ranks"] != args.expected_threshold_ranks
        ]
        if bad:
            failures.append(f"threshold ranks mismatch: {bad[:3]}")
    if args.expected_steps is not None:
        expected_ids = list(range(args.expected_steps))
        bad = [
            row["file"]
            for row in chunks
            if row["decode_step_ids"] != expected_ids
        ]
        if bad:
            failures.append(f"decode steps mismatch: {bad[:3]}")
    observed_layers = sorted({int(row["layer"]) for row in chunks})
    if args.expected_layers is not None and observed_layers != sorted(
        args.expected_layers
    ):
        failures.append(
            f"layers {observed_layers} != {sorted(args.expected_layers)}"
        )
    request_layer_pairs = {
        (str(row["request_id"]), int(row["layer_id"])) for row in manifest
    }
    observed_requests = sorted({request_id for request_id, _ in request_layer_pairs})
    if args.expected_requests is not None and len(observed_requests) != args.expected_requests:
        failures.append(
            f"requests {len(observed_requests)} != {args.expected_requests}"
        )
    if args.expected_layers is not None:
        expected_pairs = {
            (request_id, layer)
            for request_id in observed_requests
            for layer in args.expected_layers
        }
        missing_pairs = expected_pairs - request_layer_pairs
        extra_pairs = request_layer_pairs - expected_pairs
        if missing_pairs or extra_pairs:
            failures.append(
                "request-layer coverage mismatch: "
                f"missing={sorted(missing_pairs)[:3]}, extra={sorted(extra_pairs)[:3]}"
            )
    result = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "manifest_records": len(manifest),
        "validated_chunks": len(chunks),
        "validated_steps": sum(int(row["steps"]) for row in chunks),
        "full_score_chunks": sum(bool(row["full_scores_retained"]) for row in chunks),
        "total_bytes": sum(int(row["bytes"]) for row in chunks),
        "observed_requests": len(observed_requests),
        "observed_layers": observed_layers,
        "contract_failures": failures,
        "all_passed": (
            bool(chunks)
            and all(bool(row["passed"]) for row in chunks)
            and not failures
        ),
        "chunks": chunks,
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    output = args.output or run_dir / "analysis" / "compact-trace-v5-validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
