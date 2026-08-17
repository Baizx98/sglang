#!/usr/bin/env python3
"""Audit serving top-2048 against compact trace's exact FP32 ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit_chunk(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    record = torch.load(path, map_location="cpu", weights_only=False)
    if int(record["schema_version"]) != 5:
        raise ValueError(f"{path}: expected schema v5")
    canonical = record["indices"].long()
    candidates = record["candidate_indices"].long()
    candidate_scores = record["candidate_scores"].float()
    valid_counts = record["score_valid_counts"].long()
    rows: list[dict[str, Any]] = []
    for row_index, step in enumerate(record["decode_step_ids"]):
        canonical_row = canonical[row_index]
        canonical_row = canonical_row[canonical_row >= 0]
        candidate_row = candidates[row_index]
        candidate_row = candidate_row[candidate_row >= 0]
        if len(torch.unique(canonical_row)) != len(canonical_row):
            raise ValueError(f"{path}: duplicate canonical index at row {row_index}")
        if not bool(torch.all(canonical_row < valid_counts[row_index])):
            raise ValueError(f"{path}: canonical index out of range at row {row_index}")

        canonical_set = set(canonical_row.tolist())
        exact_set = set(candidate_row[:2048].tolist())
        candidate_set = set(candidate_row.tolist())
        missing_from_candidates = canonical_set - candidate_set
        rank = {token: index + 1 for index, token in enumerate(candidate_row.tolist())}
        selected_ranks = [rank[token] for token in canonical_set if token in rank]

        score_by_token = {
            int(token): float(score)
            for token, score in zip(
                candidate_row.tolist(),
                candidate_scores[row_index, : len(candidate_row)].tolist(),
            )
        }
        exact_score_sum = float(candidate_scores[row_index, :2048].sum())
        canonical_score_sum = sum(
            score_by_token[token]
            for token in canonical_set
            if token in score_by_token
        )
        overlap = len(canonical_set & exact_set)
        rows.append(
            {
                "file": path.name,
                "layer": int(record["layer_id"]),
                "step": int(step),
                "score_valid_count": int(valid_counts[row_index]),
                "canonical_count": len(canonical_set),
                "exact_top2048_overlap": overlap / len(canonical_set),
                "boundary_replacements": len(canonical_set) - overlap,
                "contained_by_exact_top4096": not missing_from_candidates,
                "missing_from_exact_top4096": len(missing_from_candidates),
                "worst_exact_rank_selected": (
                    max(selected_ranks) if selected_ranks else None
                ),
                "relative_selected_score_loss": (
                    (exact_score_sum - canonical_score_sum) / abs(exact_score_sum)
                    if exact_score_sum
                    else 0.0
                ),
            }
        )
    return rows, {
        "file": path.name,
        "layer": int(record["layer_id"]),
        "topk_backend": str(record.get("topk_backend", "unknown")),
        "steps": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    events_dir = run_dir / "events"
    manifest = read_jsonl(events_dir / "manifest.jsonl")
    all_rows: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for entry in manifest:
        rows, chunk = audit_chunk(events_dir / entry["file"])
        all_rows.extend(rows)
        chunks.append(chunk)

    layer_summaries = []
    for layer in sorted({int(row["layer"]) for row in all_rows}):
        selected = [row for row in all_rows if int(row["layer"]) == layer]
        layer_summaries.append(
            {
                "layer": layer,
                "steps": len(selected),
                "exact_top2048_overlap_mean": sum(
                    float(row["exact_top2048_overlap"]) for row in selected
                )
                / len(selected),
                "exact_top2048_overlap_min": min(
                    float(row["exact_top2048_overlap"]) for row in selected
                ),
                "boundary_replacements_max": max(
                    int(row["boundary_replacements"]) for row in selected
                ),
                "worst_exact_rank_selected_max": max(
                    int(row["worst_exact_rank_selected"] or 0) for row in selected
                ),
                "relative_selected_score_loss_mean": sum(
                    float(row["relative_selected_score_loss"]) for row in selected
                )
                / len(selected),
                "relative_selected_score_loss_max": max(
                    float(row["relative_selected_score_loss"]) for row in selected
                ),
                "all_contained_by_exact_top4096": all(
                    bool(row["contained_by_exact_top4096"]) for row in selected
                ),
            }
        )

    result = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "chunks": chunks,
        "rows": len(all_rows),
        "layers": layer_summaries,
        "all_canonical_indices_contained_by_exact_top4096": all(
            bool(row["contained_by_exact_top4096"]) for row in all_rows
        ),
    }
    output_dir = args.output_dir or run_dir / "analysis" / "topk-kernel-audit-v01"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    payload = json.dumps(result, indent=2)
    (output_dir / "summary.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
