#!/usr/bin/env python3
"""Analyze synchronous lookahead-candidate restricted-rescoring traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def token_set(row: torch.Tensor) -> set[int]:
    return set(int(value) for value in row[row >= 0].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--requests-file", type=Path)
    return parser.parse_args()


def request_metadata(
    run_dir: Path, requests_file: Path | None = None
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    requests_path = requests_file or run_dir / "requests.jsonl"
    if requests_path.exists():
        for row in read_jsonl(requests_path):
            metadata[row.get("rid", row.get("prepared_rid", ""))] = row
    selection_path = run_dir / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        selected = {row["rid"]: row for row in selection.get("selected", [])}
        for actual_rid, row in metadata.items():
            prepared_rid = row.get("prepared_rid", actual_rid)
            row.update(selected.get(prepared_rid, {}))
        for rid, row in selected.items():
            metadata.setdefault(rid, row)
    return metadata


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    events_dir = (args.events_dir or run_dir / "events").resolve()
    output_dir = (args.output_dir or run_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = request_metadata(run_dir, args.requests_file)

    rows: list[dict[str, Any]] = []
    for path in sorted(events_dir.glob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        request_id = str(record["request_id"])
        meta = metadata.get(request_id, {})
        candidate_k = int(record["candidate_k"])
        for row_index, decode_step in enumerate(record["decode_step_ids"]):
            candidate = token_set(record["candidate_indices"][row_index])
            final = token_set(record["final_indices"][row_index])
            exact = token_set(record["exact_indices"][row_index])
            denominator = max(len(exact), 1)
            candidate_hits = len(candidate & exact)
            final_hits = len(final & exact)
            union = len(final | exact)
            coverage = candidate_hits / denominator
            final_recall = final_hits / denominator
            rows.append(
                {
                    "request_id": request_id,
                    "prepared_rid": meta.get("prepared_rid", meta.get("rid")),
                    "trajectory_id": meta.get("trajectory_id"),
                    "category": meta.get("category"),
                    "split": meta.get("split"),
                    "prompt_len": meta.get("prompt_len"),
                    "target_layer": int(record["target_layer_id"]),
                    "decode_step": int(decode_step),
                    "valid_count": int(record["valid_counts"][row_index]),
                    "candidate_k": candidate_k,
                    "candidate_valid": len(candidate),
                    "final_valid": len(final),
                    "exact_valid": len(exact),
                    "candidate_coverage": coverage,
                    "final_recall": final_recall,
                    "final_jaccard": final_hits / union if union else 1.0,
                    "miss_tokens": len(exact) - final_hits,
                    "exact_containment": candidate_hits == len(exact),
                    "coverage_recall_abs": abs(coverage - final_recall),
                    "candidate_score_max_abs": float(
                        record["candidate_score_max_abs"][row_index]
                    ),
                    "source_file": path.name,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No rescore trace chunks found in {events_dir}")

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "pair_step_metrics.csv", index=False)
    frame.to_parquet(output_dir / "pair_step_metrics.parquet", index=False)
    layer = (
        frame.groupby(["split", "target_layer"], dropna=False)
        .agg(
            pair_steps=("final_recall", "size"),
            candidate_coverage_mean=("candidate_coverage", "mean"),
            final_recall_mean=("final_recall", "mean"),
            final_recall_p01=("final_recall", lambda values: values.quantile(0.01)),
            final_recall_p10=("final_recall", lambda values: values.quantile(0.10)),
            miss_tokens_mean=("miss_tokens", "mean"),
            exact_containment_rate=("exact_containment", "mean"),
            final_jaccard_mean=("final_jaccard", "mean"),
            max_score_abs=("candidate_score_max_abs", "max"),
        )
        .reset_index()
    )
    layer.to_csv(output_dir / "layer_summary.csv", index=False)
    layer.to_parquet(output_dir / "layer_summary.parquet", index=False)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "events_dir": str(events_dir),
        "candidate_k_values": sorted(int(value) for value in frame.candidate_k.unique()),
        "request_count": int(frame.request_id.nunique()),
        "layer_count": int(frame.target_layer.nunique()),
        "pair_step_count": len(frame),
        "max_candidate_score_abs": float(frame.candidate_score_max_abs.max()),
        "max_coverage_recall_abs": float(frame.coverage_recall_abs.max()),
        "coverage_recall_mismatch_count": int(
            (frame.coverage_recall_abs > 0).sum()
        ),
        "invalid_final_width_count": int((frame.final_valid != frame.exact_valid).sum()),
        "splits": {},
    }
    groups = frame.groupby("split", dropna=False)
    for split, group in groups:
        request_layer = group.groupby(["request_id", "target_layer"])[
            "final_recall"
        ].mean()
        name = "unknown" if pd.isna(split) else str(split)
        summary["splits"][name] = {
            "pair_step_count": len(group),
            "request_count": int(group.request_id.nunique()),
            "final_recall_mean": float(group.final_recall.mean()),
            "final_recall_step_p01": float(group.final_recall.quantile(0.01)),
            "final_recall_step_p05": float(group.final_recall.quantile(0.05)),
            "final_recall_step_p10": float(group.final_recall.quantile(0.10)),
            "final_recall_request_layer_p10": float(request_layer.quantile(0.10)),
            "miss_tokens_mean": float(group.miss_tokens.mean()),
            "exact_containment_rate": float(group.exact_containment.mean()),
            "final_jaccard_mean": float(group.final_jaccard.mean()),
        }
    # Equal-score tokens at the top-k boundary can be resolved differently when
    # the same scores are presented in logical-token order versus candidate order.
    # Permit at most one exchanged token while still requiring exact score equality.
    summary["tie_tolerant_property_gate"] = (
        summary["max_candidate_score_abs"] == 0.0
        and summary["max_coverage_recall_abs"] <= 1 / 2048
        and summary["invalid_final_width_count"] == 0
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    if not summary["tie_tolerant_property_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
