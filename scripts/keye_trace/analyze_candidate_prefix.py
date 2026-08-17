#!/usr/bin/env python3
"""Evaluate a smaller prefix of previously collected lookahead candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=2560)
    return parser.parse_args()


def valid_set(values: torch.Tensor) -> set[int]:
    return set(int(value) for value in values[values >= 0].tolist())


def main() -> None:
    args = parse_args()
    if args.candidate_k < 2048:
        raise ValueError("candidate-k must be at least the final top-k size 2048")

    rows: list[dict[str, object]] = []
    source_candidate_k_values: set[int] = set()
    for path in sorted(args.events_dir.resolve().glob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        source_candidate_k = int(record["candidate_k"])
        source_candidate_k_values.add(source_candidate_k)
        if args.candidate_k > source_candidate_k:
            raise ValueError(
                f"requested K={args.candidate_k} exceeds source K={source_candidate_k} "
                f"in {path.name}"
            )

        candidate_indices = record["candidate_indices"][:, : args.candidate_k]
        for row_index, decode_step in enumerate(record["decode_step_ids"]):
            candidate = valid_set(candidate_indices[row_index])
            exact = valid_set(record["exact_indices"][row_index])
            hits = len(candidate & exact)
            denominator = max(len(exact), 1)
            recall = hits / denominator
            rows.append(
                {
                    "request_id": str(record["request_id"]),
                    "target_layer": int(record["target_layer_id"]),
                    "decode_step": int(decode_step),
                    "valid_count": int(record["valid_counts"][row_index]),
                    "source_candidate_k": source_candidate_k,
                    "candidate_k": args.candidate_k,
                    "candidate_valid": len(candidate),
                    "exact_valid": len(exact),
                    "exact_hits": hits,
                    "recall": recall,
                    "miss_tokens": len(exact) - hits,
                    "exact_containment": hits == len(exact),
                    "source_file": path.name,
                }
            )

    if not rows:
        raise FileNotFoundError(f"no .pt events found in {args.events_dir}")

    frame = pd.DataFrame(rows)
    by_layer = (
        frame.groupby("target_layer", sort=True)
        .agg(
            samples=("recall", "size"),
            recall_mean=("recall", "mean"),
            recall_p01=("recall", lambda values: values.quantile(0.01)),
            recall_p05=("recall", lambda values: values.quantile(0.05)),
            recall_min=("recall", "min"),
            miss_tokens_mean=("miss_tokens", "mean"),
            miss_tokens_p99=("miss_tokens", lambda values: values.quantile(0.99)),
            exact_containment_rate=("exact_containment", "mean"),
        )
        .reset_index()
    )
    request_layer = frame.groupby(["request_id", "target_layer"])["recall"].mean()
    summary = {
        "schema_version": 1,
        "events_dir": str(args.events_dir.resolve()),
        "source_candidate_k_values": sorted(source_candidate_k_values),
        "candidate_k": args.candidate_k,
        "request_count": int(frame.request_id.nunique()),
        "layer_count": int(frame.target_layer.nunique()),
        "sample_count": len(frame),
        "recall_mean": float(frame.recall.mean()),
        "recall_p01": float(frame.recall.quantile(0.01)),
        "recall_p05": float(frame.recall.quantile(0.05)),
        "recall_min": float(frame.recall.min()),
        "request_layer_recall_p10": float(request_layer.quantile(0.10)),
        "miss_tokens_mean": float(frame.miss_tokens.mean()),
        "miss_tokens_p99": float(frame.miss_tokens.quantile(0.99)),
        "exact_containment_rate": float(frame.exact_containment.mean()),
        "worst_layers_by_mean_recall": [
            {
                "target_layer": int(row.target_layer),
                "recall_mean": float(row.recall_mean),
                "recall_p01": float(row.recall_p01),
                "miss_tokens_mean": float(row.miss_tokens_mean),
            }
            for row in by_layer.nsmallest(10, "recall_mean").itertuples()
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "step_metrics.csv", index=False)
    frame.to_parquet(args.output_dir / "step_metrics.parquet", index=False)
    by_layer.to_csv(args.output_dir / "layer_summary.csv", index=False)
    by_layer.to_parquet(args.output_dir / "layer_summary.parquet", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
