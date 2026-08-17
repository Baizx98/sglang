#!/usr/bin/env python3
"""Summarize same-row exact/candidate attention output fidelity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cosine-gate", type=float, default=0.995)
    parser.add_argument("--layer-cosine-gate", type=float)
    parser.add_argument("--relative-l2-gate", type=float, default=0.10)
    parser.add_argument("--relative-l2-p99-gate", type=float)
    parser.add_argument("--relative-l2-max-gate", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.run_dir / "attention_shadow.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    frame = pd.DataFrame(rows)
    by_layer = (
        frame.groupby("layer_id", sort=True)
        .agg(
            samples=("attention_cosine", "size"),
            cosine_mean=("attention_cosine", "mean"),
            cosine_p01=("attention_cosine", lambda value: value.quantile(0.01)),
            cosine_min=("attention_cosine", "min"),
            relative_l2_mean=("attention_relative_l2", "mean"),
            relative_l2_p99=(
                "attention_relative_l2", lambda value: value.quantile(0.99)
            ),
            relative_l2_max=("attention_relative_l2", "max"),
        )
        .reset_index()
    )
    layer_cosine_gate = (
        args.cosine_gate
        if args.layer_cosine_gate is None
        else args.layer_cosine_gate
    )
    gates = {
        "mean_cosine_at_least_threshold": float(frame.attention_cosine.mean())
        >= args.cosine_gate,
        "layer_mean_cosine_at_least_threshold": bool(
            (by_layer.cosine_mean >= layer_cosine_gate).all()
        ),
        "mean_relative_l2_at_most_threshold": float(
            frame.attention_relative_l2.mean()
        )
        <= args.relative_l2_gate,
    }
    if args.relative_l2_p99_gate is not None:
        gates["p99_relative_l2_at_most_threshold"] = float(
            frame.attention_relative_l2.quantile(0.99)
        ) <= args.relative_l2_p99_gate
    if args.relative_l2_max_gate is not None:
        gates["max_relative_l2_at_most_threshold"] = float(
            frame.attention_relative_l2.max()
        ) <= args.relative_l2_max_gate

    summary = {
        "schema_version": 1,
        "request_count": int(frame.request_id.nunique()),
        "layer_count": int(frame.layer_id.nunique()),
        "sample_count": len(frame),
        "attention_cosine_mean": float(frame.attention_cosine.mean()),
        "attention_cosine_p01": float(frame.attention_cosine.quantile(0.01)),
        "attention_cosine_min": float(frame.attention_cosine.min()),
        "attention_relative_l2_mean": float(frame.attention_relative_l2.mean()),
        "attention_relative_l2_p99": float(
            frame.attention_relative_l2.quantile(0.99)
        ),
        "attention_relative_l2_max": float(frame.attention_relative_l2.max()),
        "thresholds": {
            "mean_cosine": args.cosine_gate,
            "layer_mean_cosine": layer_cosine_gate,
            "mean_relative_l2": args.relative_l2_gate,
            "p99_relative_l2": args.relative_l2_p99_gate,
            "max_relative_l2": args.relative_l2_max_gate,
        },
        "gates": gates,
    }
    summary["quality_gate_pass"] = all(summary["gates"].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "attention_shadow.parquet", index=False)
    by_layer.to_csv(args.output_dir / "attention_shadow_by_layer.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
