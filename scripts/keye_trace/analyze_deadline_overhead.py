#!/usr/bin/env python3
"""Paired client-streaming overhead audit for CUDA deadline instrumentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def request_metrics(row: dict[str, Any]) -> dict[str, Any]:
    arrivals = row.get("stream_arrivals") or []
    if len(arrivals) < 2:
        raise ValueError(f"{row['rid']}: at least two streaming arrivals are required")
    output_tokens = int(row["output_tokens"])
    first_tokens = int(arrivals[0]["completion_tokens"])
    decode_tokens = output_tokens - first_tokens
    if decode_tokens <= 0:
        raise ValueError(f"{row['rid']}: no post-first-token decode tokens")
    decode_duration_s = float(arrivals[-1]["arrival_s"]) - float(
        arrivals[0]["arrival_s"]
    )
    per_token_gaps = []
    for arrival in arrivals[1:]:
        delta_tokens = int(arrival["delta_tokens"])
        if delta_tokens > 0:
            per_token_gaps.extend(
                [float(arrival["gap_s"]) / delta_tokens] * delta_tokens
            )
    if len(per_token_gaps) != decode_tokens:
        raise ValueError(
            f"{row['rid']}: reconstructed {len(per_token_gaps)} gaps for "
            f"{decode_tokens} decode tokens"
        )
    response = row.get("response") or {}
    return {
        "request_index": int(row["request_index"]),
        "repetition": int(row["repetition"]),
        "output_tokens": output_tokens,
        "output_ids": json.dumps(response.get("output_ids") or []),
        "ttft_ms": float(arrivals[0]["arrival_s"]) * 1000,
        "mean_tpot_ms": decode_duration_s / decode_tokens * 1000,
        "median_itl_ms": float(np.median(per_token_gaps)) * 1000,
        "e2e_ms": float(row["latency_s"]) * 1000,
        "structural_pass": bool(row["structural_pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--instrumented", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate", type=float, default=0.03)
    args = parser.parse_args()
    if args.gate < 0:
        raise ValueError("--gate must be non-negative")

    baseline = pd.DataFrame(request_metrics(row) for row in read_jsonl(args.baseline))
    instrumented = pd.DataFrame(
        request_metrics(row) for row in read_jsonl(args.instrumented)
    )
    paired = baseline.merge(
        instrumented,
        on=["request_index", "repetition"],
        how="outer",
        suffixes=("_baseline", "_instrumented"),
        validate="one_to_one",
        indicator=True,
    )
    if not (paired._merge == "both").all():
        raise ValueError("baseline/instrumented request keys do not match")
    if not (
        paired.structural_pass_baseline & paired.structural_pass_instrumented
    ).all():
        raise ValueError("one or more paired requests failed structural audit")
    if not (paired.output_tokens_baseline == paired.output_tokens_instrumented).all():
        raise ValueError("paired runs produced different output lengths")
    paired["output_ids_match"] = (
        paired.output_ids_baseline == paired.output_ids_instrumented
    )
    for metric in ["mean_tpot_ms", "median_itl_ms", "ttft_ms", "e2e_ms"]:
        paired[f"{metric}_relative_change"] = (
            paired[f"{metric}_instrumented"] / paired[f"{metric}_baseline"] - 1
        )

    median_baseline = float(np.median(paired.mean_tpot_ms_baseline))
    median_instrumented = float(np.median(paired.mean_tpot_ms_instrumented))
    perturbation = median_instrumented / median_baseline - 1
    summary = {
        "schema_version": 1,
        "pairs": len(paired),
        "metric": "median across requests of client-streamed mean TPOT",
        "baseline_median_mean_tpot_ms": median_baseline,
        "instrumented_median_mean_tpot_ms": median_instrumented,
        "relative_perturbation": perturbation,
        "absolute_relative_perturbation": abs(perturbation),
        "gate": args.gate,
        "overhead_gate_passed": abs(perturbation) <= args.gate,
        "all_output_ids_match": bool(paired.output_ids_match.all()),
        "speedup_measured": False,
        "limitation": (
            "local client streaming includes HTTP/scheduler noise; this is an "
            "instrumentation perturbation gate, not a performance result"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.drop(columns=["_merge"]).to_csv(
        args.output_dir / "paired_overhead.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    if not summary["all_output_ids_match"]:
        raise RuntimeError("deadline instrumentation changed one or more outputs")
    if not summary["overhead_gate_passed"]:
        raise RuntimeError("deadline instrumentation overhead gate failed")


if __name__ == "__main__":
    main()
