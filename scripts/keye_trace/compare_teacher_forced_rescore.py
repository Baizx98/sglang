#!/usr/bin/env python3
"""Compare candidate rescoring with an exact-repeat control on fixed prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-steps", type=Path, required=True)
    parser.add_argument("--exact-repeat-steps", type=Path, required=True)
    parser.add_argument("--candidate-steps", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def keyed(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.DataFrame(read_jsonl(path))
    columns = ["rid", "step", "top1_match", "reference_nll"]
    frame = frame[columns].copy()
    return frame.rename(
        columns={
            "top1_match": f"{prefix}_top1_match",
            "reference_nll": f"{prefix}_reference_nll",
        }
    )


def main() -> None:
    args = parse_args()
    reference = keyed(args.reference_steps, "reference")
    exact_repeat = keyed(args.exact_repeat_steps, "exact_repeat")
    candidate = keyed(args.candidate_steps, "candidate")
    frame = reference.merge(exact_repeat, on=["rid", "step"], validate="one_to_one")
    frame = frame.merge(candidate, on=["rid", "step"], validate="one_to_one")
    frame["candidate_minus_exact_nll"] = (
        frame.candidate_reference_nll - frame.exact_repeat_reference_nll
    )
    frame["candidate_better_top1"] = (
        frame.candidate_top1_match & ~frame.exact_repeat_top1_match
    )
    frame["candidate_worse_top1"] = (
        ~frame.candidate_top1_match & frame.exact_repeat_top1_match
    )

    request_ids = sorted(frame.rid.unique())
    rng = np.random.default_rng(args.seed)
    request_deltas = (
        frame.groupby("rid", sort=True)
        .agg(
            exact_top1=("exact_repeat_top1_match", "mean"),
            candidate_top1=("candidate_top1_match", "mean"),
            exact_nll=("exact_repeat_reference_nll", "mean"),
            candidate_nll=("candidate_reference_nll", "mean"),
        )
        .assign(
            top1_delta=lambda values: values.candidate_top1 - values.exact_top1,
            nll_delta=lambda values: values.candidate_nll - values.exact_nll,
        )
    )
    sample_indices = rng.integers(
        0,
        len(request_deltas),
        size=(args.bootstrap_samples, len(request_deltas)),
    )
    bootstrap = np.column_stack(
        (
            request_deltas.top1_delta.to_numpy()[sample_indices].mean(axis=1),
            request_deltas.nll_delta.to_numpy()[sample_indices].mean(axis=1),
        )
    )
    exact_top1 = float(frame.exact_repeat_top1_match.mean())
    candidate_top1 = float(frame.candidate_top1_match.mean())
    exact_nll = float(frame.exact_repeat_reference_nll.mean())
    candidate_nll = float(frame.candidate_reference_nll.mean())
    summary = {
        "schema_version": 1,
        "request_count": len(request_ids),
        "step_count": len(frame),
        "reference_nll_mean": float(frame.reference_reference_nll.mean()),
        "exact_repeat_top1_agreement": exact_top1,
        "candidate_top1_agreement": candidate_top1,
        "candidate_minus_exact_top1_pp": 100 * (candidate_top1 - exact_top1),
        "candidate_minus_exact_top1_pp_cluster_bootstrap_95ci": [
            100 * float(np.quantile(bootstrap[:, 0], 0.025)),
            100 * float(np.quantile(bootstrap[:, 0], 0.975)),
        ],
        "exact_repeat_nll_mean": exact_nll,
        "candidate_nll_mean": candidate_nll,
        "candidate_minus_exact_nll": candidate_nll - exact_nll,
        "candidate_nll_relative_change_percent": 100 * (candidate_nll / exact_nll - 1),
        "candidate_minus_exact_nll_cluster_bootstrap_95ci": [
            float(np.quantile(bootstrap[:, 1], 0.025)),
            float(np.quantile(bootstrap[:, 1], 0.975)),
        ],
        "candidate_better_top1_steps": int(frame.candidate_better_top1.sum()),
        "candidate_worse_top1_steps": int(frame.candidate_worse_top1.sum()),
        "both_top1_match_steps": int(
            (frame.candidate_top1_match & frame.exact_repeat_top1_match).sum()
        ),
        "both_top1_mismatch_steps": int(
            (~frame.candidate_top1_match & ~frame.exact_repeat_top1_match).sum()
        ),
        "absolute_top1_gate_99_5": candidate_top1 >= 0.995,
        "absolute_nll_gate_0_5_percent": candidate_nll <= 1.005 * float(
            frame.reference_reference_nll.mean()
        ),
        "relative_control_top1_not_worse_0_5pp": candidate_top1
        >= exact_top1 - 0.005,
        "relative_control_nll_not_worse_0_5_percent": candidate_nll
        <= exact_nll * 1.005,
        "decision": "inconclusive_absolute_gate_due_to_exact_nondeterminism",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "paired_steps.csv", index=False)
    frame.to_parquet(args.output_dir / "paired_steps.parquet", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
