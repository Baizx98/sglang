#!/usr/bin/env python3
"""Analyze same-batch exact/exact/candidate decode comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def top_map(response: dict[str, Any], position: int = 1) -> dict[int, float]:
    return {
        int(row["token_id"]): float(row["logprob"])
        for row in response["output_top_logprobs"][position]
    }


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / max(len(left | right), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.run_dir / "paired_decode_steps.jsonl")
    rows = []
    for record in records:
        exact_a = record["exact_a"]
        exact_b = record["exact_b"]
        candidate = record["candidate"]
        exact_token = int(exact_a["output_ids"][1])
        exact_a_top = top_map(exact_a)
        exact_b_top = top_map(exact_b)
        candidate_top = top_map(candidate)
        rows.append(
            {
                "source_rid": record["source_rid"],
                "step": int(record["step"]),
                "context_len": int(record["context_len"]),
                "prefill_exact_pair_match": exact_a["output_ids"][0]
                == exact_b["output_ids"][0],
                "prefill_all_match": len(
                    {
                        exact_a["output_ids"][0],
                        exact_b["output_ids"][0],
                        candidate["output_ids"][0],
                    }
                )
                == 1,
                "exact_pair_top1_match": exact_token == exact_b["output_ids"][1],
                "candidate_top1_match": exact_token == candidate["output_ids"][1],
                "exact_pair_top20_jaccard": jaccard(
                    set(exact_a_top), set(exact_b_top)
                ),
                "candidate_top20_jaccard": jaccard(
                    set(exact_a_top), set(candidate_top)
                ),
                "exact_top1_logprob": exact_a_top.get(exact_token),
                "exact_b_reference_logprob": exact_b_top.get(exact_token),
                "candidate_reference_logprob": candidate_top.get(exact_token),
            }
        )

    frame = pd.DataFrame(rows)
    frame["exact_pair_nll_delta"] = (
        -frame.exact_b_reference_logprob + frame.exact_top1_logprob
    )
    frame["candidate_nll_delta"] = (
        -frame.candidate_reference_logprob + frame.exact_top1_logprob
    )
    request_values = (
        frame.groupby("source_rid", sort=True)
        .agg(
            exact_top1=("exact_pair_top1_match", "mean"),
            candidate_top1=("candidate_top1_match", "mean"),
            exact_nll_delta=("exact_pair_nll_delta", "mean"),
            candidate_nll_delta=("candidate_nll_delta", "mean"),
        )
        .assign(
            top1_relative=lambda value: value.candidate_top1 - value.exact_top1,
            nll_relative=lambda value: value.candidate_nll_delta
            - value.exact_nll_delta,
        )
    )
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(
        0,
        len(request_values),
        size=(args.bootstrap_samples, len(request_values)),
    )
    top1_bootstrap = request_values.top1_relative.to_numpy()[indices].mean(axis=1)
    nll_bootstrap = request_values.nll_relative.to_numpy()[indices].mean(axis=1)

    exact_top1 = float(frame.exact_pair_top1_match.mean())
    candidate_top1 = float(frame.candidate_top1_match.mean())
    missing_exact = int(frame.exact_b_reference_logprob.isna().sum())
    missing_candidate = int(frame.candidate_reference_logprob.isna().sum())
    exact_nll_delta = float(frame.exact_pair_nll_delta.mean())
    candidate_nll_delta = float(frame.candidate_nll_delta.mean())
    summary = {
        "schema_version": 1,
        "request_count": int(frame.source_rid.nunique()),
        "paired_decode_step_count": len(frame),
        "prefill_exact_pair_agreement": float(frame.prefill_exact_pair_match.mean()),
        "prefill_all_agreement": float(frame.prefill_all_match.mean()),
        "decode_exact_pair_top1_agreement": exact_top1,
        "decode_candidate_top1_agreement": candidate_top1,
        "candidate_minus_exact_pair_top1_pp": 100 * (candidate_top1 - exact_top1),
        "candidate_minus_exact_pair_top1_pp_cluster_bootstrap_95ci": [
            100 * float(np.quantile(top1_bootstrap, 0.025)),
            100 * float(np.quantile(top1_bootstrap, 0.975)),
        ],
        "decode_exact_pair_top20_jaccard_mean": float(
            frame.exact_pair_top20_jaccard.mean()
        ),
        "decode_candidate_top20_jaccard_mean": float(
            frame.candidate_top20_jaccard.mean()
        ),
        "exact_reference_missing_from_exact_b_top20": missing_exact,
        "exact_reference_missing_from_candidate_top20": missing_candidate,
        "decode_exact_pair_nll_delta_mean": exact_nll_delta,
        "decode_candidate_nll_delta_mean": candidate_nll_delta,
        "candidate_minus_exact_pair_nll_delta": candidate_nll_delta
        - exact_nll_delta,
        "candidate_minus_exact_pair_nll_delta_cluster_bootstrap_95ci": [
            float(np.quantile(nll_bootstrap, 0.025)),
            float(np.quantile(nll_bootstrap, 0.975)),
        ],
        "gates": {
            "prefill_pairing_valid": float(frame.prefill_all_match.mean()) >= 0.995,
            "exact_baseline_reproducible": exact_top1 >= 0.995,
            "candidate_top1_at_least_99_5": candidate_top1 >= 0.995,
            "candidate_not_worse_than_exact_by_0_5pp": candidate_top1
            >= exact_top1 - 0.005,
            "reference_token_available_for_nll": missing_exact == 0
            and missing_candidate == 0,
            "candidate_nll_delta_not_over_0_005": candidate_nll_delta
            - exact_nll_delta
            <= 0.005,
        },
    }
    summary["quality_gate_pass"] = all(summary["gates"].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "paired_decode_steps.csv", index=False)
    frame.to_parquet(args.output_dir / "paired_decode_steps.parquet", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
