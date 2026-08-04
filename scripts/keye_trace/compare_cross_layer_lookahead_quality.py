#!/usr/bin/env python3
"""Compare exact and selective-lookahead BFCL generation outputs."""

from __future__ import annotations

import argparse
import difflib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def common_prefix(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def load_audit(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "analysis/inference-output-audit-v01/request_audit.jsonl"
    return {row["rid"]: row for row in jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-run", type=Path, required=True)
    parser.add_argument("--selective-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    exact_run = args.exact_run.resolve()
    selective_run = args.selective_run.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads((exact_run / "selection.json").read_text())
    selection_by_rid = {row["rid"]: row for row in selection["selected"]}
    exact_requests = {row["prepared_rid"]: row for row in jsonl(exact_run / "requests.jsonl")}
    selective_requests = {
        row["prepared_rid"]: row for row in jsonl(selective_run / "requests.jsonl")
    }
    exact_audit = load_audit(exact_run)
    selective_audit = load_audit(selective_run)
    if not (set(selection_by_rid) == set(exact_requests) == set(selective_requests)):
        raise ValueError("selection and executed request sets differ")

    rows = []
    for rid, meta in selection_by_rid.items():
        exact = exact_requests[rid]
        selective = selective_requests[rid]
        exact_ids = list(exact["response"]["output_ids"])
        selective_ids = list(selective["response"]["output_ids"])
        comparable = min(len(exact_ids), len(selective_ids))
        prefix = common_prefix(exact_ids, selective_ids)
        exact_semantic = exact_audit[rid]
        selective_semantic = selective_audit[rid]
        rows.append(
            {
                "rid": rid,
                "trajectory_id": meta["trajectory_id"],
                "category": meta["category"],
                "split": meta["split"],
                "prompt_len": int(meta["prompt_len"]),
                "exact_output_tokens": len(exact_ids),
                "selective_output_tokens": len(selective_ids),
                "output_exact_match": exact_ids == selective_ids,
                "common_prefix_tokens": prefix,
                "common_prefix_fraction": prefix / comparable if comparable else math.nan,
                "position_token_agreement": (
                    sum(left == right for left, right in zip(exact_ids, selective_ids)) / comparable
                    if comparable
                    else math.nan
                ),
                "sequence_match_ratio": difflib.SequenceMatcher(
                    None, exact_ids, selective_ids, autojunk=False
                ).ratio(),
                "exact_target_name_recall": exact_semantic["target_name_recall"],
                "selective_target_name_recall": selective_semantic["target_name_recall"],
                "exact_constant_recall": exact_semantic["constant_recall"],
                "selective_constant_recall": selective_semantic["constant_recall"],
                "exact_semantic_status": exact_semantic["semantic_status"],
                "selective_semantic_status": selective_semantic["semantic_status"],
                "exact_structural_pass": exact_semantic["structural_pass"],
                "selective_structural_pass": selective_semantic["structural_pass"],
                "exact_latency_s": float(exact["latency_s"]),
                "selective_latency_s": float(selective["latency_s"]),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "request_quality_comparison.csv", index=False)
    frame.to_parquet(output_dir / "request_quality_comparison.parquet", index=False)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "comparison": "exact DSA vs calibration-selected synchronous lookahead K=2048",
        "request_count": len(frame),
        "splits": {},
        "phase_c_proceed": False,
        "phase_c_blockers": [],
    }
    for split, group in frame.groupby("split"):
        exact_tool_success = int((group["exact_target_name_recall"] == 1.0).sum())
        selective_tool_success = int(
            (group["selective_target_name_recall"] == 1.0).sum()
        )
        exact_constants_ok = group["exact_constant_recall"].isna() | (
            group["exact_constant_recall"] == 1.0
        )
        selective_constants_ok = group["selective_constant_recall"].isna() | (
            group["selective_constant_recall"] == 1.0
        )
        exact_tool_parameter_success = int(
            ((group["exact_target_name_recall"] == 1.0) & exact_constants_ok).sum()
        )
        selective_tool_parameter_success = int(
            (
                (group["selective_target_name_recall"] == 1.0)
                & selective_constants_ok
            ).sum()
        )
        summary["splits"][split] = {
            "request_count": len(group),
            "output_exact_match_count": int(group["output_exact_match"].sum()),
            "common_prefix_tokens_median": finite_or_none(group["common_prefix_tokens"].median()),
            "position_token_agreement_mean": finite_or_none(group["position_token_agreement"].mean()),
            "sequence_match_ratio_mean": finite_or_none(group["sequence_match_ratio"].mean()),
            "exact_target_tool_success_count": exact_tool_success,
            "selective_target_tool_success_count": selective_tool_success,
            "target_tool_success_delta": selective_tool_success - exact_tool_success,
            "exact_tool_parameter_success_count": exact_tool_parameter_success,
            "selective_tool_parameter_success_count": selective_tool_parameter_success,
            "tool_parameter_success_delta": (
                selective_tool_parameter_success - exact_tool_parameter_success
            ),
            "exact_structural_pass_count": int(group["exact_structural_pass"].sum()),
            "selective_structural_pass_count": int(group["selective_structural_pass"].sum()),
            "exact_latency_mean_s": finite_or_none(group["exact_latency_s"].mean()),
            "selective_latency_mean_s": finite_or_none(group["selective_latency_s"].mean()),
            "latency_change_percent": finite_or_none(
                100
                * (group["selective_latency_s"].mean() / group["exact_latency_s"].mean() - 1)
            ),
        }
    test_summary = summary["splits"].get("test")
    if test_summary and test_summary["target_tool_success_delta"] < 0:
        summary["phase_c_blockers"].append(
            "held-out target-tool success count is lower than exact baseline"
        )
    summary["phase_c_blockers"].append(
        "pre-registered teacher-forced logits and attention-output gates were not measured"
    )
    (output_dir / "quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
