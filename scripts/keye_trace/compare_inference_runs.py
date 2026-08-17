#!/usr/bin/env python3
"""Compare saved inference outputs from two trace-run directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def by_rid(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["rid"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path}: duplicate request id")
    return result


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def answer_quality(audit: dict[str, Any]) -> tuple[str, float | bool | None]:
    if "answer_recall" in audit:
        return "answer_recall", audit["answer_recall"]
    if "correct" in audit:
        return "correct", bool(audit["correct"])
    return "unavailable", None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_path = args.reference_run.resolve() / "requests.jsonl"
    candidate_path = args.candidate_run.resolve() / "requests.jsonl"
    reference = by_rid(reference_path)
    candidate = by_rid(candidate_path)
    if set(reference) != set(candidate):
        raise ValueError("request-id sets differ")

    reference_audits = by_rid(
        args.reference_run.resolve()
        / "analysis"
        / "inference-output-audit-v01"
        / "request_audit.jsonl"
    )
    candidate_audits = by_rid(
        args.candidate_run.resolve()
        / "analysis"
        / "inference-output-audit-v01"
        / "request_audit.jsonl"
    )
    rows: list[dict[str, Any]] = []
    for request_id in sorted(reference):
        left = reference[request_id]
        right = candidate[request_id]
        if left["prompt_sha256"] != right["prompt_sha256"]:
            raise ValueError(f"{request_id}: prompt differs")
        left_ids = list(left["response"].get("output_ids") or [])
        right_ids = list(right["response"].get("output_ids") or [])
        left_quality_field, left_quality = answer_quality(
            reference_audits[request_id]
        )
        right_quality_field, right_quality = answer_quality(
            candidate_audits[request_id]
        )
        if left_quality_field != right_quality_field:
            raise ValueError(f"{request_id}: answer quality fields differ")
        rows.append(
            {
                "request_id": request_id,
                "task": left.get("task"),
                "prompt_sha256": left["prompt_sha256"],
                "reference_output_tokens": len(left_ids),
                "candidate_output_tokens": len(right_ids),
                "output_tokens_exact": left_ids == right_ids,
                "output_text_exact": left["response"].get("text")
                == right["response"].get("text"),
                "first_different_output_token": first_difference(
                    left_ids, right_ids
                ),
                "answer_quality_field": left_quality_field,
                "reference_answer_quality": left_quality,
                "candidate_answer_quality": right_quality,
                "reference_answer_recall": (
                    left_quality
                    if left_quality_field == "answer_recall"
                    else None
                ),
                "candidate_answer_recall": (
                    right_quality
                    if right_quality_field == "answer_recall"
                    else None
                ),
                "reference_structural_pass": reference_audits[request_id][
                    "structural_pass"
                ],
                "candidate_structural_pass": candidate_audits[request_id][
                    "structural_pass"
                ],
            }
        )

    result = {
        "schema_version": 1,
        "reference_run": str(args.reference_run.resolve()),
        "candidate_run": str(args.candidate_run.resolve()),
        "reference_requests_sha256": sha256(reference_path),
        "candidate_requests_sha256": sha256(candidate_path),
        "requests": len(rows),
        "exact_output_token_sequences": sum(
            bool(row["output_tokens_exact"]) for row in rows
        ),
        "exact_output_texts": sum(bool(row["output_text_exact"]) for row in rows),
        "answer_quality_matches": sum(
            row["reference_answer_quality"] == row["candidate_answer_quality"]
            for row in rows
        ),
        "answer_recall_matches": sum(
            row["answer_quality_field"] == "answer_recall"
            and row["reference_answer_recall"] == row["candidate_answer_recall"]
            for row in rows
        ),
        "all_structural_passed": all(
            row["reference_structural_pass"] and row["candidate_structural_pass"]
            for row in rows
        ),
        "all_exact": all(
            row["output_tokens_exact"]
            and row["output_text_exact"]
            and row["reference_answer_quality"] == row["candidate_answer_quality"]
            for row in rows
        ),
        "comparisons": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["all_exact"] or not result["all_structural_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
