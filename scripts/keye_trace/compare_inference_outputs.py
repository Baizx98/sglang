#!/usr/bin/env python3
"""Compare deterministic inference outputs from two otherwise identical runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def request_key(row: dict[str, Any]) -> str:
    return row.get("prepared_rid") or row["rid"]


def common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=int, default=32)
    args = parser.parse_args()
    reference = {
        request_key(row): row
        for row in read_jsonl(args.reference_run / "requests.jsonl")
    }
    candidate = {
        request_key(row): row
        for row in read_jsonl(args.candidate_run / "requests.jsonl")
    }
    keys = sorted(set(reference) & set(candidate))
    rows = []
    for key in keys:
        left = reference[key]
        right = candidate[key]
        left_response = left["response"]
        right_response = right["response"]
        left_ids = list(left_response["output_ids"])
        right_ids = list(right_response["output_ids"])
        prefix = args.prefix_tokens
        rows.append(
            {
                "request_key": key,
                "prompt_sha256_equal": left.get("prompt_sha256")
                == right.get("prompt_sha256"),
                "prompt_len_equal": left.get("prompt_len") == right.get("prompt_len"),
                "first_n_output_ids_equal": left_ids[:prefix] == right_ids[:prefix],
                "common_prefix_tokens": common_prefix_length(left_ids, right_ids),
                "all_output_ids_equal": left_ids == right_ids,
                "text_equal": left_response.get("text") == right_response.get("text"),
                "reference_output_tokens": len(left_ids),
                "candidate_output_tokens": len(right_ids),
            }
        )
    checks = {
        "same_request_keys": set(reference) == set(candidate),
        "nonempty_comparison": bool(rows),
        "same_prompts": bool(rows)
        and all(
            row["prompt_sha256_equal"] and row["prompt_len_equal"] for row in rows
        ),
        "same_first_n_output_ids": bool(rows)
        and all(row["first_n_output_ids_equal"] for row in rows),
        "same_complete_outputs": bool(rows)
        and all(
            row["all_output_ids_equal"] and row["text_equal"] for row in rows
        ),
    }
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_run": str(args.reference_run.resolve()),
        "candidate_run": str(args.candidate_run.resolve()),
        "prefix_tokens": args.prefix_tokens,
        "reference_requests": len(reference),
        "candidate_requests": len(candidate),
        "compared_requests": len(rows),
        "checks": checks,
        "rows": rows,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
