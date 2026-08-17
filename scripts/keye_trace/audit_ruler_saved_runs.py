#!/usr/bin/env python3
"""Audit completed RULER requests across one or more saved trace runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

KEYE_EOS_TOKEN_ID = 151645


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_response(prepared: dict[str, Any], executed: dict[str, Any]) -> dict[str, Any]:
    response = executed.get("response") or {}
    text = str(response.get("text") or "")
    output_ids = list(response.get("output_ids") or [])
    meta = response.get("meta_info") or {}
    finish = meta.get("finish_reason") or {}
    first_eos = next(
        (index for index, token in enumerate(output_ids) if token == KEYE_EOS_TOKEN_ID),
        None,
    )
    expected = [str(value) for value in prepared.get("expected_outputs") or []]
    matched = [value for value in expected if value.lower() in text.lower()]
    checks = {
        "nonempty_text": bool(text.strip()),
        "nonempty_output_ids": bool(output_ids),
        "request_id_matches": meta.get("id") == prepared["rid"],
        "prompt_tokens_match": int(meta.get("prompt_tokens", -1))
        == int(prepared["prompt_len"]),
        "completion_tokens_match": int(meta.get("completion_tokens", -1))
        == len(output_ids),
        "finite_latency": math.isfinite(float(executed.get("latency_s", math.nan)))
        and float(executed.get("latency_s", 0.0)) > 0,
        "valid_finish_reason": finish.get("type") in {"stop", "length"},
        "no_replacement_character": "\ufffd" not in text,
        "no_tokens_after_eos": first_eos is None or first_eos == len(output_ids) - 1,
    }
    return {
        "rid": prepared["rid"],
        "task": prepared["task"],
        "source_index": prepared.get("source_index"),
        "prompt_len": int(prepared["prompt_len"]),
        "output_tokens": len(output_ids),
        "finish_reason": finish,
        "expected_outputs": expected,
        "matched_outputs": matched,
        "answer_recall": len(matched) / len(expected) if expected else math.nan,
        "structural_checks": checks,
        "structural_pass": all(checks.values()),
        "text_preview": text[:500],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prepared: dict[str, dict[str, Any]] = {}
    executed: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        prepared_path = run_dir / "prepared_requests.jsonl"
        requests_path = run_dir / "requests.jsonl"
        prepared_rows = read_jsonl(prepared_path)
        executed_rows = read_jsonl(requests_path)
        inputs.append(
            {
                "run_dir": str(run_dir.resolve()),
                "prepared_sha256": sha256(prepared_path),
                "requests_sha256": sha256(requests_path),
                "prepared_rows": len(prepared_rows),
                "completed_rows": len(executed_rows),
            }
        )
        local_prepared = {row["rid"]: row for row in prepared_rows}
        for row in executed_rows:
            rid = str(row["rid"])
            if rid not in local_prepared:
                raise ValueError(f"{run_dir}: completed request {rid} was not prepared")
            if rid in prepared or rid in executed:
                raise ValueError(f"duplicate completed request across runs: {rid}")
            prepared[rid] = local_prepared[rid]
            executed[rid] = row

    audits = [audit_response(prepared[rid], executed[rid]) for rid in sorted(executed)]
    if not audits:
        raise ValueError("no completed requests to audit")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "request_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for row in audits:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {
        "schema_version": 1,
        "inputs": inputs,
        "audited_requests": len(audits),
        "tasks": sorted(row["task"] for row in audits),
        "structural_pass_count": sum(row["structural_pass"] for row in audits),
        "all_structural_passed": all(row["structural_pass"] for row in audits),
        "mean_answer_recall": sum(row["answer_recall"] for row in audits) / len(audits),
        "answer_recall_by_task": {
            row["task"]: row["answer_recall"] for row in audits
        },
        "finish_type_counts": dict(
            Counter(row["finish_reason"].get("type") for row in audits)
        ),
        "request_audit_sha256": sha256(audit_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["all_structural_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
