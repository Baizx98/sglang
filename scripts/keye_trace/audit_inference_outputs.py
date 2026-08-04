#!/usr/bin/env python3
"""Audit saved SGLang inference outputs for trace experiments."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KEYE_EOS_TOKEN_ID = 151645


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def call_name(call: str) -> str | None:
    match = re.match(r"\s*([A-Za-z_]\w*)\s*\(", str(call))
    return match.group(1) if match else None


def normalized_constants(call: str) -> set[str]:
    try:
        tree = ast.parse(str(call), mode="eval")
    except SyntaxError:
        return set()
    values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
            continue
        if isinstance(node.value, (int, float)):
            values.add(str(node.value).lower())
        elif isinstance(node.value, str):
            value = node.value.strip().lower()
            if 2 <= len(value) <= 80:
                values.add(value)
    return values


def longest_run(values: list[int]) -> int:
    if not values:
        return 0
    longest = current = 1
    for previous, value in zip(values, values[1:]):
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
    return longest


def repeated_ngram_fraction(values: list[int], n: int = 4) -> float:
    if len(values) < n:
        return 0.0
    ngrams = [tuple(values[index : index + n]) for index in range(len(values) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def audit_row(prepared: dict[str, Any], executed: dict[str, Any]) -> dict[str, Any]:
    rid = prepared["rid"]
    response = executed.get("response") or {}
    text = str(response.get("text") or "")
    output_ids = list(response.get("output_ids") or [])
    first_eos = next(
        (index for index, token_id in enumerate(output_ids) if token_id == KEYE_EOS_TOKEN_ID),
        None,
    )
    tokens_after_first_eos = (
        len(output_ids) - first_eos - 1 if first_eos is not None else 0
    )
    meta = response.get("meta_info") or {}
    finish = meta.get("finish_reason") or {}
    finish_type = finish.get("type")
    calls = list(prepared.get("ground_truth_calls") or executed.get("ground_truth_calls") or [])
    target_names = [name for call in calls if (name := call_name(call))]
    lower_text = text.lower()
    mentioned_names = [
        name
        for name in target_names
        if re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", lower_text)
    ]
    exact_call_names = [
        name
        for name in target_names
        if re.search(rf"(?<!\w){re.escape(name.lower())}\s*\(", lower_text)
    ]
    constants = set().union(*(normalized_constants(call) for call in calls))
    mentioned_constants = [value for value in constants if value in lower_text]
    tool_names = list(prepared.get("tool_names") or [])
    generated_known_calls = [
        name
        for name in tool_names
        if re.search(rf"(?<!\w){re.escape(name.lower())}\s*\(", lower_text)
    ]
    control_characters = [
        character
        for character in text
        if ord(character) < 32 and character not in {"\n", "\r", "\t"}
    ]
    structural_checks = {
        "response_object": isinstance(response, dict) and bool(response),
        "nonempty_text": bool(text.strip()),
        "nonempty_output_ids": bool(output_ids),
        "request_id_matches": meta.get("id") == executed.get("rid", rid),
        "prompt_tokens_match": int(meta.get("prompt_tokens", -1))
        == int(prepared["prompt_len"]),
        "completion_tokens_match": int(meta.get("completion_tokens", -1))
        == len(output_ids),
        "no_replacement_character": "\ufffd" not in text,
        "no_forbidden_control_character": not control_characters,
        "finite_latency": math.isfinite(float(executed.get("latency_s", math.nan)))
        and float(executed.get("latency_s", 0.0)) > 0,
        "no_pathological_token_run": longest_run(output_ids) <= 8,
        "no_pathological_4gram_repetition": repeated_ngram_fraction(output_ids) <= 0.5,
        "valid_finish_reason": finish_type in {"stop", "length"},
        "no_tokens_after_eos": tokens_after_first_eos == 0,
    }
    if not target_names:
        semantic_status = "no_call_expected"
    elif len(exact_call_names) == len(target_names):
        semantic_status = "exact_target_call_syntax"
    elif len(mentioned_names) == len(target_names):
        semantic_status = "all_target_tools_mentioned"
    elif mentioned_names:
        semantic_status = "partial_target_tools_mentioned"
    elif finish_type == "length":
        semantic_status = "inconclusive_length_truncation"
    elif generated_known_calls:
        semantic_status = "different_known_tool_generated"
    else:
        semantic_status = "target_tool_not_observed"
    return {
        "rid": rid,
        "source_rid": prepared.get("source_rid"),
        "category": prepared.get("category"),
        "prompt_len": int(prepared["prompt_len"]),
        "latency_s": float(executed.get("latency_s", math.nan)),
        "output_tokens": len(output_ids),
        "finish_type": finish_type,
        "length_limited": finish_type == "length",
        "first_eos_position": first_eos,
        "tokens_after_first_eos": tokens_after_first_eos,
        "target_names": target_names,
        "mentioned_target_names": mentioned_names,
        "exact_call_target_names": exact_call_names,
        "generated_known_calls": generated_known_calls,
        "ground_truth_constants": sorted(constants),
        "mentioned_ground_truth_constants": sorted(mentioned_constants),
        "target_name_recall": len(mentioned_names) / len(target_names)
        if target_names
        else math.nan,
        "constant_recall": len(mentioned_constants) / len(constants)
        if constants
        else math.nan,
        "longest_repeated_token_run": longest_run(output_ids),
        "repeated_4gram_fraction": repeated_ngram_fraction(output_ids),
        "semantic_status": semantic_status,
        "structural_checks": structural_checks,
        "structural_pass": all(structural_checks.values()),
        "text_preview": text[:500],
    }


def audit_run(
    run_dir: Path,
    *,
    min_target_mention_rate: float | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    prepared_rows = read_jsonl(run_dir / "prepared_requests.jsonl")
    executed_rows = read_jsonl(run_dir / "requests.jsonl")
    prepared = {row["rid"]: row for row in prepared_rows}
    executed: dict[str, dict[str, Any]] = {}
    for row in executed_rows:
        rid = row.get("prepared_rid") or row.get("rid")
        if rid in executed:
            raise ValueError(f"duplicate executed request {rid}")
        executed[rid] = row
    missing = sorted(set(prepared) - set(executed))
    unexpected = sorted(set(executed) - set(prepared))
    rows = [audit_row(prepared[rid], executed[rid]) for rid in prepared if rid in executed]
    statuses = Counter(row["semantic_status"] for row in rows)
    calls = [row for row in rows if row["target_names"]]
    conclusive = [
        row
        for row in calls
        if row["semantic_status"] != "inconclusive_length_truncation"
    ]
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "prepared_requests": len(prepared),
        "executed_requests": len(executed),
        "audited_requests": len(rows),
        "missing_requests": missing,
        "unexpected_requests": unexpected,
        "structural_pass_count": sum(row["structural_pass"] for row in rows),
        "structural_pass_rate": sum(row["structural_pass"] for row in rows) / len(rows)
        if rows
        else 0.0,
        "semantic_status_counts": dict(statuses),
        "requests_with_expected_calls": len(calls),
        "conclusive_call_requests": len(conclusive),
        "all_target_tool_mention_rate": sum(
            row["target_name_recall"] == 1.0 for row in calls
        )
        / len(calls)
        if calls
        else math.nan,
        "exact_target_call_syntax_rate": sum(
            row["semantic_status"] == "exact_target_call_syntax" for row in calls
        )
        / len(calls)
        if calls
        else math.nan,
        "mean_latency_s": sum(row["latency_s"] for row in rows) / len(rows)
        if rows
        else math.nan,
        "mean_output_tokens": sum(row["output_tokens"] for row in rows) / len(rows)
        if rows
        else math.nan,
        "finish_type_counts": dict(Counter(row["finish_type"] for row in rows)),
        "length_limited_count": sum(row["length_limited"] for row in rows),
        "requests_with_tokens_after_eos": sum(
            row["tokens_after_first_eos"] > 0 for row in rows
        ),
    }
    checks = {
        "all_prepared_requests_executed": not missing or allow_partial,
        "no_unexpected_requests": not unexpected,
        "all_outputs_structurally_valid": bool(rows)
        and all(row["structural_pass"] for row in rows),
        "no_different_known_tool_generated": statuses[
            "different_known_tool_generated"
        ]
        == 0,
    }
    if min_target_mention_rate is not None:
        checks["target_tool_mention_rate"] = bool(
            calls
            and summary["all_target_tool_mention_rate"]
            >= min_target_mention_rate
        )
        summary["minimum_target_tool_mention_rate"] = min_target_mention_rate
    summary["checks"] = checks
    summary["allow_partial"] = allow_partial
    summary["passed"] = all(checks.values())
    output_dir = run_dir / "analysis" / "inference-output-audit-v01"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "request_audit.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-target-mention-rate", type=float)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summary = audit_run(
        args.run_dir,
        min_target_mention_rate=args.min_target_mention_rate,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
