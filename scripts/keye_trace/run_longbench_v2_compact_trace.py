#!/usr/bin/env python3
"""Prepare and replay a deterministic LongBench-v2 subset for compact DSA tracing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

KEYE_EOS_TOKEN_ID = 151645
DEFAULT_DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
ANSWER_FIRST_SYSTEM_PROMPT = (
    'Begin with exactly "The correct answer is (X)", replacing X with A, B, C, '
    "or D. Then briefly justify the choice. Do not delay the answer until after "
    "the explanation."
)
ANSWER_ASSISTANT_PREFIX = "</think>\nThe correct answer is ("


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def format_prompt(row: dict[str, Any]) -> str:
    # Keep this byte-for-byte aligned with
    # python/sglang/test/simple_eval_longbench_v2.py.
    return f"""
Please read the following text and answer the question below.
<text>
{str(row['context']).strip()}
</text>

What is the correct answer to this question: {str(row['question']).strip()}
Choices:
(A) {str(row['choice_A']).strip()}
(B) {str(row['choice_B']).strip()}
(C) {str(row['choice_C']).strip()}
(D) {str(row['choice_D']).strip()}

Format your response as follows: "The correct answer is (insert answer here)"."""


def extract_answer(text: str) -> str | None:
    normalized = text.replace("*", "")
    patterns = [
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])\b",
        r"answer\s+is\s*\(?([A-D])\)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def post_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    body = json.dumps(payload or {}).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"POST {url} failed: HTTP {exc.code}: {detail}") from exc


def get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read().decode()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


def tokenize_candidates(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    columns = [
        "_id",
        "domain",
        "sub_domain",
        "difficulty",
        "length",
        "question",
        "choice_A",
        "choice_B",
        "choice_C",
        "choice_D",
        "answer",
        "context",
    ]
    frame = pd.read_parquet(args.dataset, columns=columns)
    max_chars = int(args.length_config * args.max_chars_per_token)
    min_tokens = math.floor(args.length_config * args.min_length_fraction)
    max_tokens = args.length_config - args.max_new_tokens
    candidates: list[dict[str, Any]] = []
    skipped_too_many_chars = 0
    for source_ordinal, raw in enumerate(frame.to_dict("records")):
        if len(str(raw["context"])) > max_chars:
            skipped_too_many_chars += 1
            continue
        prompt = format_prompt(raw)
        if args.apply_chat_template:
            messages: list[dict[str, str]] = []
            if args.answer_first_system_prompt:
                messages.append(
                    {"role": "system", "content": ANSWER_FIRST_SYSTEM_PROMPT}
                )
            messages.append({"role": "user", "content": prompt})
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            input_ids = (
                encoded["input_ids"]
                if hasattr(encoded, "keys") and "input_ids" in encoded
                else encoded
            )
        else:
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = list(input_ids)
        assistant_prefix = ""
        if args.answer_assistant_prefix:
            assistant_prefix = ANSWER_ASSISTANT_PREFIX
        elif args.close_thinking_prefix:
            assistant_prefix = "</think>\n"
        if assistant_prefix:
            input_ids.extend(tokenizer.encode(assistant_prefix, add_special_tokens=False))
        prompt_len = len(input_ids)
        if prompt_len < min_tokens or prompt_len > max_tokens:
            continue
        candidates.append(
            {
                "source_ordinal": source_ordinal,
                "source_id": str(raw["_id"]),
                "domain": str(raw["domain"]),
                "sub_domain": str(raw["sub_domain"]),
                "difficulty": str(raw["difficulty"]),
                "source_length_label": str(raw["length"]),
                "answer": str(raw["answer"]).strip().upper(),
                "assistant_prefix": assistant_prefix,
                "prompt": prompt,
                "prompt_len": prompt_len,
                "input_ids": input_ids,
            }
        )
    profile = {
        "source_rows": len(frame),
        "applied_model_chat_template": args.apply_chat_template,
        "answer_first_system_prompt": (
            ANSWER_FIRST_SYSTEM_PROMPT if args.answer_first_system_prompt else None
        ),
        "close_thinking_prefix": args.close_thinking_prefix,
        "answer_assistant_prefix": (
            ANSWER_ASSISTANT_PREFIX if args.answer_assistant_prefix else None
        ),
        "max_context_chars_considered": max_chars,
        "skipped_above_char_guard": skipped_too_many_chars,
        "eligible_prompt_tokens": [min_tokens, max_tokens],
        "eligible_rows": len(candidates),
        "eligible_by_domain": {
            domain: sum(row["domain"] == domain for row in candidates)
            for domain in sorted(frame.domain.unique())
        },
    }
    return candidates, profile


def prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not 0 < args.min_length_fraction <= 1:
        raise ValueError("--min-length-fraction must be in (0, 1]")
    candidates, profile = tokenize_candidates(args)
    excluded_source_ids: set[str] = set()
    excluded_sources: list[dict[str, Any]] = []
    for path in args.exclude_prepared:
        rows = read_jsonl(path)
        ids = {str(row["source_index"]) for row in rows}
        excluded_source_ids.update(ids)
        excluded_sources.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": len(rows),
                "source_ids": len(ids),
            }
        )
    candidates_before_exclusion = len(candidates)
    candidates = [
        row for row in candidates if row["source_id"] not in excluded_source_ids
    ]
    profile["eligible_rows_before_exclusion"] = candidates_before_exclusion
    profile["excluded_eligible_rows"] = candidates_before_exclusion - len(candidates)
    profile["eligible_rows"] = len(candidates)
    profile["eligible_by_domain_after_exclusion"] = {
        domain: sum(row["domain"] == domain for row in candidates)
        for domain in sorted({row["domain"] for row in candidates})
    }
    domains = sorted({row["domain"] for row in candidates})
    source_domains = sorted(
        pd.read_parquet(args.dataset, columns=["domain"]).domain.unique()
    )
    if domains != source_domains and not args.allow_domain_shortfall:
        missing = sorted(set(source_domains) - set(domains))
        raise ValueError(
            f"length bucket has no eligible request for domains {missing}; profile={profile}"
        )

    selected: list[dict[str, Any]] = []
    for domain in domains:
        eligible = [row for row in candidates if row["domain"] == domain]
        eligible.sort(
            key=lambda row: (
                args.length_config - row["prompt_len"],
                row["source_id"],
            )
        )
        selected.extend(eligible[: args.samples_per_domain])
        if len(eligible) < args.samples_per_domain and not args.allow_domain_shortfall:
            raise ValueError(
                f"domain {domain!r} has {len(eligible)} eligible rows, "
                f"requires {args.samples_per_domain}"
            )

    requests: list[dict[str, Any]] = []
    for row in selected:
        domain_slug = slug(row["domain"])
        rid = (
            f"{args.rid_prefix}__{args.length_config}__{domain_slug}__"
            f"{row['source_id'][:12]}"
        )
        requests.append(
            {
                "rid": rid,
                "dataset": "LongBench-v2",
                "length_config": args.length_config,
                "task": domain_slug,
                "domain": row["domain"],
                "sub_domain": row["sub_domain"],
                "difficulty": row["difficulty"],
                "source_length_label": row["source_length_label"],
                "source_index": row["source_id"],
                "source_ordinal": row["source_ordinal"],
                "prompt_len": row["prompt_len"],
                "prompt_sha256": sha256_bytes(row["prompt"].encode()),
                "input_ids_sha256": sha256_bytes(
                    json.dumps(row["input_ids"], separators=(",", ":")).encode()
                ),
                "input_ids": row["input_ids"],
                "expected_answer": row["answer"],
                "assistant_prefix": row["assistant_prefix"],
            }
        )
    requests.sort(key=lambda row: (row["task"], row["source_index"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = args.output_dir / "prepared_requests.jsonl"
    prepared.unlink(missing_ok=True)
    for row in requests:
        append_jsonl(prepared, row)
    write_json(
        args.output_dir / "dataset_snapshot.json",
        {
            "schema_version": 1,
            "dataset": "zai-org/LongBench-v2",
            "dataset_revision": args.dataset_revision,
            "source_parquet": str(args.dataset.resolve()),
            "source_parquet_sha256": sha256_file(args.dataset),
            "length_config": args.length_config,
            "selection": {
                "inputs": ["Keye prompt token length", "domain", "stable source id"],
                "does_not_use": ["answer value", "model output", "DSA trace"],
                "rule": "per domain, closest prompt below bucket cap; stable source id breaks ties",
                "min_length_fraction": args.min_length_fraction,
                "samples_per_domain": args.samples_per_domain,
                "allow_domain_shortfall": args.allow_domain_shortfall,
                "excluded_prepared": excluded_sources,
                "max_new_tokens_reserved": args.max_new_tokens,
                "applied_model_chat_template": args.apply_chat_template,
                "answer_first_system_prompt": (
                    ANSWER_FIRST_SYSTEM_PROMPT
                    if args.answer_first_system_prompt
                    else None
                ),
                "close_thinking_prefix": args.close_thinking_prefix,
                "answer_assistant_prefix": (
                    ANSWER_ASSISTANT_PREFIX if args.answer_assistant_prefix else None
                ),
            },
            "profile": profile,
            "selected": [
                {
                    key: row[key]
                    for key in [
                        "rid",
                        "domain",
                        "sub_domain",
                        "difficulty",
                        "source_length_label",
                        "source_index",
                        "source_ordinal",
                        "prompt_len",
                        "prompt_sha256",
                        "input_ids_sha256",
                        "assistant_prefix",
                    ]
                }
                for row in requests
            ],
        },
    )
    write_json(
        args.output_dir / "run_config.json",
        {
            "schema_version": 1,
            "created_unix_s": time.time(),
            "model": args.model,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sglang_commit": git_revision(Path.cwd()),
            "length": args.length_config,
            "domains": [row["domain"] for row in requests],
            "samples_per_domain": args.samples_per_domain,
            "allow_domain_shortfall": args.allow_domain_shortfall,
            "excluded_prepared": excluded_sources,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "temperature": 0,
            "applied_model_chat_template": args.apply_chat_template,
            "answer_first_system_prompt": (
                ANSWER_FIRST_SYSTEM_PROMPT
                if args.answer_first_system_prompt
                else None
            ),
            "close_thinking_prefix": args.close_thinking_prefix,
            "answer_assistant_prefix": (
                ANSWER_ASSISTANT_PREFIX if args.answer_assistant_prefix else None
            ),
            "request_limit": args.request_limit,
        },
    )
    return requests


def audit_response(prepared: dict[str, Any], executed: dict[str, Any]) -> dict[str, Any]:
    response = executed["response"]
    text = str(response.get("text") or "")
    assistant_prefix = str(prepared.get("assistant_prefix") or "")
    scored_text = assistant_prefix + text
    output_ids = list(response.get("output_ids") or [])
    meta = response.get("meta_info") or {}
    finish = meta.get("finish_reason") or {}
    first_eos = next(
        (index for index, token in enumerate(output_ids) if token == KEYE_EOS_TOKEN_ID),
        None,
    )
    extracted = extract_answer(scored_text)
    expected = str(prepared["expected_answer"])
    checks = {
        "nonempty_text": bool(text.strip()),
        "nonempty_output_ids": bool(output_ids),
        "request_id_matches": meta.get("id") == prepared["rid"],
        "prompt_tokens_match": int(meta.get("prompt_tokens", -1))
        == int(prepared["prompt_len"]),
        "completion_tokens_match": int(meta.get("completion_tokens", -1))
        == len(output_ids),
        "minimum_completion_tokens": len(output_ids)
        >= int(executed.get("min_new_tokens", 0)),
        "finite_latency": math.isfinite(float(executed["latency_s"]))
        and float(executed["latency_s"]) > 0,
        "valid_finish_reason": finish.get("type") in {"stop", "length"},
        "no_replacement_character": "\ufffd" not in text,
        "no_tokens_after_eos": first_eos is None or first_eos == len(output_ids) - 1,
    }
    return {
        "rid": prepared["rid"],
        "task": prepared["task"],
        "domain": prepared["domain"],
        "sub_domain": prepared["sub_domain"],
        "difficulty": prepared["difficulty"],
        "prompt_len": prepared["prompt_len"],
        "output_tokens": len(output_ids),
        "finish_reason": finish,
        "expected_answer": expected,
        "extracted_answer": extracted,
        "correct": extracted == expected,
        "structural_checks": checks,
        "structural_pass": all(checks.values()),
        "text_preview": text[:500],
        "assistant_prefix": assistant_prefix,
        "scored_text_preview": scored_text[:500],
    }


def run(args: argparse.Namespace, requests: list[dict[str, Any]]) -> None:
    health = get_json(f"{args.base_url}/health_generate", args.timeout)
    model_info = get_json(f"{args.base_url}/model_info", args.timeout)
    write_json(
        args.output_dir / "server_snapshot.json",
        {"health_generate": health, "model_info": model_info},
    )
    output_path = args.output_dir / "requests.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        for row in read_jsonl(output_path):
            rid = str(row["rid"])
            if rid in completed:
                raise ValueError(f"duplicate completed request ID in {output_path}: {rid}")
            completed[rid] = row
        expected_rids = {request["rid"] for request in requests}
        unexpected = sorted(set(completed) - expected_rids)
        if unexpected:
            raise ValueError(
                f"{output_path} contains request IDs absent from prepared input: "
                f"{unexpected}"
            )
    else:
        output_path.unlink(missing_ok=True)

    audit_dir = args.output_dir / "analysis" / "inference-output-audit-v01"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = audit_dir / "request_audit.jsonl"
    audit_rows.unlink(missing_ok=True)
    prepared_by_rid = {request["rid"]: request for request in requests}
    audits = [
        audit_response(prepared_by_rid[rid], completed[rid])
        for rid in sorted(completed)
    ]
    for row in audits:
        append_jsonl(audit_rows, row)

    for request in requests:
        if request["rid"] in completed:
            print(
                json.dumps({"rid": request["rid"], "status": "already_completed"}),
                flush=True,
            )
            continue
        post_json(f"{args.base_url}/flush_cache", {}, args.timeout)
        started = time.perf_counter()
        response = post_json(
            f"{args.base_url}/generate",
            {
                "input_ids": request["input_ids"],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.max_new_tokens,
                    "min_new_tokens": args.min_new_tokens,
                },
                "stream": False,
                "rid": request["rid"],
            },
            args.timeout,
        )
        executed = {
            "rid": request["rid"],
            "dataset": "LongBench-v2",
            "length_config": request["length_config"],
            "task": request["task"],
            "domain": request["domain"],
            "sub_domain": request["sub_domain"],
            "difficulty": request["difficulty"],
            "source_index": request["source_index"],
            "prompt_len": request["prompt_len"],
            "prompt_sha256": request["prompt_sha256"],
            "input_ids_sha256": request.get("input_ids_sha256"),
            "expected_answer": request["expected_answer"],
            "assistant_prefix": request.get("assistant_prefix", ""),
            "min_new_tokens": args.min_new_tokens,
            "latency_s": time.perf_counter() - started,
            "response": response,
        }
        append_jsonl(output_path, executed)
        audit = audit_response(request, executed)
        audits.append(audit)
        append_jsonl(audit_rows, audit)
        print(
            json.dumps(
                {
                    "rid": request["rid"],
                    "prompt_len": request["prompt_len"],
                    "latency_s": round(executed["latency_s"], 3),
                    "expected": audit["expected_answer"],
                    "extracted": audit["extracted_answer"],
                    "correct": audit["correct"],
                }
            ),
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "requests": len(audits),
        "structural_pass_count": sum(row["structural_pass"] for row in audits),
        "correct_count": sum(row["correct"] for row in audits),
        "accuracy": sum(row["correct"] for row in audits) / len(audits),
        "all_structural_passed": bool(audits)
        and all(row["structural_pass"] for row in audits),
    }
    write_json(audit_dir / "summary.json", summary)
    if not summary["all_structural_passed"]:
        raise RuntimeError("LongBench-v2 inference output structural audit failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length-config", type=int, required=True)
    parser.add_argument("--samples-per-domain", type=int, default=1)
    parser.add_argument(
        "--exclude-prepared",
        type=Path,
        action="append",
        default=[],
        help="exclude source_index values found in a prior prepared_requests.jsonl",
    )
    parser.add_argument(
        "--allow-domain-shortfall",
        action="store_true",
        help="keep all available rows when a domain has fewer than requested",
    )
    parser.add_argument("--rid-prefix", default="lbv2v5")
    parser.add_argument(
        "--request-limit",
        type=int,
        help="run only the first N frozen requests; intended for quality gates",
    )
    parser.add_argument("--min-length-fraction", type=float, default=0.75)
    parser.add_argument("--max-chars-per-token", type=float, default=8.0)
    parser.add_argument("--model", default="/Tan/model/Keye-VL-2.0-30B-A3B")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="wrap the official user prompt with the model chat template",
    )
    parser.add_argument(
        "--answer-first-system-prompt",
        action="store_true",
        help="add a fixed answer-first instruction before the official user prompt",
    )
    parser.add_argument(
        "--close-thinking-prefix",
        action="store_true",
        help='prefill the assistant with "</think>" before generation',
    )
    parser.add_argument(
        "--answer-assistant-prefix",
        action="store_true",
        help="prefill a fixed answer-format prefix without supplying the answer",
    )
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="retain completed requests.jsonl rows and skip their request IDs",
    )
    args = parser.parse_args()

    if not 0 <= args.min_new_tokens <= args.max_new_tokens:
        raise ValueError("--min-new-tokens must be between 0 and --max-new-tokens")

    if args.answer_first_system_prompt and not args.apply_chat_template:
        raise ValueError("--answer-first-system-prompt requires --apply-chat-template")
    if args.close_thinking_prefix and not args.apply_chat_template:
        raise ValueError("--close-thinking-prefix requires --apply-chat-template")
    if args.answer_assistant_prefix and not args.apply_chat_template:
        raise ValueError("--answer-assistant-prefix requires --apply-chat-template")
    if args.answer_assistant_prefix and args.close_thinking_prefix:
        raise ValueError(
            "--answer-assistant-prefix and --close-thinking-prefix are mutually exclusive"
        )

    if args.reuse_prepared:
        requests = read_jsonl(args.output_dir / "prepared_requests.jsonl")
    else:
        requests = prepare(args)
    if args.request_limit is not None:
        if args.request_limit <= 0:
            raise ValueError("--request-limit must be positive")
        requests = requests[: args.request_limit]
    if not args.prepare_only:
        run(args, requests)


if __name__ == "__main__":
    main()
