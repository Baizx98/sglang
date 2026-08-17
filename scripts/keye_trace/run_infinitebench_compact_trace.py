#!/usr/bin/env python3
"""Prepare and replay a deterministic InfiniteBench subset for compact tracing."""

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

from transformers import AutoTokenizer

KEYE_EOS_TOKEN_ID = 151645
DEFAULT_REVISION = "90f0394333616266d9fe85824ceaf505093cbaa5"
DEFAULT_TASKS = [
    "passkey",
    "kv_retrieval",
    "longbook_qa_eng",
    "longbook_choice_eng",
    "code_debug",
    "longdialogue_qa_eng",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


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


def official_prompt(task: str, row: dict[str, Any]) -> str:
    context, question = str(row["context"]), str(row["input"])
    options = [str(value) for value in row.get("options") or []]
    if task == "passkey":
        return "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n" + context + "\n\n" + question
    if task == "kv_retrieval":
        return "Extract the value corresponding to the specified key in the JSON object below.\n\n" + context + "\n\n" + question
    if task == "longbook_qa_eng":
        return f"Read the book below and answer a question.\n\n{context}\n\nQuestion: {question}\n\nBe very concise."
    if task == "longbook_choice_eng":
        return f"Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}"
    if task == "code_debug":
        return f"There is ONLY ONE function in the large project that is deliberately made to include an obvious error. Please find the function that contains the most obvious errors. I will give you four options to narrow your scope. You can inspect the options and think. Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n{context}\n\nWhich funtion has deliberate error?\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n\nYou should first find the functions in the options. Repeat their content, inspect through code, and at last give me your answer for the function that has the deliberate and obvious error in A, B, C, or D."
    if task == "longdialogue_qa_eng":
        return f"Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n{context}\n\n---\n\nEnd of dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else."
    raise ValueError(f"unsupported task: {task}")


def expected_choice(row: dict[str, Any]) -> str | None:
    answers = [str(value).strip() for value in row.get("answer") or []]
    options = [str(value).strip() for value in row.get("options") or []]
    if len(options) != 4:
        return None
    for answer in answers:
        for index, option in enumerate(options):
            if answer.strip('"').casefold() == option.strip('"').casefold():
                return "ABCD"[index]
    return None


def post_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    body = json.dumps(payload or {}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # SGLang's cache-management endpoints return plain text in
                # some versions even though generation endpoints return JSON.
                return raw
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"POST {url} failed: HTTP {exc.code}: {detail}") from exc


def get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read().decode()
        return json.loads(raw) if raw else None


def prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not 0 < args.min_length_fraction <= 1:
        raise ValueError("--min-length-fraction must be in (0, 1]")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    min_tokens = math.floor(args.length_config * args.min_length_fraction)
    max_tokens = args.length_config - args.max_new_tokens
    requests: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    profiles: dict[str, Any] = {}
    for task in args.tasks:
        source = args.dataset_root / f"{task}.jsonl"
        eligible: list[dict[str, Any]] = []
        source_rows = 0
        tokenized_rows = 0
        with source.open() as file:
            for source_ordinal, line in enumerate(file):
                source_rows += 1
                row = json.loads(line)
                prompt = official_prompt(task, row)
                # A generous guard prevents tokenizing contexts that cannot fit.
                if len(prompt) > args.length_config * args.max_chars_per_token:
                    continue
                if len(eligible) >= args.samples_per_task:
                    continue
                if tokenized_rows >= args.max_tokenized_rows_per_task:
                    continue
                tokenized_rows += 1
                input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if hasattr(input_ids, "keys"):
                    input_ids = input_ids["input_ids"]
                input_ids = list(input_ids)
                prompt_len = len(input_ids)
                if min_tokens <= prompt_len <= max_tokens:
                    eligible.append(
                        {
                            "source_ordinal": source_ordinal,
                            "source_id": str(row["id"]),
                            "prompt": prompt,
                            "prompt_len": prompt_len,
                            "input_ids": input_ids,
                            "expected_answers": [str(v) for v in row.get("answer") or []],
                            "expected_choice": expected_choice(row),
                        }
                    )
        if len(eligible) < args.samples_per_task and not args.allow_task_shortfall:
            raise ValueError(f"{task}: {len(eligible)} eligible rows, requires {args.samples_per_task}")
        selected = eligible[: args.samples_per_task]
        sources.append({"task": task, "path": str(source.resolve()), "sha256": sha256_file(source), "rows": source_rows})
        profiles[task] = {"source_rows": source_rows, "tokenized_rows_inspected": tokenized_rows, "eligible_rows_selected": len(eligible), "selected_prompt_tokens": [row["prompt_len"] for row in selected]}
        for row in selected:
            rid = f"{args.rid_prefix}__{args.length_config}__{task}__{int(row['source_ordinal']):04d}"
            requests.append(
                {
                    "rid": rid,
                    "dataset": "InfiniteBench",
                    "length_config": args.length_config,
                    "task": task,
                    "source_index": row["source_id"],
                    "source_ordinal": row["source_ordinal"],
                    "prompt_len": row["prompt_len"],
                    "prompt_sha256": sha256_bytes(row["prompt"].encode()),
                    "input_ids_sha256": sha256_bytes(json.dumps(row["input_ids"], separators=(",", ":")).encode()),
                    "input_ids": row["input_ids"],
                    "expected_answers": row["expected_answers"],
                    "expected_choice": row["expected_choice"],
                }
            )
    requests.sort(key=lambda row: (row["task"], row["source_index"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = args.output_dir / "prepared_requests.jsonl"
    prepared.write_text("")
    for row in requests:
        append_jsonl(prepared, row)
    write_json(args.output_dir / "dataset_snapshot.json", {
        "schema_version": 1,
        "dataset": "xinrongzhang2022/InfiniteBench",
        "dataset_revision": args.dataset_revision,
        "sources": sources,
        "selection": {"inputs": ["Keye prompt token length", "task", "stable source order"], "does_not_use": ["answer value", "model output", "DSA trace"], "rule": "per task, first fitting rows in a frozen source-order prefix", "max_tokenized_rows_per_task": args.max_tokenized_rows_per_task, "min_length_fraction": args.min_length_fraction, "samples_per_task": args.samples_per_task, "allow_task_shortfall": args.allow_task_shortfall, "max_new_tokens_reserved": args.max_new_tokens, "official_prompt_family": "gpt4_templates", "model_chat_template": True},
        "profiles": profiles,
    })
    write_json(args.output_dir / "run_config.json", {"schema_version": 1, "created_unix_s": time.time(), "model": args.model, "python": platform.python_version(), "platform": platform.platform(), "sglang_commit": git_revision(Path.cwd()), "length": args.length_config, "tasks": args.tasks, "samples_per_task": args.samples_per_task, "max_new_tokens": args.max_new_tokens, "min_new_tokens": args.min_new_tokens, "temperature": 0})
    return requests


def audit_response(prepared: dict[str, Any], executed: dict[str, Any]) -> dict[str, Any]:
    response = executed["response"]
    text = str(response.get("text") or "")
    output_ids = list(response.get("output_ids") or [])
    meta = response.get("meta_info") or {}
    finish = meta.get("finish_reason") or {}
    answers = [str(value) for value in prepared["expected_answers"]]
    matched = [answer for answer in answers if answer.strip('"').casefold() in text.casefold()]
    expected_letter = prepared.get("expected_choice")
    extracted_letter = None
    if expected_letter:
        match = re.search(r"(?:^|\b|\()([A-D])(?:\b|\))", text, re.IGNORECASE)
        extracted_letter = match.group(1).upper() if match else None
    first_eos = next((i for i, token in enumerate(output_ids) if token == KEYE_EOS_TOKEN_ID), None)
    checks = {
        "nonempty_text": bool(text.strip()),
        "nonempty_output_ids": bool(output_ids),
        "request_id_matches": meta.get("id") == prepared["rid"],
        "prompt_tokens_match": int(meta.get("prompt_tokens", -1)) == int(prepared["prompt_len"]),
        "completion_tokens_match": int(meta.get("completion_tokens", -1)) == len(output_ids),
        "minimum_completion_tokens": len(output_ids) >= int(executed.get("min_new_tokens", 0)),
        "finite_latency": math.isfinite(float(executed["latency_s"])) and float(executed["latency_s"]) > 0,
        "valid_finish_reason": finish.get("type") in {"stop", "length"},
        "no_replacement_character": "\ufffd" not in text,
        "no_tokens_after_eos": first_eos is None or first_eos == len(output_ids) - 1,
    }
    answer_match = (extracted_letter == expected_letter) if expected_letter else bool(matched)
    return {"rid": prepared["rid"], "task": prepared["task"], "prompt_len": prepared["prompt_len"], "output_tokens": len(output_ids), "finish_reason": finish, "expected_answers": answers, "matched_answers": matched, "expected_choice": expected_letter, "extracted_choice": extracted_letter, "answer_match": answer_match, "structural_checks": checks, "structural_pass": all(checks.values()), "text_preview": text[:500]}


def run(args: argparse.Namespace, requests: list[dict[str, Any]]) -> None:
    write_json(args.output_dir / "server_snapshot.json", {"health_generate": get_json(f"{args.base_url}/health_generate", args.timeout), "model_info": get_json(f"{args.base_url}/model_info", args.timeout)})
    output_path = args.output_dir / "requests.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        for row in read_jsonl(output_path):
            if row["rid"] in completed:
                raise ValueError(f"duplicate completed request ID: {row['rid']}")
            completed[row["rid"]] = row
    else:
        output_path.unlink(missing_ok=True)
    expected_rids = {row["rid"] for row in requests}
    unexpected = sorted(set(completed) - expected_rids)
    if unexpected:
        raise ValueError(f"unexpected completed request IDs: {unexpected}")
    audit_dir = args.output_dir / "analysis" / "inference-output-audit-v01"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = audit_dir / "request_audit.jsonl"
    audit_rows.unlink(missing_ok=True)
    prepared_by_rid = {row["rid"]: row for row in requests}
    audits = [audit_response(prepared_by_rid[rid], row) for rid, row in sorted(completed.items())]
    for row in audits:
        append_jsonl(audit_rows, row)
    for request in requests:
        if request["rid"] in completed:
            print(json.dumps({"rid": request["rid"], "status": "already_completed"}), flush=True)
            continue
        post_json(f"{args.base_url}/flush_cache", {}, args.timeout)
        started = time.perf_counter()
        response = post_json(f"{args.base_url}/generate", {"input_ids": request["input_ids"], "sampling_params": {"temperature": 0, "max_new_tokens": args.max_new_tokens, "min_new_tokens": args.min_new_tokens}, "stream": False, "rid": request["rid"]}, args.timeout)
        executed = {key: request[key] for key in ["rid", "dataset", "length_config", "task", "source_index", "source_ordinal", "prompt_len", "prompt_sha256", "input_ids_sha256", "expected_answers", "expected_choice"]}
        executed.update({"min_new_tokens": args.min_new_tokens, "latency_s": time.perf_counter() - started, "response": response})
        append_jsonl(output_path, executed)
        audit = audit_response(request, executed)
        audits.append(audit)
        append_jsonl(audit_rows, audit)
        print(json.dumps({"rid": request["rid"], "prompt_len": request["prompt_len"], "latency_s": round(executed["latency_s"], 3), "answer_match": audit["answer_match"]}), flush=True)
    summary = {"schema_version": 1, "requests": len(audits), "structural_pass_count": sum(row["structural_pass"] for row in audits), "answer_match_count": sum(row["answer_match"] for row in audits), "answer_match_rate": sum(row["answer_match"] for row in audits) / len(audits), "all_structural_passed": bool(audits) and all(row["structural_pass"] for row in audits)}
    write_json(audit_dir / "summary.json", summary)
    if not summary["all_structural_passed"]:
        raise RuntimeError("InfiniteBench inference output structural audit failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length-config", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--allow-task-shortfall", action="store_true")
    parser.add_argument("--min-length-fraction", type=float, default=0.5)
    parser.add_argument("--max-chars-per-token", type=float, default=8.0)
    parser.add_argument("--max-tokenized-rows-per-task", type=int, default=64)
    parser.add_argument("--rid-prefix", default="infbv5")
    parser.add_argument("--model", default="/Tan/model/Keye-VL-2.0-30B-A3B")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-new-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.min_new_tokens <= args.max_new_tokens:
        raise ValueError("--min-new-tokens must be between 0 and --max-new-tokens")
    requests = read_jsonl(args.output_dir / "prepared_requests.jsonl") if args.reuse_prepared else prepare(args)
    if not args.prepare_only:
        run(args, requests)


if __name__ == "__main__":
    main()
