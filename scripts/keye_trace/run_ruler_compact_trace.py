#!/usr/bin/env python3
"""Prepare and replay a small official-RULER subset for compact DSA tracing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

KEYE_EOS_TOKEN_ID = 151645


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


def prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    requests: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for task in args.tasks:
        source = args.dataset_root / str(args.length) / task / args.split
        rows = read_jsonl(source)
        sample_end = args.sample_start + args.samples_per_task
        if args.sample_start < 0:
            raise ValueError("--sample-start must be non-negative")
        if len(rows) < sample_end:
            raise ValueError(
                f"{source}: requested rows [{args.sample_start}, {sample_end}), "
                f"found {len(rows)}"
            )
        sources.append(
            {
                "task": task,
                "path": str(source.resolve()),
                "sha256": sha256_file(source),
                "rows": len(rows),
            }
        )
        for ordinal, row in enumerate(
            rows[args.sample_start : sample_end], start=args.sample_start
        ):
            prompt = str(row["input"])
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            rid = f"{args.rid_prefix}__{args.length}__{task}__{ordinal:03d}"
            requests.append(
                {
                    "rid": rid,
                    "dataset": "RULER",
                    "length_config": args.length,
                    "task": task,
                    "source_index": row.get("index"),
                    "prompt_len": len(input_ids),
                    "prompt_sha256": sha256_bytes(prompt.encode()),
                    "input_ids": input_ids,
                    "expected_outputs": list(row.get("outputs") or []),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = args.output_dir / "prepared_requests.jsonl"
    prepared.unlink(missing_ok=True)
    for row in requests:
        append_jsonl(prepared, row)
    write_json(
        args.output_dir / "dataset_snapshot.json",
        {
            "schema_version": 1,
            "dataset": "RULER",
            "official_repo": str(args.ruler_repo.resolve()),
            "official_commit": git_revision(args.ruler_repo),
            "length_config": args.length,
            "sources": sources,
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
            "length": args.length,
            "tasks": args.tasks,
            "sample_start": args.sample_start,
            "samples_per_task": args.samples_per_task,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "temperature": 0,
        },
    )
    return requests


def audit_response(prepared: dict[str, Any], executed: dict[str, Any]) -> dict[str, Any]:
    response = executed["response"]
    text = str(response.get("text") or "")
    output_ids = list(response.get("output_ids") or [])
    meta = response.get("meta_info") or {}
    finish = meta.get("finish_reason") or {}
    first_eos = next(
        (index for index, token in enumerate(output_ids) if token == KEYE_EOS_TOKEN_ID),
        None,
    )
    expected = [str(value) for value in prepared["expected_outputs"]]
    matched = [value for value in expected if value.lower() in text.lower()]
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
        "prompt_len": prepared["prompt_len"],
        "output_tokens": len(output_ids),
        "finish_reason": finish,
        "expected_outputs": expected,
        "matched_outputs": matched,
        "answer_recall": len(matched) / len(expected) if expected else math.nan,
        "structural_checks": checks,
        "structural_pass": all(checks.values()),
        "text_preview": text[:500],
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
            raise ValueError(f"unexpected completed request IDs: {unexpected}")
    else:
        output_path.unlink(missing_ok=True)
    audit_dir = args.output_dir / "analysis" / "inference-output-audit-v01"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = audit_dir / "request_audit.jsonl"
    audit_rows.unlink(missing_ok=True)
    prepared_by_rid = {request["rid"]: request for request in requests}
    audits: list[dict[str, Any]] = [
        audit_response(prepared_by_rid[rid], completed[rid]) for rid in sorted(completed)
    ]
    for row in audits:
        append_jsonl(audit_rows, row)
    for request in requests:
        if request["rid"] in completed:
            print(json.dumps({"rid": request["rid"], "status": "already_completed"}))
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
            "dataset": "RULER",
            "length_config": request["length_config"],
            "task": request["task"],
            "source_index": request["source_index"],
            "prompt_len": request["prompt_len"],
            "prompt_sha256": request["prompt_sha256"],
            "expected_outputs": request["expected_outputs"],
            "min_new_tokens": args.min_new_tokens,
            "latency_s": time.perf_counter() - started,
            "response": response,
        }
        append_jsonl(output_path, executed)
        audit = audit_response(request, executed)
        audits.append(audit)
        print(
            json.dumps(
                {
                    "rid": request["rid"],
                    "prompt_len": request["prompt_len"],
                    "latency_s": round(executed["latency_s"], 3),
                    "answer_recall": audit["answer_recall"],
                }
            ),
            flush=True,
        )

    audit_rows.unlink(missing_ok=True)
    for row in audits:
        append_jsonl(audit_rows, row)
    summary = {
        "schema_version": 1,
        "requests": len(audits),
        "structural_pass_count": sum(row["structural_pass"] for row in audits),
        "mean_answer_recall": sum(row["answer_recall"] for row in audits) / len(audits),
        "all_structural_passed": bool(audits)
        and all(row["structural_pass"] for row in audits),
    }
    write_json(audit_dir / "summary.json", summary)
    if not summary["all_structural_passed"]:
        raise RuntimeError("RULER inference output structural audit failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ruler-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", default=["niah_single_1"])
    parser.add_argument("--split", default="validation.jsonl")
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="Zero-based source row at which to start each task slice.",
    )
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--rid-prefix", default="rulerv5")
    parser.add_argument("--model", default="/Tan/model/Keye-VL-2.0-30B-A3B")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.min_new_tokens <= args.max_new_tokens:
        raise ValueError("--min-new-tokens must be between 0 and --max-new-tokens")

    if args.reuse_prepared:
        requests = read_jsonl(args.output_dir / "prepared_requests.jsonl")
    else:
        requests = prepare(args)
    if not args.prepare_only:
        run(args, requests)


if __name__ == "__main__":
    main()
