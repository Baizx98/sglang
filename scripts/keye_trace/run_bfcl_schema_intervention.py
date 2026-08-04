#!/usr/bin/env python3
"""Prepare and replay controlled BFCL tool-schema interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from run_bfcl_segmented import (
    ADDITIONAL_FUNCTION_PROMPT,
    AnnotatedMessage,
    append_jsonl,
    git_revision,
    get_json,
    load_tools,
    literal_assignment,
    post_json,
    read_jsonl,
    resolve_bfcl_package_root,
    system_components,
    teacher_tool_call_message,
    tokenize_with_segments,
    write_json,
)

SEED = 20260803
BUDGETS = {"2p5k": 2500, "3p5k": 3500, "full": None}
POSITIONS = ["front", "tail"]
TARGET_REQUESTS = [
    "bfclseg__base__109__round_02",
    "bfclseg__base__172__round_00",
    "bfclseg__base__125__round_02",
    "bfclseg__long_context__49__round_02",
    "bfclseg__long_context__165__round_03",
    "bfclseg__long_context__168__round_02",
    "bfclseg__miss_func__173__round_03",
    "bfclseg__miss_func__154__round_03",
    "bfclseg__miss_func__166__round_03",
    "bfclseg__miss_param__22__round_03",
    "bfclseg__miss_param__151__round_02",
    "bfclseg__miss_param__170__round_02",
]


def call_name(call: str) -> str:
    match = re.match(r"\s*([A-Za-z_]\w*)\s*\(", call)
    if match is None:
        raise ValueError(f"Cannot parse target function from {call!r}")
    return match.group(1)


def build_target_context(
    snapshot: dict[str, Any],
    target_round: int,
    data_root: Path,
) -> tuple[list[AnnotatedMessage], list[dict[str, Any]]]:
    case = snapshot["source_question"]
    all_tools = load_tools(data_root, case["involved_classes"])
    missed_by_round = {
        int(round_id): list(names)
        for round_id, names in case.get("missed_function", {}).items()
    }
    initially_missing = {
        name for names in missed_by_round.values() for name in names
    }
    current_tools = [
        tool for tool in all_tools if tool["name"] not in initially_missing
    ]
    held_out = {
        tool["name"]: tool for tool in all_tools if tool["name"] in initially_missing
    }
    ground_truth = snapshot["source_answer"]["ground_truth"]
    results_by_round = snapshot["ground_truth_results"]
    messages: list[AnnotatedMessage] = []
    for round_id in range(target_round + 1):
        if round_id in missed_by_round:
            current_tools.extend(held_out[name] for name in missed_by_round[round_id])
            current_turn = [{"role": "user", "content": ADDITIONAL_FUNCTION_PROMPT}]
        else:
            current_turn = case["question"][round_id]
        if not current_turn:
            current_turn = [{"role": "user", "content": ADDITIONAL_FUNCTION_PROMPT}]
        for message_index, message in enumerate(current_turn):
            messages.append(
                AnnotatedMessage(
                    role=message["role"],
                    content=message["content"],
                    segment_type="user_turn",
                    segment_id=f"user_turn::{round_id:02d}::{message_index:02d}",
                    source_round=round_id,
                )
            )
        if round_id == target_round:
            break
        calls = list(ground_truth[round_id])
        messages.append(
            AnnotatedMessage(
                role="assistant",
                content=teacher_tool_call_message(calls),
                segment_type="assistant_tool_call" if calls else "assistant_response",
                segment_id=(
                    f"assistant_tool_call::{round_id:02d}"
                    if calls
                    else f"assistant_response::{round_id:02d}"
                ),
                source_round=round_id,
            )
        )
        for result_index, result in enumerate(results_by_round[round_id]):
            messages.append(
                AnnotatedMessage(
                    role="tool",
                    content=result,
                    segment_type="tool_result",
                    segment_id=f"tool_result::{round_id:02d}::{result_index:02d}",
                    source_round=round_id,
                )
            )
    return messages, current_tools


def schema_cost(tokenizer: Any, tool: dict[str, Any]) -> int:
    text = f"TOOL_SCHEMA::{tool['name']}=" + json.dumps(
        tool, ensure_ascii=False, separators=(",", ":")
    )
    return len(tokenizer.encode(text + "\n\n", add_special_tokens=False))


def nested_tool_sets(
    tokenizer: Any,
    tools: list[dict[str, Any]],
    target_name: str,
    source_rid: str,
) -> dict[str, list[dict[str, Any]]]:
    target = next((tool for tool in tools if tool["name"] == target_name), None)
    if target is None:
        raise ValueError(f"{source_rid}: target tool {target_name} is unavailable")
    distractors = [tool for tool in tools if tool["name"] != target_name]
    seed = int(hashlib.sha256(source_rid.encode()).hexdigest()[:16], 16) ^ SEED
    random.Random(seed).shuffle(distractors)
    costs = [schema_cost(tokenizer, target)] + [
        schema_cost(tokenizer, tool) for tool in distractors
    ]
    sets: dict[str, list[dict[str, Any]]] = {}
    for label, budget in BUDGETS.items():
        if budget is None:
            sets[label] = distractors.copy()
            continue
        cumulative = costs[0]
        count = 0
        for cost in costs[1:]:
            if cumulative + cost > budget:
                break
            cumulative += cost
            count += 1
        sets[label] = distractors[:count]
    if not (len(sets["2p5k"]) < len(sets["3p5k"]) < len(sets["full"])):
        raise ValueError(f"{source_rid}: schema budgets do not form strict nested sets")
    return sets


def prepare_files(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_requests = {
        row["rid"]: row
        for row in read_jsonl(args.source_run / "prepared_requests.jsonl")
    }
    snapshots = {
        row["case_id"]: row
        for row in json.loads((args.source_run / "dataset_snapshot.json").read_text())
    }
    bfcl_root = resolve_bfcl_package_root(args.bfcl_root)
    data_root = bfcl_root / "bfcl_eval" / "data"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    long_context_source = (
        bfcl_root
        / "bfcl_eval"
        / "eval_checker"
        / "multi_turn_eval"
        / "func_source_code"
        / "long_context.py"
    )
    long_context_distractor = literal_assignment(
        long_context_source, "FILE_CONTENT_EXTENSION"
    )
    requests: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for source_rid in TARGET_REQUESTS:
        source = source_requests[source_rid]
        if len(source["ground_truth_calls"]) != 1:
            raise ValueError(f"{source_rid}: expected exactly one target call")
        case_id = source["trajectory_id"]
        snapshot = snapshots[case_id]
        messages, available_tools = build_target_context(
            snapshot, int(source["round_id"]), data_root
        )
        target_name = call_name(source["ground_truth_calls"][0])
        tool_sets = nested_tool_sets(
            tokenizer, available_tools, target_name, source_rid
        )
        target_tool = next(
            tool for tool in available_tools if tool["name"] == target_name
        )
        for budget_label, desired_budget in BUDGETS.items():
            distractors = tool_sets[budget_label]
            for position in POSITIONS:
                variant_tools = (
                    [target_tool, *distractors]
                    if position == "front"
                    else [*distractors, target_tool]
                )
                system_parts = system_components(
                    snapshot["source_question"],
                    variant_tools,
                    long_context_distractor=long_context_distractor,
                )
                prompt_messages = [
                    AnnotatedMessage(
                        role="system",
                        content="\n\n".join(part["text"] for part in system_parts),
                        segment_type="system_instruction",
                        segment_id="system_instruction",
                        source_round=None,
                    ),
                    *messages,
                ]
                short_source = source_rid.removeprefix("bfclseg__")
                rid = f"bfclint__{short_source}__b_{budget_label}__p_{position}"
                trajectory_id = f"schema_intervention::{rid}"
                prompt, prompt_segments = tokenize_with_segments(
                    tokenizer,
                    prompt_messages,
                    variant_tools,
                    system_parts,
                    trajectory_id=trajectory_id,
                    category=source["category"],
                    round_id=int(source["round_id"]),
                    rid=rid,
                )
                prompt.update(
                    {
                        "source_rid": source_rid,
                        "budget_label": budget_label,
                        "desired_schema_budget": desired_budget,
                        "target_position": position,
                        "target_tool": target_name,
                        "distractor_tool_names": [tool["name"] for tool in distractors],
                        "ground_truth_calls": source["ground_truth_calls"],
                        "ground_truth_results": source["ground_truth_results"],
                    }
                )
                actual_schema_tokens = sum(
                    row["token_count"]
                    for row in prompt_segments
                    if row["segment_type"] == "tool_schema"
                )
                prompt["actual_schema_tokens"] = actual_schema_tokens
                prompt["target_schema_tokens"] = sum(
                    row["token_count"]
                    for row in prompt_segments
                    if row["segment_id"] == f"tool_schema::{target_name}"
                )
                requests.append(prompt)
                segments.extend(prompt_segments)
                selection_rows.append(
                    {
                        "rid": rid,
                        "source_rid": source_rid,
                        "category": source["category"],
                        "source_round": source["round_id"],
                        "target_tool": target_name,
                        "budget_label": budget_label,
                        "desired_schema_budget": desired_budget,
                        "actual_schema_tokens": actual_schema_tokens,
                        "target_position": position,
                        "distractor_count": len(distractors),
                        "prompt_len": prompt["prompt_len"],
                    }
                )
    if len(requests) != 72:
        raise ValueError(f"expected 72 variants, prepared {len(requests)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["prepared_requests.jsonl", "segments.jsonl"]:
        (args.output_dir / name).unlink(missing_ok=True)
    for row in requests:
        append_jsonl(args.output_dir / "prepared_requests.jsonl", row)
    for row in segments:
        append_jsonl(args.output_dir / "segments.jsonl", row)
    write_json(
        args.output_dir / "selection.json",
        {
            "schema_version": 1,
            "seed": SEED,
            "source_run": str(args.source_run.resolve()),
            "target_requests": TARGET_REQUESTS,
            "budget_targets": BUDGETS,
            "positions": POSITIONS,
            "variants": selection_rows,
        },
    )
    write_json(
        args.output_dir / "run_config.json",
        {
            "schema_version": 1,
            "created_unix_s": time.time(),
            "seed": SEED,
            "model": args.model,
            "bfcl_root": str(bfcl_root),
            "bfcl_commit": git_revision(bfcl_root),
            "sglang_commit": git_revision(Path.cwd()),
            "python": platform.python_version(),
            "sampling": {
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "ignore_eos": args.ignore_eos,
                "concurrency": 1,
            },
        },
    )
    return requests


def run_requests(args: argparse.Namespace, requests: list[dict[str, Any]]) -> None:
    health = get_json(f"{args.base_url}/health_generate", args.timeout)
    model_info = get_json(f"{args.base_url}/model_info", args.timeout)
    write_json(
        args.output_dir / "server_snapshot.json",
        {"health_generate": health, "model_info": model_info},
    )
    requests_path = args.output_dir / "requests.jsonl"
    requests_path.unlink(missing_ok=True)
    for index, request in enumerate(requests, 1):
        post_json(f"{args.base_url}/flush_cache", {}, args.timeout)
        started = time.perf_counter()
        response = post_json(
            f"{args.base_url}/generate",
            {
                "input_ids": request["input_ids"],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.max_new_tokens,
                    "ignore_eos": args.ignore_eos,
                },
                "stream": False,
                "rid": request["rid"],
            },
            args.timeout,
        )
        latency_s = time.perf_counter() - started
        append_jsonl(
            requests_path,
            {
                key: request[key]
                for key in [
                    "rid",
                    "source_rid",
                    "category",
                    "round_id",
                    "prompt_len",
                    "prompt_sha256",
                    "budget_label",
                    "desired_schema_budget",
                    "actual_schema_tokens",
                    "target_position",
                    "target_tool",
                    "ground_truth_calls",
                ]
            }
            | {"latency_s": latency_s, "response": response},
        )
        print(
            f"[{index:02d}/{len(requests):02d}] {request['rid']} "
            f"prompt={request['prompt_len']} latency={latency_s:.3f}s",
            flush=True,
        )
    from audit_inference_outputs import audit_run

    audit = audit_run(
        args.output_dir,
        min_target_mention_rate=args.min_target_mention_rate,
        allow_partial=args.request_limit is not None,
    )
    print(json.dumps({"inference_output_audit": audit}, ensure_ascii=False), flush=True)
    if not audit["passed"]:
        raise RuntimeError("inference output audit failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfcl-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/Tan/model/Keye-VL-2.0-30B-A3B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="force a fixed decode length; disabled by default to preserve valid output",
    )
    parser.add_argument("--min-target-mention-rate", type=float, default=0.9)
    parser.add_argument("--request-limit", type=int)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requests = (
        read_jsonl(args.output_dir / "prepared_requests.jsonl")
        if args.reuse_prepared
        else prepare_files(args)
    )
    if args.request_limit is not None:
        requests = requests[: args.request_limit]
    if not args.prepare_only:
        run_requests(args, requests)


if __name__ == "__main__":
    main()
