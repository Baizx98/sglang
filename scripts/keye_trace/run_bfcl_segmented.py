#!/usr/bin/env python3
"""Prepare and replay segmented BFCL trajectories through SGLang /generate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

SEED = 20260731
DEFAULT_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
]
FUNCTION_DOC_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}
SYSTEM_INSTRUCTION = (
    "You are a tool-using agent in the Berkeley Function Calling Leaderboard "
    "(BFCL). Use only the provided tools. The experiment deterministically "
    "replays the ground-truth tool calls and tool results between LLM rounds."
)
ADDITIONAL_FUNCTION_PROMPT = (
    "I have updated some more functions you can choose from. What about now?"
)


@dataclass
class AnnotatedMessage:
    role: str
    content: str
    segment_type: str
    segment_id: str
    source_round: int | None
    tool_name: str | None = None

    def clean(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def resolve_bfcl_package_root(root: Path) -> Path:
    candidates = [root, root / "berkeley-function-call-leaderboard"]
    for candidate in candidates:
        if (candidate / "bfcl_eval" / "data").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate bfcl_eval/data below {root}")


def literal_assignment(path: Path, variable: str) -> Any:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == variable
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(f"{variable} not found in {path}")


def load_tools(data_root: Path, involved_classes: list[str]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    func_root = data_root / "multi_turn_func_doc"
    for class_name in involved_classes:
        tools.extend(read_jsonl(func_root / FUNCTION_DOC_MAPPING[class_name]))
    return tools


def system_components(
    case: dict[str, Any],
    tools: list[dict[str, Any]],
    *,
    long_context_distractor: str,
) -> list[dict[str, Any]]:
    components = [
        {
            "segment_type": "system_instruction",
            "segment_id": "system_instruction",
            "text": SYSTEM_INSTRUCTION,
        },
    ]
    components.extend(
        {
            "segment_type": "tool_schema",
            "segment_id": f"tool_schema::{tool['name']}",
            "tool_name": tool["name"],
            "text": f"TOOL_SCHEMA::{tool['name']}=" + compact_json(tool),
        }
        for tool in tools
    )
    components.append(
        {
            "segment_type": "initial_state",
            "segment_id": "initial_state",
            "text": "INITIAL_STATE=" + compact_json(case["initial_config"]),
        }
    )
    if "long_context" in case["id"]:
        components.append(
            {
                "segment_type": "long_context_distractor",
                "segment_id": "long_context_distractor",
                "text": "LONG_CONTEXT_DISTRACTOR=" + long_context_distractor,
            }
        )
    return components


def locate_spans(
    rendered: str,
    system_parts: list[dict[str, Any]],
    messages: list[AnnotatedMessage],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    system_content = "\n\n".join(part["text"] for part in system_parts)
    system_start = rendered.index(system_content)
    component_cursor = system_start
    for part in system_parts:
        start = rendered.index(part["text"], component_cursor)
        end = start + len(part["text"])
        spans.append(
            {
                "char_start": start,
                "char_end": end,
                "segment_type": part["segment_type"],
                "segment_id": part["segment_id"],
                "source_round": None,
                "tool_name": part.get("tool_name"),
            }
        )
        component_cursor = end

    message_cursor = component_cursor
    for message in messages[1:]:
        start = rendered.index(message.content, message_cursor)
        end = start + len(message.content)
        spans.append(
            {
                "char_start": start,
                "char_end": end,
                "segment_type": message.segment_type,
                "segment_id": message.segment_id,
                "source_round": message.source_round,
                "tool_name": message.tool_name,
            }
        )
        message_cursor = end
    return spans


def tokenize_with_segments(
    tokenizer: Any,
    messages: list[AnnotatedMessage],
    tools: list[dict[str, Any]],
    system_parts: list[dict[str, Any]],
    *,
    trajectory_id: str,
    category: str,
    round_id: int,
    rid: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clean_messages = [message.clean() for message in messages]
    rendered = tokenizer.apply_chat_template(
        clean_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    templated = tokenizer.apply_chat_template(
        clean_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    templated_ids = list(templated["input_ids"])
    if input_ids != templated_ids:
        raise ValueError(f"Rendered/tokenized prompt mismatch for {rid}")

    spans = locate_spans(rendered, system_parts, messages)
    default_label = {
        "segment_type": "chat_template",
        "segment_id": "chat_template",
        "source_round": None,
        "tool_name": None,
    }
    token_labels = []
    for token_start, token_end in offsets:
        best_label = default_label
        best_overlap = 0
        for span in spans:
            overlap = max(
                0,
                min(token_end, span["char_end"]) - max(token_start, span["char_start"]),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = span
        token_labels.append(
            {
                "segment_type": best_label["segment_type"],
                "segment_id": best_label["segment_id"],
                "source_round": best_label["source_round"],
                "tool_name": best_label["tool_name"],
            }
        )
    if len(token_labels) != len(input_ids):
        raise AssertionError("Each prompt token must have exactly one label")

    local_offsets: dict[str, int] = {}
    segment_rows: list[dict[str, Any]] = []
    range_start = 0
    for index in range(1, len(input_ids) + 1):
        boundary = index == len(input_ids) or token_labels[index] != token_labels[index - 1]
        if not boundary:
            continue
        label = token_labels[index - 1]
        segment_id = label["segment_id"]
        local_start = local_offsets.get(segment_id, 0)
        token_count = index - range_start
        segment_rows.append(
            {
                "trajectory_id": trajectory_id,
                "category": category,
                "round_id": round_id,
                "rid": rid,
                "segment_type": label["segment_type"],
                "segment_id": segment_id,
                "source_round": label["source_round"],
                "round_age": (
                    round_id - int(label["source_round"])
                    if label["source_round"] is not None
                    else None
                ),
                "tool_name": label["tool_name"],
                "token_start": range_start,
                "token_end": index,
                "token_count": token_count,
                "local_token_start": local_start,
                "local_token_end": local_start + token_count,
                "token_ids_sha256": sha256_json(input_ids[range_start:index]),
            }
        )
        local_offsets[segment_id] = local_start + token_count
        range_start = index

    prompt = {
        "trajectory_id": trajectory_id,
        "category": category,
        "round_id": round_id,
        "rid": rid,
        "prompt_len": len(input_ids),
        "prompt_sha256": sha256_json(input_ids),
        "input_ids": input_ids,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "segment_count": len(segment_rows),
        "tool_names": [tool["name"] for tool in tools],
    }
    return prompt, segment_rows


def teacher_tool_call_message(calls: list[str]) -> str:
    if not calls:
        return "Teacher-forced response: no executable tool call in this turn."
    return "Teacher-forced tool calls:\n" + "\n".join(f"- {call}" for call in calls)


def execute_ground_truth(
    bfcl_package_root: Path,
    case: dict[str, Any],
    calls: list[str],
    executor_name: str,
) -> list[str]:
    sys.path.insert(0, str(bfcl_package_root))
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # noqa: PLC0415
        execute_multi_turn_func_call,
    )

    results, _ = execute_multi_turn_func_call(
        calls,
        case["initial_config"],
        case["involved_classes"],
        executor_name,
        case["id"],
        long_context="long_context" in case["id"],
        is_evaL_run=False,
    )
    return results


def build_case_requests(
    tokenizer: Any,
    bfcl_package_root: Path,
    category: str,
    case: dict[str, Any],
    answer: dict[str, Any],
    long_context_distractor: str,
    *,
    limit_rounds: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data_root = bfcl_package_root / "bfcl_eval" / "data"
    all_tools = load_tools(data_root, case["involved_classes"])
    missed_by_round = {
        int(round_id): list(names)
        for round_id, names in case.get("missed_function", {}).items()
    }
    initially_missing = {
        function_name
        for function_names in missed_by_round.values()
        for function_name in function_names
    }
    current_tools = [
        tool for tool in all_tools if tool["name"] not in initially_missing
    ]
    held_out_tools = {
        tool["name"]: tool for tool in all_tools if tool["name"] in initially_missing
    }

    messages: list[AnnotatedMessage] = []
    trajectory_id = case["id"]
    executor_name = f"segtrace_{uuid.uuid4().hex}"
    requests: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    observations: list[list[str]] = []
    total_rounds = len(case["question"])
    if limit_rounds is not None:
        total_rounds = min(total_rounds, limit_rounds)

    for round_id in range(total_rounds):
        if round_id in missed_by_round:
            for function_name in missed_by_round[round_id]:
                current_tools.append(held_out_tools[function_name])
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

        system_parts = system_components(
            case,
            current_tools,
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
        rid = (
            f"bfclseg__{category.replace('multi_turn_', '')}__"
            f"{case['id'].rsplit('_', 1)[-1]}__round_{round_id:02d}"
        )
        prompt, prompt_segments = tokenize_with_segments(
            tokenizer,
            prompt_messages,
            current_tools,
            system_parts,
            trajectory_id=trajectory_id,
            category=category,
            round_id=round_id,
            rid=rid,
        )
        calls = list(answer["ground_truth"][round_id])
        results = execute_ground_truth(
            bfcl_package_root, case, calls, executor_name
        )
        observations.append(results)
        prompt["ground_truth_calls"] = calls
        prompt["ground_truth_results"] = results
        requests.append(prompt)
        segment_rows.extend(prompt_segments)

        messages.append(
            AnnotatedMessage(
                role="assistant",
                content=teacher_tool_call_message(calls),
                segment_type=(
                    "assistant_tool_call" if calls else "assistant_response"
                ),
                segment_id=(
                    f"assistant_tool_call::{round_id:02d}"
                    if calls
                    else f"assistant_response::{round_id:02d}"
                ),
                source_round=round_id,
            )
        )
        for result_index, result in enumerate(results):
            messages.append(
                AnnotatedMessage(
                    role="tool",
                    content=result,
                    segment_type="tool_result",
                    segment_id=f"tool_result::{round_id:02d}::{result_index:02d}",
                    source_round=round_id,
                )
            )

    snapshot = {
        "category": category,
        "case_id": case["id"],
        "rounds": total_rounds,
        "source_question": case,
        "source_answer": answer,
        "ground_truth_results": observations,
        "prompt_token_lengths": [request["prompt_len"] for request in requests],
        "involved_classes": case["involved_classes"],
    }
    return requests, segment_rows, snapshot


def load_category_entries(
    data_root: Path, category: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    question_path = data_root / f"BFCL_v4_{category}.json"
    answer_path = data_root / "possible_answer" / f"BFCL_v4_{category}.json"
    questions = read_jsonl(question_path)
    answers = {row["id"]: row for row in read_jsonl(answer_path)}
    return questions, answers


def select_and_prepare(
    tokenizer: Any,
    bfcl_package_root: Path,
    categories: list[str],
    trajectories_per_category: int,
    long_context_distractor: str,
    *,
    case_ids: list[str] | None,
    limit_rounds: int | None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    data_root = bfcl_package_root / "bfcl_eval" / "data"
    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    prepared_requests: list[dict[str, Any]] = []
    all_segments: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    used_suffixes: set[int] = set()
    explicit_ids = set(case_ids or [])

    for category in categories:
        questions, answers = load_category_entries(data_root, category)
        candidates = [
            case
            for case in questions
            if len(case["question"]) >= 4
            and (not explicit_ids or case["id"] in explicit_ids)
        ]
        rng.shuffle(candidates)
        category_count = 0
        for case in candidates:
            suffix = int(case["id"].rsplit("_", 1)[-1])
            if not explicit_ids and suffix in used_suffixes:
                exclusions.append(
                    {"case_id": case["id"], "reason": "duplicate_base_suffix"}
                )
                continue
            requests, segments, snapshot = build_case_requests(
                tokenizer,
                bfcl_package_root,
                category,
                case,
                answers[case["id"]],
                long_context_distractor,
                limit_rounds=limit_rounds,
            )
            lengths = [request["prompt_len"] for request in requests]
            if min(lengths) <= 2048:
                exclusions.append(
                    {
                        "case_id": case["id"],
                        "reason": "prompt_not_sparse",
                        "prompt_token_lengths": lengths,
                    }
                )
                continue
            if max(lengths) > 8000:
                exclusions.append(
                    {
                        "case_id": case["id"],
                        "reason": "prompt_exceeds_8000",
                        "prompt_token_lengths": lengths,
                    }
                )
                continue
            selected.append(
                {
                    "category": category,
                    "case_id": case["id"],
                    "rounds": len(requests),
                    "prompt_token_lengths": lengths,
                    "involved_classes": case["involved_classes"],
                }
            )
            prepared_requests.extend(requests)
            all_segments.extend(segments)
            snapshots.append(snapshot)
            used_suffixes.add(suffix)
            category_count += 1
            if explicit_ids or category_count == trajectories_per_category:
                if explicit_ids:
                    continue
                break
        expected = (
            sum(case_id.startswith(category + "_") for case_id in explicit_ids)
            if explicit_ids
            else trajectories_per_category
        )
        if category_count != expected:
            raise RuntimeError(
                f"Selected {category_count}/{expected} trajectories for {category}"
            )

    selection = {
        "schema_version": 1,
        "seed": SEED,
        "categories": categories,
        "trajectories_per_category": trajectories_per_category,
        "selected_count": len(selected),
        "request_count": len(prepared_requests),
        "selected": selected,
        "exclusions": exclusions,
    }
    return selection, prepared_requests, all_segments, snapshots


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
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def prepare_files(args: argparse.Namespace) -> list[dict[str, Any]]:
    bfcl_package_root = resolve_bfcl_package_root(args.bfcl_root)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    long_context_source = (
        bfcl_package_root
        / "bfcl_eval"
        / "eval_checker"
        / "multi_turn_eval"
        / "func_source_code"
        / "long_context.py"
    )
    long_context_distractor = literal_assignment(
        long_context_source, "FILE_CONTENT_EXTENSION"
    )
    categories = args.categories
    if args.case_ids:
        category_by_case = {
            case_id: case_id.rsplit("_", 1)[0] for case_id in args.case_ids
        }
        categories = list(dict.fromkeys(category_by_case.values()))
    selection, requests, segments, snapshots = select_and_prepare(
        tokenizer,
        bfcl_package_root,
        categories,
        args.trajectories_per_category,
        long_context_distractor,
        case_ids=args.case_ids,
        limit_rounds=args.limit_rounds,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "selection.json", selection)
    write_json(args.output_dir / "dataset_snapshot.json", snapshots)
    for path in [
        args.output_dir / "prepared_requests.jsonl",
        args.output_dir / "segments.jsonl",
    ]:
        path.unlink(missing_ok=True)
    for request in requests:
        append_jsonl(args.output_dir / "prepared_requests.jsonl", request)
    for segment in segments:
        append_jsonl(args.output_dir / "segments.jsonl", segment)

    run_config = {
        "schema_version": 1,
        "created_unix_s": time.time(),
        "seed": SEED,
        "model": args.model,
        "bfcl_root": str(bfcl_package_root),
        "bfcl_commit": git_revision(bfcl_package_root),
        "sglang_repo": str(Path.cwd()),
        "sglang_commit": git_revision(Path.cwd()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "categories": categories,
        "trajectories_per_category": args.trajectories_per_category,
        "case_ids": args.case_ids,
        "limit_rounds": args.limit_rounds,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
        "temperature": 0,
    }
    write_json(args.output_dir / "run_config.json", run_config)
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

    previous_trajectory = None
    for request in requests:
        trajectory_id = request["trajectory_id"]
        if trajectory_id != previous_trajectory:
            post_json(f"{args.base_url}/flush_cache", {}, args.timeout)
            previous_trajectory = trajectory_id
        actual_rid = request["rid"]
        if args.rid_tag:
            actual_rid = f"{actual_rid}__{args.rid_tag}"
        payload = {
            "input_ids": request["input_ids"],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "ignore_eos": args.ignore_eos,
            },
            "stream": False,
            "rid": actual_rid,
        }
        started = time.perf_counter()
        response = post_json(f"{args.base_url}/generate", payload, args.timeout)
        latency_s = time.perf_counter() - started
        append_jsonl(
            requests_path,
            {
                "trajectory_id": trajectory_id,
                "category": request["category"],
                "round_id": request["round_id"],
                "rid": actual_rid,
                "prepared_rid": request["rid"],
                "prompt_len": request["prompt_len"],
                "prompt_sha256": request["prompt_sha256"],
                "latency_s": latency_s,
                "ground_truth_calls": request["ground_truth_calls"],
                "ground_truth_results": request["ground_truth_results"],
                "response": response,
            },
        )
        print(
            json.dumps(
                {
                    "rid": request["rid"],
                    "actual_rid": actual_rid,
                    "prompt_len": request["prompt_len"],
                    "latency_s": round(latency_s, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    from audit_inference_outputs import audit_run

    audit = audit_run(args.output_dir)
    print(json.dumps({"inference_output_audit": audit}, ensure_ascii=False), flush=True)
    if not audit["passed"]:
        raise RuntimeError("inference output audit failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfcl-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/Tan/model/Keye-VL-2.0-30B-A3B")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        choices=DEFAULT_CATEGORIES,
    )
    parser.add_argument("--trajectories-per-category", type=int, default=6)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--limit-rounds", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="force a fixed decode length; disabled by default to preserve valid output",
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--rid-tag", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse_prepared:
        requests = read_jsonl(args.output_dir / "prepared_requests.jsonl")
    else:
        requests = prepare_files(args)
    if not args.prepare_only:
        run_requests(args, requests)


if __name__ == "__main__":
    main()
