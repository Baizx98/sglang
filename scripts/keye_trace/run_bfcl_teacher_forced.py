#!/usr/bin/env python3
"""Replay two deterministic BFCL multi-turn trajectories against an OpenAI API."""

from __future__ import annotations

import argparse
import ast
import json
import platform
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

SEED = 20260730
BASE_CASE_ID = "multi_turn_base_2"
LONG_CASE_ID = "multi_turn_long_context_0"


@dataclass
class Trajectory:
    name: str
    category: str
    case_id: str
    messages: list[dict[str, str]]
    ground_truth: list[list[str]]
    observations: list[str]
    source_question: dict[str, Any]
    source_answer: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def select_entry(path: Path, entry_id: str) -> dict[str, Any]:
    for entry in read_jsonl(path):
        if entry["id"] == entry_id:
            return entry
    raise KeyError(f"{entry_id} not found in {path}")


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


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_system_prompt(
    case: dict[str, Any], tools: list[dict[str, Any]], distractor: str = ""
) -> str:
    prompt = (
        "You are an agent in the Berkeley Function Calling Leaderboard (BFCL). "
        "For each user turn, briefly state the necessary action and emit the "
        "tool calls "
        "as Python-like expressions. Do not invent tools or arguments. The experiment "
        "will replace your answer with a deterministic teacher-forced action and tool "
        "observation before the next LLM round.\n\n"
        f"AVAILABLE_TOOLS={compact_json(tools)}\n\n"
        f"INITIAL_STATE={compact_json(case['initial_config'])}"
    )
    if distractor:
        prompt += "\n\nOFFICIAL_BFCL_LONG_CONTEXT_FILE_EXTENSION=" + distractor
    return prompt


def teacher_message(calls: list[str]) -> str:
    return "Teacher-forced action:\n" + "\n".join(f"- {call}" for call in calls)


def build_trajectories(bfcl_root: Path) -> list[Trajectory]:
    data_root = bfcl_root / "bfcl_eval" / "data"
    func_root = data_root / "multi_turn_func_doc"
    answer_root = data_root / "possible_answer"

    base_case = select_entry(data_root / "BFCL_v4_multi_turn_base.json", BASE_CASE_ID)
    base_answer = select_entry(
        answer_root / "BFCL_v4_multi_turn_base.json", BASE_CASE_ID
    )
    long_case = select_entry(
        data_root / "BFCL_v4_multi_turn_long_context.json", LONG_CASE_ID
    )
    long_answer = select_entry(
        answer_root / "BFCL_v4_multi_turn_long_context.json", LONG_CASE_ID
    )

    fs_tools = read_jsonl(func_root / "gorilla_file_system.json")
    posting_tools = read_jsonl(func_root / "posting_api.json")
    long_context_source = (
        bfcl_root
        / "bfcl_eval"
        / "eval_checker"
        / "multi_turn_eval"
        / "func_source_code"
        / "long_context.py"
    )
    file_extension = literal_assignment(long_context_source, "FILE_CONTENT_EXTENSION")

    base_system = build_system_prompt(base_case, fs_tools)
    long_system = build_system_prompt(
        long_case, fs_tools + posting_tools, file_extension
    )

    base_observations = [
        "cd => /simona/documents; touch => TeamNotes.txt created.",
        (
            "echo => TeamNotes.txt now contains: Collaboration leads to success. "
            "Innovation ignites growth."
        ),
        "diff => no line differences between ideas.txt and TeamNotes.txt.",
        (
            "cp => TeamNotes.txt copied to Archived; cd => "
            "/simona/documents/Archived; mv => IdeasArchive.txt."
        ),
        "cat => Collaboration leads to success. Innovation ignites growth.",
    ]
    long_observations = [
        (
            "cd => /workspace/document; mkdir => temp created; mv => "
            "final_report.pdf moved into temp."
        ),
        (
            "cd => /workspace/document/temp; grep => Year2024 This is the final "
            "report content including budget analysis and other sections."
        ),
        (
            "sort => Year2024 This is the final report content including budget "
            "analysis and other sections."
        ),
        (
            "cd => /workspace/document; mv => previous_report.pdf moved into temp; "
            "cd => temp; diff => Year2024 differs from Year203 and both "
            "budget-analysis lines differ."
        ),
    ]

    def initial_messages(system_prompt: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system_prompt}]

    return [
        Trajectory(
            name="T1",
            category="multi_turn_base",
            case_id=BASE_CASE_ID,
            messages=initial_messages(base_system),
            ground_truth=base_answer["ground_truth"],
            observations=base_observations,
            source_question=base_case,
            source_answer=base_answer,
        ),
        Trajectory(
            name="T2",
            category="multi_turn_long_context",
            case_id=LONG_CASE_ID,
            messages=initial_messages(long_system),
            ground_truth=long_answer["ground_truth"],
            observations=long_observations,
            source_question=long_case,
            source_answer=long_answer,
        ),
    ]


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
        return json.loads(response.read().decode())


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def estimate_prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    # The multimodal server converts string content to OpenAI content blocks
    # before templating. AutoTokenizer alone does not perform that conversion.
    serialized = "\n".join(
        f"<{message['role']}>\n{message['content']}" for message in messages
    )
    return len(tokenizer.encode(serialized))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfcl-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/Tan/model/Keye-VL-2.0-30B-A3B")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    trajectories = build_trajectories(args.bfcl_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    snapshot = {
        trajectory.name: {
            "category": trajectory.category,
            "case_id": trajectory.case_id,
            "source_question": trajectory.source_question,
            "source_answer": trajectory.source_answer,
            "initial_system_prompt": trajectory.messages[0]["content"],
            "observations": trajectory.observations,
        }
        for trajectory in trajectories
    }
    (args.output_dir / "dataset_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
    )

    config = {
        "seed": SEED,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "concurrency": 1,
        "teacher_forced": True,
        "flush_between_trajectories": True,
        "preserve_radix_within_trajectory": True,
        "model": args.model,
        "base_url": args.base_url,
        "code_revision": git_revision(Path.cwd()),
        "bfcl_revision": git_revision(args.bfcl_root.parent),
        "python": platform.python_version(),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )

    estimates = {}
    for trajectory in trajectories:
        messages = list(trajectory.messages)
        round_estimates = []
        for round_id, turn in enumerate(trajectory.source_question["question"]):
            messages.extend(turn)
            round_estimates.append(estimate_prompt_tokens(tokenizer, messages))
            messages.append(
                {
                    "role": "assistant",
                    "content": teacher_message(trajectory.ground_truth[round_id]),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "[DETERMINISTIC_TOOL_RESULTS]\n"
                    + trajectory.observations[round_id],
                }
            )
        estimates[trajectory.name] = round_estimates
    (args.output_dir / "prompt_token_estimates.json").write_text(
        json.dumps(estimates, indent=2) + "\n"
    )
    print("prompt-token estimates:", estimates, flush=True)
    if args.prepare_only:
        return

    server_info = get_json(f"{args.base_url}/model_info", args.timeout)
    (args.output_dir / "server_info.json").write_text(
        json.dumps(server_info, indent=2, ensure_ascii=False) + "\n"
    )

    request_log = args.output_dir / "requests.jsonl"
    for trajectory in trajectories:
        post_json(f"{args.base_url}/flush_cache", {}, args.timeout)
        messages = list(trajectory.messages)
        for round_id, turn in enumerate(trajectory.source_question["question"]):
            messages.extend(turn)
            rid = f"bfcl__{trajectory.name}__round_{round_id:02d}"
            payload = {
                "model": args.model,
                "messages": messages,
                "temperature": 0,
                "seed": SEED,
                "max_tokens": args.max_tokens,
                "stream": False,
                "rid": rid,
            }
            start = time.perf_counter()
            response = post_json(
                f"{args.base_url}/v1/chat/completions", payload, args.timeout
            )
            latency_s = time.perf_counter() - start
            record = {
                "trajectory": trajectory.name,
                "category": trajectory.category,
                "case_id": trajectory.case_id,
                "round_id": round_id,
                "rid": rid,
                "latency_s": latency_s,
                "prompt_token_estimate": estimate_prompt_tokens(tokenizer, messages),
                "payload": payload,
                "response": response,
                "teacher_action": trajectory.ground_truth[round_id],
                "teacher_observation": trajectory.observations[round_id],
            }
            with request_log.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"{rid}: latency={latency_s:.3f}s usage={response.get('usage')}",
                flush=True,
            )

            # Do not feed the sampled answer into the next round. This is the
            # defining teacher-forcing step and makes repeated runs comparable.
            messages.append(
                {
                    "role": "assistant",
                    "content": teacher_message(trajectory.ground_truth[round_id]),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "[DETERMINISTIC_TOOL_RESULTS]\n"
                    + trajectory.observations[round_id],
                }
            )


if __name__ == "__main__":
    main()
