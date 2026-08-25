#!/usr/bin/env python3
"""Prepare current tau3-bench base prompts with instance-level region labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


DOMAINS = ("airline", "retail")
AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()
SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(compact_json(row) + "\n")


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def official_result_path(tau_root: Path, domain: str) -> Path:
    matches = sorted(
        (tau_root / "data/tau2/results/final").glob(
            f"gpt-4.1-mini-2025-04-14_{domain}_base_*trials.json"
        )
    )
    if len(matches) != 1:
        raise ValueError(f"expected one official base result for {domain}, got {matches}")
    return matches[0]


def load_current_tool_schemas(tau_root: Path, domain: str) -> list[dict[str, Any]]:
    """Export schemas through the pinned tau3-bench environment implementation."""
    code = (
        f"from tau2.domains.{domain}.environment import get_environment;"
        "import json;"
        "print('SCHEMA_JSON='+json.dumps([t.openai_schema for t in "
        "get_environment().get_tools()],separators=(',',':')))"
    )
    result = subprocess.run(
        ["uv", "run", "--frozen", "python", "-c", code],
        cwd=tau_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("SCHEMA_JSON=")]
    if len(lines) != 1:
        raise ValueError(
            f"failed to export {domain} tool schemas (rc={result.returncode}): "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
    return json.loads(lines[0].split("=", 1)[1])


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message["role"]
    normalized: dict[str, Any] = {"role": role}
    if message.get("content") is not None:
        normalized["content"] = str(message["content"])
    elif role == "assistant":
        normalized["content"] = None
    if role == "tool":
        normalized["name"] = message.get("name")
        normalized["tool_call_id"] = message.get("tool_call_id")
    calls = message.get("tool_calls") or []
    if calls:
        normalized["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": compact_json(call.get("arguments") or {}),
                },
            }
            for call in calls
        ]
    return normalized


def assistant_positions(messages: list[dict[str, Any]]) -> list[int]:
    return [index for index, message in enumerate(messages) if message["role"] == "assistant"]


def select_sessions(
    tau_root: Path, *, sessions_per_domain: int, min_invocations: int, max_invocations: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for domain in DOMAINS:
        current_path = tau_root / f"data/tau2/domains/{domain}/tasks.json"
        current = {str(row["id"]): row for row in json.loads(current_path.read_text())}
        result_path = official_result_path(tau_root, domain)
        result = json.loads(result_path.read_text())
        historical = {str(row["id"]): row for row in result["tasks"]}
        compatible_ids = {
            task_id
            for task_id in current.keys() & historical.keys()
            if current[task_id].get("user_scenario")
            == historical[task_id].get("user_scenario")
            and current[task_id].get("initial_state")
            == historical[task_id].get("initial_state")
        }
        candidates = []
        for simulation in result["simulations"]:
            task_id = str(simulation["task_id"])
            positions = assistant_positions(simulation["messages"])
            reward = float(simulation["reward_info"].get("reward", 0.0))
            if (
                task_id in compatible_ids
                and int(simulation["trial"]) == 0
                and reward == 1.0
                and min_invocations <= len(positions) <= max_invocations
            ):
                candidates.append((len(positions), int(task_id), simulation))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if len(candidates) < sessions_per_domain:
            raise ValueError(f"{domain}: only {len(candidates)} compatible sessions")
        for num_invocations, _, simulation in candidates[:sessions_per_domain]:
            selected.append(
                {
                    "domain": domain,
                    "task_id": str(simulation["task_id"]),
                    "trial": int(simulation["trial"]),
                    "historical_reward": 1.0,
                    "historical_result_commit": result.get("info", {}).get("git_commit"),
                    "source_file": str(result_path),
                    "messages": [normalize_message(row) for row in simulation["messages"]],
                    "num_invocations": num_invocations,
                }
            )
        audit[domain] = {
            "current_task_count": len(current),
            "compatible_prompt_task_count": len(compatible_ids),
            "compatible_definition": "exact equality of user_scenario and initial_state",
            "source_result": str(result_path),
            "source_result_git_commit": result.get("info", {}).get("git_commit"),
        }
    return sorted(selected, key=lambda row: (row["domain"], int(row["task_id"]))), audit


def tool_schema_spans(
    rendered: str, tools: list[dict[str, Any]], session_id: str
) -> list[dict[str, Any]]:
    start_marker, end_marker = "<tools>\n", "\n</tools>"
    start = rendered.index(start_marker) + len(start_marker)
    end = rendered.index(end_marker, start)
    cursor = start
    spans = []
    by_name = {tool["function"]["name"]: index for index, tool in enumerate(tools)}
    observed = set()
    for line in rendered[start:end].splitlines():
        if not line.strip():
            cursor += len(line) + 1
            continue
        line_start = rendered.index(line, cursor, end)
        name = json.loads(line)["function"]["name"]
        spans.append(
            {
                "char_start": line_start,
                "char_end": line_start + len(line),
                "region_id": f"{session_id}:tool-schema:{by_name[name]:03d}",
                "region_type": "tool_schema",
                "region_label": name,
                "created_turn": 0,
            }
        )
        observed.add(name)
        cursor = line_start + len(line) + 1
    if observed != set(by_name):
        raise ValueError(f"tool schema render mismatch for {session_id}")
    return spans


def message_region(message: dict[str, Any]) -> str:
    if message["role"] == "user":
        return "user_turn"
    if message["role"] == "tool":
        return "tool_result"
    if message["role"] == "assistant" and message.get("tool_calls"):
        return "assistant_tool_call"
    if message["role"] == "assistant":
        return "assistant_response"
    if message["role"] == "system":
        return "system_instruction"
    raise ValueError(f"unsupported role: {message['role']}")


def message_created_turn(messages: list[dict[str, Any]], index: int) -> int:
    user_count = sum(message["role"] == "user" for message in messages[: index + 1])
    return max(0, user_count - 1)


def locate_message_spans(
    rendered: str, messages: list[dict[str, Any]], session_id: str
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    cursor = 0
    for message_index, message in enumerate(messages):
        region_type = message_region(message)
        if message_index == 0 and message["role"] == "system":
            content = str(message.get("content") or "")
            start = rendered.index(content, cursor)
            spans.append(
                {
                    "char_start": start,
                    "char_end": start + len(content),
                    "region_id": f"{session_id}:system:000",
                    "region_type": region_type,
                    "region_label": "system",
                    "created_turn": 0,
                }
            )
            cursor = start + len(content)
            continue
        region_id = f"{session_id}:message:{message_index:03d}"
        created_turn = message_created_turn(messages, message_index)
        content = message.get("content")
        if content:
            start = rendered.index(str(content), cursor)
            spans.append(
                {
                    "char_start": start,
                    "char_end": start + len(str(content)),
                    "region_id": region_id,
                    "region_type": region_type,
                    "region_label": str(message.get("name") or region_type),
                    "created_turn": created_turn,
                }
            )
            cursor = start + len(str(content))
        if message.get("tool_calls"):
            start = rendered.index("<tool_call>", cursor)
            end = rendered.index("</tool_call>", start) + len("</tool_call>")
            spans.append(
                {
                    "char_start": start,
                    "char_end": end,
                    "region_id": region_id,
                    "region_type": region_type,
                    "region_label": message["tool_calls"][0]["function"]["name"],
                    "created_turn": created_turn,
                }
            )
            cursor = end
    return spans


def tokenize_prompt(
    tokenizer: Any,
    *,
    session: dict[str, Any],
    tools: list[dict[str, Any]],
    invocation_id: int,
    assistant_message_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session_id = session["session_id"]
    history = session["messages"][:assistant_message_index]
    turn_id = max(0, sum(message["role"] == "user" for message in history) - 1)
    messages = [{"role": "system", "content": session["system_prompt"]}, *history]
    rendered = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    templated = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True
    )
    templated_ids = list(templated["input_ids"] if hasattr(templated, "keys") else templated)
    if input_ids != templated_ids:
        raise ValueError(f"render/tokenize mismatch for {session_id} invocation {invocation_id}")
    spans = tool_schema_spans(rendered, tools, session_id)
    spans.extend(locate_message_spans(rendered, messages, session_id))
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    token_regions: dict[str, list[int]] = defaultdict(list)
    span_by_region = {span["region_id"]: span for span in spans}
    for token_index, (token_start, token_end) in enumerate(offsets):
        best_region, best_overlap = None, 0
        for span in spans:
            overlap = max(
                0,
                min(token_end, span["char_end"])
                - max(token_start, span["char_start"]),
            )
            if overlap > best_overlap:
                best_overlap, best_region = overlap, span["region_id"]
        if best_region is not None:
            token_regions[best_region].append(token_index)
    region_rows = []
    for region_id, token_indices in token_regions.items():
        token_start, token_end = min(token_indices), max(token_indices) + 1
        if token_indices != list(range(token_start, token_end)):
            raise ValueError(f"non-contiguous region span: {region_id}")
        span = span_by_region[region_id]
        region_rows.append(
            {
                "session_id": session_id,
                "domain": session["domain"],
                "task_id": session["task_id"],
                "invocation_id": invocation_id,
                "turn_id": turn_id,
                "region_id": region_id,
                "region_type": span["region_type"],
                "region_label": span["region_label"],
                "created_turn": int(span["created_turn"]),
                "turn_age": turn_id - int(span["created_turn"]),
                "token_start": token_start,
                "token_end": token_end,
                "num_tokens": token_end - token_start,
                "token_ids_sha256": sha256_json(input_ids[token_start:token_end]),
            }
        )
    rid = f"fig3tau3__{session_id}__inv{invocation_id:02d}__turn{turn_id:02d}"
    return (
        {
            "schema_version": 1,
            "rid": rid,
            "session_id": session_id,
            "domain": session["domain"],
            "task_id": session["task_id"],
            "trial": session["trial"],
            "invocation_id": invocation_id,
            "turn_id": turn_id,
            "assistant_message_index": assistant_message_index,
            "prompt_len": len(input_ids),
            "prompt_sha256": sha256_json(input_ids),
            "input_ids": input_ids,
        },
        sorted(region_rows, key=lambda row: row["token_start"]),
    )


def prepare(args: argparse.Namespace) -> None:
    tau_root = args.tau_root.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True
    )
    selected, compatibility_audit = select_sessions(
        tau_root,
        sessions_per_domain=args.sessions_per_domain,
        min_invocations=args.min_invocations,
        max_invocations=args.max_invocations,
    )
    tools_by_domain = {domain: load_current_tool_schemas(tau_root, domain) for domain in DOMAINS}
    policies = {
        domain: (tau_root / f"data/tau2/domains/{domain}/policy.md").read_text()
        for domain in DOMAINS
    }
    requests: list[dict[str, Any]] = []
    span_rows: list[dict[str, Any]] = []
    session_manifest = []
    for selection_index, session in enumerate(selected):
        session_id = (
            f"{session['domain']}-task{int(session['task_id']):03d}-"
            f"trial{session['trial']}"
        )
        session["session_id"] = session_id
        session["system_prompt"] = SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION, domain_policy=policies[session["domain"]]
        )
        positions = assistant_positions(session["messages"])
        for invocation_id, message_index in enumerate(positions):
            request, regions = tokenize_prompt(
                tokenizer,
                session=session,
                tools=tools_by_domain[session["domain"]],
                invocation_id=invocation_id,
                assistant_message_index=message_index,
            )
            requests.append(request)
            span_rows.extend(regions)
        session_manifest.append(
            {
                "selection_index": selection_index,
                "session_id": session_id,
                "domain": session["domain"],
                "task_id": session["task_id"],
                "trial": session["trial"],
                "historical_reward": session["historical_reward"],
                "num_invocations": len(positions),
                "message_count": len(session["messages"]),
                "source_file": session["source_file"],
                "source_result_commit": session["historical_result_commit"],
            }
        )
    canonical: dict[str, tuple[int, int, int, str]] = {}
    for row in sorted(span_rows, key=lambda value: (value["invocation_id"], value["token_start"])):
        signature = (
            int(row["token_start"]),
            int(row["token_end"]),
            int(row["num_tokens"]),
            row["token_ids_sha256"],
        )
        previous = canonical.setdefault(row["region_id"], signature)
        if previous != signature:
            raise ValueError(f"region token range drifted: {row['region_id']}")
    write_jsonl(args.output_dir / "prepared_requests.jsonl", requests)
    write_jsonl(args.output_dir / "region_spans.jsonl", span_rows)
    write_json(args.output_dir / "tool_schemas.json", tools_by_domain)
    write_json(
        args.output_dir / "selection_manifest.json",
        {
            "schema_version": 1,
            "workload": "tau3-bench",
            "task_split": "base",
            "tau3_repository": str(tau_root),
            "tau3_commit": git_revision(tau_root),
            "model_path": str(args.model_path.resolve()),
            "selection_rule": "longest successful trial-0 sessions among prompt-compatible tasks",
            "sessions_per_domain": args.sessions_per_domain,
            "min_invocations": args.min_invocations,
            "max_invocations": args.max_invocations,
            "compatibility_audit": compatibility_audit,
            "sessions": session_manifest,
            "request_count": len(requests),
            "region_span_rows": len(span_rows),
            "agent_visible_initial_state_region": False,
            "initial_state_note": (
                "Selected tasks have no agent-visible initial-state prompt segment; hidden "
                "environment state was not exposed or synthesized."
            ),
        },
    )
    print(
        json.dumps(
            {
                "sessions": len(session_manifest),
                "requests": len(requests),
                "region_span_rows": len(span_rows),
                "prompt_len_min": min(row["prompt_len"] for row in requests),
                "prompt_len_max": max(row["prompt_len"] for row in requests),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions-per-domain", type=int, default=2)
    parser.add_argument("--min-invocations", type=int, default=10)
    parser.add_argument("--max-invocations", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
