"""Freeze SWE-bench Verified and BFCL prompts for decode-only trace collection."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
from urllib.request import urlopen

from transformers import AutoTokenizer

MODEL = "/Tan/model/Keye-VL-2.0-30B-A3B"
REV = "773748a7c1222e8a642a7059821498e14293562a"
SOURCE = f"https://huggingface.co/datasets/tarsur385/swebench-verified-trajectories/resolve/{REV}"
COHORT = (
    "astropy__astropy-12907", "sympy__sympy-15349",
    "pytest-dev__pytest-10051", "sphinx-doc__sphinx-10323",
    "pydata__xarray-4356", "scikit-learn__scikit-learn-12682",
    "astropy__astropy-14182", "sympy__sympy-21596",
    "pytest-dev__pytest-7571", "sphinx-doc__sphinx-7440",
    "pydata__xarray-3151", "scikit-learn__scikit-learn-25102",
)
TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Execute a shell command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]},
}}]


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch(source, case):
    name = f"{case}.traj.json"
    url = f"{SOURCE}/swebench_verified_raw/gpt-5-mini/{case}/{name}"
    path = source / "normalized" / name
    if not path.exists():
        with urlopen(url, timeout=30) as response:
            data = response.read()
        path.write_bytes(data)
    payload = path.read_bytes()
    row = json.loads(payload)
    if row["instance_id"] != case:
        raise ValueError(f"Wrong trajectory for {case}")
    return row["messages"], url, hashlib.sha256(payload).hexdigest()


def normalize(message):
    result = {"role": message["role"], "content": message["content"] or ""}
    calls = []
    for call in message.get("tool_calls") or []:
        call = json.loads(call) if isinstance(call, str) else call
        function = call["function"]
        if isinstance(function, str):
            function = {"name": function, "arguments": call["arguments"]}
        calls.append({"function": function})
    if calls:
        result["tool_calls"] = calls
    return result


def render(tokenizer, messages, turns):
    prompt = tokenizer.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
    )
    spans = []
    schema_begin = prompt.index("<tools>") + len("<tools>")
    schema_end = prompt.index("</tools>", schema_begin)
    cursor = 0
    for message, origin in zip(messages, turns, strict=True):
        role, content = message["role"], message["content"]
        if content:
            begin = prompt.index(content, cursor)
            end = begin + len(content)
            spans.append((begin, end, role if role != "tool" else "tool_result", origin))
            cursor = end
        if role == "system":
            spans.append((schema_begin, schema_end, "tool_schema", None))
            cursor = schema_end
        for _ in message.get("tool_calls", ()):
            begin = prompt.index("\n<tool_call>\n", cursor)
            end = prompt.index("</tool_call>", begin) + len("</tool_call>")
            spans.append((begin, end, "tool_call", origin))
            cursor = end
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded.input_ids
    labels = [("chat_template", None)] * len(ids)
    cursor = 0
    for i, (begin, end) in enumerate(encoded.offset_mapping):
        while cursor < len(spans) and spans[cursor][1] <= begin:
            cursor += 1
        if cursor < len(spans):
            start, stop, role, origin = spans[cursor]
            if start <= begin < end <= stop:
                labels[i] = role, origin
    segments = []
    for (role, origin), group in itertools.groupby(labels):
        begin = segments[-1]["end"] if segments else 0
        end = begin + sum(1 for _ in group)
        segments.append(dict(role=role, source_turn=origin, start=begin, end=end))
    return ids, segments


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def swe_rows(tokenizer, case, data, split):
    raw_messages, url, digest = data
    messages, turns, previous = [], [], []
    turn = 0
    for raw in raw_messages:
        message = normalize(raw)
        if message["role"] == "assistant":
            ids, segments = render(tokenizer, messages, turns)
            yield dict(rid=f"agenttrace__swe__{case}__{turn:03d}",
                       dataset="SWE-bench Verified", session_id=case, turn=turn,
                       split=split, source_url=url, source_sha256=digest,
                       input_ids=ids, segments=segments, prompt_tokens=len(ids),
                       shared_prefix_tokens=common_prefix(previous, ids))
            previous = ids
            turn += 1
        messages.append(message)
        turns.append(max(0, turn - 1) if message["role"] in ("assistant", "tool") else 0)


def bfcl_rows(root):
    segments = {}
    for row in read_jsonl(root / "segments.jsonl"):
        segments.setdefault(row["rid"], []).append(dict(
            role=row["segment_type"], start=row["token_start"], end=row["token_end"],
            source_turn=row["source_round"]))
    previous = {}
    for row in sorted(read_jsonl(root / "prepared_requests.jsonl"),
                      key=lambda value: (value["trajectory_id"], value["round_id"])):
        session = row["trajectory_id"]
        ids = row["input_ids"]
        yield dict(rid="agenttrace__" + row["rid"], dataset="BFCL", session_id=session,
                   turn=row["round_id"], input_ids=ids, prompt_tokens=len(ids),
                   shared_prefix_tokens=common_prefix(previous.get(session, []), ids),
                   segments=segments[row["rid"]])
        previous[session] = ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bfcl", type=Path, required=True)
    args = parser.parse_args()
    (args.out / "source/normalized").mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        data = list(pool.map(lambda case: fetch(args.out / "source", case), COHORT))
    swe = []
    for i, (case, trajectory) in enumerate(zip(COHORT, data, strict=True)):
        split = "pilot" if i < 2 else "development" if i < 6 else "heldout"
        rows = list(swe_rows(tokenizer, case, trajectory, split))
        if len(rows) < 2 or max(row["prompt_tokens"] for row in rows) + 256 >= 131328:
            raise ValueError(f"Trajectory outside the frozen replay limits: {case}")
        swe.extend(rows)
    bfcl = list(bfcl_rows(args.bfcl))
    for name, rows in (("swe", swe), ("bfcl", bfcl)):
        write_jsonl(args.out / f"{name}-requests.jsonl", rows)
        print(name, len({r["session_id"] for r in rows}), len(rows),
              min(r["prompt_tokens"] for r in rows), max(r["prompt_tokens"] for r in rows),
              flush=True)


if __name__ == "__main__":
    main()
