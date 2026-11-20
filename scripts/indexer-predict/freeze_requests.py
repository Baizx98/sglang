"""Freeze distinct target-model requests before tuning the predictor."""

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer


MODEL = "/Tan/model/Keye-VL-2.0-30B-A3B"


def fingerprint(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def balanced(rows, count, rng):
    if len(rows) < count:
        raise ValueError(f"cohort needs {count} distinct inputs, found {len(rows)}")
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda item: item["case_id"]):
        groups[row["dataset"], row["task"]].append(row)
    order = sorted(groups)
    rng.shuffle(order)
    active = deque(order)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while len(result) < count:
        task = active.popleft()
        result.append(groups[task].pop())
        if groups[task]:
            active.append(task)
    return result


def write(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def run(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    cohort = []
    source = [json.loads(line) for line in args.previous.read_text().splitlines()]
    for bucket in (65536, 131072):
        for split, count in (("calibration", 4), ("test", 16)):
            rows = [row for row in source if row["length_bucket_upper"] == bucket
                    and row["split"] == split and not row.get("truncated", False)]
            for ordinal, row in enumerate(balanced(rows, count, rng)):
                cohort.append(dict(
                    row, rid=f"predict-full-{bucket}-{split}-{ordinal:02}",
                    source_request=row["rid"], source_file=str(args.previous),
                ))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    long_rows = []
    for ordinal, line in enumerate(args.near256.read_text().splitlines()):
        source_row = json.loads(line)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": source_row["input"]}], tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert 250000 < len(ids) <= 262128, len(ids)
        split = "calibration" if ordinal < 4 else "test"
        long_rows.append(dict(
            rid=f"predict-full-262144-{split}-{ordinal:02}", split=split,
            task="niah_multikey_1", dataset="RULER", length_bucket_upper=262144,
            input_ids=ids, input_ids_sha256=fingerprint(ids), prompt_tokens=len(ids),
            expected_answer=source_row["outputs"],
            source_file=str(args.near256), source_line=ordinal + 1,
        ))
    assert len(long_rows) == 20
    all_rows = cohort + long_rows
    assert len({fingerprint(row["input_ids"]) for row in all_rows}) == len(all_rows)
    write(root / "near256-gate.jsonl", long_rows[:1])
    write(root / "near256-remaining.jsonl", long_rows[1:])
    write(root / "shorter.jsonl", cohort)
    write(root / "manifest.jsonl", [{k: v for k, v in row.items() if k != "input_ids"} for row in all_rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--near256", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261120)
    run(parser.parse_args())
