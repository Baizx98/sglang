#!/usr/bin/env python3
"""Collect exact decode logprobs along an existing exact reference trajectory."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw.decode(errors="replace")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--top-logprobs-num", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--split", choices=["calibration", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared_run = args.prepared_run.resolve()
    reference_run = args.reference_run.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = {
        row["rid"]: row for row in read_jsonl(prepared_run / "prepared_requests.jsonl")
    }
    reference = {
        row["prepared_rid"]: row for row in read_jsonl(reference_run / "requests.jsonl")
    }
    selection = json.loads((prepared_run / "selection.json").read_text())
    selected = [
        row
        for row in selection["selected"]
        if args.split is None or row["split"] == args.split
    ]

    output_path = args.output_dir / "teacher_forced_steps.jsonl"
    requests_path = args.output_dir / "requests.jsonl"
    output_path.unlink(missing_ok=True)
    requests_path.unlink(missing_ok=True)
    old_reference_mismatch_count = 0
    for request_index, meta in enumerate(selected):
        rid = meta["rid"]
        post_json(f"{args.base_url}/flush_cache", {}, args.timeout)
        old_reference_ids = list(
            reference[rid]["response"]["output_ids"][: args.steps]
        )
        payload = {
            "input_ids": prepared[rid]["input_ids"],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": len(old_reference_ids),
                "ignore_eos": True,
            },
            "stream": False,
            "rid": f"tfexact__{request_index:02d}",
            "return_logprob": True,
            "top_logprobs_num": args.top_logprobs_num,
        }
        started = time.perf_counter()
        response = post_json(f"{args.base_url}/generate", payload, args.timeout)
        latency_s = time.perf_counter() - started
        output_ids = list(response["output_ids"])
        old_reference_match = output_ids == old_reference_ids
        old_reference_mismatch_count += int(not old_reference_match)
        append_jsonl(
            requests_path,
            {
                "prepared_rid": rid,
                "rid": payload["rid"],
                "trajectory_id": meta["trajectory_id"],
                "category": meta["category"],
                "round_id": meta["round_id"],
                "prompt_len": meta["prompt_len"],
                "latency_s": latency_s,
                "response": response,
                "old_reference_match": old_reference_match,
            },
        )
        meta_info = response["meta_info"]
        logprobs = meta_info["output_token_logprobs"]
        top_logprobs = meta_info["output_top_logprobs"]
        for step, reference_token in enumerate(output_ids):
            append_jsonl(
                output_path,
                {
                    "variant": "exact",
                    "rid": rid,
                    "trajectory_id": meta["trajectory_id"],
                    "category": meta["category"],
                    "split": meta["split"],
                    "prompt_len": meta["prompt_len"],
                    "step": step,
                    "context_len": meta["prompt_len"] + step,
                    "reference_token": int(reference_token),
                    "predicted_token": int(output_ids[step]),
                    "top1_match": True,
                    "reference_logprob": float(logprobs[step][0]),
                    "reference_nll": -float(logprobs[step][0]),
                    "top_logprobs": [
                        {"logprob": float(entry[0]), "token_id": int(entry[1])}
                        for entry in top_logprobs[step]
                    ],
                    "request_latency_s": latency_s,
                },
            )
        print(
            json.dumps(
                {
                    "rid": rid,
                    "steps": len(output_ids),
                    "latency_s": latency_s,
                    "old_reference_match": old_reference_match,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    rows = read_jsonl(output_path)
    summary = {
        "schema_version": 1,
        "variant": "exact",
        "request_count": len(selected),
        "step_count": len(rows),
        "top1_agreement": 1.0,
        "reference_nll_mean": sum(row["reference_nll"] for row in rows) / len(rows),
        "old_reference_mismatch_count": old_reference_mismatch_count,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
