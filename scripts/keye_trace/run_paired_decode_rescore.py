#!/usr/bin/env python3
"""Run exact/exact/candidate rows together and compare a real decode step."""

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


def compact_response(response: dict[str, Any]) -> dict[str, Any]:
    meta = response["meta_info"]
    return {
        "rid": meta["id"],
        "output_ids": [int(value) for value in response["output_ids"]],
        "output_token_logprobs": [float(row[0]) for row in meta["output_token_logprobs"]],
        "output_top_logprobs": [
            [
                {"logprob": float(entry[0]), "token_id": int(entry[1])}
                for entry in step
            ]
            for step in meta["output_top_logprobs"]
        ],
        "finish_reason": meta.get("finish_reason"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--top-logprobs-num", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--split", choices=["calibration", "test"], default="test")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--exact-rid-prefix", default="pair_exact__")
    parser.add_argument("--candidate-rid-prefix", default="pair_candidate__")
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
    selected = [row for row in selection["selected"] if row["split"] == args.split]
    if args.max_requests is not None:
        selected = selected[: args.max_requests]

    output_path = args.output_dir / "paired_decode_steps.jsonl"
    output_path.unlink(missing_ok=True)
    for request_index, meta in enumerate(selected):
        source_rid = meta["rid"]
        prompt_ids = list(prepared[source_rid]["input_ids"])
        reference_ids = list(reference[source_rid]["response"]["output_ids"])
        step_count = min(args.steps, len(reference_ids))
        for step in range(step_count):
            # The first generated token comes from prefill. The second token is
            # the decode step where candidate rescoring is active.
            input_ids = prompt_ids + reference_ids[:step]
            suffix = f"{request_index:02d}__step_{step:03d}"
            rids = [
                f"{args.exact_rid_prefix}a__{suffix}",
                f"{args.exact_rid_prefix}b__{suffix}",
                f"{args.candidate_rid_prefix}{suffix}",
            ]
            payload = {
                "input_ids": [input_ids, input_ids, input_ids],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 2,
                    "ignore_eos": True,
                },
                "stream": False,
                "rid": rids,
                "return_logprob": True,
                "top_logprobs_num": args.top_logprobs_num,
            }
            post_json(f"{args.base_url}/flush_cache", {}, args.timeout)
            started = time.perf_counter()
            raw_responses = post_json(f"{args.base_url}/generate", payload, args.timeout)
            latency_s = time.perf_counter() - started
            if not isinstance(raw_responses, list) or len(raw_responses) != 3:
                raise RuntimeError(
                    f"expected three batched responses, got {type(raw_responses).__name__}"
                )
            by_rid = {
                response["meta_info"]["id"]: compact_response(response)
                for response in raw_responses
            }
            if set(by_rid) != set(rids):
                raise RuntimeError(f"response ids differ: expected={rids}, got={list(by_rid)}")
            if any(len(by_rid[rid]["output_ids"]) != 2 for rid in rids):
                raise RuntimeError(f"paired request did not return two tokens: {rids}")
            row = {
                "schema_version": 1,
                "source_rid": source_rid,
                "trajectory_id": meta["trajectory_id"],
                "category": meta["category"],
                "split": meta["split"],
                "step": step,
                "prompt_len": meta["prompt_len"],
                "context_len": len(input_ids),
                "latency_s": latency_s,
                "exact_a": by_rid[rids[0]],
                "exact_b": by_rid[rids[1]],
                "candidate": by_rid[rids[2]],
            }
            append_jsonl(output_path, row)
            print(
                json.dumps(
                    {
                        "source_rid": source_rid,
                        "step": step,
                        "prefill_tokens": [by_rid[rid]["output_ids"][0] for rid in rids],
                        "decode_tokens": [by_rid[rid]["output_ids"][1] for rid in rids],
                        "latency_s": latency_s,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    metadata = {
        "schema_version": 1,
        "prepared_run": str(prepared_run),
        "reference_run": str(reference_run),
        "split": args.split,
        "request_count": len(selected),
        "step_count": sum(1 for _ in output_path.open()),
        "generated_tokens_per_row": 2,
        "compared_output_position": 1,
        "exact_rid_prefix": args.exact_rid_prefix,
        "candidate_rid_prefix": args.candidate_rid_prefix,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
