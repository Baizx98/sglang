#!/usr/bin/env python3
"""Measure next-token fidelity on exact-reference prefixes through /generate."""

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
    parser.add_argument("--variant", required=True)
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
    selected = {
        row["rid"]: row
        for row in selection["selected"]
        if args.split is None or row["split"] == args.split
    }
    if not set(selected) <= set(prepared) & set(reference):
        raise ValueError("selected, prepared, and reference request sets differ")

    output_path = args.output_dir / "teacher_forced_steps.jsonl"
    output_path.unlink(missing_ok=True)
    request_summaries = []
    for request_index, (rid, meta) in enumerate(selected.items()):
        post_json(f"{args.base_url}/flush_cache", {}, args.timeout)
        prompt_ids = list(prepared[rid]["input_ids"])
        reference_ids = list(reference[rid]["response"]["output_ids"])
        step_count = min(args.steps, len(reference_ids))
        top1_matches = 0
        nll_values = []
        for step in range(step_count):
            reference_token = int(reference_ids[step])
            payload = {
                "input_ids": prompt_ids + reference_ids[:step],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
                "stream": False,
                "rid": f"tfrescore__{request_index:02d}__step_{step:03d}",
                "return_logprob": True,
                "top_logprobs_num": args.top_logprobs_num,
                "token_ids_logprob": [reference_token],
            }
            started = time.perf_counter()
            response = post_json(f"{args.base_url}/generate", payload, args.timeout)
            latency_s = time.perf_counter() - started
            predicted_token = int(response["output_ids"][0])
            token_entries = response["meta_info"]["output_token_ids_logprobs"][0]
            matching = [entry for entry in token_entries if int(entry[1]) == reference_token]
            if len(matching) != 1:
                raise RuntimeError(f"missing reference token logprob for {rid} step {step}")
            reference_logprob = float(matching[0][0])
            top_entries = response["meta_info"].get("output_top_logprobs", [[]])[0]
            top_distribution = [
                {"logprob": float(entry[0]), "token_id": int(entry[1])}
                for entry in top_entries
            ]
            matched = predicted_token == reference_token
            top1_matches += int(matched)
            nll_values.append(-reference_logprob)
            append_jsonl(
                output_path,
                {
                    "variant": args.variant,
                    "rid": rid,
                    "trajectory_id": meta["trajectory_id"],
                    "category": meta["category"],
                    "split": meta["split"],
                    "prompt_len": meta["prompt_len"],
                    "step": step,
                    "context_len": len(prompt_ids) + step,
                    "reference_token": reference_token,
                    "predicted_token": predicted_token,
                    "top1_match": matched,
                    "reference_logprob": reference_logprob,
                    "reference_nll": -reference_logprob,
                    "top_logprobs": top_distribution,
                    "latency_s": latency_s,
                },
            )
        request_summary = {
            "rid": rid,
            "split": meta["split"],
            "steps": step_count,
            "top1_agreement": top1_matches / max(step_count, 1),
            "reference_nll_mean": sum(nll_values) / max(len(nll_values), 1),
        }
        request_summaries.append(request_summary)
        print(json.dumps(request_summary, ensure_ascii=False), flush=True)

    rows = read_jsonl(output_path)
    summary = {
        "schema_version": 1,
        "variant": args.variant,
        "prepared_run": str(prepared_run),
        "reference_run": str(reference_run),
        "request_count": len(request_summaries),
        "step_count": len(rows),
        "top1_agreement": sum(row["top1_match"] for row in rows) / len(rows),
        "reference_nll_mean": sum(row["reference_nll"] for row in rows) / len(rows),
        "request_summaries": request_summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
