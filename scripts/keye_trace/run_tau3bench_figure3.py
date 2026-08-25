#!/usr/bin/env python3
"""Replay prepared tau3-bench invocation prompts through SGLang for Figure 3."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"POST {url} failed: HTTP {exc.code}: {detail}") from exc
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--request-limit", type=int)
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()
    if args.decode_steps < 17:
        raise ValueError("--decode-steps must be at least 17 for delta-step 16")
    rows = read_jsonl(args.prepared_requests)
    if args.request_limit is not None:
        rows = rows[: args.request_limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for request_index, row in enumerate(rows):
            payload = {
                "input_ids": row["input_ids"],
                "rid": row["rid"],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.decode_steps,
                    "min_new_tokens": args.decode_steps,
                    "ignore_eos": True,
                },
            }
            started = time.perf_counter()
            response = post_json(f"{args.base_url}/generate", payload, args.timeout)
            record = {
                "schema_version": 1,
                "request_index": request_index,
                "rid": row["rid"],
                "session_id": row["session_id"],
                "domain": row["domain"],
                "invocation_id": row["invocation_id"],
                "turn_id": row["turn_id"],
                "prompt_len": row["prompt_len"],
                "latency_s": time.perf_counter() - started,
                "output_ids": list(response.get("output_ids") or []),
                "text": response.get("text"),
                "meta_info": response.get("meta_info"),
            }
            if len(record["output_ids"]) != args.decode_steps:
                raise RuntimeError(
                    f"{row['rid']} returned {len(record['output_ids'])} tokens, "
                    f"expected {args.decode_steps}"
                )
            output.write(compact_json(record) + "\n")
            output.flush()
            print(
                f"[{request_index + 1}/{len(rows)}] {row['rid']} "
                f"prompt={row['prompt_len']} latency={record['latency_s']:.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
