#!/usr/bin/env python3
"""Issue reproducible requests for DSA deadline tracing and overhead checks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read().decode()
        return json.loads(raw) if raw else None


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
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def post_stream(
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[Any, list[dict[str, Any]]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    arrivals: list[dict[str, Any]] = []
    final = None
    previous_time = started
    previous_tokens = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line or not line.startswith("data:"):
                    continue
                value = line[len("data:") :].strip()
                if value == "[DONE]":
                    continue
                message = json.loads(value)
                final = message
                now = time.perf_counter()
                meta = message.get("meta_info") or {}
                completion_tokens = int(meta.get("completion_tokens", previous_tokens))
                delta_tokens = completion_tokens - previous_tokens
                arrivals.append(
                    {
                        "message_index": len(arrivals),
                        "completion_tokens": completion_tokens,
                        "delta_tokens": delta_tokens,
                        "arrival_s": now - started,
                        "gap_s": now - previous_time,
                    }
                )
                previous_time = now
                previous_tokens = completion_tokens
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"POST {url} failed: HTTP {exc.code}: {detail}") from exc
    if final is None:
        raise RuntimeError("streaming response did not contain a JSON data event")
    return final, arrivals


def build_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prepared_requests is not None:
        prepared = list(enumerate(read_jsonl(args.prepared_requests)))
        if args.request_indices:
            indices = [
                int(value.strip())
                for value in args.request_indices.split(",")
                if value.strip()
            ]
            if not indices or len(indices) != len(set(indices)):
                raise ValueError("--request-indices must contain unique integers")
            if min(indices) < 0 or max(indices) >= len(prepared):
                raise ValueError(
                    "--request-indices is outside the prepared request range"
                )
            prepared = [prepared[index] for index in indices]
        if args.request_limit is not None:
            prepared = prepared[: args.request_limit]
        return [
            {
                "rid_base": f"{args.rid_prefix}__{row['task']}",
                "prepared_request_index": prepared_index,
                "input_ids": row["input_ids"],
                "task": row["task"],
                "source_rid": row["rid"],
                "prompt_len": row.get("prompt_len"),
            }
            for prepared_index, row in prepared
        ]
    if args.prompt is None:
        raise ValueError("provide either --prepared-requests or --prompt")
    return [
        {
            "rid_base": args.rid_prefix,
            "text": args.prompt * args.prompt_repeat,
        }
    ]


def issue_request(
    *,
    args: argparse.Namespace,
    request_spec: dict[str, Any],
    request_index: int,
    repetition: int,
    wave_index: int,
    slot_index: int,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    rid = (
        f"{request_spec['rid_base']}__rep{repetition:02d}"
        f"__wave{wave_index:02d}__slot{slot_index:02d}"
    )
    payload: dict[str, Any] = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
        },
        "stream": args.stream,
        "rid": rid,
    }
    if "input_ids" in request_spec:
        payload["input_ids"] = request_spec["input_ids"]
    else:
        payload["text"] = request_spec["text"]

    barrier.wait()
    started_unix_s = time.time()
    started = time.perf_counter()
    if args.stream:
        response, arrivals = post_stream(
            f"{args.base_url}/generate", payload, args.timeout
        )
    else:
        response = post_json(f"{args.base_url}/generate", payload, args.timeout)
        arrivals = []
    latency_s = time.perf_counter() - started
    output_ids = list(response.get("output_ids") or [])
    return {
        "schema_version": 1,
        "rid": rid,
        "request_index": request_index,
        "repetition": repetition,
        "wave_index": wave_index,
        "slot_index": slot_index,
        "requested_concurrency": args.concurrency,
        "source_rid": request_spec.get("source_rid"),
        "task": request_spec.get("task"),
        "expected_prompt_len": request_spec.get("prompt_len"),
        "started_unix_s": started_unix_s,
        "latency_s": latency_s,
        "output_tokens": len(output_ids),
        "structural_pass": bool(output_ids)
        and bool(str(response.get("text") or "").strip())
        and math.isfinite(latency_s)
        and latency_s > 0,
        "stream_arrivals": arrivals,
        "response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prepared-requests", type=Path)
    parser.add_argument("--request-limit", type=int)
    parser.add_argument("--request-indices", default="")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--rid-prefix", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=36)
    parser.add_argument("--min-new-tokens", type=int, default=36)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()

    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.prompt_repeat <= 0:
        raise ValueError("--prompt-repeat must be positive")
    if not 0 <= args.min_new_tokens <= args.max_new_tokens:
        raise ValueError("min-new-tokens must be between zero and max-new-tokens")
    if args.request_limit is not None and args.request_limit <= 0:
        raise ValueError("--request-limit must be positive")

    requests = build_requests(args)
    scheduled = [
        (
            int(request_spec.get("prepared_request_index", request_index)),
            repetition,
            request_spec,
        )
        for repetition in range(args.repetitions)
        for request_index, request_spec in enumerate(requests)
    ]
    if len(scheduled) % args.concurrency:
        raise ValueError(
            "request count times repetitions must be divisible by --concurrency"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "requests.jsonl"
    output_path.unlink(missing_ok=True)
    write_json(
        args.output_dir / "run_config.json",
        {
            "schema_version": 1,
            "base_url": args.base_url,
            "prepared_requests": (
                str(args.prepared_requests.resolve())
                if args.prepared_requests is not None
                else None
            ),
            "request_limit": args.request_limit,
            "request_indices": args.request_indices or None,
            "rid_prefix": args.rid_prefix,
            "repetitions": args.repetitions,
            "concurrency": args.concurrency,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "stream": args.stream,
            "flush_cache": args.flush_cache,
            "speedup_measurement": False,
        },
    )
    write_json(
        args.output_dir / "server_snapshot.json",
        {
            "health_generate": get_json(
                f"{args.base_url}/health_generate", args.timeout
            ),
            "model_info": get_json(f"{args.base_url}/model_info", args.timeout),
        },
    )

    results = []
    waves = []
    for wave_index, start in enumerate(range(0, len(scheduled), args.concurrency)):
        group = scheduled[start : start + args.concurrency]
        if args.flush_cache:
            post_json(f"{args.base_url}/flush_cache", {}, args.timeout)
        barrier = threading.Barrier(len(group))
        wave_started_unix_s = time.time()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(group)
        ) as executor:
            futures = [
                executor.submit(
                    issue_request,
                    args=args,
                    request_spec=request_spec,
                    request_index=request_index,
                    repetition=repetition,
                    wave_index=wave_index,
                    slot_index=slot_index,
                    barrier=barrier,
                )
                for slot_index, (
                    request_index,
                    repetition,
                    request_spec,
                ) in enumerate(group)
            ]
            wave_results = [future.result() for future in futures]
        wave_finished_unix_s = time.time()
        for result in sorted(wave_results, key=lambda row: row["slot_index"]):
            append_jsonl(output_path, result)
            results.append(result)
            meta = result["response"].get("meta_info") or {}
            print(
                json.dumps(
                    {
                        "rid": result["rid"],
                        "wave_index": wave_index,
                        "latency_s": round(result["latency_s"], 4),
                        "output_tokens": result["output_tokens"],
                        "server_e2e_latency": meta.get("e2e_latency"),
                        "structural_pass": result["structural_pass"],
                    }
                ),
                flush=True,
            )
        waves.append(
            {
                "wave_index": wave_index,
                "requested_concurrency": len(group),
                "started_unix_s": wave_started_unix_s,
                "finished_unix_s": wave_finished_unix_s,
                "request_ids": [row["rid"] for row in wave_results],
                "start_skew_ms": 1000
                * (
                    max(row["started_unix_s"] for row in wave_results)
                    - min(row["started_unix_s"] for row in wave_results)
                ),
            }
        )

    write_json(args.output_dir / "wave_manifest.json", {"waves": waves})

    summary = {
        "schema_version": 1,
        "requests": len(results),
        "requested_concurrency": args.concurrency,
        "waves": len(waves),
        "maximum_request_start_skew_ms": max(
            wave["start_skew_ms"] for wave in waves
        ),
        "all_structural_passed": all(row["structural_pass"] for row in results),
        "mean_latency_s": sum(row["latency_s"] for row in results) / len(results),
        "speedup_measured": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    if not summary["all_structural_passed"]:
        raise RuntimeError(
            "one or more deadline-trace requests failed structure checks"
        )


if __name__ == "__main__":
    main()
