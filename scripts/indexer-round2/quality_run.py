"""Run paired RULER generations and save per-token streaming latency."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SOURCE = Path("data/horizonkv-predictive/2026-11-13_1150_quality-expansion/deep-inputs.jsonl")
MODES = ("exact", "step", "layer")
SEED = 20261119
CLOSED_THINK = [151667, 271, 151668, 271]


def freeze_requests(source=SOURCE):
    tasks = {"niah_single_1", "niah_multivalue", "qa_1"}
    requests = []
    with source.open() as file:
        for line in file:
            row = json.loads(line)
            if (row["dataset"].lower() == "ruler" and row["split"] == "test"
                    and row["task"] in tasks and row["length_bucket_upper"] in {8192, 32768}):
                requests.append(row)
    requests.sort(key=lambda row: (row["length_bucket_upper"], row["task"]))
    if len(requests) != 6:
        raise ValueError("expected one held-out case per RULER task and length")
    return requests


def api(url, name, body=None, timeout=600):
    if body is None:
        return urlopen(f"{url}/{name}", timeout=timeout)
    post = Request(f"{url}/{name}", data=json.dumps(body).encode(),
                   headers={"Content-Type": "application/json"})
    return urlopen(post, timeout=timeout)


def readiness(url):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            with api(url, "health", timeout=4) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(3)
    raise TimeoutError("Keye server readiness")


def flush_idle(url):
    deadline = time.monotonic() + 12
    while True:
        try:
            with api(url, "flush_cache", {}, timeout=12) as response:
                response.read()
            return
        except HTTPError as error:
            if error.code != 400 or time.monotonic() >= deadline:
                raise
            # Pipeline rank 0 can still retire a request after its final SSE.
            time.sleep(0.2)


def sample(url, inputs, name, max_tokens, force_length):
    flush_idle(url)
    started = time.perf_counter()
    payload = {
        "input_ids": inputs, "rid": name, "stream": True,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens,
                            "ignore_eos": force_length},
    }
    output = None
    stamps = []
    with api(url, "generate", payload) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            content = line[5:].strip()
            if content == b"[DONE]":
                break
            output = json.loads(content)
            n = output["meta_info"]["completion_tokens"]
            if not stamps or n > stamps[-1][0]:
                stamps.append((n, time.perf_counter() - started))
    elapsed = time.perf_counter() - started
    if output is None or not stamps or not output["output_ids"]:
        raise RuntimeError(f"{name}: empty server generation")
    finish = output["meta_info"]["finish_reason"]["type"]
    length = len(output["output_ids"])
    if output["meta_info"]["id"] != name or output["meta_info"]["prompt_tokens"] != len(inputs):
        raise RuntimeError(f"{name}: server request identity or input length changed")
    if finish not in {"stop", "length"} or stamps[-1][0] != length:
        raise RuntimeError(f"{name}: incomplete generation, finish={finish}, output={length}, SSE={stamps[-1]}")
    if force_length and length != max_tokens:
        raise RuntimeError(f"{name}: fixed length expected {max_tokens}, got {length}")
    return output, elapsed, stamps


def score(task, text, references):
    text = re.sub(r"[\x00-\x1f]", "\n", text.strip()).strip()
    hit = [answer.lower() in text.lower() for answer in references]
    return max(hit) if task.startswith("qa_") else statistics.mean(hit)


def annotate(case, output, elapsed, stamps, mode, repetition):
    text = output["text"]
    has_open_think = "<think>" in text and "</think>" not in text
    answer = "" if has_open_think else text.rsplit("</think>", 1)[-1].strip()
    references = case["expected_answer"]
    decode = (stamps[-1][1] - stamps[0][1]) / (stamps[-1][0] - 1) if stamps[0][0] == 1 and stamps[-1][0] > 1 else None
    refresh_step = []
    reuse_step = []
    for (previous_n, previous_t), (n, t) in zip(stamps, stamps[1:]):
        if n != previous_n + 1:
            continue
        # Completion 1 is prefill. Completion 2 is the first decode refresh.
        (reuse_step if n % 2 else refresh_step).append(t - previous_t)
    return {
        "case_id": case["case_id"], "case_rid": case["rid"], "task": case["task"],
        "length_bucket": case["length_bucket_upper"], "prompt_tokens": case["prompt_tokens"],
        "mode": mode, "repetition": repetition, "expected": references,
        "task_raw_score": score(case["task"], text, references),
        "task_final_score": score(case["task"], answer, references),
        "end_of_think_observed": "</think>" in text,
        "unfinished_think": has_open_think,
        "final_answer": answer,
        "ttft_s": stamps[0][1] if stamps[0][0] == 1 else None,
        "elapsed_s": elapsed, "decode_s_per_token": decode,
        "stream_timestamps_s": stamps,
        "refresh_parity_s": refresh_step,
        "reuse_parity_s": reuse_step,
        "response": output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:31249")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--phase", choices=("quality", "performance"), default="quality")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--length", nargs="+", type=int, default=(8192, 32768))
    parser.add_argument("--task", nargs="+", default=("niah_single_1", "niah_multivalue", "qa_1"))
    parser.add_argument("--modes", nargs="+",
                        choices=(*MODES, "int4direct", "int4fused", "int4", "int4wide", "skip4"),
                        default=MODES)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--skip-warmups", action="store_true")
    parser.add_argument("--closed-think", action="store_true",
                        help="Append the Keye tokenizer's empty thinking block to every prompt.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a run from its existing completed output rows.")
    parser.add_argument("--audit", action="store_true",
                        help="Mark requests for server-side exact selection diagnostics.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    phases = list(args.length)
    if args.requests is None:
        source = freeze_requests()
    else:
        source = [json.loads(line) for line in args.requests.read_text().splitlines()]
    requests = [item for item in source if item["length_bucket_upper"] in phases and item["task"] in args.task]
    if not requests:
        raise ValueError("empty quality selection")
    if args.closed_think:
        requests = [
            dict(req, source_prompt_tokens=req["prompt_tokens"],
                 prompt_tokens=req["prompt_tokens"] + len(CLOSED_THINK),
                 input_ids=req["input_ids"] + CLOSED_THINK) for req in requests
        ]
    frozen = "".join(json.dumps(req) + "\n" for req in requests)
    request_path = args.output / "requests.jsonl"
    if args.resume and request_path.read_text() != frozen:
        raise ValueError("retry requests differ from the original frozen inputs")
    request_path.write_text(frozen)
    config = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "seed": SEED,
        "requests_sha256": hashlib.sha256((args.output / "requests.jsonl").read_bytes()).hexdigest(),
        "selection": [row["rid"] for row in requests],
    }
    config_path = args.output / "config.json"
    if not args.resume:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    else:
        saved = json.loads(config_path.read_text())["arguments"]
        for key in ("phase", "max_tokens", "modes", "closed_think", "audit"):
            current = config["arguments"][key]
            if isinstance(current, tuple):
                current = list(current)
            if saved.get(key, False) != current:
                raise ValueError(f"retry changed {key}")
    readiness(args.url)

    if not args.skip_warmups and not args.resume:
        for mode in args.modes:
            print("warmup", mode, flush=True)
            warmup_tokens = {"int4": 4352, "int4wide": 8448}.get(mode, 2560)
            sample(args.url, requests[0]["input_ids"][:warmup_tokens],
                   f"quality__{mode}__warm", 6, True)
    mode_order = random.Random(SEED)
    output_path = args.output / "outputs.jsonl"
    completed = set()
    if args.resume and output_path.exists():
        old = [json.loads(line) for line in output_path.read_text().splitlines()]
        completed = {(row["case_id"], row["mode"], row["repetition"]) for row in old}
    with output_path.open("a" if args.resume else "w", buffering=1) as outcomes:
        for repetition in range(args.repeat):
            for i, case in enumerate(requests):
                order = list(args.modes)
                mode_order.shuffle(order)
                for mode in order:
                    if (case["case_id"], mode, repetition) in completed:
                        continue
                    prefix = "audit_" if args.audit else ""
                    name = f"quality__{mode}__{prefix}{args.phase}_{i}_{repetition}"
                    print("start", name, case["task"], case["prompt_tokens"], flush=True)
                    output, elapsed, stamps = sample(args.url, case["input_ids"], name, args.max_tokens,
                                                     args.phase == "performance")
                    row = annotate(case, output, elapsed, stamps, mode, repetition)
                    outcomes.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print("done", name, "s", round(elapsed, 3), "score", row["task_final_score"],
                          "tokens", len(output["output_ids"]), flush=True)


if __name__ == "__main__":
    main()
