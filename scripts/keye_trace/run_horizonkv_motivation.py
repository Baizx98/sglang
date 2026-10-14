"""Run resumable exact-Keye HBM / DRAM / LRU measurement waves.

Default is a full workload run. --pilot fixes a small outcome-independent set.
Raw files and server logs stay under ignored data/. No new predictor is used.
"""

import argparse
import hashlib
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GPUS = (
    "GPU-79886749-8f57-63a5-8c8e-8a68a085565a,GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822"
)


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def run_wave(url, rows, prefix, max_tokens, timeout):
    # Submit one batch request so the entire wave is queued together, instead
    # of depending on Python thread arrival order during chunked prefill.
    payload = {
        "input_ids": [v["input_ids"] for v in rows],
        "rid": [prefix + "__" + v["case_id"] for v in rows],
        "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
        "stream": True,
    }
    request = urllib.request.Request(
        url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    finals = {}
    arrivals = defaultdict(list)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            message = json.loads(line[5:])
            meta = message.get("meta_info") or {}
            rid = meta.get("id")
            if rid is None:
                raise ValueError("Missing request ID in batched stream")
            finals[rid] = message
            arrivals[rid].append(
                {
                    "elapsed_s": time.perf_counter() - started,
                    "completion_tokens": meta.get("completion_tokens"),
                }
            )
    records = []
    for row in rows:
        rid = prefix + "__" + row["case_id"]
        value = finals[rid]
        if not value.get("output_ids"):
            raise ValueError(f"Empty generated token IDs for {rid}")
        records.append(
            {
                "case_id": row["case_id"],
                "rid": rid,
                "task": row["task"],
                "workload": row["workload"],
                "prompt_tokens": row["prompt_tokens"],
                "requested_concurrency": len(rows),
                "response": value,
                "stream_arrivals": arrivals[rid],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--storage-configs", default="hbm,dram,lru2048,lru4096,lru6144,lru8192"
    )
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--logical-pool-tokens", type=int, default=140000)
    parser.add_argument("--port", type=int, default=31014)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if (
        min(args.concurrencies) <= 0
        or args.repetitions <= 0
        or args.max_new_tokens <= 0
    ):
        raise ValueError("concurrency, repetitions and token limit must be positive")
    configs = args.storage_configs.split(",")
    if configs[0] != "hbm":
        raise ValueError("HBM must run first as the exact output reference")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.requests.read_text().splitlines() if line]
    if args.pilot:
        chosen = {}
        for row in sorted(
            rows, key=lambda v: (abs(v["length_bucket_upper"] - 32768), v["case_id"])
        ):
            chosen.setdefault(row["workload"], row)
        rows = list(chosen.values())
        if max(args.concurrencies) > 1:
            pressure = [
                v
                for v in [
                    json.loads(line) for line in args.requests.read_text().splitlines()
                ]
                if v["dataset"] == "RULER-v1"
                and v["task"] == "niah_multikey_1"
                and v["length_bucket_upper"] == 32768
            ]
            rows = list(
                {
                    v["case_id"]: v for v in rows + pressure[: max(args.concurrencies)]
                }.values()
            )
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["task"], row["length_bucket_upper"])].append(row)
    waves, skipped = [], []
    for key, values in sorted(groups.items()):
        for concurrency in args.concurrencies:
            for start in range(0, len(values), concurrency):
                batch = values[start : start + concurrency]
                if len(batch) != concurrency:
                    skipped.append(
                        {
                            "group": key,
                            "concurrency": concurrency,
                            "reason": "insufficient_distinct_requests",
                        }
                    )
                elif (
                    sum(v["prompt_tokens"] + args.max_new_tokens + 64 for v in batch)
                    > args.logical_pool_tokens // 64 * 64
                ):
                    skipped.append(
                        {
                            "group": key,
                            "concurrency": concurrency,
                            "reason": "configured_logical_pool_limit",
                        }
                    )
                else:
                    waves.append(batch)
    plan = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "pilot": args.pilot,
        "repetitions": args.repetitions,
        "max_new_tokens": args.max_new_tokens,
        "configs": configs,
        "waves": [[v["case_id"] for v in wave] for wave in waves],
        "skipped": skipped,
        "gpu_uuids": GPUS,
        "timing_scope": "eager real Keye; CPU-managed exact host prototype",
        "record_contract": "CUDA stream spans include host idle; nesting must be subtracted; not kernel sum",
    }
    plan_path = args.run_dir / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Run plan changed; use a new run directory")
    save(plan_path, plan)
    reference = {}
    rng = random.Random(20261014)
    for config in configs:
        directory = args.run_dir / config
        directory.mkdir(exist_ok=True)
        output = directory / "requests.jsonl"
        existing = (
            [json.loads(line) for line in output.read_text().splitlines()]
            if output.exists()
            else []
        )
        completed_groups = defaultdict(set)
        for v in existing:
            completed_groups[(v["wave_index"], v["repetition"], v["phase"])].add(
                v["case_id"]
            )
        done = {
            key
            for key, case_ids in completed_groups.items()
            if case_ids == {v["case_id"] for v in waves[key[0]]}
        }
        if config == "hbm":
            for v in existing:
                reference[(v["case_id"], v["wave_index"], v["repetition"])] = v[
                    "response"
                ]["output_ids"]
        phases = (
            ["control", "timing", "selection"]
            if config.startswith("lru")
            else ["control", "timing"]
        )
        tasks = [
            (i, r, phase)
            for r in range(args.repetitions)
            for i in range(len(waves))
            for phase in phases
        ]
        rng.shuffle(tasks)
        pending = [v for v in tasks if v not in done]
        if not pending:
            continue
        env = os.environ.copy()
        for k in list(env):
            if k.startswith(("KEYE_", "SGLANG_")):
                env.pop(k)
        mode = "lru" if config.startswith("lru") else config
        env.update(
            CUDA_VISIBLE_DEVICES=GPUS,
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            KEYE_SM80_DSA="1",
            KEYE_SM80_EXACT_TOPK="1",
            KEYE_HORIZON_STORAGE=mode,
            KEYE_HORIZON_CACHE_TOKENS=config[3:] if mode == "lru" else "0",
            KEYE_HORIZON_PROBE_DIR=str((directory / "events").resolve()),
            KEYE_HORIZON_RID_PREFIX="horizon__",
            PYTHONUNBUFFERED="1",
        )
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            "/Tan/model/Keye-VL-2.0-30B-A3B",
            "--trust-remote-code",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--tp-size",
            "2",
            "--attention-backend",
            "keye_sa",
            "--context-length",
            "131328",
            "--max-total-tokens",
            str(args.logical_pool_tokens),
            "--max-running-requests",
            "8",
            "--chunked-prefill-size",
            "1024",
            "--mem-fraction-static",
            "0.90",
            "--disable-cuda-graph",
            "--disable-custom-all-reduce",
            "--disable-radix-cache",
            "--random-seed",
            "20261014",
        ]
        log = directory / "server.log"
        try:
            with socket.create_connection(("127.0.0.1", args.port), timeout=2):
                occupied = True
        except ConnectionRefusedError:
            occupied = False
        if occupied:
            raise RuntimeError(
                "Port already serves a process; do not interfere with it"
            )
        with log.open("w") as stdout:
            proc = subprocess.Popen(
                command,
                env=env,
                cwd=REPO,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            save(
                directory / "startup.json",
                {
                    "pid": proc.pid,
                    "command": command,
                    "environment": {k: env[k] for k in env if k.startswith("KEYE_")},
                    "source_commit": plan["source_commit"],
                },
            )
            try:
                ready = False
                for _ in range(300):
                    if proc.poll() is not None:
                        raise RuntimeError(f"Server failed: {log}")
                    if (
                        log.exists()
                        and "The server is fired up and ready to roll!"
                        in log.read_text()[-6000:]
                    ):
                        ready = True
                        break
                    time.sleep(2)
                if not ready:
                    raise RuntimeError(f"Server readiness timeout: {log}")
                with output.open("a", buffering=1) as sink:
                    for wave_index, repetition, phase in pending:
                        prefix = f"{'horizon' if phase == 'timing' else phase}__{config}__wave{wave_index:03d}__rep{repetition:02d}"
                        result = run_wave(
                            f"http://127.0.0.1:{args.port}",
                            waves[wave_index],
                            prefix,
                            args.max_new_tokens,
                            args.timeout,
                        )
                        for value in result:
                            value.update(
                                config=config,
                                wave_index=wave_index,
                                repetition=repetition,
                                phase=phase,
                                instrumented=phase != "control",
                            )
                            key = (value["case_id"], wave_index, repetition)
                            if config == "hbm":
                                value["output_ids_equal_hbm"] = (
                                    key not in reference
                                    or reference[key] == value["response"]["output_ids"]
                                )
                                if value["output_ids_equal_hbm"]:
                                    reference[key] = value["response"]["output_ids"]
                            else:
                                value["output_ids_equal_hbm"] = (
                                    value["response"]["output_ids"] == reference[key]
                                )
                            sink.write(
                                json.dumps(
                                    value, ensure_ascii=False, separators=(",", ":")
                                )
                                + "\n"
                            )
                        if not all(v["output_ids_equal_hbm"] for v in result):
                            raise RuntimeError(
                                "Exact output gate failed; retain failure data and stop timing claims"
                            )
                        save(
                            args.run_dir / "state.json",
                            {
                                "status": "running",
                                "config": config,
                                "last_wave": wave_index,
                                "last_repetition": repetition,
                                "phase": phase,
                                "updated_unix_s": time.time(),
                            },
                        )
                        print(
                            json.dumps(
                                {
                                    "config": config,
                                    "wave": wave_index,
                                    "rep": repetition,
                                    "phase": phase,
                                    "requests": len(result),
                                }
                            ),
                            flush=True,
                        )
                time.sleep(3)  # drain already-completed CUDA events and raw writer
            except BaseException as exc:
                save(
                    args.run_dir / "state.json",
                    {
                        "status": "failed",
                        "config": config,
                        "error": str(exc),
                        "updated_unix_s": time.time(),
                    },
                )
                raise
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
    save(
        args.run_dir / "state.json",
        {"status": "completed", "updated_unix_s": time.time()},
    )


if __name__ == "__main__":
    main()
