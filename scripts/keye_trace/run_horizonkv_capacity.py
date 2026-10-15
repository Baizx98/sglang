"""Frozen C4/32K capacity collection, replay and three online audits."""

import argparse
import csv
import hashlib
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from horizonkv_capacity_replay import CAPACITIES, analyze
from run_horizonkv_motivation import GPUS, run_wave, save

REPO = Path(__file__).resolve().parents[2]


def export(path, values):
    if not values:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def collect(root, plan, config, waves, port):
    directory = root / config
    directory.mkdir(exist_ok=True)
    output = directory / "requests.jsonl"
    if output.exists():
        raise RuntimeError(f"Preserve existing collection: {output}; use a new run")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip():
        raise RuntimeError("Commit source changes before collection")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        != plan["source_commit"]
    ):
        raise RuntimeError("Source drift")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError("Port occupied; do not interrupt another server")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("KEYE_", "SGLANG_"))
    }
    env.update(
        CUDA_VISIBLE_DEVICES=GPUS,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        KEYE_SM80_DSA="1",
        KEYE_SM80_EXACT_TOPK="1",
        KEYE_HORIZON_STORAGE="hbm" if config == "hbm" else "lru",
        KEYE_HORIZON_CACHE_TOKENS="0" if config == "hbm" else config[3:],
        KEYE_HORIZON_CAPACITY_DIR=str(directory / "trace"),
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
        str(port),
        "--tp-size",
        "2",
        "--attention-backend",
        "keye_sa",
        "--context-length",
        "131328",
        "--max-total-tokens",
        "140000",
        "--max-running-requests",
        "4",
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
    with log.open("w") as stream:
        server = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        save(
            directory / "startup.json",
            {
                "pid": server.pid,
                "command": command,
                "source_commit": plan["source_commit"],
                "environment": {
                    key: value
                    for key, value in env.items()
                    if key.startswith(("KEYE_", "CUDA_"))
                },
            },
        )
        try:
            for _ in range(600):
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited: {log}")
                if (
                    "The server is fired up and ready to roll!"
                    in log.read_text()[-6000:]
                ):
                    break
                time.sleep(2)
            else:
                raise RuntimeError("Server readiness timeout")
            with output.open("w", buffering=1) as sink:
                for wave, rows in enumerate(waves):
                    prefix = f"capacity__{config}__wave{wave:03d}"
                    result = run_wave(
                        f"http://127.0.0.1:{port}",
                        rows,
                        prefix,
                        plan["max_new_tokens"],
                        3600,
                    )
                    for record in result:
                        record.update(config=config, wave_index=wave)
                        sink.write(json.dumps(record, separators=(",", ":")) + "\n")
                    save(
                        root / "state.json",
                        {
                            "status": "running",
                            "phase": "collect",
                            "config": config,
                            "last_wave": wave,
                            "updated_unix_s": time.time(),
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "collected": config,
                                "wave": wave,
                                "outputs": [
                                    len(v["response"]["output_ids"]) for v in result
                                ],
                            }
                        ),
                        flush=True,
                    )
            # Wait for both binary writers to settle at a complete model step.
            previous, stable = None, 0
            for _ in range(120):
                time.sleep(1)
                errors = list((directory / "trace").glob("*.error"))
                if errors:
                    raise RuntimeError(errors[0].read_text())
                statuses = [
                    directory / "trace" / f"rank_{rank:02d}.status.json"
                    for rank in (0, 1)
                ]
                value = (
                    [json.loads(path.read_text()) for path in statuses]
                    if all(path.exists() for path in statuses)
                    else None
                )
                stable = stable + 1 if value is not None and value == previous else 0
                previous = value
                if stable >= 5:
                    break
            else:
                raise RuntimeError("Capacity writers did not settle")
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def main():
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=31015)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--phase", choices=["all", "collect", "replay", "online"], default="all"
    )
    args = parser.parse_args()
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        selected = [
            json.loads(line)
            for line in (root / "frozen_requests.jsonl").read_text().splitlines()
        ]
    else:
        selected = [
            row
            for row in (json.loads(line) for line in args.requests.open())
            if row["dataset"] == "RULER-v1"
            and row["task"] == "niah_multikey_1"
            and row["length_bucket_upper"] == 32768
        ]
        selected.sort(key=lambda value: value["case_id"])
        assert len(selected) == 20 and len({row["case_id"] for row in selected}) == 20
        random.Random(20261014).shuffle(selected)
        for start in range(0, 20, 4):
            assert (
                sum(
                    (row["prompt_tokens"] + args.max_new_tokens + 63) // 64 * 64
                    for row in selected[start : start + 4]
                )
                <= 139968
            )
        frozen = root / "frozen_requests.jsonl"
        frozen.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected)
        )
        plan = {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "capacities": CAPACITIES,
            "concurrency": 4,
            "max_new_tokens": args.max_new_tokens,
            "seed": 20261014,
            "frozen_requests_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
            "waves": [
                [row["case_id"] for row in selected[start : start + 4]]
                for start in range(0, 20, 4)
            ],
            "kind": "fixed measured trace, CPU batch-LRU replay, online count audit; no timing claims",
        }
        save(plan_path, plan)
    waves = [selected[start : start + 4] for start in range(0, 20, 4)]
    try:
        if args.phase in ("all", "collect"):
            collect(root, plan, "hbm", waves, args.port)
        if args.phase in ("all", "replay"):
            save(
                root / "state.json",
                {"status": "running", "phase": "replay", "updated_unix_s": time.time()},
            )
            result, layers, audit = analyze(root / "hbm", root / "analysis")
            export(root / "analysis" / "capacity_by_wave.csv", result)
            export(root / "analysis" / "capacity_by_layer.csv", layers)
            assert audit["requests"] == 20
        if args.phase in ("all", "online"):
            reference = {
                v["case_id"]: v["response"]["output_ids"]
                for v in (
                    json.loads(line)
                    for line in (root / "hbm" / "requests.jsonl").open()
                )
                if v["wave_index"] == 0
            }
            gates = []
            for capacity in (2048, 8192, 32768):
                config = f"lru{capacity}"
                collect(root, plan, config, waves[:1], args.port)
                _, _, audit = analyze(
                    root / config, root / config / "analysis", [capacity]
                )
                assert audit["online_audited_layer_request_steps"] > 0
                output = [
                    json.loads(line)
                    for line in (root / config / "requests.jsonl").open()
                ]
                equality = {
                    v["case_id"]: v["response"]["output_ids"] == reference[v["case_id"]]
                    for v in output
                }
                gates.append(
                    {
                        "capacity": capacity,
                        "count_audit": "passed",
                        "layer_request_steps": audit[
                            "online_audited_layer_request_steps"
                        ],
                        "output_equal_hbm": equality,
                    }
                )
                save(
                    root / "online-audit.json",
                    {
                        "gates": gates,
                        "mismatched_outputs_are_not_pooled_with_main_curve": True,
                    },
                )
        save(
            root / "state.json",
            {
                "status": "completed" if args.phase == "all" else "phase_completed",
                "phase": args.phase,
                "updated_unix_s": time.time(),
            },
        )
    except BaseException as exc:
        save(
            root / "state.json",
            {"status": "failed", "error": repr(exc), "updated_unix_s": time.time()},
        )
        raise


if __name__ == "__main__":
    main()
