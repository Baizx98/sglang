#!/usr/bin/env python3
"""Continue the frozen Gate E0 expanded collection in a safe serial order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(bool(line.strip()) for line in path.read_text().splitlines())


def wait_for_existing(pid: int, output: Path, expected: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(15)
    actual = line_count(output)
    if actual != expected:
        raise RuntimeError(
            f"prerequisite runner {pid} stopped with {actual}/{expected} outputs"
        )
    print(f"prerequisite runner {pid}: complete ({actual}/{expected})", flush=True)


def run(command: list[str], expected_output: Path, expected: int) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    actual = line_count(expected_output)
    if actual != expected:
        raise RuntimeError(f"{expected_output}: expected {expected}, found {actual}")
    print(f"complete: {expected_output.parent.name} ({actual}/{expected})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()

    python = ".venv/bin/python"
    ruler128 = args.run_dir / "ruler-131072/requests.jsonl"
    wait_for_existing(args.wait_pid, ruler128, 12)

    jobs = [
        (
            [
                python,
                "scripts/keye_trace/run_longbench_v2_compact_trace.py",
                "--dataset",
                "/Tan/dataset/LongBench-v2/0000.parquet",
                "--output-dir",
                str(args.run_dir / "longbench-v2-65536"),
                "--length-config",
                "65536",
                "--max-new-tokens",
                "64",
                "--min-new-tokens",
                "32",
                "--reuse-prepared",
                "--resume",
                "--apply-chat-template",
                "--answer-assistant-prefix",
            ],
            args.run_dir / "longbench-v2-65536/requests.jsonl",
            10,
        ),
        (
            [
                python,
                "scripts/keye_trace/run_longbench_v2_compact_trace.py",
                "--dataset",
                "/Tan/dataset/LongBench-v2/0000.parquet",
                "--output-dir",
                str(args.run_dir / "longbench-v2-131072"),
                "--length-config",
                "131072",
                "--max-new-tokens",
                "64",
                "--min-new-tokens",
                "32",
                "--reuse-prepared",
                "--resume",
                "--apply-chat-template",
                "--answer-assistant-prefix",
            ],
            args.run_dir / "longbench-v2-131072/requests.jsonl",
            11,
        ),
        (
            [
                python,
                "scripts/keye_trace/run_infinitebench_compact_trace.py",
                "--dataset-root",
                "/Tan/dataset/InfiniteBench",
                "--output-dir",
                str(args.run_dir / "infinitebench-131072"),
                "--length-config",
                "131072",
                "--max-new-tokens",
                "64",
                "--min-new-tokens",
                "32",
                "--reuse-prepared",
                "--resume",
            ],
            args.run_dir / "infinitebench-131072/requests.jsonl",
            12,
        ),
    ]
    for command, output, expected in jobs:
        run(command, output, expected)


if __name__ == "__main__":
    main()
