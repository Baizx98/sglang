#!/usr/bin/env python3
"""Wait for Gate E0 collection, then run frozen E0/E1 validation and analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


RUNS = [
    "ruler-65536",
    "ruler-131072",
    "longbench-v2-65536",
    "longbench-v2-131072",
    "infinitebench-131072",
]


def wait_for_pid(pid: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(15)


def run(command: list[str], records: list[dict[str, Any]]) -> None:
    print("running:", " ".join(command), flush=True)
    started = datetime.now(timezone.utc)
    subprocess.run(command, check=True)
    records.append(
        {
            "command": command,
            "started_at": started.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("scripts/keye_trace/configs/gate_e1_resident_protection_v1.json"),
    )
    args = parser.parse_args()
    wait_for_pid(args.wait_pid)
    print(f"collection controller {args.wait_pid}: stopped", flush=True)

    python = ".venv/bin/python"
    analysis = args.run_dir / "analysis"
    e0 = analysis / "generalization-v01"
    quality = analysis / "output-quality-v01"
    e1 = analysis / "gate-e1-v01"
    records: list[dict[str, Any]] = []
    commands = [
        [
            python,
            "scripts/keye_trace/audit_gate_e0_collection.py",
            "--run-dir",
            str(args.run_dir),
        ],
        [
            python,
            "scripts/keye_trace/validate_compact_trace_v5.py",
            "--run-dir",
            str(args.run_dir),
            "--expected-compact-k",
            "4096",
            "--expected-threshold-ranks",
            "2048,2560,3072,4096",
            "--expected-layers",
            "0,7,15,23,31,39,47",
            "--expected-steps",
            "32",
            "--expected-requests",
            "57",
        ],
        [
            python,
            "scripts/keye_trace/analyze_layer_sampling_bias.py",
            "--run-dir",
            str(args.run_dir / "full48-paired"),
            "--output-dir",
            str(
                args.run_dir
                / "full48-paired/analysis/layer-sampling-bias-p4-v01"
            ),
            "--page-size",
            "4",
            "--sampled-layers",
            "0,7,15,23,31,39,47",
        ],
    ]
    commands.extend(
        [
            python,
            "scripts/keye_trace/audit_inference_outputs.py",
            "--run-dir",
            str(args.run_dir / name),
        ]
        for name in RUNS
    )
    commands.extend(
        [
            [
                python,
                "scripts/keye_trace/audit_gate_e0_outputs.py",
                "--run-dir",
                str(args.run_dir),
                "--output-dir",
                str(quality),
            ],
            [
                python,
                "scripts/keye_trace/analyze_gate_e0_generalization.py",
                "--run-dir",
                str(args.run_dir),
                "--output-dir",
                str(e0),
                "--page-size",
                "4",
                "--full48-table",
                str(
                    args.run_dir
                    / "full48-paired/analysis/layer-sampling-bias-p4-v01/tables/by_request_layer_step.parquet"
                ),
            ],
            [
                python,
                "scripts/keye_trace/audit_gate_e0_analysis.py",
                "--analysis-dir",
                str(e0),
                "--full48-table",
                str(
                    args.run_dir
                    / "full48-paired/analysis/layer-sampling-bias-p4-v01/tables/by_request_layer_step.parquet"
                ),
                "--full48-summary",
                str(
                    args.run_dir
                    / "full48-paired/analysis/layer-sampling-bias-p4-v01/summary.json"
                ),
            ],
            [
                python,
                "scripts/keye_trace/plot_gate_e0_generalization.py",
                "--analysis-dir",
                str(e0),
                "--full48-table",
                str(
                    args.run_dir
                    / "full48-paired/analysis/layer-sampling-bias-p4-v01/tables/by_request_layer_step.parquet"
                ),
                "--output-dir",
                str(e0 / "figures"),
            ],
            [
                python,
                "scripts/keye_trace/analyze_resident_page_protection.py",
                "--run-dir",
                str(args.run_dir),
                "--policy",
                str(args.policy),
                "--output-dir",
                str(e1),
            ],
            [
                python,
                "scripts/keye_trace/audit_resident_page_protection.py",
                "--analysis-dir",
                str(e1),
                "--policy",
                str(args.policy),
            ],
            [
                python,
                "scripts/keye_trace/plot_resident_page_protection.py",
                "--analysis-dir",
                str(e1),
                "--output-dir",
                str(e1 / "figures"),
            ],
        ]
    )
    status_path = analysis / "gate-e0-e1-finalization-status.json"
    try:
        for command in commands:
            run(command, records)
    except Exception as error:
        payload = {
            "schema_version": 1,
            "run_dir": str(args.run_dir.resolve()),
            "status": "failed",
            "error": repr(error),
            "records": records,
        }
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(payload, indent=2) + "\n")
        raise
    payload = {
        "schema_version": 1,
        "run_dir": str(args.run_dir.resolve()),
        "status": "passed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    status_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
