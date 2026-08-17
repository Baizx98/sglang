#!/usr/bin/env python3
"""Wait for full48 Gate E1 replay, then compare it with seven-layer replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def wait_for_pid(pid: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--full48-analysis", type=Path, required=True)
    parser.add_argument("--sampled-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    wait_for_pid(args.wait_pid)
    summary_path = args.full48_analysis / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"full48 replay stopped without {summary_path}")
    summary = json.loads(summary_path.read_text())
    if not summary.get("coverage_complete") or not summary["contracts"].get(
        "all_contracts_passed"
    ):
        raise RuntimeError("full48 replay coverage or contracts failed")
    python = ".venv/bin/python"
    subprocess.run(
        [
            python,
            "scripts/keye_trace/analyze_resident_protection_layer_sampling.py",
            "--full48-analysis",
            str(args.full48_analysis),
            "--sampled-analysis",
            str(args.sampled_analysis),
            "--output-dir",
            str(args.output_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            python,
            "scripts/keye_trace/plot_resident_protection_layer_sampling.py",
            "--analysis-dir",
            str(args.output_dir),
            "--output-dir",
            str(args.output_dir / "figures"),
        ],
        check=True,
    )
    status = {
        "schema_version": 1,
        "status": "passed",
        "full48_analysis": str(args.full48_analysis.resolve()),
        "sampled_analysis": str(args.sampled_analysis.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }
    (args.output_dir / "finalization_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
