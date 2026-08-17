#!/usr/bin/env python3
"""Synthetic integration test for deadline-trace validation and plotting."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        trace_dir = root / "trace"
        output_dir = root / "analysis"
        trace_dir.mkdir()
        for rank in (0, 1):
            (trace_dir / f"deadline_metadata_rank_{rank:02d}.json").write_text(
                json.dumps({"schema_version": 1, "global_rank": rank}) + "\n"
            )
            rows = []
            for layer in (0, 7):
                for step in (0, 1):
                    rows.append(
                        {
                            "schema_version": 1,
                            "kind": "previous_step_same_layer",
                            "request_id": "synthetic-request",
                            "layer_id": layer,
                            "producer_decode_step": step,
                            "consumer_decode_step": step + 1,
                            "interval_ms": 5.0 + layer + step + rank,
                            "context_tokens": 4096 + step,
                            "batch_size": 1,
                            "global_rank": rank,
                            "device_index": rank,
                        }
                    )
            (trace_dir / f"deadline_events_rank_{rank:02d}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

        subprocess.run(
            [
                sys.executable,
                str(script_dir / "analyze_deadline_trace.py"),
                "--trace-dir",
                str(trace_dir),
                "--output-dir",
                str(output_dir),
                "--expected-layers",
                "0,7",
                "--expected-kinds",
                "previous_step_same_layer",
                "--expected-intervals",
                "2",
                "--context-label",
                "synthetic",
                "--requested-concurrency",
                "1",
            ],
            check=True,
        )
        paired = pd.read_csv(output_dir / "paired_intervals.csv")
        validation = json.loads((output_dir / "validation.json").read_text())
        assert len(paired) == 4
        assert paired.interval_ms.tolist() == [5.0, 6.0, 12.0, 13.0]
        assert validation["tp_pairing"]["all_intervals_have_all_ranks"]
        assert validation["batching"]["fraction_at_requested_concurrency"] == 1.0
        for name in [
            "deadline_quantiles_by_layer.pdf",
            "deadline_quantiles_by_layer.png",
            "deadline_attainment_by_layer.pdf",
            "deadline_attainment_by_layer.png",
        ]:
            assert (output_dir / name).is_file()
        mismatch = subprocess.run(
            [
                sys.executable,
                str(script_dir / "analyze_deadline_trace.py"),
                "--trace-dir",
                str(trace_dir),
                "--output-dir",
                str(root / "invalid-analysis"),
                "--expected-layers",
                "0,7",
                "--expected-kinds",
                "previous_step_same_layer",
                "--expected-intervals",
                "2",
                "--context-label",
                "synthetic",
                "--requested-concurrency",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        assert mismatch.returncode != 0
        assert "does not sustain requested concurrency" in mismatch.stderr
    print(json.dumps({"passed": True, "paired_intervals": 4}))


if __name__ == "__main__":
    main()
