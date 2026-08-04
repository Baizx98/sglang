#!/usr/bin/env python3
"""Select one long, tool-bearing request per BFCL trajectory for lookahead tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

SEED = 20260804


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_requests(source_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(source_dir / "prepared_requests.jsonl"):
        grouped[row["trajectory_id"]].append(row)

    selected = []
    for trajectory_id, rows in grouped.items():
        eligible = [row for row in rows if row.get("ground_truth_calls")]
        if not eligible:
            raise ValueError(f"{trajectory_id} has no request with a ground-truth call")
        selected.append(max(eligible, key=lambda row: (int(row["prompt_len"]), int(row["round_id"]))))

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_category[row["category"]].append(row)
    split_by_rid: dict[str, str] = {}
    rng = random.Random(SEED)
    for category, rows in sorted(by_category.items()):
        rows.sort(key=lambda row: (int(row["prompt_len"]), row["trajectory_id"]))
        if len(rows) != 6:
            raise ValueError(f"expected 6 trajectories for {category}, found {len(rows)}")
        # Pair adjacent prompt lengths and place one member of every pair in each split.
        for offset in range(0, len(rows), 2):
            pair = rows[offset : offset + 2]
            rng.shuffle(pair)
            split_by_rid[pair[0]["rid"]] = "calibration"
            split_by_rid[pair[1]["rid"]] = "test"

    selected.sort(key=lambda row: (row["category"], row["trajectory_id"]))
    return selected, split_by_rid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["calibration", "test"],
        help="optionally materialize only one pre-registered split",
    )
    args = parser.parse_args()

    source_dir = args.source_run.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, split_by_rid = select_requests(source_dir)
    if args.split:
        selected = [row for row in selected if split_by_rid[row["rid"]] == args.split]
    selected_rids = {row["rid"] for row in selected}
    selected_trajectories = {row["trajectory_id"] for row in selected}
    prepared = [dict(row, split=split_by_rid[row["rid"]]) for row in selected]
    segments = [
        row
        for row in read_jsonl(source_dir / "segments.jsonl")
        if row["rid"] in selected_rids
    ]
    snapshots = [
        row
        for row in json.loads((source_dir / "dataset_snapshot.json").read_text())
        if row["case_id"] in selected_trajectories
    ]
    if len(snapshots) != len(selected):
        raise ValueError(f"expected {len(selected)} snapshots, found {len(snapshots)}")

    write_jsonl(output_dir / "prepared_requests.jsonl", prepared)
    write_jsonl(output_dir / "segments.jsonl", segments)
    write_json(output_dir / "dataset_snapshot.json", snapshots)
    selection_rows = [
        {
            "trajectory_id": row["trajectory_id"],
            "category": row["category"],
            "round_id": int(row["round_id"]),
            "rid": row["rid"],
            "prompt_len": int(row["prompt_len"]),
            "split": split_by_rid[row["rid"]],
            "ground_truth_call_count": len(row["ground_truth_calls"]),
        }
        for row in prepared
    ]
    write_json(
        output_dir / "selection.json",
        {
            "schema_version": 1,
            "seed": SEED,
            "policy": "longest request with nonempty ground_truth_calls per trajectory",
            "split_policy": "within each category, pair adjacent prompt lengths and seeded-shuffle one into each split",
            "source_run": str(source_dir),
            "source_prepared_sha256": sha256(source_dir / "prepared_requests.jsonl"),
            "request_count": len(selection_rows),
            "split_counts": {
                split: sum(row["split"] == split for row in selection_rows)
                for split in ["calibration", "test"]
            },
            "selected": selection_rows,
        },
    )
    source_config = source_dir / "run_config.json"
    if source_config.exists():
        shutil.copy2(source_config, output_dir / "source_run_config.json")

    print(json.dumps({"output_dir": str(output_dir), "requests": len(prepared), "segments": len(segments)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
