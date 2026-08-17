#!/usr/bin/env python3
"""Audit frozen Gate E0 request, output, manifest, and file coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNS = {
    "ruler-65536": 12,
    "ruler-131072": 12,
    "longbench-v2-65536": 10,
    "longbench-v2-131072": 11,
    "infinitebench-131072": 12,
}
LAYERS = [0, 7, 15, 23, 31, 39, 47]
STEPS = list(range(32))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def unique_map(rows: list[dict[str, Any]], key: str, source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"{source}: duplicate {key}={value}")
        result[value] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    prepared: dict[str, dict[str, Any]] = {}
    executed: dict[str, dict[str, Any]] = {}
    per_run: dict[str, Any] = {}
    for name, expected_count in RUNS.items():
        directory = args.run_dir / name
        prepared_path = directory / "prepared_requests.jsonl"
        requests_path = directory / "requests.jsonl"
        prepared_rows = read_jsonl(prepared_path)
        executed_rows = read_jsonl(requests_path) if requests_path.exists() else []
        prepared_part = unique_map(prepared_rows, "rid", prepared_path)
        executed_part = unique_map(executed_rows, "rid", requests_path)
        if len(prepared_part) != expected_count:
            failures.append(f"{name}: prepared={len(prepared_part)} expected={expected_count}")
        missing = sorted(set(prepared_part) - set(executed_part))
        unexpected = sorted(set(executed_part) - set(prepared_part))
        if unexpected:
            failures.append(f"{name}: unexpected outputs={unexpected[:3]}")
        if missing and not args.allow_incomplete:
            failures.append(f"{name}: missing outputs={len(missing)}")
        overlap = set(prepared) & set(prepared_part)
        if overlap:
            failures.append(f"{name}: prepared IDs reused across runs={sorted(overlap)[:3]}")
        prepared.update(prepared_part)
        executed.update(executed_part)
        per_run[name] = {
            "expected": expected_count,
            "prepared": len(prepared_part),
            "executed": len(executed_part),
            "missing": len(missing),
            "unexpected": len(unexpected),
        }

    events_dir = args.run_dir / "events"
    manifest_path = events_dir / "manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    event_ids = [int(row["event_id"]) for row in manifest]
    if len(event_ids) != len(set(event_ids)):
        failures.append("manifest: duplicate event_id")
    if event_ids != list(range(len(event_ids))):
        failures.append("manifest: event_id is not contiguous from zero")
    pairs: set[tuple[str, int]] = set()
    duplicate_pairs: list[tuple[str, int]] = []
    bad_contract_rows: list[str] = []
    for row in manifest:
        rid = str(row["request_id"])
        layer = int(row["layer_id"])
        pair = (rid, layer)
        if pair in pairs:
            duplicate_pairs.append(pair)
        pairs.add(pair)
        metadata = prepared.get(rid)
        contract_ok = bool(
            metadata
            and int(row["schema_version"]) == 5
            and row["topk_backend"] == "torch_exact"
            and int(row["topk_width"]) == 2048
            and int(row["compact_k"]) == 4096
            and [int(value) for value in row["score_threshold_ranks"]]
            == [2048, 2560, 3072, 4096]
            and not bool(row["full_scores_retained"])
            and int(row["num_steps"]) == 32
            and [int(value) for value in row["decode_step_ids"]] == STEPS
            and int(row["score_valid_min"]) == int(metadata["prompt_len"]) + 1
            and int(row["score_valid_max"]) == int(metadata["prompt_len"]) + 32
            and row["request_ids"] == [rid]
        )
        file_path = events_dir / str(row["file"])
        contract_ok = contract_ok and file_path.exists() and file_path.stat().st_size == int(row["bytes"])
        if not contract_ok:
            bad_contract_rows.append(str(row.get("file")))
    if duplicate_pairs:
        failures.append(f"manifest: duplicate request-layer pairs={duplicate_pairs[:3]}")
    if bad_contract_rows:
        failures.append(f"manifest: bad contract rows={bad_contract_rows[:3]}")

    complete_ids = {
        rid for rid in prepared if all((rid, layer) in pairs for layer in LAYERS)
    }
    partial_ids = {
        rid
        for rid in prepared
        if 0 < sum((rid, layer) in pairs for layer in LAYERS) < len(LAYERS)
    }
    unexpected_trace_ids = sorted({rid for rid, _ in pairs} - set(prepared))
    if partial_ids:
        failures.append(f"manifest: partial requests={sorted(partial_ids)[:3]}")
    if unexpected_trace_ids:
        failures.append(f"manifest: unexpected trace requests={unexpected_trace_ids[:3]}")
    if not args.allow_incomplete and complete_ids != set(prepared):
        failures.append(f"manifest: complete requests={len(complete_ids)} expected={len(prepared)}")
    if complete_ids != set(executed):
        failures.append(
            "manifest/output mismatch: "
            f"trace_only={sorted(complete_ids-set(executed))[:3]}, "
            f"output_only={sorted(set(executed)-complete_ids)[:3]}"
        )
    manifest_files = {str(row["file"]) for row in manifest}
    disk_files = {path.name for path in events_dir.glob("*.pt")}
    orphan_files = sorted(disk_files - manifest_files)
    missing_files = sorted(manifest_files - disk_files)
    if orphan_files or missing_files:
        failures.append(
            f"trace files: orphan={orphan_files[:3]}, missing={missing_files[:3]}"
        )

    coverage_complete = (
        len(prepared) == sum(RUNS.values())
        and len(executed) == sum(RUNS.values())
        and len(manifest) == sum(RUNS.values()) * len(LAYERS)
        and len(complete_ids) == sum(RUNS.values())
    )
    result = {
        "schema_version": 1,
        "run_dir": str(args.run_dir.resolve()),
        "expected_runs": RUNS,
        "per_run": per_run,
        "prepared_requests": len(prepared),
        "executed_requests": len(executed),
        "complete_trace_requests": len(complete_ids),
        "partial_trace_requests": len(partial_ids),
        "manifest_records": len(manifest),
        "expected_manifest_records": sum(RUNS.values()) * len(LAYERS),
        "layers": LAYERS,
        "orphan_trace_files": orphan_files,
        "missing_trace_files": missing_files,
        "coverage_complete": coverage_complete,
        "failures": failures,
        "passed": not failures and coverage_complete,
    }
    output = args.output or args.run_dir / "analysis/gate-e0-collection-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures or (not coverage_complete and not args.allow_incomplete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
