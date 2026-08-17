#!/usr/bin/env python3
"""Export cleaned Gate E0/E1 tables for the research-note repository."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_dataset(value: str) -> str:
    return {
        "longbench-v2": "LongBench-v2",
        "ruler": "RULER",
        "infinitebench": "InfiniteBench",
    }.get(value.lower(), value)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e0-analysis", type=Path, required=True)
    parser.add_argument("--quality-analysis", type=Path, required=True)
    parser.add_argument("--e1-expanded-analysis", type=Path, required=True)
    parser.add_argument("--e1-full48-analysis", type=Path, required=True)
    parser.add_argument("--e1-layer-calibration", type=Path, required=True)
    parser.add_argument("--e1-layer-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timestamp",
        required=True,
        help="filename timestamp in YYYY-MM-DD_HHMM format",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{4}", args.timestamp):
        raise ValueError("--timestamp must use YYYY-MM-DD_HHMM")

    inputs: list[Path] = []
    outputs: list[Path] = []
    e0_summary_path = args.e0_analysis / "summary.json"
    e0_audit_path = args.e0_analysis / "independent_audit.json"
    quality_summary_path = args.quality_analysis / "summary.json"
    e1_expanded_summary_path = args.e1_expanded_analysis / "summary.json"
    e1_expanded_audit_path = args.e1_expanded_analysis / "independent_audit.json"
    e1_full48_summary_path = args.e1_full48_analysis / "summary.json"
    e1_full48_audit_path = args.e1_full48_analysis / "independent_audit.json"
    e1_calibration_summary_path = args.e1_layer_calibration / "summary.json"
    e1_profile_summary_path = args.e1_layer_profile / "summary.json"
    validation_paths = [
        e0_summary_path,
        e0_audit_path,
        quality_summary_path,
        e1_expanded_summary_path,
        e1_expanded_audit_path,
        e1_full48_summary_path,
        e1_full48_audit_path,
        e1_calibration_summary_path,
        e1_profile_summary_path,
    ]
    validation = {str(path): load_json(path) for path in validation_paths}
    e0_summary = validation[str(e0_summary_path)]
    quality_summary = validation[str(quality_summary_path)]
    e1_expanded_summary = validation[str(e1_expanded_summary_path)]
    e1_full48_summary = validation[str(e1_full48_summary_path)]
    e1_calibration_summary = validation[str(e1_calibration_summary_path)]
    e1_profile_summary = validation[str(e1_profile_summary_path)]
    contracts = {
        "e0_complete_and_p4": bool(
            e0_summary.get("all_contracts_passed")
            and e0_summary.get("coverage_complete")
            and int(e0_summary.get("page_size", -1)) == 4
        ),
        "e0_independent_audit_passed": bool(
            validation[str(e0_audit_path)].get("passed")
        ),
        "output_quality_audit_passed": bool(
            quality_summary.get("passed") and quality_summary.get("coverage_complete")
        ),
        "e1_expanded_independent_audit_passed": bool(
            validation[str(e1_expanded_audit_path)].get("passed")
        ),
        "e1_full48_independent_audit_passed": bool(
            validation[str(e1_full48_audit_path)].get("passed")
        ),
        "e1_expanded_is_seven_layer": len(
            e1_expanded_summary.get("setting", {}).get("layers", [])
        )
        == 7,
        "e1_full48_is_all_layer": e1_full48_summary.get("setting", {}).get(
            "layers"
        )
        == list(range(48)),
        "e1_layer_pairs_matched": bool(
            e1_calibration_summary.get("all_request_pairs_matched")
        ),
        "e1_full48_layer_profile_complete": bool(
            e1_profile_summary.get("all_contracts_passed")
            and e1_profile_summary.get("layers") == list(range(48))
        ),
    }
    if not all(contracts.values()):
        failed = [key for key, value in contracts.items() if not value]
        raise ValueError(f"core-data export validation failed: {failed}")
    inputs.extend(validation_paths)
    e0_path = args.e0_analysis / "tables/dataset_length_summary.csv"
    inputs.append(e0_path)
    e0 = pd.read_csv(e0_path)
    e0["dataset"] = e0.dataset.map(normalize_dataset)
    e0["evidence_kind"] = e0.estimator.map(
        {
            "full48": "full48_measurement",
            "seven_layer_nearest_weighted": "seven_layer_estimate",
        }
    )
    e0 = e0[
        [
            "evidence_kind",
            "dataset",
            "length_config",
            "metric",
            "requests",
            "tasks",
            "mean",
            "ci95_low",
            "ci95_high",
            "task_balanced_mean",
            "task_cluster_ci95_low",
            "task_cluster_ci95_high",
            "median",
            "p10",
            "p90",
        ]
    ].sort_values(["evidence_kind", "dataset", "length_config", "metric"])
    path = args.output_dir / f"{args.timestamp}_gate-e0-generalization-core-data_v01_final.csv"
    write_csv(e0, path)
    outputs.append(path)

    e0_calibration_path = args.e0_analysis / "tables/layer_sampling_calibration_summary.csv"
    inputs.append(e0_calibration_path)
    e0_calibration = pd.read_csv(e0_calibration_path)
    e0_calibration["dataset"] = e0_calibration.dataset.map(normalize_dataset)
    path = args.output_dir / f"{args.timestamp}_gate-e0-layer-sampling-calibration-core-data_v01_final.csv"
    write_csv(
        e0_calibration.sort_values(["dataset", "length_config", "metric"]), path
    )
    outputs.append(path)

    quality_path = args.quality_analysis / "quality_by_task.csv"
    inputs.append(quality_path)
    quality = pd.read_csv(quality_path)
    quality["dataset"] = quality.dataset.map(normalize_dataset)
    path = args.output_dir / f"{args.timestamp}_gate-e0-output-quality-core-data_v01_final.csv"
    write_csv(quality.sort_values(["dataset", "length_config", "task"]), path)
    outputs.append(path)

    e1_frames: list[pd.DataFrame] = []
    for evidence, analysis_dir in [
        ("full48_measurement", args.e1_full48_analysis),
        ("expanded_seven_layer_estimate", args.e1_expanded_analysis),
    ]:
        source = analysis_dir / "tables/dataset_length_summary.csv"
        inputs.append(source)
        frame = pd.read_csv(source)
        frame["dataset"] = frame.dataset.map(normalize_dataset)
        frame.insert(0, "evidence_kind", evidence)
        e1_frames.append(frame)
    e1 = pd.concat(e1_frames, ignore_index=True).sort_values(
        ["evidence_kind", "dataset", "context_config", "metric"]
    )
    path = args.output_dir / f"{args.timestamp}_gate-e1-resident-protection-core-data_v01_final.csv"
    write_csv(e1, path)
    outputs.append(path)

    e1_calibration_path = args.e1_layer_calibration / "sampling_error_summary.csv"
    inputs.append(e1_calibration_path)
    e1_calibration = pd.read_csv(e1_calibration_path)
    e1_calibration["dataset"] = e1_calibration.dataset.map(normalize_dataset)
    path = args.output_dir / f"{args.timestamp}_gate-e1-layer-sampling-calibration-core-data_v01_final.csv"
    write_csv(
        e1_calibration.sort_values(["dataset", "context_config", "metric"]), path
    )
    outputs.append(path)

    e1_profile_path = args.e1_layer_profile / "layer_summary.csv"
    inputs.append(e1_profile_path)
    e1_profile = pd.read_csv(e1_profile_path)
    path = args.output_dir / f"{args.timestamp}_gate-e1-full48-layer-profile-core-data_v01_final.csv"
    write_csv(e1_profile.sort_values(["metric", "layer"]), path)
    outputs.append(path)

    manifest = {
        "schema_version": 1,
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in inputs
        ],
        "outputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in outputs
        ],
        "evidence_semantics": {
            "full48_measurement": "all 48 DSA layers replayed",
            "seven_layer_estimate": "nearest-layer weighted estimate; use calibration table",
            "expanded_seven_layer_estimate": "frozen external request set; seven sampled layers",
        },
        "validation_contracts": contracts,
    }
    manifest_path = args.output_dir / f"{args.timestamp}_gate-e0-e1-core-data-provenance_v01_final.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
