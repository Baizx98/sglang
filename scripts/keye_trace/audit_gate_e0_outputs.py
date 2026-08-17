#!/usr/bin/env python3
"""Audit Gate E0 saved generations with dataset-appropriate metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string
import subprocess
from typing import Any

import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def token_f1(prediction: str, references: list[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    best = 0.0
    for reference in references:
        reference_tokens = normalize_answer(reference).split()
        prediction_counts = Counter(prediction_tokens)
        reference_counts = Counter(reference_tokens)
        common = sum((prediction_counts & reference_counts).values())
        if not common:
            continue
        precision = common / len(prediction_tokens)
        recall = common / len(reference_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def choice_match(prediction: str, answer: str) -> bool:
    prediction = prediction.strip().upper()
    matches = re.findall(r"\b[A-D]\b", prediction)
    if matches:
        return matches[-1] == answer.upper()
    return bool(prediction) and prediction[0] == answer.upper()


def ruler_score(task: str, prediction: str, references: list[str]) -> float:
    matches = [reference.lower() in prediction.lower() for reference in references]
    if task.startswith("qa_"):
        return float(any(matches))
    return sum(matches) / len(matches)


def infinitebench_score(
    task: str,
    prediction: str,
    references: list[str],
    expected_choice: str | None,
) -> tuple[float, str]:
    if task == "passkey":
        match = re.search(r"\d+", prediction)
        return float(bool(match) and match.group(0) == str(references[0])), "exact_first_integer"
    if task == "longbook_choice_eng":
        return float(choice_match(prediction, str(expected_choice))), "exact_choice"
    if task == "longdialogue_qa_eng":
        upper = prediction.strip().upper()
        return float(any(reference.upper() in upper for reference in references)), "substring_accuracy"
    if task == "longbook_qa_eng":
        return token_f1(prediction, references), "official_normalized_token_f1"
    raise ValueError(f"unsupported InfiniteBench task: {task}")


def audit_run(path: Path, allow_incomplete: bool) -> tuple[list[dict[str, Any]], int]:
    prepared_rows = read_jsonl(path / "prepared_requests.jsonl")
    executed_rows = read_jsonl(path / "requests.jsonl")
    prepared = {str(row["rid"]): row for row in prepared_rows}
    executed = {str(row["rid"]): row for row in executed_rows}
    if len(prepared) != len(prepared_rows) or len(executed) != len(executed_rows):
        raise ValueError(f"{path}: duplicate request IDs")
    missing = set(prepared) - set(executed)
    unexpected = set(executed) - set(prepared)
    if unexpected or (missing and not allow_incomplete):
        raise ValueError(
            f"{path}: request mismatch missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    rows: list[dict[str, Any]] = []
    for rid in sorted(set(prepared) & set(executed)):
        source = prepared[rid]
        output = executed[rid]
        prediction = str((output.get("response") or {}).get("text") or "")
        dataset = str(source["dataset"])
        task = str(source["task"])
        if dataset.lower() == "ruler":
            score = ruler_score(task, prediction, list(source["expected_outputs"]))
            metric = "official_string_match"
        elif dataset.lower() == "longbench-v2":
            score = float(choice_match(prediction, str(source["expected_answer"])))
            metric = "exact_choice"
        elif dataset.lower() == "infinitebench":
            score, metric = infinitebench_score(
                task,
                prediction,
                [str(value) for value in source["expected_answers"]],
                source.get("expected_choice"),
            )
        else:
            raise ValueError(f"unsupported dataset: {dataset}")
        response = output.get("response") or {}
        metadata = response.get("meta_info") or {}
        rows.append(
            {
                "rid": rid,
                "dataset": dataset,
                "length_config": int(source["length_config"]),
                "task": task,
                "prompt_len": int(source["prompt_len"]),
                "output_tokens": len(response.get("output_ids") or []),
                "finish_type": (metadata.get("finish_reason") or {}).get("type"),
                "latency_s": float(output["latency_s"]),
                "metric": metric,
                "score": score,
                "text_preview": prediction[:500],
            }
        )
    return rows, len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    run_dirs = sorted(
        path.parent
        for path in args.run_dir.glob("*/prepared_requests.jsonl")
        if path.parent.name != "full48-paired" and (path.parent / "requests.jsonl").exists()
    )
    all_rows: list[dict[str, Any]] = []
    missing_requests = 0
    for run_dir in run_dirs:
        part, missing = audit_run(run_dir, args.allow_incomplete)
        all_rows.extend(part)
        missing_requests += missing
    rows = pd.DataFrame(all_rows)
    if rows.empty:
        raise ValueError("no completed runs")
    summary = (
        rows.groupby(["dataset", "length_config", "task", "metric"], as_index=False)
        .agg(
            requests=("rid", "size"),
            mean_score=("score", "mean"),
            mean_latency_s=("latency_s", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
        )
        .sort_values(["dataset", "length_config", "task"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "quality_by_request.csv", index=False)
    summary.to_csv(args.output_dir / "quality_by_task.csv", index=False)
    payload = {
        "schema_version": 1,
        "run_dir": str(args.run_dir.resolve()),
        "audited_run_dirs": [str(path.resolve()) for path in run_dirs],
        "audited_requests": len(rows),
        "missing_requests_in_started_runs": missing_requests,
        "datasets": sorted(rows.dataset.unique().tolist()),
        "all_outputs_nonempty": bool((rows.output_tokens > 0).all()),
        "all_latencies_positive": bool((rows.latency_s > 0).all()),
        "metric_provenance": {
            "RULER": {
                "revision": git_revision(Path("data/external/RULER")),
                "source": "scripts/eval/synthetic/constants.py",
            },
            "InfiniteBench": {
                "revision": git_revision(Path("data/external/InfiniteBench")),
                "source": "src/compute_scores.py",
            },
            "LongBench-v2": {
                "metric": "multiple-choice exact answer letter",
            },
        },
        "coverage_complete": missing_requests == 0,
        "passed": bool(
            (rows.output_tokens > 0).all()
            and (rows.latency_s > 0).all()
            and missing_requests == 0
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
