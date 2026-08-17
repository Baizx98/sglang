#!/usr/bin/env python3
"""Summarize the batch/concurrency deadline gate without claiming speedup.

The script joins each analyzed CUDA-event table with the retained inference
responses.  It restricts comparisons to the task intersection within each
context length, audits the observed decode batch size, checks answer-first
outputs, and records configurations that are capacity-blocked rather than
silently omitting them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPECTED_LAYERS = (0, 7, 15, 23, 31, 39, 47)
EXPECTED_INTERVALS_PER_REQUEST_LAYER = 32
CHOICE_PATTERN = re.compile(r"^\s*([A-D])(?:\)|\.|\s)")


@dataclass(frozen=True)
class RunSpec:
    label: str
    context_tokens: int
    concurrency: int
    analysis_csv: Path
    requests_jsonl: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_run(values: list[str]) -> RunSpec:
    label, context, concurrency, analysis_csv, requests_jsonl = values
    return RunSpec(
        label=label,
        context_tokens=int(context),
        concurrency=int(concurrency),
        analysis_csv=Path(analysis_csv).resolve(),
        requests_jsonl=Path(requests_jsonl).resolve(),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def load_requests(spec: RunSpec) -> pd.DataFrame:
    run_config_path = spec.requests_jsonl.parent / "run_config.json"
    if not run_config_path.exists():
        raise ValueError(f"missing run_config.json for {spec.label}")
    run_config = json.loads(run_config_path.read_text())
    prepared_path = Path(run_config["prepared_requests"]).resolve()
    prepared = load_jsonl(prepared_path)
    prepared_by_rid = {str(row["rid"]): row for row in prepared}
    prepared_by_task = {str(row["task"]): row for row in prepared}

    rows: list[dict[str, Any]] = []
    for ordinal, request in enumerate(load_jsonl(spec.requests_jsonl)):
        source = prepared_by_rid.get(str(request.get("source_rid", "")))
        task = request.get("task") or (source or {}).get("task")
        if task is None:
            raise ValueError(f"cannot resolve task for {request.get('rid')}")
        task = str(task)
        expected = (source or prepared_by_task.get(task, {})).get(
            "expected_answer"
        )
        response = request.get("response") or {}
        text = str(response.get("text", ""))
        output_ids = [int(value) for value in response.get("output_ids", [])]
        match = CHOICE_PATTERN.match(text)
        rows.append(
            {
                "run_label": spec.label,
                "context_tokens": spec.context_tokens,
                "requested_concurrency": spec.concurrency,
                "request_id": str(request["rid"]),
                "task": task,
                "request_ordinal": ordinal,
                "prompt_tokens": int(request["expected_prompt_len"]),
                "reserved_output_tokens": int(run_config["max_new_tokens"]),
                "actual_output_tokens": int(request["output_tokens"]),
                "structural_pass": bool(request["structural_pass"]),
                "text_nonempty": bool(text.strip()),
                "has_replacement_character": "\ufffd" in text,
                "answer_choice": match.group(1) if match else None,
                "expected_answer": expected,
                "answer_correct": bool(
                    expected and match and match.group(1) == expected
                ),
                "output_ids": output_ids,
                "prepared_requests": str(prepared_path),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.request_id.duplicated().any():
        raise ValueError(f"duplicate request IDs in {spec.label}")
    if int(run_config.get("concurrency", 1)) != spec.concurrency:
        raise ValueError(f"run_config concurrency mismatch for {spec.label}")
    return frame


def load_timing(spec: RunSpec, request_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(spec.analysis_csv)
    required = {
        "kind",
        "request_id",
        "layer_id",
        "producer_decode_step",
        "interval_ms",
        "batch_size",
        "tp_rank_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.label} timing table lacks columns: {missing}")
    frame = frame[
        (frame.kind == "previous_step_same_layer")
        & frame.request_id.isin(request_ids)
    ].copy()
    if set(int(value) for value in frame.layer_id.unique()) != set(EXPECTED_LAYERS):
        raise ValueError(f"sampled layer mismatch for {spec.label}")
    if not (frame.tp_rank_count == 2).all():
        raise ValueError(f"one or more {spec.label} intervals lack a TP rank")
    if not (frame.batch_size == spec.concurrency).all():
        observed = sorted(int(value) for value in frame.batch_size.unique())
        raise ValueError(
            f"{spec.label} did not sustain batch={spec.concurrency}: {observed}"
        )
    counts = frame.groupby(["request_id", "layer_id"]).size()
    if not (counts == EXPECTED_INTERVALS_PER_REQUEST_LAYER).all():
        raise ValueError(f"incomplete request/layer timing groups for {spec.label}")
    if not np.isfinite(frame.interval_ms).all() or not (frame.interval_ms >= 0).all():
        raise ValueError(f"invalid interval values for {spec.label}")
    return frame


def timing_row(spec: RunSpec, timing: pd.DataFrame) -> dict[str, Any]:
    values = timing.interval_ms.to_numpy(dtype=np.float64)
    return {
        "context_tokens": spec.context_tokens,
        "context_label": f"{spec.context_tokens // 1024}K",
        "concurrency": spec.concurrency,
        "run_status": "measured",
        "evidence_type": "cuda_event",
        "run_label": spec.label,
        "requests": int(timing.request_id.nunique()),
        "tasks": int(timing.task.nunique()),
        "layers": int(timing.layer_id.nunique()),
        "intervals": len(timing),
        "observed_batch_min": int(timing.batch_size.min()),
        "observed_batch_max": int(timing.batch_size.max()),
        "mean_ms": float(values.mean()),
        "p1_ms": float(np.quantile(values, 0.01)),
        "p10_ms": float(np.quantile(values, 0.10)),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p90_ms": float(np.quantile(values, 0.90)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "fraction_ge_40ms": float((values >= 40.0).mean()),
        "fraction_ge_50ms": float((values >= 50.0).mean()),
    }


def summarize_per_layer(spec: RunSpec, timing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for layer_id, part in timing.groupby("layer_id", sort=True):
        values = part.interval_ms.to_numpy(dtype=np.float64)
        rows.append(
            {
                "context_tokens": spec.context_tokens,
                "context_label": f"{spec.context_tokens // 1024}K",
                "concurrency": spec.concurrency,
                "run_label": spec.label,
                "layer_id": int(layer_id),
                "requests": int(part.request_id.nunique()),
                "intervals": len(part),
                "p1_ms": float(np.quantile(values, 0.01)),
                "p10_ms": float(np.quantile(values, 0.10)),
                "p50_ms": float(np.quantile(values, 0.50)),
                "min_ms": float(values.min()),
                "fraction_ge_40ms": float((values >= 40.0).mean()),
                "fraction_ge_50ms": float((values >= 50.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_output_audit(
    specs: list[RunSpec], requests: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_frames = []
    for spec in specs:
        frame = requests[spec.label].copy()
        baseline_spec = min(
            (
                candidate
                for candidate in specs
                if candidate.context_tokens == spec.context_tokens
            ),
            key=lambda candidate: candidate.concurrency,
        )
        baseline = requests[baseline_spec.label].set_index("task")
        frame["baseline_run_label"] = baseline_spec.label
        frame["exact_output_vs_c1"] = frame.apply(
            lambda row: row.output_ids == baseline.loc[row.task, "output_ids"], axis=1
        )
        frame["common_prefix_tokens_vs_c1"] = frame.apply(
            lambda row: common_prefix_length(
                row.output_ids, baseline.loc[row.task, "output_ids"]
            ),
            axis=1,
        )
        frame["answer_choice_matches_c1"] = frame.apply(
            lambda row: row.answer_choice == baseline.loc[row.task, "answer_choice"],
            axis=1,
        )
        detail_frames.append(frame)
    detail = pd.concat(detail_frames, ignore_index=True)
    detail_export = detail.drop(columns=["output_ids"])
    rows = []
    for (context, concurrency, label), part in detail.groupby(
        ["context_tokens", "requested_concurrency", "run_label"], sort=True
    ):
        rows.append(
            {
                "context_tokens": int(context),
                "context_label": f"{int(context) // 1024}K",
                "concurrency": int(concurrency),
                "run_label": label,
                "requests": len(part),
                "structural_pass_fraction": float(part.structural_pass.mean()),
                "nonempty_text_fraction": float(part.text_nonempty.mean()),
                "replacement_character_fraction": float(
                    part.has_replacement_character.mean()
                ),
                "fixed_length_fraction": float(
                    (
                        part.actual_output_tokens
                        == part.reserved_output_tokens
                    ).mean()
                ),
                "answer_extracted_fraction": float(part.answer_choice.notna().mean()),
                "answer_accuracy": float(part.answer_correct.mean()),
                "answer_choice_matches_c1_fraction": float(
                    part.answer_choice_matches_c1.mean()
                ),
                "exact_output_vs_c1_fraction": float(
                    part.exact_output_vs_c1.mean()
                ),
                "mean_common_prefix_tokens_vs_c1": float(
                    part.common_prefix_tokens_vs_c1.mean()
                ),
            }
        )
    return detail_export, pd.DataFrame(rows)


def capacity_rows(
    specs: list[RunSpec],
    requests: dict[str, pd.DataFrame],
    common_tasks: dict[int, set[str]],
    token_pool: int,
    sweep: list[int],
) -> pd.DataFrame:
    measured = {(spec.context_tokens, spec.concurrency) for spec in specs}
    rows: list[dict[str, Any]] = []
    for context in sorted(common_tasks):
        reference_spec = min(
            (spec for spec in specs if spec.context_tokens == context),
            key=lambda spec: spec.concurrency,
        )
        reference = requests[reference_spec.label].sort_values("request_ordinal")
        reference = reference[reference.task.isin(common_tasks[context])]
        prompt = reference.prompt_tokens.astype(int).tolist()
        reserve = reference.reserved_output_tokens.astype(int).tolist()
        for concurrency in sweep:
            wave_demands = [
                sum(prompt[start : start + concurrency])
                + sum(reserve[start : start + concurrency])
                for start in range(0, len(prompt), concurrency)
            ]
            demand = max(wave_demands)
            is_measured = (context, concurrency) in measured
            feasible = demand <= token_pool
            rows.append(
                {
                    "context_tokens": context,
                    "context_label": f"{context // 1024}K",
                    "concurrency": concurrency,
                    "tasks": len(prompt),
                    "maximum_wave_reserved_tokens": demand,
                    "server_kv_token_pool": token_pool,
                    "pool_utilization": demand / token_pool,
                    "capacity_feasible": feasible,
                    "run_status": (
                        "measured"
                        if is_measured
                        else "capacity_blocked"
                        if not feasible
                        else "not_measured"
                    ),
                    "evidence_type": (
                        "request_manifest+server_pool"
                        if is_measured
                        else "capacity_model"
                    ),
                }
            )
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_gate(
    timing: pd.DataFrame, capacity: pd.DataFrame, output_dir: Path
) -> None:
    configure_style()
    colors = {32768: "#0072B2", 65536: "#D55E00"}
    labels = {32768: "32K", 65536: "64K"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)

    axis = axes[0]
    for context, part in timing.groupby("context_tokens", sort=True):
        part = part.sort_values("concurrency")
        color = colors[int(context)]
        axis.plot(
            part.concurrency,
            part.p1_ms,
            color=color,
            marker="o",
            label=f"{labels[int(context)]} P1",
        )
        axis.plot(
            part.concurrency,
            part.p10_ms,
            color=color,
            marker="s",
            linestyle="--",
            label=f"{labels[int(context)]} P10",
        )
    axis.axhline(40.0, color="#4D4D4D", linestyle=":", linewidth=1.3)
    axis.text(4.04, 41.8, "40 ms gate", ha="right", va="bottom", color="#4D4D4D")
    blocked = capacity[
        (capacity.context_tokens == 65536)
        & (capacity.run_status == "capacity_blocked")
    ]
    if len(blocked):
        axis.annotate(
            "64K c4: capacity-blocked",
            xy=(4, 0),
            xycoords=axis.get_xaxis_transform(),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#9E9E9E",
        )
    axis.set_xticks([1, 2, 4])
    axis.set_xlabel("Decode concurrency")
    axis.set_ylabel("Same-layer window (ms, higher is better)")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.legend(ncol=2, loc="upper left")
    axis.text(-0.14, 1.04, "(a)", transform=axis.transAxes, fontweight="bold")

    axis = axes[1]
    x = np.arange(3, dtype=float)
    width = 0.34
    for offset, context in [(-width / 2, 32768), (width / 2, 65536)]:
        part = capacity[capacity.context_tokens == context].set_index("concurrency")
        values = [
            part.loc[value, "maximum_wave_reserved_tokens"] / 1000
            for value in (1, 2, 4)
        ]
        bars = axis.bar(
            x + offset,
            values,
            width,
            color=colors[context],
            edgecolor="white",
            linewidth=0.7,
            label=labels[context],
        )
        if context == 65536 and not bool(part.loc[4, "capacity_feasible"]):
            bars[-1].set_facecolor("white")
            bars[-1].set_edgecolor(colors[context])
            bars[-1].set_hatch("///")
            bars[-1].set_linewidth(1.0)
    pool = float(capacity.server_kv_token_pool.iloc[0]) / 1000
    axis.axhline(pool, color="#4D4D4D", linestyle=":", linewidth=1.3, label="KV pool")
    axis.set_xticks(x, ["1", "2", "4"])
    axis.set_xlabel("Requested concurrency")
    axis.set_ylabel("Max wave reservation (K tokens)")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.legend(ncol=2, loc="upper left")
    axis.text(-0.14, 1.04, "(b)", transform=axis.transAxes, fontweight="bold")

    stem = output_dir / "concurrency_deadline_gate"
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "matplotlib"},
    )
    plt.close(fig)


def git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=5,
        required=True,
        metavar=("LABEL", "CONTEXT", "CONCURRENCY", "PAIRED_CSV", "REQUESTS_JSONL"),
    )
    parser.add_argument("--server-kv-token-pool", type=int, required=True)
    parser.add_argument("--capacity-concurrency", default="1,2,4")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    specs = [parse_run(values) for values in args.run]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = [int(value) for value in args.capacity_concurrency.split(",")]

    requests = {spec.label: load_requests(spec) for spec in specs}
    common_tasks: dict[int, set[str]] = {}
    for context in sorted({spec.context_tokens for spec in specs}):
        task_sets = [
            set(requests[spec.label].task)
            for spec in specs
            if spec.context_tokens == context
        ]
        common_tasks[context] = set.intersection(*task_sets)
        if not common_tasks[context]:
            raise ValueError(f"no common tasks for context={context}")

    timing_rows = []
    per_layer_frames = []
    filtered_requests: dict[str, pd.DataFrame] = {}
    for spec in specs:
        request_frame = requests[spec.label]
        request_frame = request_frame[
            request_frame.task.isin(common_tasks[spec.context_tokens])
        ].copy()
        filtered_requests[spec.label] = request_frame
        timing = load_timing(spec, set(request_frame.request_id))
        task_by_request = request_frame.set_index("request_id").task
        timing["task"] = timing.request_id.map(task_by_request)
        if timing.task.isna().any():
            raise ValueError(f"unmapped timing requests for {spec.label}")
        timing_rows.append(timing_row(spec, timing))
        per_layer_frames.append(summarize_per_layer(spec, timing))

    timing_summary = pd.DataFrame(timing_rows).sort_values(
        ["context_tokens", "concurrency"]
    )
    per_layer = pd.concat(per_layer_frames, ignore_index=True).sort_values(
        ["context_tokens", "concurrency", "layer_id"]
    )
    output_detail, output_summary = build_output_audit(specs, filtered_requests)
    capacity = capacity_rows(
        specs,
        filtered_requests,
        common_tasks,
        args.server_kv_token_pool,
        sweep,
    )
    gate = capacity.merge(
        timing_summary,
        on=["context_tokens", "context_label", "concurrency", "run_status"],
        how="left",
        suffixes=("_capacity", "_timing"),
    ).sort_values(["context_tokens", "concurrency"])

    timing_summary.to_csv(output_dir / "timing_summary.csv", index=False)
    per_layer.to_csv(output_dir / "per_layer_summary.csv", index=False)
    output_detail.to_csv(output_dir / "output_audit_detail.csv", index=False)
    output_summary.to_csv(output_dir / "output_audit_summary.csv", index=False)
    capacity.to_csv(output_dir / "capacity_summary.csv", index=False)
    gate.to_csv(output_dir / "gate_summary.csv", index=False)
    plot_gate(timing_summary, capacity, output_dir)

    validation = {
        "schema_version": 1,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "speedup_measurement": False,
        "timing_evidence": "real CUDA events, minimum across two TP ranks",
        "capacity_evidence": (
            "request prompt + reserved output tokens versus logged server KV token pool"
        ),
        "common_tasks_by_context": {
            str(context): sorted(tasks) for context, tasks in common_tasks.items()
        },
        "all_measured_configs_sustain_requested_batch": True,
        "all_measured_intervals_ge_40ms": bool(
            (timing_summary.fraction_ge_40ms == 1.0).all()
        ),
        "all_measured_intervals_ge_50ms": bool(
            (timing_summary.fraction_ge_50ms == 1.0).all()
        ),
        "capacity_blocked_configs": capacity.loc[
            capacity.run_status == "capacity_blocked",
            ["context_label", "concurrency"],
        ].to_dict("records"),
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "inputs": [
            {
                "label": spec.label,
                "context_tokens": spec.context_tokens,
                "concurrency": spec.concurrency,
                "analysis_csv": str(spec.analysis_csv),
                "analysis_sha256": sha256_file(spec.analysis_csv),
                "requests_jsonl": str(spec.requests_jsonl),
                "requests_sha256": sha256_file(spec.requests_jsonl),
            }
            for spec in specs
        ],
    }
    write_json(output_dir / "validation.json", validation)
    manifest = {
        "schema_version": 1,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "server_kv_token_pool": args.server_kv_token_pool,
        "outputs": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
