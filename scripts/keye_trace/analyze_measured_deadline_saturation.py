#!/usr/bin/env python3
"""Join measured DSA windows with the frozen LongBench-v2 transfer model.

This analysis does not rerun serving and does not estimate end-to-end speedup.
It asks whether the modeled prefetch transfer fits inside the CUDA-event window
measured for the same context, task, decode transition, and sampled layers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACTION_K = (2048, 2560, 3072, 4096)
FIXED_DEADLINES_MS = (20.0, 50.0, 100.0)
EXPECTED_LAYERS = (0, 7, 15, 23, 31, 39, 47)
CONTEXT_LABEL_TO_TOKENS = {"64K": 65536, "128K": 131072}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_request_steps(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["source_table"] = str(path.resolve())
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    required = {
        "request_id",
        "task",
        "context_config",
        "method",
        "candidate_k",
        "step",
        "prefetch_ms",
        "correction_ms",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"request-step table is missing columns: {missing}")
    data = data[
        (data.method == "lru-demand")
        | (
            (data.method == "previous-score")
            & data.candidate_k.isin(ACTION_K)
        )
    ].copy()
    if set(int(value) for value in data.context_config.unique()) != set(
        CONTEXT_LABEL_TO_TOKENS.values()
    ):
        raise ValueError("expected exactly the 64K and 128K context configs")
    key = ["context_config", "task", "method", "candidate_k", "step"]
    if data.duplicated(key).any():
        raise ValueError("duplicate request-step policy rows")
    counts = data.groupby(key[:-1]).size()
    if not (counts == 31).all():
        raise ValueError("every context/task/policy must contain 31 transitions")
    return data


def _parse_task(request_id: str) -> str:
    marker = "__rep"
    if marker not in request_id:
        raise ValueError(f"deadline request ID lacks repetition suffix: {request_id}")
    prefix, _ = request_id.rsplit(marker, 1)
    parts = prefix.split("__", 1)
    if len(parts) != 2 or not parts[0].startswith("deadline-lbv2-"):
        raise ValueError(f"unexpected deadline request ID: {request_id}")
    return parts[1]


def load_measured_windows(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(path)
    required = {
        "kind",
        "request_id",
        "layer_id",
        "producer_decode_step",
        "consumer_decode_step",
        "interval_ms",
        "context_label",
        "tp_rank_count",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"deadline table is missing columns: {missing}")
    data = data[data.kind == "previous_step_same_layer"].copy()
    data = data[data.producer_decode_step.between(0, 30)].copy()
    data["context_config"] = data.context_label.map(CONTEXT_LABEL_TO_TOKENS)
    if data.context_config.isna().any():
        raise ValueError("unexpected context label in deadline table")
    data["context_config"] = data.context_config.astype(int)
    data["task"] = data.request_id.map(_parse_task)
    if set(int(value) for value in data.layer_id.unique()) != set(EXPECTED_LAYERS):
        raise ValueError("deadline layer set does not match the seven-layer contract")
    if not (data.tp_rank_count == 2).all():
        raise ValueError("every deadline interval must contain both TP ranks")
    group = ["context_config", "task", "producer_decode_step"]
    layer_counts = data.groupby(group).layer_id.nunique()
    if not (layer_counts == len(EXPECTED_LAYERS)).all():
        raise ValueError("every request transition must contain all sampled layers")
    windows = (
        data.groupby(group, as_index=False)
        .agg(
            measured_window_ms=("interval_ms", "min"),
            measured_window_max_layer_ms=("interval_ms", "max"),
        )
        .rename(columns={"producer_decode_step": "step"})
    )
    if len(windows) != 2 * 6 * 31:
        raise ValueError(f"expected 372 measured request-step windows, got {len(windows)}")
    return data, windows


def join_and_compute(
    request_steps: pd.DataFrame,
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    previous = request_steps[request_steps.method == "previous-score"].copy()
    joined = previous.merge(
        windows,
        on=["context_config", "task", "step"],
        how="left",
        validate="many_to_one",
    )
    if joined.measured_window_ms.isna().any():
        raise ValueError("one or more policy transitions lack a measured deadline")

    baseline = request_steps[request_steps.method == "lru-demand"][
        ["context_config", "task", "step", "correction_ms"]
    ].rename(columns={"correction_ms": "lru_stall_ms"})
    joined = joined.merge(
        baseline,
        on=["context_config", "task", "step"],
        validate="many_to_one",
    )
    for deadline in FIXED_DEADLINES_MS:
        label = f"{deadline:g}ms"
        modeled_stall = joined.correction_ms + np.maximum(
            joined.prefetch_ms - deadline, 0.0
        )
        source_column = f"stall_{label}"
        if source_column in joined and not np.allclose(
            joined[source_column], modeled_stall
        ):
            raise ValueError(
                f"source {source_column} does not match the transfer contract"
            )
        joined[source_column] = modeled_stall
        joined[f"fully_hidden_{label}"] = joined.prefetch_ms <= deadline
    joined["stall_measured"] = joined.correction_ms + np.maximum(
        joined.prefetch_ms - joined.measured_window_ms, 0.0
    )
    joined["fully_hidden_measured"] = (
        joined.prefetch_ms <= joined.measured_window_ms
    )
    joined["unhidden_prefetch_measured_ms"] = (
        joined.stall_measured - joined.correction_ms
    )
    if len(joined) != len(windows) * len(ACTION_K):
        raise ValueError(f"unexpected joined row count: {len(joined)}")
    return joined, baseline


def summarize(
    joined: pd.DataFrame,
    baseline: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    baseline_by_group = (
        baseline.groupby(group_columns, as_index=False)
        .lru_stall_ms.mean()
        .rename(columns={"lru_stall_ms": "lru_stall_ms_mean"})
    )
    rows: list[dict[str, Any]] = []
    for key, part in joined.groupby(group_columns + ["candidate_k"], sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns + ["candidate_k"], key, strict=True))
        row.update(
            {
                "requests": int(part.task.nunique()),
                "request_steps": len(part),
                "prefetch_ms_mean": float(part.prefetch_ms.mean()),
                "prefetch_ms_p95": float(part.prefetch_ms.quantile(0.95)),
                "prefetch_ms_max": float(part.prefetch_ms.max()),
                "correction_floor_ms_mean": float(part.correction_ms.mean()),
                "measured_window_ms_min": float(part.measured_window_ms.min()),
                "measured_window_ms_p10": float(
                    part.measured_window_ms.quantile(0.10)
                ),
                "stall_measured_ms_mean": float(part.stall_measured.mean()),
                "fully_hidden_measured_fraction": float(
                    part.fully_hidden_measured.mean()
                ),
                "unhidden_prefetch_measured_ms_mean": float(
                    part.unhidden_prefetch_measured_ms.mean()
                ),
            }
        )
        for deadline in FIXED_DEADLINES_MS:
            label = f"{deadline:g}ms"
            row[f"stall_{label}_mean"] = float(part[f"stall_{label}"].mean())
            row[f"fully_hidden_{label}_fraction"] = float(
                part[f"fully_hidden_{label}"].mean()
            )
        rows.append(row)
    result = pd.DataFrame(rows).merge(
        baseline_by_group, on=group_columns, validate="many_to_one"
    )
    for deadline in FIXED_DEADLINES_MS:
        label = f"{deadline:g}ms"
        result[f"reduction_vs_lru_{label}"] = 1 - (
            result[f"stall_{label}_mean"] / result.lru_stall_ms_mean
        )
    result["reduction_vs_lru_measured"] = 1 - (
        result.stall_measured_ms_mean / result.lru_stall_ms_mean
    )
    return result.sort_values(group_columns + ["candidate_k"])


def select_best(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    settings = [*(f"{value:g}ms" for value in FIXED_DEADLINES_MS), "measured"]
    for context, part in summary.groupby("context_config", sort=True):
        for setting in settings:
            metric = f"stall_{setting}_mean" if setting != "measured" else "stall_measured_ms_mean"
            best = part.loc[part[metric].idxmin()]
            reduction = (
                f"reduction_vs_lru_{setting}"
                if setting != "measured"
                else "reduction_vs_lru_measured"
            )
            rows.append(
                {
                    "context_config": int(context),
                    "deadline_setting": setting,
                    "best_candidate_k": int(best.candidate_k),
                    "modeled_stall_ms_mean": float(best[metric]),
                    "reduction_vs_lru": float(best[reduction]),
                    "lru_stall_ms_mean": float(best.lru_stall_ms_mean),
                    "correction_floor_ms_mean": float(
                        best.correction_floor_ms_mean
                    ),
                    "prefetch_ms_max": float(best.prefetch_ms_max),
                    "measured_window_ms_min": float(
                        best.measured_window_ms_min
                    ),
                    "fully_hidden_fraction": float(
                        best[
                            f"fully_hidden_{setting}_fraction"
                            if setting != "measured"
                            else "fully_hidden_measured_fraction"
                        ]
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
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_summary(joined: pd.DataFrame, best: pd.DataFrame, output_dir: Path) -> None:
    configure_style()
    colors = {65536: "#0072B2", 131072: "#D55E00"}
    labels = {65536: "64K", 131072: "128K"}
    fig, axes = plt.subplots(
        1, 2, figsize=(6.8, 2.8), constrained_layout=True
    )

    axis = axes[0]
    for context, part in joined[joined.candidate_k == 4096].groupby(
        "context_config", sort=True
    ):
        values = np.sort(part.prefetch_ms.to_numpy(dtype=np.float64))
        cdf = np.arange(1, len(values) + 1) / len(values)
        axis.step(
            values,
            cdf,
            where="post",
            color=colors[int(context)],
            label=f"{labels[int(context)]} (max {values.max():.1f} ms)",
        )
    axis.axvline(20, color="#4D4D4D", linestyle="--", linewidth=1.2, label="20 ms")
    axis.set_xlim(0, 40)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Modeled K=4096 prefetch time (ms)")
    axis.set_ylabel("Empirical CDF")
    axis.set_title("(a) Transfer demand within one decode step")
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.legend(loc="lower right")

    axis = axes[1]
    settings = ["20ms", "50ms", "100ms", "measured"]
    x = np.arange(len(settings))
    width = 0.36
    for offset, context in zip((-0.5, 0.5), (65536, 131072), strict=True):
        part = best[best.context_config == context].set_index("deadline_setting")
        values = [part.loc[setting, "reduction_vs_lru"] * 100 for setting in settings]
        axis.bar(
            x + offset * width,
            values,
            width,
            color=colors[context],
            edgecolor="white",
            linewidth=0.7,
            hatch="" if context == 65536 else "//",
            label=labels[context],
        )
    axis.set_xticks(x, ["20", "50", "100", "Measured"])
    axis.set_ylim(0, 55)
    axis.set_xlabel("Available overlap window (ms)")
    axis.set_ylabel("Modeled stall reduction vs. LRU (%)\nhigher is better")
    axis.set_title("(b) Benefit saturates at the correction floor")
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.legend(loc="lower right")

    for suffix, kwargs in (
        ("pdf", {"metadata": {"CreationDate": None, "ModDate": None}}),
        ("png", {"dpi": 300}),
    ):
        fig.savefig(
            output_dir / f"measured_deadline_saturation.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def self_test() -> None:
    request_rows = []
    deadline_rows = []
    for context_label, context in CONTEXT_LABEL_TO_TOKENS.items():
        task = f"task-{context_label.lower()}"
        for step in range(31):
            request_rows.append(
                {
                    "context_config": context,
                    "task": task,
                    "method": "lru-demand",
                    "candidate_k": 0,
                    "step": step,
                    "prefetch_ms": 0.0,
                    "correction_ms": 10.0,
                }
            )
            for candidate_k in ACTION_K:
                request_rows.append(
                    {
                        "context_config": context,
                        "task": task,
                        "method": "previous-score",
                        "candidate_k": candidate_k,
                        "step": step,
                        "prefetch_ms": candidate_k / 1024,
                        "correction_ms": 5.0,
                    }
                )
            for layer in EXPECTED_LAYERS:
                deadline_rows.append(
                    {
                        "context_config": context,
                        "task": task,
                        "step": step,
                        "layer_id": layer,
                        "interval_ms": 100.0 + layer,
                    }
                )
    request_steps = pd.DataFrame(request_rows)
    windows = (
        pd.DataFrame(deadline_rows)
        .groupby(["context_config", "task", "step"], as_index=False)
        .agg(
            measured_window_ms=("interval_ms", "min"),
            measured_window_max_layer_ms=("interval_ms", "max"),
        )
    )
    joined, baseline = join_and_compute(request_steps, windows)
    summary = summarize(joined, baseline, ["context_config"])
    best = select_best(summary)
    assert joined.fully_hidden_measured.all()
    assert np.allclose(joined.stall_measured, joined.correction_ms)
    assert len(best) == 8
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        plot_summary(joined, best, output)
        assert (output / "measured_deadline_saturation.pdf").is_file()
        assert (output / "measured_deadline_saturation.png").is_file()
    print(json.dumps({"passed": True, "joined_rows": len(joined)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-step-tables", nargs="+", type=Path)
    parser.add_argument("--deadline-intervals", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.request_step_tables or args.deadline_intervals is None or args.output_dir is None:
        raise SystemExit(
            "--request-step-tables, --deadline-intervals, and --output-dir are required"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_steps = load_request_steps(args.request_step_tables)
    raw_deadlines, windows = load_measured_windows(args.deadline_intervals)
    joined, baseline = join_and_compute(request_steps, windows)
    overall = summarize(joined, baseline, ["context_config"])
    by_task = summarize(joined, baseline, ["context_config", "task"])
    best = select_best(overall)

    joined.to_parquet(args.output_dir / "joined_request_step.parquet", index=False)
    windows.to_csv(args.output_dir / "measured_window_by_request_step.csv", index=False)
    overall.to_csv(args.output_dir / "summary_overall.csv", index=False)
    by_task.to_csv(args.output_dir / "summary_by_task.csv", index=False)
    best.to_csv(args.output_dir / "best_k_by_deadline.csv", index=False)
    plot_summary(joined, best, args.output_dir)

    k4096 = overall[overall.candidate_k == 4096].copy()
    validation = {
        "schema_version": 1,
        "analysis_kind": "measured deadline joined with modeled transfer; not measured speedup",
        "contexts": sorted(int(value) for value in joined.context_config.unique()),
        "tasks": int(joined.task.nunique()),
        "action_k": list(ACTION_K),
        "fixed_deadlines_ms": list(FIXED_DEADLINES_MS),
        "sampled_layers": list(EXPECTED_LAYERS),
        "deadline_contract": (
            "minimum across TP ranks, then minimum across seven sampled layers "
            "for the same request transition"
        ),
        "source_20ms_formula_checked": True,
        "joined_policy_rows": len(joined),
        "measured_request_step_windows": len(windows),
        "raw_deadline_rows_used": len(raw_deadlines),
        "all_k4096_prefetch_fits_measured_window": bool(
            (k4096.fully_hidden_measured_fraction == 1.0).all()
        ),
        "all_measured_stall_at_correction_floor": bool(
            np.allclose(joined.stall_measured, joined.correction_ms)
        ),
        "speedup_measured": False,
        "limitations": [
            "prefetch and correction times come from the frozen bandwidth/latency model",
            "deadline events were collected in eager exact-topk TP=2 batch=1 serving",
            "seven sampled layers are scaled to 48 in the source transfer model",
            "no DMA, cache manager, contention, throughput, or end-to-end speedup is measured",
        ],
    }
    input_paths = [*args.request_step_tables, args.deadline_intervals]
    artifacts = [
        args.output_dir / "joined_request_step.parquet",
        args.output_dir / "measured_window_by_request_step.csv",
        args.output_dir / "summary_overall.csv",
        args.output_dir / "summary_by_task.csv",
        args.output_dir / "best_k_by_deadline.csv",
        args.output_dir / "measured_deadline_saturation.pdf",
        args.output_dir / "measured_deadline_saturation.png",
    ]
    validation["inputs"] = {
        str(path.resolve()): sha256_file(path) for path in input_paths
    }
    validation["artifacts"] = {
        path.name: sha256_file(path) for path in artifacts
    }
    write_json(args.output_dir / "validation.json", validation)
    write_json(
        args.output_dir / "reproducibility.json",
        {
            "script": str(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "parameters": {
                "action_k": list(ACTION_K),
                "fixed_deadlines_ms": list(FIXED_DEADLINES_MS),
                "sampled_layers": list(EXPECTED_LAYERS),
            },
            "inputs": validation["inputs"],
            "outputs": validation["artifacts"],
        },
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
