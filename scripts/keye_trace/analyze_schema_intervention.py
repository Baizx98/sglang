#!/usr/bin/env python3
"""Analyze the controlled BFCL tool-schema budget and position intervention."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from cycler import cycler

LAYERS = list(range(48))
STEPS = 16
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_SAMPLES = 2000
BUDGET_ORDER = ["2p5k", "3p5k", "full"]
BUDGET_LABELS = {"2p5k": "2.5k", "3p5k": "3.5k", "full": "Full"}
COLORS = {"front": "#0072B2", "tail": "#D55E00"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    sampled = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(
    frame: pd.DataFrame, groups: list[str], metric: str
) -> pd.DataFrame:
    unit = (
        frame.groupby(["source_rid", *groups], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for keys, values in unit.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        array = values[metric].to_numpy(dtype=float)
        low, high = bootstrap_ci(array, rng)
        rows.append(
            {
                **dict(zip(groups, keys)),
                "metric": metric,
                "source_request_count": int(values.source_rid.nunique()),
                "mean": float(np.nanmean(array)),
                "ci95_low": low,
                "ci95_high": high,
                "p10": float(np.nanquantile(array, 0.10)),
                "p50": float(np.nanquantile(array, 0.50)),
                "p90": float(np.nanquantile(array, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def build_manifest(run_dir: Path, requests: list[dict[str, Any]]) -> dict[tuple[str, int], Path]:
    requested = {row["rid"] for row in requests}
    lookup: dict[tuple[str, int], Path] = {}
    for row in read_jsonl(run_dir / "events" / "manifest.jsonl"):
        rid = row.get("request_id")
        if rid not in requested:
            continue
        key = (rid, int(row["layer_id"]))
        if key in lookup:
            raise ValueError(f"duplicate trace chunk {key}")
        lookup[key] = run_dir / "events" / row["file"]
    expected = len(requests) * len(LAYERS)
    if len(lookup) != expected:
        raise ValueError(f"expected {expected} chunks, found {len(lookup)}")
    return lookup


def output_prefix_lengths(run_dir: Path, requests: list[dict[str, Any]]) -> dict[str, int]:
    executed = {row["rid"]: row for row in read_jsonl(run_dir / "requests.jsonl")}
    prefixes: dict[str, int] = {}
    by_source: dict[str, list[list[int]]] = {}
    for request in requests:
        response = executed[request["rid"]]["response"]
        ids = list(response["output_ids"][:STEPS])
        if len(ids) != STEPS:
            raise ValueError(f"{request['rid']}: expected {STEPS} generated tokens")
        by_source.setdefault(request["source_rid"], []).append(ids)
    for source_rid, variants in by_source.items():
        common = 0
        for step in range(STEPS):
            if len({tuple(row[: step + 1]) for row in variants}) != 1:
                break
            common += 1
        prefixes[source_rid] = common
    return prefixes


def analyze(
    run_dir: Path,
    requests: list[dict[str, Any]],
    segments: dict[str, list[dict[str, Any]]],
    lookup: dict[tuple[str, int], Path],
    common_prefix: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    partition_checks = 0
    threshold_checks = 0
    total = len(requests) * len(LAYERS)
    processed = 0
    for request in requests:
        rid = request["rid"]
        prompt_len = int(request["prompt_len"])
        target_segment_id = f"tool_schema::{request['target_tool']}"
        target_ranges = [
            row for row in segments[rid] if row["segment_id"] == target_segment_id
        ]
        if len(target_ranges) != 1:
            raise ValueError(f"{rid}: expected one target schema range")
        target = target_ranges[0]
        target_start, target_end = int(target["token_start"]), int(target["token_end"])
        irrelevant = [
            row
            for row in segments[rid]
            if row["segment_type"] == "tool_schema"
            and row["segment_id"] != target_segment_id
        ]
        irrelevant_mask = np.zeros(prompt_len, dtype=bool)
        for segment in irrelevant:
            irrelevant_mask[int(segment["token_start"]):int(segment["token_end"])] = True
        irrelevant_tokens = int(irrelevant_mask.sum())
        for layer in LAYERS:
            processed += 1
            record = torch.load(lookup[(rid, layer)], weights_only=True)
            if record["schema_version"] != 4:
                raise ValueError(f"{rid}, L{layer}: expected schema v4")
            if record["decode_step_ids"] != list(range(STEPS)):
                raise ValueError(f"{rid}, L{layer}: non-continuous decode steps")
            indices = record["indices"].numpy(force=True).astype(np.int32)
            score_lens = record["score_valid_counts"].numpy(force=True).astype(int)
            valid_counts = record["valid_counts"].numpy(force=True).astype(int)
            for step in range(STEPS):
                selected = indices[step]
                history = selected[(selected >= 0) & (selected < prompt_len)]
                decode = selected[
                    (selected >= prompt_len) & (selected < score_lens[step])
                ]
                if len(history) + len(decode) != valid_counts[step]:
                    raise ValueError(f"{rid}, L{layer}, S{step}: top-k partition")
                partition_checks += 1
                if valid_counts[step] != min(2048, score_lens[step]):
                    raise ValueError(f"{rid}, L{layer}, S{step}: invalid top-k count")
                threshold_checks += 1
                history_k = len(history)
                baseline = history_k / prompt_len
                target_hits = int(
                    np.count_nonzero((history >= target_start) & (history < target_end))
                )
                irrelevant_hits = int(np.count_nonzero(irrelevant_mask[history]))
                target_tokens = target_end - target_start
                target_coverage = target_hits / target_tokens
                irrelevant_coverage = irrelevant_hits / irrelevant_tokens
                rows.append(
                    {
                        "rid": rid,
                        "source_rid": request["source_rid"],
                        "category": request["category"],
                        "source_round": int(request["round_id"]),
                        "layer": layer,
                        "step": step,
                        "prefix_comparable": step <= common_prefix[request["source_rid"]],
                        "budget_label": request["budget_label"],
                        "desired_schema_budget": request["desired_schema_budget"],
                        "actual_schema_tokens": int(request["actual_schema_tokens"]),
                        "target_position": request["target_position"],
                        "target_tool": request["target_tool"],
                        "prompt_len": prompt_len,
                        "history_k": history_k,
                        "target_schema_tokens": target_tokens,
                        "target_selected_tokens": target_hits,
                        "target_coverage": target_coverage,
                        "target_lift": target_coverage / baseline,
                        "irrelevant_schema_tokens": irrelevant_tokens,
                        "irrelevant_selected_tokens": irrelevant_hits,
                        "irrelevant_coverage": irrelevant_coverage,
                        "irrelevant_lift": irrelevant_coverage / baseline,
                        "target_excess_coverage": target_coverage - irrelevant_coverage,
                    }
                )
            if processed % 96 == 0 or processed == total:
                print(f"[{processed:04d}/{total}] {rid}, L{layer}", flush=True)
    validation = {
        "request_layer_chunks": processed,
        "expected_request_layer_chunks": total,
        "topk_partition_checks": partition_checks,
        "expected_topk_partition_checks": total * STEPS,
        "topk_count_checks": threshold_checks,
        "expected_topk_count_checks": total * STEPS,
    }
    return pd.DataFrame(rows), validation


def paired_effects(step0: pd.DataFrame) -> pd.DataFrame:
    key = ["source_rid", "category", "layer"]
    pivot = step0.pivot_table(
        index=key,
        columns=["budget_label", "target_position"],
        values="target_coverage",
        aggfunc="first",
    )
    rows: list[dict[str, Any]] = []
    for index, values in pivot.iterrows():
        source_rid, category, layer = index
        for position in ["front", "tail"]:
            rows.append(
                {
                    "source_rid": source_rid,
                    "category": category,
                    "layer": layer,
                    "effect": "full_minus_2p5k",
                    "condition": position,
                    "value": values[("full", position)] - values[("2p5k", position)],
                }
            )
        for budget in BUDGET_ORDER:
            rows.append(
                {
                    "source_rid": source_rid,
                    "category": category,
                    "layer": layer,
                    "effect": "tail_minus_front",
                    "condition": budget,
                    "value": values[(budget, "tail")] - values[(budget, "front")],
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: matplotlib.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_results(
    step0_summary: pd.DataFrame,
    effect_summary: pd.DataFrame,
    prefix_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    for position in ["front", "tail"]:
        for budget, linestyle in zip(BUDGET_ORDER, [":", "--", "-"]):
            values = step0_summary[
                (step0_summary.target_position == position)
                & (step0_summary.budget_label == budget)
            ].sort_values("layer")
            axes[0].plot(
                values.layer,
                values["mean"],
                color=COLORS[position],
                linestyle=linestyle,
                label=f"{position}, {BUDGET_LABELS[budget]}",
            )
    axes[0].set(xlabel="DSA layer", ylabel="Target-schema coverage at step 0")
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2)
    axes[0].grid(alpha=0.25)
    overall = step0_summary.groupby(["budget_label", "target_position"], as_index=False)["mean"].mean()
    for offset, position in zip([-0.09, 0.09], ["front", "tail"]):
        values = overall[overall.target_position == position].set_index("budget_label").reindex(BUDGET_ORDER)
        axes[1].plot(
            np.arange(3) + offset,
            values["mean"],
            marker="o",
            color=COLORS[position],
            label=position,
        )
    axes[1].set_xticks(range(3), [BUDGET_LABELS[x] for x in BUDGET_ORDER])
    axes[1].set(xlabel="Tool-schema token budget", ylabel="Mean target coverage across layers")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    save_figure(fig, figure_dir / "target_schema_budget_position")

    fig, ax = plt.subplots(figsize=(6.8, 3.5), layout="constrained")
    effect_order = [
        ("full_minus_2p5k", "front"),
        ("full_minus_2p5k", "tail"),
        ("tail_minus_front", "2p5k"),
        ("tail_minus_front", "3p5k"),
        ("tail_minus_front", "full"),
    ]
    labels = ["Full-low\n(front)", "Full-low\n(tail)", "Tail-front\n(2.5k)", "Tail-front\n(3.5k)", "Tail-front\n(full)"]
    for index, (effect, condition) in enumerate(effect_order):
        values = effect_summary[
            (effect_summary.effect == effect) & (effect_summary.condition == condition)
        ].sort_values("layer")
        ax.plot(values.layer, values["mean"], label=labels[index].replace("\n", " "))
    ax.axhline(0, color="#4D4D4D", linestyle="--")
    ax.set(xlabel="DSA layer", ylabel="Paired target-coverage difference")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "paired_intervention_effect_by_layer")

    fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
    counts = prefix_summary.common_prefix_tokens.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#0072B2")
    ax.set(xlabel="Common generated-prefix length across 6 variants", ylabel="Source requests")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figure_dir / "generated_prefix_consistency")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "analysis" / "schema-intervention-v01"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stderr(io.StringIO()):
        plt.style.use(args.style)
    plt.rcParams["axes.prop_cycle"] = cycler(
        color=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    )
    requests = read_jsonl(args.run_dir / "prepared_requests.jsonl")
    if len(requests) != 72:
        raise ValueError(f"expected 72 requests, found {len(requests)}")
    segments: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(args.run_dir / "segments.jsonl"):
        segments.setdefault(row["rid"], []).append(row)
    for request in requests:
        cursor = 0
        for segment in sorted(segments[request["rid"]], key=lambda row: row["token_start"]):
            if int(segment["token_start"]) != cursor:
                raise ValueError(f"{request['rid']}: segment gap or overlap")
            cursor = int(segment["token_end"])
        if cursor != int(request["prompt_len"]):
            raise ValueError(f"{request['rid']}: incomplete segment coverage")
    common_prefix = output_prefix_lengths(args.run_dir, requests)
    lookup = build_manifest(args.run_dir, requests)
    metrics, validation = analyze(args.run_dir, requests, segments, lookup, common_prefix)
    step0 = metrics[metrics.step == 0].copy()
    effects = paired_effects(step0)
    step0_summary = summarize(
        step0, ["layer", "budget_label", "target_position"], "target_coverage"
    )
    excess_summary = summarize(
        step0,
        ["layer", "budget_label", "target_position"],
        "target_excess_coverage",
    )
    effect_summary = summarize(effects, ["layer", "effect", "condition"], "value")
    prefix_summary = pd.DataFrame(
        [
            {
                "source_rid": source_rid,
                "category": next(
                    row["category"] for row in requests if row["source_rid"] == source_rid
                ),
                "common_prefix_tokens": count,
            }
            for source_rid, count in common_prefix.items()
        ]
    )
    tables = {
        "schema_intervention_step_metrics": metrics,
        "paired_intervention_effects": effects,
        "summary_step0_target_coverage": step0_summary,
        "summary_step0_target_excess": excess_summary,
        "summary_paired_effects": effect_summary,
        "generated_prefix_consistency": prefix_summary,
    }
    for name, frame in tables.items():
        frame.to_parquet(table_dir / f"{name}.parquet", index=False)
    plot_results(step0_summary, effect_summary, prefix_summary, figure_dir)
    validation.update(
        {
            "requests": len(requests),
            "source_requests": len(common_prefix),
            "complete_prompt_segment_coverage": True,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    checks = {
        "all_72_requests": len(requests) == 72,
        "all_12_sources": len(common_prefix) == 12,
        "all_request_layer_chunks": validation["request_layer_chunks"] == 72 * 48,
        "all_topk_partitions": validation["topk_partition_checks"] == 72 * 48 * STEPS,
        "all_topk_counts": validation["topk_count_checks"] == 72 * 48 * STEPS,
        "all_effect_pairs": len(effects) == 12 * 48 * 5,
    }
    validation["checks"] = checks
    if not all(checks.values()):
        raise ValueError(f"validation failed: {validation}")
    overall_effects = effects.groupby(["source_rid", "effect", "condition"]).value.mean().reset_index()
    overall_summary = summarize(overall_effects, ["effect", "condition"], "value")
    overall_summary.to_parquet(table_dir / "summary_paired_effects_overall.parquet", index=False)
    summary = {
        "requests": 72,
        "source_requests": 12,
        "layers": 48,
        "steps": 16,
        "common_prefix_tokens_by_source": common_prefix,
        "primary_step": 0,
        "overall_paired_effects": overall_summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    reproducibility = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_commit": git_revision(Path.cwd()),
        "run_dir": str(args.run_dir.resolve()),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "cluster": "source_rid",
        },
        "primary_analysis": "decode step 0; source-request-paired",
        "later_step_rule": "report only while all six variants share the generated prefix",
        "versions": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "figure_formats": ["PDF", "PNG 300 dpi"],
    }
    (output_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
