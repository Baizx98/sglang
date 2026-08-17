#!/usr/bin/env python3
"""Test whether token-position shifts add useful next-step KV page predictions.

This analysis is intentionally stricter than a raw ``S_t + delta`` overlap:

* it evaluates only tokens newly entering the next-step top-k;
* it normalizes token precision by the random opportunity among non-selected
  history tokens;
* it compares shift candidates with previous-step score rank at equal token or
  page budget; and
* it reports the incremental page-miss coverage above simply retaining the
  previous-step active pages.

Compact trace v5 is required because its exact top-4096 prefix provides the
score-ranked baseline without retaining every history score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

TARGET_K = 2048
DELTAS = [delta for delta in range(-8, 9) if delta]
PAGE_SIZES = [4, 16, 64, 256]
SHIFT_METHODS = {
    "shift+1": (1,),
    "shift+-1": (-1, 1),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def shifted_frontier(
    previous: np.ndarray, previous_mask: np.ndarray, deltas: Iterable[int]
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for delta in deltas:
        shifted = previous + delta
        shifted = shifted[(shifted >= 0) & (shifted < len(previous_mask))]
        shifted = shifted[~previous_mask[shifted]]
        if len(shifted):
            parts.append(shifted)
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)


def exact_prefix_hits(
    ranked: np.ndarray, target_mask: np.ndarray, width: int
) -> int:
    prefix = np.unique(ranked[:width])
    return int(np.count_nonzero(target_mask[prefix]))


def select_score_pages(
    ranked: np.ndarray,
    previous_pages: set[int],
    page_size: int,
    new_page_budget: int,
) -> set[int]:
    selected: set[int] = set()
    if new_page_budget <= 0:
        return selected
    for token in ranked:
        page = int(token) // page_size
        if page in previous_pages or page in selected:
            continue
        selected.add(page)
        if len(selected) == new_page_budget:
            break
    return selected


def page_metrics(
    target_tokens: np.ndarray,
    previous_pages: set[int],
    prefetched_pages: set[int],
    page_size: int,
) -> dict[str, float | int]:
    target_pages = {int(token) // page_size for token in target_tokens}
    missing_pages = target_pages - previous_pages
    useful_prefetch = prefetched_pages & missing_pages
    resident = previous_pages | prefetched_pages
    covered_tokens = sum(
        (int(token) // page_size) in resident for token in target_tokens
    )
    return {
        "target_pages": len(target_pages),
        "base_missing_pages": len(missing_pages),
        "prefetched_pages": len(prefetched_pages),
        "useful_prefetched_pages": len(useful_prefetch),
        "covered_target_pages": len(target_pages & resident),
        "covered_topk_tokens": covered_tokens,
        "page_recall": ratio(len(target_pages & resident), len(target_pages)),
        "miss_page_coverage": ratio(len(useful_prefetch), len(missing_pages)),
        "prefetch_precision": ratio(len(useful_prefetch), len(prefetched_pages)),
        "topk_token_recall_from_pages": ratio(covered_tokens, len(target_tokens)),
    }


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[int, Path]]:
    requests = read_jsonl(run_dir / "prepared_requests.jsonl")
    if len(requests) != 1:
        raise ValueError(f"{run_dir}: pilot analyzer expects exactly one request")
    request = requests[0]
    lookup: dict[int, Path] = {}
    for row in read_jsonl(run_dir / "events" / "manifest.jsonl"):
        if row["request_id"] != request["rid"]:
            continue
        layer = int(row["layer_id"])
        if layer in lookup:
            raise ValueError(f"{run_dir}: duplicate layer {layer}")
        lookup[layer] = run_dir / "events" / row["file"]
    if not lookup:
        raise ValueError(f"{run_dir}: no trace chunks")
    return request, lookup


def load_chunk(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    record = torch.load(path, weights_only=False)
    if int(record["schema_version"]) != 5:
        raise ValueError(f"{path}: expected compact trace schema v5")
    if record.get("topk_backend") != "torch_exact":
        raise ValueError(f"{path}: exact top-k trace required")
    if record["decode_step_ids"] != list(range(32)):
        raise ValueError(f"{path}: expected decode steps 0..31")
    indices = record["indices"].numpy(force=True).astype(np.int64)
    ranked = record["candidate_indices"].numpy(force=True).astype(np.int64)
    valid_counts = record["score_valid_counts"].numpy(force=True).astype(np.int64)
    full_scores = None
    if bool(record.get("full_scores_retained", False)):
        full_scores = record["scores"].numpy(force=True).astype(np.float32)
    if indices.shape != (32, TARGET_K) or ranked.shape[1] < 2 * TARGET_K:
        raise ValueError(f"{path}: unexpected top-k shapes")
    for step in range(32):
        if set(indices[step]) != set(ranked[step, :TARGET_K]):
            raise ValueError(f"{path}: exact prefix mismatch at step {step}")
    return indices, ranked, valid_counts, full_scores


def analyze_run(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    request, lookup = load_run(run_dir)
    context_config = int(request["length_config"])
    prompt_tokens = int(request["prompt_len"])
    token_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []

    for layer, chunk_path in sorted(lookup.items()):
        indices, ranked, valid_counts, full_scores = load_chunk(chunk_path)
        for step in range(31):
            common = min(int(valid_counts[step]), int(valid_counts[step + 1]))
            previous = np.unique(indices[step][indices[step] < common])
            current = np.unique(indices[step + 1][indices[step + 1] < common])
            previous_ranked = ranked[step]
            previous_ranked = previous_ranked[
                (previous_ranked >= 0) & (previous_ranked < common)
            ]
            full_ranked = None
            if full_scores is not None:
                scores = full_scores[step, :common]
                if not np.isfinite(scores).all():
                    raise ValueError(f"{chunk_path}: non-finite full scores")
                full_ranked = np.argsort(-scores, kind="stable")
            if len(previous) != TARGET_K or len(current) not in {
                TARGET_K - 1,
                TARGET_K,
            }:
                raise ValueError(
                    f"{chunk_path}: invalid width at step {step}: "
                    f"{len(previous)}, {len(current)}"
                )

            previous_mask = np.zeros(common, dtype=bool)
            previous_mask[previous] = True
            current_mask = np.zeros(common, dtype=bool)
            current_mask[current] = True
            entries = current[~previous_mask[current]]
            overlap = len(current) - len(entries)
            nonselected = common - len(previous)
            prevalence = ratio(len(entries), nonselected)

            base = {
                "run_dir": str(run_dir),
                "request_id": request["rid"],
                "task": request.get("task", "unknown"),
                "context_config": context_config,
                "prompt_tokens": prompt_tokens,
                "layer": layer,
                "step": step,
                "common_tokens": common,
                "new_entries": len(entries),
                "nonselected_tokens": nonselected,
                "current_tokens": len(current),
                "previous_overlap": overlap,
                "base_token_recall": ratio(overlap, len(current)),
            }

            for delta in DELTAS:
                candidates = shifted_frontier(previous, previous_mask, (delta,))
                hits = int(np.count_nonzero(current_mask[candidates]))
                precision = ratio(hits, len(candidates))
                union_width = len(previous) + len(candidates)
                score_rank_hits = exact_prefix_hits(
                    previous_ranked, current_mask, union_width
                )
                token_rows.append(
                    {
                        **base,
                        "delta": delta,
                        "shift_candidates": len(candidates),
                        "shift_hits": hits,
                        "score_rank_hits": score_rank_hits,
                        "entry_recall": ratio(hits, len(entries)),
                        "candidate_precision": precision,
                        "entry_prevalence": prevalence,
                        "precision_lift": ratio(precision, prevalence),
                        "union_width": union_width,
                        "shift_union_recall": ratio(overlap + hits, len(current)),
                        "score_rank_recall_equal_token_budget": ratio(
                            score_rank_hits, len(current)
                        ),
                    }
                )

            # Equal-I/O page comparison requires a full score ranking. Compact
            # top-4096 alone may not contain enough distinct pages to match the
            # shift predictor's page budget, especially for 4-token pages.
            if full_ranked is None:
                continue
            for page_size in PAGE_SIZES:
                previous_pages = {
                    int(token) // page_size for token in previous
                }
                base_metrics = page_metrics(
                    current, previous_pages, set(), page_size
                )
                page_rows.append(
                    {
                        **base,
                        "page_size": page_size,
                        "method": "previous-only",
                        "new_page_budget": 0,
                        **base_metrics,
                    }
                )
                for method, deltas in SHIFT_METHODS.items():
                    candidates = shifted_frontier(previous, previous_mask, deltas)
                    candidate_pages = {
                        int(token) // page_size for token in candidates
                    }
                    shifted_pages = candidate_pages - previous_pages
                    shift_metrics = page_metrics(
                        current, previous_pages, shifted_pages, page_size
                    )
                    page_rows.append(
                        {
                            **base,
                            "page_size": page_size,
                            "method": method,
                            "new_page_budget": len(shifted_pages),
                            **shift_metrics,
                        }
                    )

                    score_pages = select_score_pages(
                        full_ranked,
                        previous_pages,
                        page_size,
                        len(shifted_pages),
                    )
                    if len(score_pages) != len(shifted_pages):
                        raise ValueError(
                            f"{chunk_path}: top-4096 cannot fill page budget "
                            f"{len(shifted_pages)} at step {step}, p={page_size}"
                        )
                    score_metrics = page_metrics(
                        current, previous_pages, score_pages, page_size
                    )
                    page_rows.append(
                        {
                            **base,
                            "page_size": page_size,
                            "method": f"score-rank@{method}-budget",
                            "new_page_budget": len(score_pages),
                            **score_metrics,
                        }
                    )

                    missing_pages = base_metrics["base_missing_pages"]
                    oracle_pages = min(len(shifted_pages), int(missing_pages))
                    page_rows.append(
                        {
                            **base,
                            "page_size": page_size,
                            "method": f"oracle@{method}-budget",
                            "new_page_budget": len(shifted_pages),
                            "target_pages": base_metrics["target_pages"],
                            "base_missing_pages": missing_pages,
                            "prefetched_pages": len(shifted_pages),
                            "useful_prefetched_pages": oracle_pages,
                            "covered_target_pages": (
                                int(base_metrics["target_pages"])
                                - int(missing_pages)
                                + oracle_pages
                            ),
                            "covered_topk_tokens": float("nan"),
                            "page_recall": ratio(
                                int(base_metrics["target_pages"])
                                - int(missing_pages)
                                + oracle_pages,
                                int(base_metrics["target_pages"]),
                            ),
                            "miss_page_coverage": ratio(oracle_pages, missing_pages),
                            "prefetch_precision": ratio(
                                oracle_pages, len(shifted_pages)
                            ),
                            # A page oracle does not specify which top-k tokens
                            # share each chosen page, so token recall is undefined.
                            "topk_token_recall_from_pages": float("nan"),
                        }
                    )

    validation = {
        "run_dir": str(run_dir),
        "request_id": request["rid"],
        "task": request.get("task", "unknown"),
        "context_config": context_config,
        "prompt_tokens": prompt_tokens,
        "layers": sorted(lookup),
        "page_layers": sorted(
            {
                int(row["layer"])
                for row in page_rows
            }
        ),
        "chunks": len(lookup),
        "transitions": len(lookup) * 31,
        "token_rows": len(token_rows),
        "page_rows": len(page_rows),
    }
    expected_token_rows = validation["transitions"] * len(DELTAS)
    expected_page_rows = (
        len(validation["page_layers"])
        * 31
        * len(PAGE_SIZES)
        * (1 + 3 * len(SHIFT_METHODS))
    )
    if len(token_rows) != expected_token_rows:
        raise AssertionError("token row count mismatch")
    if len(page_rows) != expected_page_rows:
        raise AssertionError("page row count mismatch")
    return token_rows, page_rows, validation


def aggregate_token(rows: pd.DataFrame) -> pd.DataFrame:
    groups = ["context_config", "prompt_tokens", "delta"]
    count_columns = [
        "new_entries",
        "nonselected_tokens",
        "current_tokens",
        "previous_overlap",
        "shift_candidates",
        "shift_hits",
        "score_rank_hits",
    ]
    counts = rows.groupby(groups, as_index=False)[count_columns].sum()
    widths = rows.groupby(groups, as_index=False).union_width.mean()
    result = counts.merge(widths, on=groups, validate="1:1")
    result["entry_recall"] = result.shift_hits / result.new_entries
    result["candidate_precision"] = result.shift_hits / result.shift_candidates
    result["entry_prevalence"] = result.new_entries / result.nonselected_tokens
    result["precision_lift"] = (
        result.candidate_precision / result.entry_prevalence
    )
    result["base_token_recall"] = (
        result.previous_overlap / result.current_tokens
    )
    result["shift_union_recall"] = (
        result.previous_overlap + result.shift_hits
    ) / result.current_tokens
    result["score_rank_recall_equal_token_budget"] = (
        result.score_rank_hits / result.current_tokens
    )
    return result.sort_values(["context_config", "delta"])


def aggregate_page(rows: pd.DataFrame) -> pd.DataFrame:
    groups = ["context_config", "prompt_tokens", "page_size", "method"]
    count_columns = [
        "target_pages",
        "base_missing_pages",
        "prefetched_pages",
        "useful_prefetched_pages",
        "covered_target_pages",
        "current_tokens",
    ]
    counts = rows.groupby(groups, as_index=False, dropna=False)[count_columns].sum()
    budgets = rows.groupby(groups, as_index=False, dropna=False).new_page_budget.mean()
    covered_tokens = (
        rows.groupby(groups, as_index=False, dropna=False)
        .covered_topk_tokens.sum(min_count=1)
    )
    result = counts.merge(budgets, on=groups, validate="1:1").merge(
        covered_tokens, on=groups, validate="1:1"
    )
    result["page_recall"] = result.covered_target_pages / result.target_pages
    result["miss_page_coverage"] = np.where(
        result.base_missing_pages > 0,
        result.useful_prefetched_pages / result.base_missing_pages,
        np.nan,
    )
    result["prefetch_precision"] = np.where(
        result.prefetched_pages > 0,
        result.useful_prefetched_pages / result.prefetched_pages,
        np.nan,
    )
    result["topk_token_recall_from_pages"] = (
        result.covered_topk_tokens / result.current_tokens
    )
    return result.sort_values(["context_config", "page_size", "method"])


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_token_summary(summary: pd.DataFrame, output: Path) -> None:
    focus = summary[summary.delta == 1].sort_values("context_config")
    x = focus.context_config / 1024
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)
    axes[0].plot(x, focus.precision_lift, color="#0072B2", marker="o")
    axes[0].axhline(1.0, color="#9E9E9E", linestyle="--", linewidth=1.2)
    axes[0].set(
        xlabel="Context configuration (K tokens)",
        ylabel="New-entry precision lift vs. chance (x)",
        xticks=x,
    )
    axes[1].plot(
        x,
        focus.shift_union_recall * 100,
        color="#D55E00",
        marker="o",
        label="Previous top-k + shift+1",
    )
    axes[1].plot(
        x,
        focus.score_rank_recall_equal_token_budget * 100,
        color="#4D4D4D",
        marker="s",
        linestyle="--",
        label="Previous score rank (same tokens)",
    )
    axes[1].set(
        xlabel="Context configuration (K tokens)",
        ylabel="Next-step top-2048 recall (%)",
        xticks=x,
    )
    axes[1].legend(frameon=False)
    save_figure(fig, output / "shift_signal_vs_context_length")


def plot_page_summary(summary: pd.DataFrame, output: Path) -> None:
    methods = [
        ("shift+1", "Shift +1", "#0072B2", "o", "-"),
        (
            "score-rank@shift+1-budget",
            "Previous score rank (same +1 page budget)",
            "#4D4D4D",
            "s",
            "--",
        ),
    ]
    contexts = sorted(summary.context_config.unique())
    fig, axes = plt.subplots(
        1, len(contexts), figsize=(6.8, 2.65), sharey=True, constrained_layout=True
    )
    if len(contexts) == 1:
        axes = [axes]
    for axis, context in zip(axes, contexts, strict=True):
        part = summary[summary.context_config == context]
        for method, label, color, marker, linestyle in methods:
            values = part[part.method == method].sort_values("page_size")
            axis.plot(
                values.page_size,
                values.miss_page_coverage * 100,
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
            )
        axis.set_xscale("log", base=4)
        axis.set_xticks(PAGE_SIZES, labels=PAGE_SIZES)
        axis.set_xlabel("KV page size (tokens)")
        axis.text(
            0.04,
            0.94,
            f"{context // 1024}K",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    axes[0].set_ylabel("New page misses covered (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=2,
    )
    save_figure(fig, output / "shift_page_prefetch_value")


def run_self_test() -> None:
    previous = np.array([0, 2], dtype=np.int64)
    mask = np.zeros(8, dtype=bool)
    mask[previous] = True
    assert shifted_frontier(previous, mask, (1,)).tolist() == [1, 3]
    assert shifted_frontier(previous, mask, (-1, 1)).tolist() == [1, 3]
    target = np.array([1, 2], dtype=np.int64)
    metrics = page_metrics(target, {0, 1}, {2}, page_size=2)
    assert metrics["base_missing_pages"] == 0
    assert metrics["page_recall"] == 1.0
    assert select_score_pages(np.array([0, 2, 4, 6]), {0, 1}, 2, 2) == {2, 3}
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.run_dirs or args.output_dir is None:
        parser.error("--run-dirs and --output-dir are required unless --self-test")

    run_dirs = [path.resolve() for path in args.run_dirs]
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    token_records: list[dict[str, Any]] = []
    page_records: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        tokens, pages, validation = analyze_run(run_dir)
        token_records.extend(tokens)
        page_records.extend(pages)
        validations.append(validation)
        print(
            f"analyzed {validation['context_config'] // 1024}K: "
            f"{validation['layers']}, {validation['transitions']} transitions",
            flush=True,
        )

    token_rows = pd.DataFrame(token_records)
    page_rows = pd.DataFrame(page_records)
    token_summary = aggregate_token(token_rows)
    page_summary = aggregate_page(page_rows)
    token_rows.to_parquet(table_dir / "token_shift_by_transition.parquet", index=False)
    page_rows.to_parquet(table_dir / "page_prefetch_by_transition.parquet", index=False)
    token_summary.to_csv(table_dir / "token_shift_summary.csv", index=False)
    page_summary.to_csv(table_dir / "page_prefetch_summary.csv", index=False)

    set_style()
    plot_token_summary(token_summary, figure_dir)
    plot_page_summary(page_summary, figure_dir)

    delta_one = token_summary[token_summary.delta == 1]
    selected_page = page_summary[
        page_summary.method.isin(
            ["shift+1", "score-rank@shift+1-budget", "previous-only"]
        )
    ]
    summary = {
        "metric_definitions": {
            "precision_lift": (
                "P(new next-topk entry | shifted previous-topk position) / "
                "P(new entry | previous non-topk position)"
            ),
            "equal_token_budget": (
                "previous-score prefix with the same number of token candidates "
                "as previous-topk union shift candidates"
            ),
            "miss_page_coverage": (
                "next-step pages absent from previous-step pages that are fetched "
                "by the predictor"
            ),
            "equal_page_budget": (
                "previous-score ranked pages using exactly the same number of newly "
                "fetched pages as the shift predictor"
            ),
        },
        "delta_plus_one_by_context": delta_one.to_dict("records"),
        "page_gate": selected_page.to_dict("records"),
        "validation": validations,
        "limitations": [
            "one RULER niah_single_1 request per context length",
            "seven sampled layers and 31 adjacent transitions per layer",
            (
                "equal-page-budget comparison uses only full-score layer L23 "
                "(31 transitions per context)"
            ),
            (
                "descriptive means only; layer-step observations are not treated "
                "as independent confidence samples"
            ),
            "trace-driven prediction analysis, not measured I/O latency or end-to-end speed",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    validation_path = output_dir / "validation.json"
    validation_path.write_text(
        json.dumps(validations, indent=2, ensure_ascii=False) + "\n"
    )
    reproducibility = {
        "run_dirs": [str(path) for path in run_dirs],
        "output_dir": str(output_dir),
        "script": str(Path(__file__).resolve()),
        "code_commit": git_revision(Path(__file__).resolve().parents[2]),
        "target_k": TARGET_K,
        "deltas": DELTAS,
        "page_sizes_tokens": PAGE_SIZES,
        "aggregation": (
            "descriptive mean over seven sampled layers and 31 adjacent "
            "transitions; no inferential CI"
        ),
        "output_sha256": {},
    }
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "reproducibility.json":
            reproducibility["output_sha256"][str(path.relative_to(output_dir))] = sha256(path)
    (output_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
