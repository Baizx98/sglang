#!/usr/bin/env python3
"""Evaluate frozen K4096 recurrent-resident page protection (Gate E1)."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from simulate_multitier_prefetch import (
    Chunk,
    TransferCounts,
    candidate_pages,
    page_slots,
    read_jsonl,
    target_for_transition,
    warm_cache,
)

BOOTSTRAP_SEED = 20260810
BOOTSTRAP_SAMPLES = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text())
    forbidden = [
        "uses_task_label",
        "uses_dataset_label",
        "uses_future_trace",
        "uses_test_trace_for_tuning",
    ]
    if any(bool(policy.get(key)) for key in forbidden):
        raise ValueError("resident protection policy uses forbidden information")
    if int(policy["baseline_candidate_k"]) != 4096:
        raise ValueError("Gate E1 requires K4096")
    return policy


def load_shared_chunks(
    run_dir: Path,
    layers: list[int],
    allow_incomplete: bool,
) -> tuple[list[Chunk], dict[str, int]]:
    prepared: dict[str, dict[str, Any]] = {}
    completed: set[str] = set()
    prepared_paths = sorted(run_dir.glob("*/prepared_requests.jsonl"))
    if not prepared_paths and (run_dir / "prepared_requests.jsonl").exists():
        prepared_paths = [run_dir / "prepared_requests.jsonl"]
    if not prepared_paths:
        raise ValueError(f"{run_dir}: no prepared request sets")
    for prepared_path in prepared_paths:
        child = prepared_path.parent
        prepared_rows = read_jsonl(prepared_path)
        executed_rows = (
            read_jsonl(child / "requests.jsonl")
            if (child / "requests.jsonl").exists()
            else []
        )
        child_prepared = {str(row["rid"]): row for row in prepared_rows}
        child_completed = {str(row["rid"]) for row in executed_rows}
        if len(child_prepared) != len(prepared_rows) or len(child_completed) != len(executed_rows):
            raise ValueError(f"{child}: duplicate request IDs")
        if child_completed - set(child_prepared):
            raise ValueError(f"{child}: unexpected completed request")
        overlap = set(prepared) & set(child_prepared)
        if overlap:
            raise ValueError(f"request IDs reused across runs: {sorted(overlap)[:3]}")
        prepared.update(child_prepared)
        completed.update(child_completed)
    if not allow_incomplete and completed != set(prepared):
        raise ValueError(f"incomplete collection: {len(completed)}/{len(prepared)} requests")

    wanted = set(layers)
    lookup: dict[tuple[str, int], Path] = {}
    for row in read_jsonl(run_dir / "events/manifest.jsonl"):
        rid = str(row["request_id"])
        layer = int(row["layer_id"])
        if rid not in completed or layer not in wanted:
            continue
        key = (rid, layer)
        if key in lookup:
            raise ValueError(f"duplicate request-layer chunk: {key}")
        lookup[key] = run_dir / "events" / str(row["file"])
    complete = {
        rid for rid in completed if all((rid, layer) in lookup for layer in layers)
    }
    partial = {
        rid
        for rid in completed
        if 0 < sum((rid, layer) in lookup for layer in layers) < len(layers)
    }
    if partial:
        raise ValueError(f"partial trace requests: {sorted(partial)[:3]}")
    if not allow_incomplete and complete != completed:
        raise ValueError(f"complete traces: {len(complete)}/{len(completed)} outputs")

    chunks: list[Chunk] = []
    for rid in sorted(complete):
        request = prepared[rid]
        for layer in layers:
            path = lookup[(rid, layer)]
            record = torch.load(path, map_location="cpu", weights_only=False)
            if (
                int(record["schema_version"]) != 5
                or record.get("topk_backend") != "torch_exact"
                or int(record["compact_k"]) != 4096
                or record["decode_step_ids"] != list(range(32))
            ):
                raise ValueError(f"{path}: Gate E1 compact trace contract mismatch")
            topk = record["indices"].numpy(force=True).astype(np.int64)
            ranked = record["candidate_indices"].numpy(force=True).astype(np.int64)
            valid = record["score_valid_counts"].numpy(force=True).astype(np.int64)
            for step in range(32):
                canonical = set(int(value) for value in topk[step] if int(value) >= 0)
                prefix = set(int(value) for value in ranked[step, :2048] if int(value) >= 0)
                if canonical != prefix:
                    raise ValueError(f"{path}: exact prefix mismatch at step {step}")
            chunks.append(
                Chunk(
                    request_id=rid,
                    dataset=str(request["dataset"]),
                    task=str(request["task"]),
                    source_index=request.get("source_index"),
                    context_config=int(request["length_config"]),
                    prompt_tokens=int(request["prompt_len"]),
                    layer=layer,
                    topk=topk,
                    ranked=ranked,
                    valid_counts=valid,
                    source_file=path,
                    ranked_scores=None,
                )
            )
    return chunks, {
        "prepared_requests": len(prepared),
        "completed_outputs": len(completed),
        "complete_trace_requests": len(complete),
        "chunks": len(chunks),
    }


def recent_priority(history: deque[set[int]]) -> tuple[Counter[int], dict[int, int]]:
    counts: Counter[int] = Counter()
    most_recent_age: dict[int, int] = {}
    for age, page_set in enumerate(reversed(history)):
        for page in page_set:
            counts[page] += 1
            most_recent_age.setdefault(page, age)
    return counts, most_recent_age


def run_protected_policy(
    chunk: Chunk,
    policy: dict[str, Any],
    hbm_pages: int,
    dram_pages: int,
    page_size: int,
    kv_bytes_per_token_layer: int,
) -> list[dict[str, Any]]:
    total_tokens = int(chunk.valid_counts.max())
    cache = warm_cache(chunk, page_size, hbm_pages, dram_pages)
    history: deque[set[int]] = deque(maxlen=int(policy["history_window_steps"]))
    initial_common = int(chunk.valid_counts[0])
    history.append(set(candidate_pages(chunk, 0, page_size, 2048, initial_common)))
    minimum_hits = int(policy["minimum_hits_in_window"])
    rows: list[dict[str, Any]] = []

    for step in range(31):
        target_tokens, target_order, common = target_for_transition(chunk, step, page_size)
        target_set = set(target_order)
        score_order = candidate_pages(chunk, step, page_size, 4096, common)
        baseline_selected = score_order[: cache.hbm_capacity]
        hbm_before = set(cache.hbm)
        counts, ages = recent_priority(history)
        protected = sorted(
            (
                page
                for page in hbm_before
                if counts[page] >= minimum_hits
            ),
            key=lambda page: (-counts[page], ages[page], page),
        )
        protected_set = set(protected)
        missing_protected = [page for page in protected if page not in baseline_selected]
        selected = list(baseline_selected)
        swaps = 0
        for page in missing_protected:
            removable = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if selected[index] not in protected_set
                ),
                None,
            )
            if removable is None:
                break
            selected.pop(removable)
            selected.append(page)
            swaps += 1
        selected_set = set(selected)
        priority = [page for page in protected if page in selected_set]
        priority.extend(page for page in baseline_selected if page in selected_set and page not in protected_set)
        if len(priority) != len(set(priority)):
            raise AssertionError("duplicate protected admission priority")

        counterfactual_nonresident = [
            page for page in baseline_selected if cache.tier(page) != "hbm"
        ]
        page_budget = len(counterfactual_nonresident)
        byte_budget = sum(
            page_slots(page, page_size, total_tokens) * kv_bytes_per_token_layer
            for page in counterfactual_nonresident
        )
        resident_selected = [page for page in priority if cache.tier(page) == "hbm"]
        candidate_admits = [page for page in priority if cache.tier(page) != "hbm"]
        admitted: list[int] = []
        admitted_bytes = 0
        for page in candidate_admits:
            size = page_slots(page, page_size, total_tokens) * kv_bytes_per_token_layer
            if len(admitted) == page_budget or admitted_bytes + size > byte_budget:
                continue
            admitted.append(page)
            admitted_bytes += size
        admitted_set = set(admitted)
        resident_selected_set = set(resident_selected)
        final_priority = [
            page
            for page in priority
            if page in resident_selected_set or page in admitted_set
        ]
        final_set = set(final_priority)
        # Both baseline and protection use the same atomic admission. Touching
        # selected residents first prevents a page selected by this same batch
        # from being evicted and retransferred while nonresidents are admitted.
        for page in reversed(resident_selected):
            if cache.promote(page) != "hbm":
                raise AssertionError("protected resident unexpectedly absent")
        prefetch = TransferCounts()
        for page in reversed(admitted):
            source = cache.promote(page)
            if source == "hbm":
                raise AssertionError("admitted page unexpectedly resident")
            prefetch.add(
                source,
                page_slots(page, page_size, total_tokens) * kv_bytes_per_token_layer,
            )
        if not final_set.issubset(cache.hbm):
            raise AssertionError("selected resident page evicted during admission")
        for page in reversed(final_priority):
            if cache.promote(page) != "hbm":
                raise AssertionError("selected page unexpectedly requires retransmission")
        prefetch_bytes = prefetch.dram_bytes + prefetch.ssd_bytes
        if len(admitted) > page_budget or prefetch_bytes > byte_budget:
            raise AssertionError("resident policy exceeded K4096 counterfactual budget")

        hbm_ready = set(cache.hbm)
        hit_pages = target_set & hbm_ready
        hit_tokens = sum((int(token) // page_size) in hbm_ready for token in target_tokens)
        critical = TransferCounts()
        for page in target_order:
            source = cache.tier(page)
            if source != "hbm":
                critical.add(
                    source,
                    page_slots(page, page_size, total_tokens) * kv_bytes_per_token_layer,
                )
        for page in reversed(target_order):
            cache.promote(page)
        next_valid = int(chunk.valid_counts[step + 1])
        if next_valid > common:
            cache.create_in_hbm((next_valid - 1) // page_size)
        cache.check()
        history.append(target_set)

        critical_bytes = critical.dram_bytes + critical.ssd_bytes
        rows.append(
            {
                "request_id": chunk.request_id,
                "dataset": chunk.dataset,
                "task": chunk.task,
                "context_config": chunk.context_config,
                "prompt_tokens": chunk.prompt_tokens,
                "layer": chunk.layer,
                "step": step,
                "method": "resident-protection-v1",
                "target_tokens": len(target_tokens),
                "target_pages": len(target_set),
                "hbm_hit_tokens": hit_tokens,
                "hbm_hit_pages": len(hit_pages),
                "prefetch_admit_pages": len(admitted),
                "prefetch_bytes": prefetch_bytes,
                "critical_miss_pages": critical.dram_pages + critical.ssd_pages,
                "critical_miss_bytes": critical_bytes,
                "total_pcie_bytes": prefetch_bytes + critical_bytes,
                "protected_resident_pages": len(protected),
                "protected_pages_outside_baseline": len(missing_protected),
                "protection_swaps": swaps,
                "counterfactual_prefetch_page_budget": page_budget,
                "counterfactual_prefetch_byte_budget": byte_budget,
                "unused_prefetch_pages": len(set(admitted) - target_set),
                "pollution_miss_pages": len((hbm_before & target_set) - hbm_ready),
                "source_file": chunk.source_file.name,
            }
        )
    return rows


def baseline_rows(
    chunk: Chunk,
    policy: dict[str, Any],
    hbm_pages: int,
    dram_pages: int,
    page_size: int,
    kv_bytes_per_token_layer: int,
) -> list[dict[str, Any]]:
    baseline_policy = dict(policy)
    baseline_policy["minimum_hits_in_window"] = (
        int(policy["history_window_steps"]) + 1
    )
    rows = run_protected_policy(
        chunk,
        baseline_policy,
        hbm_pages,
        dram_pages,
        page_size,
        kv_bytes_per_token_layer,
    )
    for row in rows:
        row["method"] = "baseline-rank4096"
        row["protected_resident_pages"] = 0
        row["protected_pages_outside_baseline"] = 0
        row["protection_swaps"] = 0
    return rows


def aggregate(detail: pd.DataFrame, layers: list[int], tp: int) -> pd.DataFrame:
    keys = ["dataset", "context_config", "task", "request_id", "method"]
    sums = detail.groupby(keys, as_index=False).agg(
        transitions=("step", "size"),
        target_tokens=("target_tokens", "sum"),
        target_pages=("target_pages", "sum"),
        hbm_hit_tokens=("hbm_hit_tokens", "sum"),
        hbm_hit_pages=("hbm_hit_pages", "sum"),
        prefetch_bytes=("prefetch_bytes", "sum"),
        critical_miss_bytes=("critical_miss_bytes", "sum"),
        total_pcie_bytes=("total_pcie_bytes", "sum"),
        protected_resident_pages=("protected_resident_pages", "sum"),
        protected_pages_outside_baseline=("protected_pages_outside_baseline", "sum"),
        protection_swaps=("protection_swaps", "sum"),
        unused_prefetch_pages=("unused_prefetch_pages", "sum"),
        pollution_miss_pages=("pollution_miss_pages", "sum"),
    )
    sums["hbm_token_recall"] = sums.hbm_hit_tokens / sums.target_tokens
    sums["hbm_page_recall"] = sums.hbm_hit_pages / sums.target_pages
    scale = 48 / len(layers) / tp / 2**20
    decode_tokens = sums.transitions / len(layers)
    for source, target in [
        ("prefetch_bytes", "prefetch_mib_per_token_per_gpu"),
        ("critical_miss_bytes", "correction_mib_per_token_per_gpu"),
        ("total_pcie_bytes", "total_pcie_mib_per_token_per_gpu"),
    ]:
        sums[target] = sums[source] * scale / decode_tokens
    return sums


def paired_requests(request: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "context_config", "task", "request_id"]
    metrics = [
        "hbm_token_recall",
        "hbm_page_recall",
        "prefetch_mib_per_token_per_gpu",
        "correction_mib_per_token_per_gpu",
        "total_pcie_mib_per_token_per_gpu",
        "protected_resident_pages",
        "protected_pages_outside_baseline",
        "protection_swaps",
        "unused_prefetch_pages",
        "pollution_miss_pages",
    ]
    baseline = request[request.method == "baseline-rank4096"][keys + metrics]
    protected = request[request.method == "resident-protection-v1"][keys + metrics]
    paired = baseline.merge(protected, on=keys, suffixes=("_baseline", "_protected"), validate="one_to_one")
    for metric in metrics:
        paired[f"{metric}_delta"] = paired[f"{metric}_protected"] - paired[f"{metric}_baseline"]
    paired["hbm_token_recall_delta_pp"] = 100 * paired.hbm_token_recall_delta
    paired["hbm_page_recall_delta_pp"] = 100 * paired.hbm_page_recall_delta
    return paired


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = [
        "hbm_token_recall_delta_pp",
        "hbm_page_recall_delta_pp",
        "correction_mib_per_token_per_gpu_delta",
        "total_pcie_mib_per_token_per_gpu_delta",
    ]
    rows: list[dict[str, Any]] = []
    for values, part in paired.groupby(["dataset", "context_config"], sort=True):
        for metric in metrics:
            vector = part[metric].to_numpy(float)
            low, high = bootstrap_ci(vector, rng)
            task_vector = part.groupby("task", sort=True)[metric].mean().to_numpy(float)
            task_low, task_high = bootstrap_ci(task_vector, rng)
            rows.append(
                {
                    "dataset": values[0],
                    "context_config": values[1],
                    "metric": metric,
                    "requests": len(part),
                    "tasks": part.task.nunique(),
                    "request_mean": float(vector.mean()),
                    "request_ci95_low": low,
                    "request_ci95_high": high,
                    "task_balanced_mean": float(task_vector.mean()),
                    "task_cluster_ci95_low": task_low,
                    "task_cluster_ci95_high": task_high,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--layers",
        help="optional comma-separated layer override; default is the frozen policy set",
    )
    args = parser.parse_args()

    policy = load_policy(args.policy)
    contract = policy["simulation_contract"]
    layers = (
        sorted({int(value) for value in args.layers.split(",") if value.strip()})
        if args.layers
        else [int(value) for value in contract["sampled_layers"]]
    )
    if not layers or layers[0] < 0 or layers[-1] >= int(contract["scaled_model_layers"]):
        raise ValueError("--layers must be a nonempty subset of model layers")
    page_size = int(contract["page_size_tokens"])
    kv_bytes = int(contract["kv_bytes_per_token_layer"])
    model_layers = int(contract["scaled_model_layers"])
    tp = int(contract["tensor_parallel_size"])
    bytes_per_model_page = page_size * kv_bytes * model_layers
    hbm_pages = max(1, int(float(contract["hbm_logical_gib"]) * 2**30 // bytes_per_model_page))
    dram_pages = max(0, int(float(contract["dram_logical_gib"]) * 2**30 // bytes_per_model_page))
    chunks, coverage = load_shared_chunks(args.run_dir, layers, args.allow_incomplete)
    if not chunks:
        raise ValueError("no complete Gate E1 chunks")

    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        rows.extend(
            baseline_rows(
                chunk,
                policy,
                hbm_pages,
                dram_pages,
                page_size,
                kv_bytes,
            )
        )
        rows.extend(run_protected_policy(chunk, policy, hbm_pages, dram_pages, page_size, kv_bytes))
        if index % 35 == 0:
            print(f"[{index}/{len(chunks)}] chunks", flush=True)
    detail = pd.DataFrame(rows)
    protected_detail = detail[detail.method == "resident-protection-v1"]
    budget_pages_ok = bool(
        (protected_detail.prefetch_admit_pages <= protected_detail.counterfactual_prefetch_page_budget).all()
    )
    budget_bytes_ok = bool(
        (protected_detail.prefetch_bytes <= protected_detail.counterfactual_prefetch_byte_budget).all()
    )
    request = aggregate(detail, layers, tp)
    paired = paired_requests(request)
    summary = summarize(paired)

    correction = paired.correction_mib_per_token_per_gpu_delta.to_numpy(float)
    correction_low, correction_high = bootstrap_ci(correction, np.random.default_rng(BOOTSTRAP_SEED))
    gate_contract = policy["gate"]
    gate_checks = {
        "mean_correction_ci95_high": correction_high
        <= float(gate_contract["mean_correction_byte_delta_request_bootstrap_ci95_high_max"]),
        "mean_total_pcie_not_higher": float(paired.total_pcie_mib_per_token_per_gpu_delta.mean())
        <= float(gate_contract["mean_total_pcie_byte_delta_max"]),
        "request_hbm_page_recall_no_worse_fraction": float((paired.hbm_page_recall_delta_pp >= 0).mean())
        >= float(gate_contract["request_hbm_page_recall_no_worse_fraction_min"]),
        "worst_request_hbm_page_recall_delta_pp": float(paired.hbm_page_recall_delta_pp.min())
        >= float(gate_contract["worst_request_hbm_page_recall_delta_pp_min"]),
    }
    coverage_complete = coverage["prepared_requests"] == coverage["complete_trace_requests"]
    all_contracts = budget_pages_ok and budget_bytes_ok
    output_tables = args.output_dir / "tables"
    output_tables.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(output_tables / "by_request_layer_step.parquet", index=False)
    request.to_csv(output_tables / "by_request_method.csv", index=False)
    paired.to_csv(output_tables / "paired_by_request.csv", index=False)
    summary.to_csv(output_tables / "dataset_length_summary.csv", index=False)
    result = {
        "schema_version": 1,
        "analysis_kind": "Gate E1 frozen K4096 resident-page protection shadow replay; no speed measurement",
        "policy": str(args.policy.resolve()),
        "policy_sha256": sha256(args.policy),
        "coverage": coverage,
        "coverage_complete": coverage_complete,
        "setting": {
            "layers": layers,
            "page_size_tokens": page_size,
            "hbm_capacity_pages_per_layer": hbm_pages,
            "dram_capacity_pages_per_layer": dram_pages,
            "tensor_parallel_size": tp,
        },
        "contracts": {
            "exact_top2048_demand_preserved": True,
            "all_corrections_materialized_before_attention": True,
            "atomic_admission_shared_by_baseline_and_protection": True,
            "prefetch_page_budget_never_exceeded": budget_pages_ok,
            "prefetch_byte_budget_never_exceeded": budget_bytes_ok,
            "all_contracts_passed": all_contracts,
        },
        "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES},
        "overall": {
            "requests": len(paired),
            "hbm_page_recall_delta_pp_mean": float(paired.hbm_page_recall_delta_pp.mean()),
            "hbm_page_recall_no_worse_fraction": float((paired.hbm_page_recall_delta_pp >= 0).mean()),
            "hbm_page_recall_delta_pp_worst": float(paired.hbm_page_recall_delta_pp.min()),
            "correction_mib_per_token_per_gpu_delta_mean": float(correction.mean()),
            "correction_delta_ci95": [correction_low, correction_high],
            "total_pcie_mib_per_token_per_gpu_delta_mean": float(
                paired.total_pcie_mib_per_token_per_gpu_delta.mean()
            ),
            "requests_with_any_protection_swap": int((paired.protection_swaps_protected > 0).sum()),
        },
        "gate_checks": gate_checks,
        "gate_passed": bool(all(gate_checks.values()) and all_contracts and coverage_complete),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
