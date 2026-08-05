#!/usr/bin/env python3
"""Validate the SM8x logical-candidate scorer against the full exact scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sglang.srt.layers.attention.keye_topk.ampere_indexer import (
    keye_indexer_score_candidates,
    keye_indexer_score_paged,
)


TOPK = 2048
NUM_HEADS = 16
HEAD_DIM = 64
SCALE = HEAD_DIM**-0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def valid_set(row: torch.Tensor) -> set[int]:
    return set(row[row >= 0].tolist())


@torch.inference_mode()
def run_case(
    *,
    device: torch.device,
    batch: int,
    context: int,
    candidate_k: int,
    generator: torch.Generator,
) -> dict[str, object]:
    seqlens = torch.tensor(
        [max(1, context - 37 * row) for row in range(batch)],
        dtype=torch.int32,
        device=device,
    )
    physical_slots = batch * context + 257
    key_cache = torch.randn(
        physical_slots,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    page_table = torch.stack(
        [
            torch.randperm(
                physical_slots, device=device, generator=generator, dtype=torch.int64
            )[:context]
            for _ in range(batch)
        ]
    )
    query = torch.randn(
        batch,
        NUM_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    weights = torch.randn(
        batch,
        NUM_HEADS,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    full_scores = keye_indexer_score_paged(
        query, key_cache, page_table, weights, seqlens, SCALE
    )
    full_scores_for_topk = full_scores.clone()
    exact_values, exact_indices = torch.topk(
        full_scores_for_topk, min(TOPK, context), dim=1
    )
    exact_indices = exact_indices.to(torch.int32).masked_fill(
        ~exact_values.isfinite(), -1
    )

    # Include all exact top-k positions that fit, then add random valid positions.
    candidates = torch.full(
        (batch, candidate_k), -1, dtype=torch.int32, device=device
    )
    for row in range(batch):
        seq_len = int(seqlens[row].item())
        exact = exact_indices[row]
        exact = exact[exact >= 0]
        keep = min(exact.numel(), max(TOPK - 17, 1), candidate_k)
        prefix = exact[:keep]
        permutation = torch.randperm(
            seq_len, device=device, generator=generator, dtype=torch.int64
        ).to(torch.int32)
        merged = torch.unique(torch.cat((prefix, permutation)), sorted=False)
        take = min(candidate_k, seq_len, merged.numel())
        candidates[row, :take] = merged[:take]

    candidate_scores = keye_indexer_score_candidates(
        query,
        key_cache,
        page_table,
        candidates,
        weights,
        seqlens,
        SCALE,
    )
    safe_candidates = candidates.clamp_min(0).long()
    gathered_scores = full_scores.gather(1, safe_candidates)
    valid_candidates = candidates >= 0
    score_difference = torch.where(
        valid_candidates,
        (candidate_scores - gathered_scores).abs(),
        torch.zeros_like(candidate_scores),
    )
    invalid_are_negative_inf = bool(
        torch.isneginf(candidate_scores[~valid_candidates]).all().item()
    )

    candidate_values, candidate_offsets = torch.topk(
        candidate_scores, min(TOPK, candidate_k), dim=1
    )
    final_indices = candidates.gather(1, candidate_offsets.long()).masked_fill(
        ~candidate_values.isfinite(), -1
    )
    coverage = []
    final_recall = []
    for row in range(batch):
        exact_set = valid_set(exact_indices[row].cpu())
        candidate_set = valid_set(candidates[row].cpu())
        final_set = valid_set(final_indices[row].cpu())
        denominator = max(len(exact_set), 1)
        coverage.append(len(exact_set & candidate_set) / denominator)
        final_recall.append(len(exact_set & final_set) / denominator)

    max_abs = float(score_difference.max().item())
    coverage_error = max(
        abs(left - right) for left, right in zip(coverage, final_recall, strict=True)
    )
    passed = max_abs == 0.0 and invalid_are_negative_inf and coverage_error == 0.0
    return {
        "batch": batch,
        "context": context,
        "candidate_k": candidate_k,
        "max_candidate_score_abs": max_abs,
        "invalid_are_negative_inf": invalid_are_negative_inf,
        "max_coverage_final_recall_abs": coverage_error,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    batches = [1, 2] if args.quick else [1, 2, 8]
    contexts = [2049, 3072] if args.quick else [2049, 2560, 3072, 4096, 6144, 8192]
    cases = [
        run_case(
            device=device,
            batch=batch,
            context=context,
            candidate_k=candidate_k,
            generator=generator,
        )
        for batch in batches
        for context in contexts
        for candidate_k in (2560, 3072)
    ]
    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "case_count": len(cases),
        "passed_count": sum(bool(case["passed"]) for case in cases),
        "all_passed": all(bool(case["passed"]) for case in cases),
        "cases": cases,
    }
    payload = json.dumps(summary, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
