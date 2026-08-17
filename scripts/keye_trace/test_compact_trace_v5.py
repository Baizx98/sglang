#!/usr/bin/env python3
"""CPU smoke test for compact Keye score-trace schema v5."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

os.environ.setdefault("KEYE_SM80_DSA", "1")

from sglang.srt.layers.attention.keye_topk.keye_indexer import KeyeIndexer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["KEYE_SM80_TRACE_SCORE_BLOCK_SIZE"] = "256"
    os.environ["KEYE_SM80_EXACT_TOPK"] = "1"

    generator = torch.Generator().manual_seed(args.seed)
    indexer = KeyeIndexer.__new__(KeyeIndexer)
    configurations = [
        (4096, (2048, 2560, 3072, 4096), (5000, 5001)),
        (8192, (2048, 2560, 3072, 4096, 6144, 8192), (9000, 9001)),
    ]
    config_checks: list[dict[str, object]] = []
    for compact_k, ranks, valid_rows in configurations:
        os.environ["KEYE_SM80_TRACE_COMPACT_K"] = str(compact_k)
        os.environ["KEYE_SM80_TRACE_COMPACT_RANKS"] = ",".join(
            str(rank) for rank in ranks
        )
        row_checks: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="keye-trace-v5-") as temp_dir:
            trace_dir = Path(temp_dir)
            for step, valid_tokens in enumerate(valid_rows):
                scores = torch.randn(valid_tokens, generator=generator)
                canonical = torch.topk(scores, 2048).indices.to(torch.int32)
                payload = indexer._compact_trace_row_sm80(scores, valid_tokens)
                candidate_indices = payload["candidate_indices"]
                candidate_scores = payload["candidate_scores"]
                thresholds = payload["score_thresholds"]
                block_counts = payload["block_valid_counts"]
                threshold_counts = payload["block_threshold_counts"]
                assert isinstance(candidate_indices, torch.Tensor)
                assert isinstance(candidate_scores, torch.Tensor)
                assert isinstance(thresholds, torch.Tensor)
                assert isinstance(block_counts, torch.Tensor)
                assert isinstance(threshold_counts, torch.Tensor)

                candidate_prefix = set(candidate_indices[:2048].tolist())
                canonical_set = set(canonical.tolist())
                assert candidate_prefix == canonical_set
                assert bool(
                    torch.all(candidate_scores[:-1] >= candidate_scores[1:])
                )
                assert int(block_counts.sum()) == valid_tokens
                for threshold_index, rank in enumerate(ranks):
                    assert torch.equal(
                        thresholds[threshold_index], candidate_scores[rank - 1]
                    )
                    assert int(
                        threshold_counts[:, threshold_index].sum()
                    ) == int((scores >= thresholds[threshold_index]).sum())

                indexer._append_trace_chunk_row(
                    trace_dir=trace_dir,
                    trace_mode="compact",
                    request_id=f"compact-smoke-{compact_k}__0",
                    layer_id=7,
                    decode_step_id=step,
                    indices=canonical,
                    score=None,
                    compact_payload=payload,
                    keep_full_score=False,
                    valid_count=2048,
                    score_valid_count=valid_tokens,
                    input_id=torch.tensor(step),
                    position=torch.tensor([step, step, step]),
                    chunk_steps=2,
                )
                row_checks.append(
                    {
                        "step": step,
                        "valid_tokens": valid_tokens,
                        "candidate_top2048_matches_canonical_set": True,
                        "block_valid_count_sum": int(block_counts.sum()),
                    }
                )

            chunks = list(trace_dir.glob("chunk_*.pt"))
            assert len(chunks) == 1
            record = torch.load(chunks[0], weights_only=False)
            assert record["schema_version"] == 5
            assert record["topk_backend"] == "torch_exact"
            assert record["candidate_indices"].shape == (2, compact_k)
            assert record["candidate_scores"].shape == (2, compact_k)
            assert record["score_thresholds"].shape == (2, len(ranks))
            expected_blocks = (max(valid_rows) + 255) // 256
            assert record["block_threshold_counts"].shape == (
                2,
                expected_blocks,
                len(ranks),
            )
            assert record["scores"] is None
            assert not record["full_scores_retained"]
            config_checks.append(
                {
                    "compact_k": compact_k,
                    "threshold_ranks": ranks,
                    "chunk_bytes": chunks[0].stat().st_size,
                    "row_checks": row_checks,
                }
            )

    result = {
        "schema_version": 1,
        "seed": args.seed,
        "passed": True,
        "trace_schema": 5,
        "configurations": config_checks,
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
