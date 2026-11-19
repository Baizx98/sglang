"""Exact sampled cutoff with fused Keye scoring and candidate compaction."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from optimize import K, _make_scoring_variant, _selector


_SAMPLE = 2048
_CAPACITY = 4096
_TILE = 128


@triton.jit
def _score_and_compact(
    q_ptr, key_ptr, table_ptr, gates_ptr, lengths_ptr,
    sample_ptr, pivot_ptr, count_ptr, scores_ptr, indices_ptr,
    q_b: tl.constexpr, q_h: tl.constexpr, q_d: tl.constexpr,
    k_s: tl.constexpr, k_d: tl.constexpr,
    p_b: tl.constexpr, p_n: tl.constexpr,
    w_b: tl.constexpr, w_h: tl.constexpr,
    length_stride: tl.constexpr, width: tl.constexpr, scale: tl.constexpr,
    capacity: tl.constexpr, samples: tl.constexpr, block: tl.constexpr,
    only_sample: tl.constexpr,
):
    row = tl.program_id(1)
    length = tl.load(lengths_ptr + row * length_stride)
    tile = tl.program_id(0) * block
    if (length > 2048) & (only_sample | (tile < length)):
        lane = tl.arange(0, block)
        offset = tile + lane
        if only_sample:
            # One deterministic sample per equal-width interval.
            logical = ((2 * offset + 1).to(tl.int64) * length // (2 * samples)).to(tl.int32)
            live = offset < samples
        else:
            logical = offset
            live = (logical < length) & (logical < width)
        slot = tl.load(table_ptr + row * p_b + logical * p_n, live, other=-1).to(tl.int64)
        live &= slot >= 0

        head = tl.arange(0, 16)
        dim = tl.arange(0, 64)
        q = tl.load(q_ptr + row * q_b + head[:, None] * q_h + dim[None, :] * q_d)
        key = tl.load(
            key_ptr + slot[:, None] * k_s + dim[None, :] * k_d,
            live[:, None], other=0,
        )
        gates = tl.load(gates_ptr + row * w_b + head * w_h).to(tl.float32)
        dot = tl.dot(q, tl.trans(key), allow_tf32=False)
        dot = dot.to(tl.bfloat16).to(tl.float32)
        score = tl.sum(tl.maximum(dot * scale, 0.0) * gates[:, None], 0)
        if only_sample:
            tl.store(
                sample_ptr + row * samples + offset,
                tl.where(live, score, -float("inf")),
                offset < samples,
            )
        else:
            pivot = tl.load(pivot_ptr + row)
            keep = live & (score >= pivot)
            taken = tl.sum(keep.to(tl.int32), 0)
            if taken:
                base = tl.atomic_add(count_ptr + row, taken, sem="relaxed")
                output = base + tl.cumsum(keep.to(tl.int32), 0) - 1
                store = keep & (output < capacity)
                tl.store(scores_ptr + row * capacity + output, score, store)
                tl.store(indices_ptr + row * capacity + output, logical, store)


@triton.jit
def _sample_cutoff(
    sample_ptr, lengths_ptr, pivot_ptr, count_ptr,
    length_stride: tl.constexpr, samples: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(lengths_ptr + row * length_stride)
    if length > 2048:
        offset = tl.arange(0, samples)
        values = tl.load(sample_ptr + row * samples + offset)
        ranked = tl.sort(values, descending=True)
        # Aim for ~3072 candidates while tolerating sampling error both ways.
        rank = tl.minimum((3072 * samples + length - 1) // length, samples)
        pivot = tl.max(tl.where(offset == rank - 1, ranked, -float("inf")), 0)
        tl.store(pivot_ptr + row, pivot)
    tl.store(count_ptr + row, 0)


@triton.jit
def _route_rows(
    lengths_ptr, count_ptr, compact_lengths_ptr, fallback_lengths_ptr,
    length_stride: tl.constexpr, batch: tl.constexpr, capacity: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.arange(0, block)
    live = row < batch
    length = tl.load(lengths_ptr + row * length_stride, live, other=0)
    count = tl.load(count_ptr + row, live, other=0)
    full_topk = length > 2048
    certified = (count >= 2048) & (count <= capacity)
    tl.store(compact_lengths_ptr + row, tl.where(full_topk & certified, count, 0), live)
    tl.store(fallback_lengths_ptr + row, tl.where(full_topk & ~certified, length, 0), live)


@triton.jit
def _merge_rows(
    lengths_ptr, compact_lengths_ptr, compact_ptr, fallback_ptr, output_ptr,
    length_stride: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.arange(0, 2048)
    length = tl.load(lengths_ptr + row * length_stride)
    compact_length = tl.load(compact_lengths_ptr + row)
    if length <= 2048:
        result = tl.where(pos < length, pos, -1)
    elif compact_length > 0:
        result = tl.load(compact_ptr + row * 2048 + pos)
    else:
        result = tl.load(fallback_ptr + row * 2048 + pos)
    tl.store(output_ptr + row * 2048 + pos, result)


def prepare(q, key_cache, page_table, weights, lengths, scale):
    """Select exact logical indices, including when the sampled cutoff fails."""
    from flashinfer.topk import top_k_page_table_transform

    batch, width = page_table.shape
    device = q.device
    candidate_scores = torch.empty((batch, _CAPACITY), dtype=torch.float32, device=device)
    candidate_indices = torch.empty_like(candidate_scores, dtype=torch.int32)
    samples = torch.empty((batch, _SAMPLE), dtype=torch.float32, device=device)
    pivot = torch.empty((batch,), dtype=torch.float32, device=device)
    counts = torch.empty((batch,), dtype=torch.int32, device=device)
    compact_lengths = torch.empty_like(counts)
    fallback_lengths = torch.empty_like(counts)
    output = torch.empty((batch, K), dtype=torch.int32, device=device)
    fallback_score = _make_scoring_variant(
        q, key_cache, page_table, weights, fallback_lengths,
        scale, _TILE, 4,
    )
    fallback_select = _selector(fallback_lengths)
    length_stride = lengths.stride(0)

    def score(only_sample):
        tiles = triton.cdiv(_SAMPLE if only_sample else width, _TILE)
        _score_and_compact[(tiles, batch)](
            q, key_cache, page_table, weights, lengths, samples, pivot, counts,
            candidate_scores, candidate_indices,
            *q.stride(), *key_cache.stride(), *page_table.stride(), *weights.stride(),
            length_stride, width, float(scale), _CAPACITY, _SAMPLE, _TILE,
            only_sample, num_warps=4,
        )

    def select():
        score(only_sample=True)
        _sample_cutoff[(batch,)](
            samples, lengths, pivot, counts, length_stride, _SAMPLE, num_warps=4,
        )
        score(only_sample=False)
        _route_rows[(1,)](
            lengths, counts, compact_lengths, fallback_lengths, length_stride,
            batch, _CAPACITY, triton.next_power_of_2(batch), num_warps=1,
        )
        candidates = top_k_page_table_transform(
            candidate_scores, candidate_indices, compact_lengths, K
        )
        # The raw fallback kernel is launched for graph capture. Certified rows
        # receive length zero and never materialize their dense FP32 scores.
        fallback = fallback_select(fallback_score())
        _merge_rows[(batch,)](
            lengths, compact_lengths, candidates, fallback, output, length_stride, num_warps=4,
        )
        return output

    select.candidate_counts = counts
    select.fallback_lengths = fallback_lengths
    return {"fused_threshold_exact_flashinfer": select}
