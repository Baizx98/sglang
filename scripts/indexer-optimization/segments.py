"""INT8 coarse scoring and BF16 refinement for the Keye decode indexer."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.keye_topk.ampere_indexer import (
    keye_indexer_score_candidates,
)

TOPK = 2048


@triton.jit
def _quantize(x, quantized, summary, lengths, pages, rows, stride_row, stride_d,
              page_batch, page_col, BLOCK_R: tl.constexpr, D: tl.constexpr,
              APPEND: tl.constexpr):
    row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    d = tl.arange(0, D)
    valid = row < rows
    if APPEND:
        length = tl.load(lengths + row, valid, other=0)
        slot = tl.load(pages + row * page_batch + (length - 1) * page_col,
                       valid & (length > 0), other=-1)
        valid &= slot >= 0
    else:
        slot = row
    value = tl.load(x + slot[:, None] * stride_row + d[None, :] * stride_d,
                    valid[:, None], other=0).to(tl.float32)
    peak = tl.max(tl.abs(value), axis=1)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(peak / 127.0, 1.0e-20))))
    code = tl.extra.cuda.libdevice.rint(value / scale[:, None])
    code = tl.maximum(-127.0, tl.minimum(127.0, code)).to(tl.int8)
    reconstructed = code.to(tl.float32) * scale[:, None]
    error = value - reconstructed
    norm = tl.sqrt(tl.sum(value * value, axis=1))
    reconstructed_norm = tl.sqrt(tl.sum(reconstructed * reconstructed, axis=1))
    error_norm = tl.sqrt(tl.sum(error * error, axis=1))
    tl.store(quantized + slot[:, None] * D + d[None, :], code, valid[:, None])
    tl.store(summary + slot * 4, scale, valid)
    tl.store(summary + slot * 4 + 1, norm, valid)
    tl.store(summary + slot * 4 + 2, reconstructed_norm, valid)
    tl.store(summary + slot * 4 + 3, error_norm, valid)


@triton.jit
def _score(
    query, key, quant_q, quant_k, query_summary, key_summary, pages, weights,
    lengths, threshold, scores, q_batch, q_head, q_dim, k_slot, k_dim,
    page_batch, page_col, w_batch, w_head, width, full_width,
    SCALE: tl.constexpr, N: tl.constexpr, MODE: tl.constexpr, SAMPLE_STEP: tl.constexpr,
    REDUCE_SEGMENT: tl.constexpr,
):
    batch = tl.program_id(0)
    column = tl.program_id(1) * N + tl.arange(0, N)
    head = tl.arange(0, 16)
    d = tl.arange(0, 64)
    logical = column * SAMPLE_STEP
    keep = (logical < tl.load(lengths + batch)) & (logical < full_width) & (column < width)
    slot = tl.load(pages + batch * page_batch + logical * page_col, keep, other=-1)
    keep &= slot >= 0
    slot = tl.maximum(slot, 0)
    gate = tl.load(weights + batch * w_batch + head * w_head).to(tl.float32)

    if MODE != "sample":
        q_row = batch * 16 + head
        q_code = tl.load(quant_q + q_row[:, None] * 64 + d[None, :])
        k_code = tl.load(quant_k + slot[:, None] * 64 + d[None, :],
                         keep[:, None], other=0)
        q_scale = tl.load(query_summary + q_row * 4)
        k_scale = tl.load(key_summary + slot * 4, keep, other=0)
        dot = tl.dot(q_code, tl.trans(k_code), out_dtype=tl.int32).to(tl.float32)
        estimate = dot * q_scale[:, None] * k_scale[None, :]

        if MODE == "bounded":
            q_norm = tl.load(query_summary + q_row * 4 + 1)
            q_error = tl.load(query_summary + q_row * 4 + 3)
            k_norm = tl.load(key_summary + slot * 4 + 1, keep, other=0)
            k_reconstructed = tl.load(key_summary + slot * 4 + 2, keep, other=0)
            k_error = tl.load(key_summary + slot * 4 + 3, keep, other=0)
            # q·k - qhat·khat = q·(k-khat) + (q-qhat)·khat.
            # Inflate residual norms and allow rounding in FP32 accumulation,
            # bound arithmetic, the BF16 cast, and FP32 gate/reduction.
            error = (
                q_norm[:, None] * k_error[None, :]
                + q_error[:, None] * k_reconstructed[None, :]
            ) * 1.0002 + q_norm[:, None] * k_norm[None, :] * 0.0002 + 1.0e-20
            limiting_dot = tl.where(gate[:, None] >= 0,
                                    estimate + error, estimate - error)
            upper_term = (
                tl.maximum(limiting_dot.to(tl.bfloat16).to(tl.float32) * SCALE, 0.0)
                * gate[:, None]
            )
            upper = tl.sum(upper_term, 0) + tl.sum(tl.abs(upper_term), 0) * 0.00002
            cutoff = tl.load(threshold + batch)
            finite_bound = (upper > -float("inf")) & (upper < float("inf"))
            keep &= (upper >= cutoff) | ~finite_bound

    if MODE == "approx":
        per_head = estimate.to(tl.bfloat16).to(tl.float32)
    else:
        q = tl.load(query + batch * q_batch + head[:, None] * q_head + d[None, :] * q_dim)
        k = tl.load(key + slot[:, None] * k_slot + d[None, :] * k_dim,
                    keep[:, None], other=0)
        per_head = tl.dot(q, tl.trans(k), allow_tf32=False).to(tl.bfloat16).to(tl.float32)

    score = tl.sum(tl.maximum(per_head * SCALE, 0.0) * gate[:, None], axis=0)
    score = tl.where(keep, score, -float("inf"))
    if REDUCE_SEGMENT:
        segment = tl.program_id(1)
        tl.store(scores + batch * tl.cdiv(width, N) + segment, tl.max(score, axis=0))
    else:
        tl.store(scores + batch * width + column, score, column < width)


def _update_summary(keys, quantized, summary):
    rows, dim = keys.shape
    _quantize[(triton.cdiv(rows, 32),)](
        keys, quantized, summary, summary, summary, rows, *keys.stride(),
        0, 0, 32, dim, False, num_warps=4,
    )


def _allocate_summary(keys):
    quantized = torch.empty(keys.shape, device=keys.device, dtype=torch.int8)
    summary = torch.empty((keys.shape[0], 4), device=keys.device, dtype=torch.float32)
    return quantized, summary


def _select(scores, width):
    valid_width = min(width, scores.shape[1])
    values, indices = scores.topk(valid_width, dim=1, sorted=False)
    indices = indices.int().masked_fill(~values.isfinite(), -1)
    if valid_width < width:
        indices = torch.nn.functional.pad(indices, (0, width - valid_width), value=-1)
    return indices


@torch.inference_mode()
def _prepare(q, key_cache, page_table, weights, lengths, scale):
    """Return graph-compatible Keye INT8+BFloat16 benchmark callables.

    Preparation is solely a function of K. To maintain the index online, run
    the quantization on new keys during append and refresh reused physical
    slots. The INT8 codes and summary require 80 bytes per physical key.
    """
    batch, heads, dim = q.shape
    if (heads, dim) != (16, 64) or page_table.shape[1] == 0:
        return {}, {}

    width = page_table.shape[1]
    k_quant, k_summary = _allocate_summary(key_cache)
    def precompute():
        _update_summary(key_cache, k_quant, k_summary)

    precompute()
    q_quant, q_summary = _allocate_summary(q.new_empty((batch * heads, dim)))

    def append():
        _quantize[(triton.cdiv(batch, 32),)](
            key_cache, k_quant, k_summary, lengths, page_table, batch,
            *key_cache.stride(), *page_table.stride(), 32, dim, True, num_warps=4,
        )

    append.summary = k_summary
    append.quantized = k_quant

    def update_query():
        # The projection output may have a batch gap for the per-head gates.
        _update_summary(q.reshape(-1, dim), q_quant, q_summary)

    def score(mode, width_out, step=1, cutoff=lengths, segment_size=0):
        block = segment_size or (64 if mode == "bounded" else 128)
        tiles = triton.cdiv(width_out, block)
        out = torch.empty((batch, tiles if segment_size else width_out),
                          device=q.device, dtype=torch.float32)
        _score[(batch, tiles)](
            q, key_cache, q_quant, k_quant, q_summary, k_summary, page_table, weights,
            lengths, cutoff, out, *q.stride(), *key_cache.stride(), *page_table.stride(),
            *weights.stride(), width_out, width, float(scale), block, mode, step,
            bool(segment_size), num_warps=4, num_stages=2,
        )
        return out

    def coarse_scores():
        update_query()
        return score("approx", width)

    def segment_scores(segment_size, fused, sample_size=0):
        if fused:
            update_query()
            return score("approx", width, segment_size=segment_size)
        segment_count = triton.cdiv(width, segment_size)
        if sample_size:
            step = segment_size // sample_size
            coarse = score("sample", triton.cdiv(width, step), step)
        else:
            coarse = coarse_scores()
        group_width = sample_size or segment_size
        padding = segment_count * group_width - coarse.shape[1]
        if padding:
            coarse = torch.nn.functional.pad(coarse, (0, padding), value=-float("inf"))
        return coarse.view(batch, segment_count, group_width).amax(dim=2)

    def refine(candidates):
        exact = keye_indexer_score_candidates(q, key_cache, page_table, candidates,
                                             weights, lengths, scale)
        positions = _select(exact, TOPK)
        return candidates.gather(1, positions.clamp_min(0).long()).masked_fill(positions < 0, -1)

    def approximate():
        return refine(_select(coarse_scores(), 4096))

    def segment_refinement(segment_size, budget_pct, fused=False, sample_size=0):
        segment_count = triton.cdiv(width, segment_size)
        retain = min(segment_count, max(triton.cdiv(TOPK, segment_size),
                                       triton.cdiv(segment_count * budget_pct, 100)))
        in_segment = torch.arange(segment_size, device=q.device, dtype=torch.int32)

        def select_segments():
            ranked_segments = segment_scores(segment_size, fused, sample_size)
            chosen = _select(ranked_segments, retain)
            logical = chosen[:, :, None] * segment_size + in_segment[None, None, :]
            logical = logical.masked_fill(logical >= width, -1)
            return logical.flatten(1)

        def run():
            return refine(select_segments())

        run.candidates = select_segments
        return run

    def bounded_scores():
        update_query()
        step = triton.cdiv(width, 8192)
        sample_width = triton.cdiv(width, step)
        sample = score("sample", sample_width, step)
        if sample_width >= TOPK:
            sample_topk = sample.topk(TOPK, dim=1, sorted=False).values
            cutoff = sample_topk.amin(dim=1)
        else:
            cutoff = torch.full((batch,), -float("inf"), device=q.device)
        return score("bounded", width, cutoff=cutoff)

    def bounded():
        return _select(bounded_scores(), TOPK)

    # Score output validates that all omitted keys fall below the exact cutoff.
    bounded.scores = bounded_scores
    approximate.scores = coarse_scores
    calls = {
        "int8_approx4096_bf16_refine": approximate,
        "int8_segment64_budget25": segment_refinement(64, 25),
        "int8_segment64_budget50": segment_refinement(64, 50),
        "int8_segment128_budget50": segment_refinement(128, 50),
        "int8_fused_segment64_budget50": segment_refinement(64, 50, fused=True),
        "int8_fused_segment128_budget50": segment_refinement(128, 50, fused=True),
        "bf16_sample4_segment64_budget50": segment_refinement(64, 50, sample_size=4),
        "int8_bounded_bf16_refine": bounded,
    }
    stages = {
        "int8_key_summary_precompute": precompute,
        "int8_key_summary_append": append,
        "int8_query_quantize": update_query,
        "int8_full_coarse_only": coarse_scores,
        "int8_segment128_coarse_only": lambda: segment_scores(128, False),
        "int8_fused_segment128_coarse_only": lambda: segment_scores(128, True),
    }
    return calls, stages


def prepare(q, key_cache, page_table, weights, lengths, scale):
    return _prepare(q, key_cache, page_table, weights, lengths, scale)[0]


def prepare_stages(q, key_cache, page_table, weights, lengths, scale):
    return _prepare(q, key_cache, page_table, weights, lengths, scale)[1]
