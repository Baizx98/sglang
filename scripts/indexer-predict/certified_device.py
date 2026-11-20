"""FP32 screening of historical Keye scores using compact INT6/INT8 keys."""

import math

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.keye_topk.ampere_indexer import (
    keye_indexer_score_candidates, keye_indexer_score_paged,
)


@triton.jit
def _load_codes(codes, planes, slots, live, bits: tl.constexpr, block: tl.constexpr):
    d = tl.arange(0, 64)
    if bits == 8:
        return tl.load(codes + slots[:, None] * 64 + d, live[:, None], other=0)
    packed = tl.load(codes + slots[:, None] * 32 + tl.arange(0, 32),
                     live[:, None], other=0).to(tl.int32)
    nibble = tl.interleave(packed & 15, packed >> 4)
    tail = tl.load(planes + slots[:, None] * 16 + tl.arange(0, 16),
                   live[:, None], other=0).to(tl.int32)
    shifts = 2 * tl.arange(0, 4)
    low = tl.reshape((tail[:, :, None] >> shifts[None, None, :]) & 3, (block, 64))
    return ((nibble << 2) + low - 32).to(tl.int8)


@triton.jit
def _half_upper(value):
    half = value.to(tl.float16)
    word = half.to(tl.uint16, bitcast=True)
    up = word + (half.to(tl.float32) < value).to(tl.uint16)
    return up.to(tl.float16, bitcast=True)


@triton.jit
def _store_norms(raw_k, codes, planes, scales, norms, slot, live,
                 bits: tl.constexpr, block: tl.constexpr):
    scale = tl.load(scales + slot, live, other=0).to(tl.float32)
    key = _load_codes(codes, planes, slot, live, bits, block).to(tl.float32) * scale[:, None]
    residual = raw_k - key
    # Leave room for the preceding FP32 products and reduction, then round
    # the two nonnegative norms outward when storing them as FP16.
    margin: tl.constexpr = 1 + 2 ** -15
    key_norm = _half_upper(tl.sqrt(tl.sum(key * key, 1)) * margin)
    error = _half_upper(tl.sqrt(tl.sum(residual * residual, 1)) * margin)
    tl.store(norms + slot * 2, key_norm, live)
    tl.store(norms + slot * 2 + 1, error, live)


@triton.jit
def _key_norms(raw, pages, lengths, bases, codes, planes, scales, norms,
               page_stride: tl.constexpr, raw_stride: tl.constexpr,
               bits: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(1)
    ids = tl.program_id(0) * block + tl.arange(0, block)
    live = ids < tl.load(lengths + row)
    slot = tl.load(bases + row) + ids
    physical = tl.load(pages + row * page_stride + ids, live, other=-1)
    live &= physical >= 0
    raw_k = tl.load(raw + physical[:, None] * raw_stride + tl.arange(0, 64),
                    live[:, None], other=0).to(tl.float32)
    _store_norms(raw_k, codes, planes, scales, norms, slot, live, bits, block)


@triton.jit
def _append_norms(new_keys, slots, codes, planes, scales, norms, rows: tl.constexpr,
                  key_stride: tl.constexpr, bits: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0) * block + tl.arange(0, block)
    slot = tl.load(slots + row, row < rows, other=-1)
    live = slot >= 0
    raw = tl.load(new_keys + row[:, None] * key_stride + tl.arange(0, 64),
                  live[:, None], other=0).to(tl.float32)
    _store_norms(raw, codes, planes, scales, norms, slot, live, bits, block)


@triton.jit
def _query_norms(query, codes, scales, norms,
                 q_stride: tl.constexpr, head_stride: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 64)
    ix = row * 16 + head
    q = tl.load(query + row * q_stride + head * head_stride + d).to(tl.float32)
    code = tl.load(codes + ix * 64 + d).to(tl.float32)
    scale = tl.load(scales + ix)
    residual = q - code * scale
    margin: tl.constexpr = 1 + 2 ** -15
    tl.store(norms + ix * 2, tl.sqrt(tl.sum(q * q, 0)) * margin)
    tl.store(norms + ix * 2 + 1, tl.sqrt(tl.sum(residual * residual, 0)) * margin)


@triton.jit
def _bound_scores(q, q_scales, q_norms, k, planes, k_scales, k_norms,
                  gates, lengths, bases, approx, upper,
                  gate_stride: tl.constexpr, width: tl.constexpr, scale: tl.constexpr,
                  bits: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(1)
    start = tl.program_id(0) * block
    token = start + tl.arange(0, block)
    length = tl.load(lengths + row)
    out = row * width + token
    if start < length:
        valid = token < length
        slots = tl.multiple_of(tl.load(bases + row), 128) + token
        head = tl.arange(0, 16)
        hd = row * 16 + head
        q_data = tl.load(q + hd[:, None] * 64 + tl.arange(0, 64))
        key = _load_codes(k, planes, slots, valid, bits, block)
        dot = tl.dot(q_data, tl.trans(key), out_dtype=tl.int32).to(tl.float32)
        q_scale = tl.load(q_scales + hd)
        k_scale = tl.load(k_scales + slots, valid, other=0).to(tl.float32)
        center = dot * q_scale[:, None] * k_scale[None, :]
        weight = tl.load(gates + gate_stride * row + head).to(tl.float32)
        compact_score = tl.sum(tl.maximum(center.to(tl.bfloat16).to(tl.float32) * scale, 0.0)
                               * weight[:, None], 0)

        q_norm = tl.load(q_norms + hd * 2)
        q_error = tl.load(q_norms + hd * 2 + 1)
        k_norm = tl.load(k_norms + slots * 2, valid, other=0).to(tl.float32)
        k_error = tl.load(k_norms + slots * 2 + 1, valid, other=0).to(tl.float32)
        # BF16 dot rounding, FP32 scaling/reductions, and the error from
        # evaluating the center with FP32 instead of the FP64 audit.
        rho: tl.constexpr = 2 ** -8 + 2 ** -12
        margin = ((1 + rho) * q_norm)[:, None] * k_error[None, :]
        margin += (q_error + rho * q_norm)[:, None] * k_norm[None, :]
        lo = tl.maximum(center - margin, 0.0) * weight[:, None]
        hi = tl.maximum(center + margin, 0.0) * weight[:, None]
        error = tl.sum(tl.maximum(tl.abs(lo), tl.abs(hi)), 0) * (2 ** -15)
        bound = (tl.sum(tl.maximum(lo, hi), 0) + error) * scale
        # An unrepresentable norm or bound keeps all affected positions.
        bound = tl.where((bound == bound) & (tl.abs(bound) != float("inf")), bound, float("inf"))
        tl.store(approx + out, tl.where(valid, compact_score, -float("inf")))
        tl.store(upper + out, tl.where(valid, bound, -float("inf")))
    else:
        tl.store(approx + out, -float("inf"))
        tl.store(upper + out, -float("inf"))


@triton.jit
def _compact_ids(keep, counts, ids, width: tl.constexpr, cap: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(1)
    token = tl.program_id(0) * block + tl.arange(0, block)
    selected = tl.load(keep + row * width + token, token < width, other=0)
    count = selected.to(tl.int32)
    used = tl.sum(count, 0)
    if used > 0:
        base = tl.atomic_add(counts + row, used, sem="relaxed")
        pos = base + tl.cumsum(count, 0) - 1
        tl.store(ids + row * cap + pos, token, selected & (pos < cap))


@triton.jit
def _scatter_scores(ids, scores, out, count: tl.constexpr, width: tl.constexpr,
                    row_stride: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(1)
    offset = tl.program_id(0) * block + tl.arange(0, block)
    token = tl.load(ids + row * count + offset, offset < count, other=-1)
    value = tl.load(scores + row * row_stride + offset, offset < count, other=0)
    tl.store(out + row * width + token, value, (token >= 0) & (token < width))


class CertifiedRows:
    """Buffers for one fixed ragged history; current target Key stays external."""

    def __init__(self, state, keys, pages, lengths, scale):
        if state.bits not in (6, 8) or state.groups != 1 or state.fp4 or state.pair_pq:
            raise ValueError("the device bounds cover single-group INT6 or INT8")
        if scale <= 0 or pages.stride(1) != 1 or keys.stride(1) != 1:
            raise ValueError("expected positive scale and contiguous page or key rows")
        self.state = state
        self.keys, self.pages, self.lengths, self.scale = keys, pages, lengths, scale
        self.batch, self.width = pages.shape
        self.block = 128
        device = keys.device
        self.k_norms = torch.empty((state.plan.capacity, 2), dtype=torch.float16, device=device)
        self.q_norms = torch.empty((self.batch, 16, 2), dtype=torch.float32, device=device)
        self.tiled_width = triton.cdiv(self.width, self.block) * self.block
        self.approx = torch.empty((self.batch, self.tiled_width), dtype=torch.float32, device=device)
        self.upper = torch.empty_like(self.approx)
        self.final_scores = torch.empty((self.batch, self.width), dtype=torch.float32, device=device)
        _key_norms[(triton.cdiv(self.width, 32), self.batch)](
            keys, pages, lengths, state.plan.bases, state.codes, state.planes, state.scales, self.k_norms,
            pages.stride(0), keys.stride(0), state.bits, 32, num_warps=4, enable_fp_fusion=False,
        )

    def score_and_bound(self, q, gates):
        state = self.state
        state.encode_query(q)
        _query_norms[(self.batch, 16)](
            q, state.query_codes, state.query_scales, self.q_norms,
            q.stride(0), q.stride(1), num_warps=2, enable_fp_fusion=False,
        )
        _bound_scores[(self.tiled_width // self.block, self.batch)](
            state.query_codes, state.query_scales, self.q_norms, state.codes, state.planes,
            state.scales, self.k_norms, gates, self.lengths, state.plan.bases, self.approx, self.upper,
            gates.stride(0), self.tiled_width, self.scale, state.bits, self.block,
            num_warps=4, enable_fp_fusion=False,
        )
        return self.approx[:, :self.width], self.upper[:, :self.width]

    def append(self, keys, slots):
        state = self.state
        state.append(keys, slots)
        _append_norms[(triton.cdiv(keys.size(0), 32),)](
            keys, slots, state.codes, state.planes, state.scales, self.k_norms,
            keys.size(0), keys.stride(0), state.bits, 32, num_warps=4, enable_fp_fusion=False,
        )

    def exact_subset(self, q, gates, ids):
        return keye_indexer_score_candidates(
            q, self.keys, self.pages, ids, gates, self.lengths, self.scale,
        )

    def compact_ids(self, keep, capacity):
        if keep.shape != self.final_scores.shape or not keep.is_contiguous():
            raise ValueError("expected a contiguous history mask")
        counts = torch.zeros_like(self.lengths)
        ids = torch.full((self.batch, capacity), -1, dtype=torch.int32, device=keep.device)
        _compact_ids[(triton.cdiv(self.width, 256), self.batch)](
            keep, counts, ids, self.width, capacity, 256, num_warps=4,
        )
        overflow = counts > capacity
        return ids.masked_fill(overflow[:, None], -1), counts, overflow

    def scatter_scores(self, ids, values):
        count = ids.size(1)
        _scatter_scores[(triton.cdiv(count, 256), self.batch)](
            ids, values, self.final_scores, count, self.width, values.stride(0),
            256, num_warps=4,
        )

    def prepare(self, project, candidates, capacity_factor=4, reference_width=None):
        """Prepare full prediction with fixed-capacity device-side fallback."""
        k = candidates - 1
        seed_size = math.ceil(k * 1.25)
        capacity = min(self.width, math.ceil(k * capacity_factor))
        if capacity <= 0 or seed_size > int(self.lengths.min()):
            raise ValueError("history or compact buffer is too short for the chosen Top-M")
        current = self.lengths[:, None]
        logical = torch.arange(self.width, dtype=torch.int32, device=self.lengths.device)[None, :]
        live = logical < current
        reference_width = self.width if reference_width is None else reference_width

        def predict():
            q, gates = project()
            approx, upper = self.score_and_bound(q, gates)
            seed = approx.topk(seed_size).indices
            seed_scores = self.exact_subset(q, gates, seed)
            threshold = seed_scores.topk(k).values[:, -1:]
            present = live & (upper >= threshold)
            remaining = present.scatter(1, seed, False)
            ids, compact_counts, overflow = self.compact_ids(remaining, capacity)
            scores = self.exact_subset(q, gates, ids)

            self.final_scores.fill_(-float("inf"))
            self.scatter_scores(seed, seed_scores.masked_fill(~present.gather(1, seed), -float("inf")))
            self.scatter_scores(ids, scores)
            reference_scores = self.final_scores[:, :reference_width].contiguous()
            ranked = reference_scores.topk(k, sorted=True)
            cutoff = ranked.values[:, -1:]
            tied = ((reference_scores == cutoff).sum(1) != (ranked.values == cutoff).sum(1))
            fallback_rows = overflow | tied
            predict.compaction = (compact_counts, overflow, fallback_rows)
            fallback_lengths = self.lengths.masked_fill(~fallback_rows, 0)
            fallback_scores = keye_indexer_score_paged(
                q, self.keys, self.pages, gates, fallback_lengths, self.scale,
            )
            fallback = fallback_scores[:, :reference_width].contiguous().topk(k, sorted=True).indices
            older = torch.where(fallback_rows[:, None], fallback, ranked.indices)
            return torch.cat((current, older.int()), dim=1)

        predict.seed_size = seed_size
        return predict
