"""Top-M of FP16 index scores using a sharded radix-prefix histogram."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _radix_bin(value, trailing: tl.constexpr):
    raw = value.to(tl.uint16, bitcast=True)
    raw = tl.where((raw & 0x7fff) == 0, 0, raw).to(tl.int32)
    ordered = tl.where((raw & 0x8000) != 0, raw ^ 0xffff, raw ^ 0x8000)
    return ordered >> trailing


@triton.jit
def _accumulate_histogram(data, live, histogram, row, tile, nbin: tl.constexpr,
                          trailing: tl.constexpr, shards: tl.constexpr):
    bins = _radix_bin(data, trailing)
    counts = tl.histogram(bins, nbin, mask=live & (data == data))
    shard = tile % shards
    target = histogram + (row * shards + shard) * nbin + tl.arange(0, nbin)
    tl.atomic_add(target, counts, sem="relaxed", mask=counts > 0)


@triton.jit
def _histogram(scores, lengths, histogram,
               stride: tl.constexpr, width: tl.constexpr, nbin: tl.constexpr,
               trailing: tl.constexpr, shards: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(1)
    tile = tl.program_id(0)
    length = tl.load(lengths + row)
    if tile * block < length:
        token = tile * block + tl.arange(0, block)
        live = (token < length) & (token < width)
        value = tl.load(scores + row * stride + token, live, other=0)
        _accumulate_histogram(value, live, histogram, row, tile, nbin, trailing, shards)


@triton.jit
def _pick_prefix(histogram, lengths, pivots, observed, compact, fallback, offsets,
                 nbin: tl.constexpr, shards: tl.constexpr, k: tl.constexpr, cap: tl.constexpr):
    row = tl.program_id(0)
    b = tl.arange(0, nbin)
    shard = tl.arange(0, shards)
    t = tl.load(histogram + (row * shards + shard[:, None]) * nbin + b[None, :])
    histogram_row = tl.sum(t, 0)
    suffix = tl.cumsum(histogram_row, axis=0, reverse=True)
    pivot = tl.max(tl.where(suffix >= k, b, 0), 0)
    count = tl.sum(tl.where(b == pivot, suffix, 0), 0)
    length = tl.load(lengths + row)
    valid = (count >= k) & (count <= cap) & (tl.sum(histogram_row, 0) == length)
    tl.store(pivots + row, pivot)
    tl.store(observed + row, count)
    tl.store(compact + row, tl.where(valid, count, 0))
    tl.store(fallback + row, tl.where(valid, 0, length))
    tl.store(offsets + row, 0)


@triton.jit
def _compact(scores, lengths, pivots, compact, offsets, values, ids,
             score_stride: tl.constexpr, width: tl.constexpr,
             cap: tl.constexpr, trailing: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(1)
    start = tl.program_id(0) * block
    length = tl.load(lengths + row)
    if (start < length) & (tl.load(compact + row) > 0):
        token = start + tl.arange(0, block)
        live = (token < length) & (token < width)
        data = tl.load(scores + row * score_stride + token, live, other=0)
        take = live & (_radix_bin(data, trailing) >= tl.load(pivots + row))
        take_int = take.to(tl.int32)
        count = tl.sum(take_int, 0)
        if count > 0:
            first = tl.atomic_add(offsets + row, count, sem="relaxed")
            slot = first + tl.cumsum(take_int, axis=0) - 1
            tl.store(values + row * cap + slot, data, take & (slot < cap))
            tl.store(ids + row * cap + slot, token, take & (slot < cap))


class PrefixTopM:
    """Select by score after compaction, preserving a dense fallback per row."""

    def __init__(self, lengths, width, candidates, bits=8, capacity_factor=2, shards=16):
        if not 1 <= bits <= 11 or shards < 1 or shards & (shards - 1):
            raise ValueError("invalid radix width or shard count")
        self.lengths = lengths
        self.batch = lengths.numel()
        self.width = width
        self.k = candidates - 1
        self.trailing = 16 - bits
        self.bins = 1 << bits
        self.capacity = min(width, triton.cdiv(math.ceil(self.k * capacity_factor), 128) * 128)
        self.shards = shards
        self.block = 256
        if self.capacity < self.k or self.k <= 0:
            raise ValueError("insufficient compact capacity for the requested Top-M")
        device = lengths.device
        self.histogram = torch.empty((self.batch, shards, self.bins), dtype=torch.int32, device=device)
        self.candidates = torch.empty((self.batch, self.capacity), dtype=torch.float16, device=device)
        self.ids = torch.empty_like(self.candidates, dtype=torch.int32)
        self.pivot = torch.empty_like(lengths)
        self.observed = torch.empty_like(lengths)
        self.compact_lengths = torch.empty_like(lengths)
        self.fallback_lengths = torch.empty_like(lengths)
        self.offsets = torch.empty_like(lengths)
        self.zero = torch.zeros_like(lengths)

    @property
    def workspace_bytes(self):
        arrays = (self.histogram, self.candidates, self.ids, self.pivot, self.observed,
                  self.compact_lengths, self.fallback_lengths, self.offsets, self.zero)
        return sum(t.nbytes for t in arrays)

    def __call__(self, scores):
        self._check_scores(scores)
        grid = triton.cdiv(self.width, self.block), self.batch
        self.histogram.zero_()
        _histogram[grid](
            scores, self.lengths, self.histogram, scores.stride(0), self.width,
            self.bins, self.trailing, self.shards, self.block, num_warps=4,
        )
        return self.select_from_histogram(scores)

    def _check_scores(self, scores):
        if scores.shape != (self.batch, self.width) or scores.dtype != torch.float16 or scores.stride(1) != 1:
            raise ValueError("prefix selection requires its prepared contiguous FP16 score rows")

    def select_from_histogram(self, scores):
        from flashinfer import top_k_page_table_transform, top_k_ragged_transform

        self._check_scores(scores)
        grid = triton.cdiv(self.width, self.block), self.batch
        _pick_prefix[(self.batch,)](
            self.histogram, self.lengths, self.pivot, self.observed, self.compact_lengths,
            self.fallback_lengths, self.offsets, self.bins, self.shards, self.k, self.capacity,
            num_warps=4,
        )
        _compact[grid](
            scores, self.lengths, self.pivot, self.compact_lengths, self.offsets,
            self.candidates, self.ids, scores.stride(0), self.width, self.capacity, self.trailing,
            self.block, num_warps=4,
        )
        compact = top_k_page_table_transform(self.candidates, self.ids, self.compact_lengths, self.k)
        dense = top_k_ragged_transform(scores, self.zero, self.fallback_lengths, self.k)
        return torch.where(self.fallback_lengths[:, None] > 0, dense, compact)
