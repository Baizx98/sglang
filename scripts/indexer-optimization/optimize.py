"""Exact Keye 16 x 64 indexer scoring and FlashInfer selection prototypes."""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl


K = 2048
_MAX_EXCEPTION_SLOTS = (1 << 23) - 1


class EncodedKeys(NamedTuple):
    """A lossless, query-independent encoding of the physical BF16 key cache."""

    mantissa_sign: torch.Tensor
    exponent_pairs: torch.Tensor
    headers: torch.Tensor
    exceptions: torch.Tensor

    @property
    def storage_bytes(self):
        return sum(
            value.numel() * value.element_size()
            for value in (self.mantissa_sign, self.exponent_pairs, self.headers, self.exceptions)
        )


@triton.jit
def _pack_values(bits, block_rows: tl.constexpr):
    exp = (bits >> 7) & 255
    min_exp = tl.minimum(tl.min(tl.where(exp == 0, 255, exp), 1), 254)
    max_exp = tl.max(exp, 1)
    # Code zero preserves exponent zero, including signed zeros and subnormals.
    delta = tl.where(exp == 0, 0, exp - min_exp[:, None] + 1)
    exception = max_exp.to(tl.int32) - min_exp.to(tl.int32) > 14
    hi0, hi1 = tl.split(tl.reshape(delta, (block_rows, 32, 2)))
    exps = (hi0 | (hi1 << 4)).to(tl.uint8)
    ms = ((bits & 127) | ((bits >> 8) & 128)).to(tl.uint8)
    return ms, exps, min_exp.to(tl.int32), exception


@triton.jit
def _pack_key_rows(
    cache_ptr,
    mantissa_sign_ptr,
    exponent_pairs_ptr,
    header_ptr,
    rows: tl.constexpr,
    block_rows: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    if rows * 64 >= 2**31:
        row = row.to(tl.int64)
    col = tl.arange(0, 64)
    col_pair = tl.arange(0, 32)
    address = row[:, None] * 64 + col[None, :]
    bits = tl.load(cache_ptr + address, row[:, None] < rows, other=0).to(
        tl.uint16, bitcast=True
    ).to(tl.uint32)
    ms, exps, base, exception = _pack_values(bits, block_rows)
    keep = (row < rows) & (~exception)
    tl.store(mantissa_sign_ptr + address, ms, keep[:, None])
    tl.store(
        exponent_pairs_ptr + row[:, None] * 32 + col_pair[None, :], exps, keep[:, None]
    )
    tl.store(header_ptr + row, tl.where(exception, -1, base), row < rows)


@torch.no_grad()
def encode_keys(
    cache: torch.Tensor, *, tail_reserve: int = 0, slot_capacity: int | None = None,
) -> EncodedKeys:
    """Pack physical cache slots once, then compact the exact exceptions."""
    if cache.dtype != torch.bfloat16 or cache.ndim != 2 or cache.size(1) != 64:
        raise ValueError("key compression requires BF16 cache [physical_slots, 64]")
    if not cache.is_contiguous() or not cache.is_cuda:
        raise ValueError("key compression requires a contiguous CUDA buffer")
    rows = cache.size(0)
    capacity = rows if slot_capacity is None else slot_capacity
    if capacity < rows or tail_reserve < 0:
        raise ValueError("physical capacity and exception reserve must cover the initial cache")
    options = {"device": cache.device}
    mantissa_sign = torch.empty((capacity, 64), dtype=torch.uint8, **options)
    exponent_pairs = torch.empty((capacity, 32), dtype=torch.uint8, **options)
    headers = torch.empty((capacity,), dtype=torch.int32, **options)
    headers[rows:].zero_()
    if rows:
        _pack_key_rows[(triton.cdiv(rows, 32),)](
            cache, mantissa_sign, exponent_pairs, headers, rows, 32, num_warps=4
        )
    exception_slots = torch.nonzero(headers[:rows] < 0, as_tuple=True)[0]
    count = exception_slots.numel()
    if count + tail_reserve > _MAX_EXCEPTION_SLOTS:
        raise ValueError("packed cache has too many full-BF16 exception rows")
    exceptions = torch.empty((count + tail_reserve, 64), dtype=cache.dtype, **options)
    if count:
        torch.index_select(cache, 0, exception_slots, out=exceptions[:count])
        reservations = (torch.arange(count, dtype=torch.int32, **options) + 1) << 8
        headers[exception_slots] = reservations | -(1 << 31)
    return EncodedKeys(mantissa_sign, exponent_pairs, headers, exceptions)


@triton.jit
def _exception_row(header):
    return ((header.to(tl.uint32) >> 8) & 0x7FFFFF).to(tl.int32) - 1


@triton.jit
def _write_packed_row(
    row, slot, values, ms_ptr, exponent_pairs_ptr, headers_ptr,
    exceptions_ptr, next_exception_ptr, overflow_ptr,
    exception_capacity: tl.constexpr,
):
    dims = tl.arange(0, 64)
    pairs = tl.arange(0, 32)
    bits = tl.expand_dims(values.to(tl.uint16, bitcast=True).to(tl.uint32), 0)
    ms, exps, base, exception = _pack_values(bits, 1)
    old = tl.load(headers_ptr + slot)
    # Bits 8–30 remember a physical slot's exception reservation through
    # retirement and normal overwrites. Bit 31 selects the currently live form.
    reservation = old & 0x7FFFFF00
    if tl.sum(exception.to(tl.int32), 0):
        if reservation:
            exception_id = _exception_row(old)
        else:
            exception_id = tl.atomic_add(next_exception_ptr, 1, sem="relaxed")
            if exception_id >= exception_capacity:
                # Leave the old key intact and retain a usable arena for retry.
                tl.atomic_add(next_exception_ptr, -1, sem="relaxed")
        overflow = exception_id >= exception_capacity
        tl.store(overflow_ptr + row, overflow.to(tl.int32))
        if not overflow:
            tl.store(exceptions_ptr + exception_id * 64 + dims, values)
            tl.store(headers_ptr + slot, ((exception_id + 1) << 8) | -2147483648)
    else:
        tl.store(ms_ptr + slot * 64 + dims, tl.reshape(ms, (64,)))
        tl.store(exponent_pairs_ptr + slot * 32 + pairs, tl.reshape(exps, (32,)))
        tl.store(headers_ptr + slot, reservation | tl.sum(base, 0))
        tl.store(overflow_ptr + row, 0)


@triton.jit
def _pack_tail_keys(
    values_ptr, table_ptr, lengths_ptr, ms_ptr, exponent_pairs_ptr, headers_ptr,
    exceptions_ptr, next_exception_ptr, overflow_ptr,
    table_batch_stride: tl.constexpr, table_token_stride: tl.constexpr,
    length_stride: tl.constexpr, exception_capacity: tl.constexpr,
    slot_capacity: tl.constexpr, width: tl.constexpr, wide_address: tl.constexpr,
    projected_values: tl.constexpr,
    values_stride: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(lengths_ptr + row * length_stride)
    slot = tl.load(
        table_ptr + row * table_batch_stride + (length - 1) * table_token_stride,
        mask=(length > 0) & (length <= width),
        other=-1,
    )
    if (slot >= 0) & (slot < slot_capacity):
        if wide_address:
            slot = slot.to(tl.int64)
        dims = tl.arange(0, 64)
        if projected_values:
            value_offset = row * values_stride
        else:
            value_offset = slot * 64
        values = tl.load(values_ptr + value_offset + dims)
        _write_packed_row(
            row, slot, values, ms_ptr, exponent_pairs_ptr, headers_ptr, exceptions_ptr,
            next_exception_ptr, overflow_ptr, exception_capacity,
        )
    else:
        invalid = (length != 0) | (slot != -1)
        tl.store(overflow_ptr + row, invalid.to(tl.int32))


@triton.jit
def _append_packed_keys(
    values_ptr, slots_ptr, ms_ptr, exponent_pairs_ptr, headers_ptr,
    exceptions_ptr, next_exception_ptr, overflow_ptr,
    value_stride: tl.constexpr, slot_stride: tl.constexpr,
    slot_capacity: tl.constexpr, exception_capacity: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row * slot_stride)
    if (slot >= 0) & (slot < slot_capacity):
        if slot_capacity * 64 >= 2**31:
            slot = slot.to(tl.int64)
        values = tl.load(values_ptr + row * value_stride + tl.arange(0, 64))
        _write_packed_row(
            row, slot, values, ms_ptr, exponent_pairs_ptr, headers_ptr, exceptions_ptr,
            next_exception_ptr, overflow_ptr, exception_capacity,
        )
    else:
        tl.store(overflow_ptr + row, (slot != -1).to(tl.int32))


class PackedKeyArena:
    """Physical key slots with a growable, persistent arena of exceptional rows.

    The caller owns the page table and writes each physical slot at most once
    per call. Inactive lanes use slot -1. Graph users reserve the upper bound
    of new slots touched during replay before capture and use write_reserved.
    """

    def __init__(self, keys: EncodedKeys, reserve: int):
        self.keys = keys
        self.next_exception = torch.full(
            (), keys.exceptions.size(0) - reserve, dtype=torch.int32, device=keys.headers.device,
        )
        self._available = reserve

    @classmethod
    def from_cache(cls, cache, *, reserve=0, slot_capacity=None):
        return cls(
            encode_keys(cache, tail_reserve=reserve, slot_capacity=slot_capacity), reserve
        )

    def reserve_insertions(self, count: int):
        """Ensure capacity even if all upcoming writes need new exception rows."""
        if count < 0:
            raise ValueError("insertion reserve must be nonnegative")
        if count > self._available:
            used = int(self.next_exception.item())
            capacity = self.keys.exceptions.size(0)
            if used + count > _MAX_EXCEPTION_SLOTS:
                raise ValueError("packed cache exception index capacity exceeded")
            if used + count > capacity:
                grown = min(_MAX_EXCEPTION_SLOTS, max(used + count, capacity * 2))
                extended = self.keys.exceptions.new_empty((grown, 64))
                extended[:used].copy_(self.keys.exceptions[:used])
                self.keys = self.keys._replace(exceptions=extended)
                capacity = grown
            self._available = capacity - used
        self._available -= count

    def write_reserved(self, values, slots, overflow):
        if slots.numel():
            _append_packed_keys[(slots.numel(),)](
                values, slots, *self.keys, self.next_exception, overflow,
                values.stride(0), slots.stride(0), self.keys.headers.numel(),
                self.keys.exceptions.size(0), num_warps=4,
            )
        return overflow

    def write(self, values, slots):
        """Reserve a worst-case bound and asynchronously insert projected BF16."""
        if (
            values.shape != (slots.numel(), 64)
            or values.dtype != torch.bfloat16
            or values.stride(1) != 1
            or slots.ndim != 1
            or slots.dtype not in (torch.int32, torch.int64)
            or values.device != self.keys.headers.device
            or slots.device != values.device
        ):
            raise ValueError("expected device-local BF16 values [N,64] and integer slots [N]")
        self.reserve_insertions(slots.numel())
        overflow = torch.empty_like(slots, dtype=torch.int32)
        return self.write_reserved(values, slots, overflow)


def _prepare_packed_tail(key_cache, page_table, lengths, encoded, tail_reserve):
    device = key_cache.device
    next_exception = torch.full(
        (), encoded.exceptions.size(0) - tail_reserve, dtype=torch.int32, device=device
    )
    overflow = torch.zeros_like(lengths, dtype=torch.int32)

    def refresh_tail():
        _pack_tail_keys[(lengths.numel(),)](
            key_cache, page_table, lengths, *encoded, next_exception, overflow,
            *page_table.stride(), lengths.stride(0), encoded.exceptions.size(0),
            encoded.headers.numel(), page_table.size(1), key_cache.numel() >= 2**31,
            False, 64, num_warps=4,
        )
        return overflow

    return refresh_tail


@triton.jit
def _load_packed_key(
    ms_ptr,
    exponent_pairs_ptr,
    headers_ptr,
    exceptions_ptr,
    slot,
    live,
    dim,
    wide_address: tl.constexpr,
):
    header = tl.load(headers_ptr + slot, live, other=0)
    normal = live & (header >= 0)
    mask = normal[:, None]
    ms = tl.load(
        ms_ptr + slot[:, None] * 64 + dim[None, :], mask, other=0,
        cache_modifier=".cg",
    ).to(tl.uint32)
    pair = tl.arange(0, 32)
    pairs = tl.load(
        exponent_pairs_ptr + slot[:, None] * 32 + pair[None, :], mask, other=0,
        cache_modifier=".cg",
    ).to(tl.uint32)
    delta = tl.interleave(pairs & 15, pairs >> 4)
    exp = tl.where(delta == 0, 0, delta + (header[:, None] & 255) - 1)
    bits = ((ms & 127) | ((ms & 128) << 8) | (exp << 7)).to(tl.uint16)
    key = bits.to(tl.bfloat16, bitcast=True)
    exceptional = live & (header < 0)
    if tl.sum(exceptional.to(tl.int32), 0) > 0:
        exception_row = _exception_row(header)
        if wide_address:
            exception_row = exception_row.to(tl.int64)
        original = tl.load(
            exceptions_ptr + exception_row[:, None] * 64 + dim[None, :],
            exceptional[:, None],
            other=0,
        )
        key = tl.where(exceptional[:, None], original, key)
    return key


@triton.jit
def _unpack_key_rows(
    ms_ptr,
    exponent_pairs_ptr,
    headers_ptr,
    exceptions_ptr,
    output_ptr,
    rows: tl.constexpr,
    block_rows: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    if rows * 64 >= 2**31:
        row = row.to(tl.int64)
    col = tl.arange(0, 64)
    values = _load_packed_key(
        ms_ptr, exponent_pairs_ptr, headers_ptr, exceptions_ptr,
        row, row < rows, col, rows * 64 >= 2**31,
    )
    tl.store(output_ptr + row[:, None] * 64 + col[None, :], values, row[:, None] < rows)


def decode_keys(keys: EncodedKeys, out: torch.Tensor | None = None) -> torch.Tensor:
    """Materialize exact BF16 into a caller-provided or newly allocated buffer."""
    if out is None:
        out = torch.empty_like(keys.mantissa_sign, dtype=torch.bfloat16)
    elif (
        out.shape != keys.mantissa_sign.shape
        or out.device != keys.mantissa_sign.device
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        raise ValueError("output must be a matching contiguous BF16 buffer")
    rows = out.size(0)
    if rows:
        _unpack_key_rows[(triton.cdiv(rows, 32),)](
            keys.mantissa_sign, keys.exponent_pairs, keys.headers, keys.exceptions,
            out, rows, 32, num_warps=4,
        )
    return out


@triton.jit
def _describe_runs(
    table_ptr,
    lengths_ptr,
    runs_ptr,
    table_batch_stride: tl.constexpr,
    table_token_stride: tl.constexpr,
    length_stride: tl.constexpr,
    max_len: tl.constexpr,
    tiles: tl.constexpr,
    block_n: tl.constexpr,
    tail_only: tl.constexpr,
):
    """Describe a tile by its first physical slot when all live slots are consecutive."""
    batch = tl.program_id(1)
    length = tl.load(lengths_ptr + batch * length_stride)
    if tail_only:
        tile = tl.maximum(length - 1, 0) // block_n
    else:
        tile = tl.program_id(0)
    offset = tl.arange(0, block_n)
    logical = tile * block_n + offset
    base_ptr = table_ptr + batch * table_batch_stride
    live = (logical < length) & (logical < max_len)
    slots = tl.load(base_ptr + logical * table_token_stride, mask=live, other=-1)
    first = tl.min(tl.where(offset == 0, slots, 0x7FFFFFFF), 0)
    consecutive = (first >= 0) & tl.min(
        ((~live) | (slots == first + offset)).to(tl.int32), 0
    ).to(tl.int1)
    tl.store(runs_ptr + batch * tiles + tile, tl.where(consecutive, first, -1))


@triton.jit
def _score_tile(
    query_ptr,
    cache_ptr,
    table_ptr,
    weights_ptr,
    lengths_ptr,
    runs_ptr,
    scores_ptr,
    exponent_pairs_ptr,
    headers_ptr,
    exceptions_ptr,
    query_batch_stride: tl.constexpr,
    query_head_stride: tl.constexpr,
    query_dim_stride: tl.constexpr,
    cache_slot_stride: tl.constexpr,
    cache_dim_stride: tl.constexpr,
    table_batch_stride: tl.constexpr,
    table_token_stride: tl.constexpr,
    weight_batch_stride: tl.constexpr,
    weight_head_stride: tl.constexpr,
    length_stride: tl.constexpr,
    max_len: tl.constexpr,
    score_stride: tl.constexpr,
    tiles: tl.constexpr,
    scale: tl.constexpr,
    block_n: tl.constexpr,
    use_runs: tl.constexpr,
    reverse_grid: tl.constexpr,
    skip_short_rows: tl.constexpr,
    wide_address: tl.constexpr,
    cache_modifier: tl.constexpr,
    packed: tl.constexpr,
):
    if reverse_grid:
        tile = tl.program_id(0)
        batch = tl.program_id(1)
    else:
        batch = tl.program_id(0)
        tile = tl.program_id(1)
    length = tl.load(lengths_ptr + batch * length_stride)
    if (tile * block_n < length) & ((not skip_short_rows) | (length > 2048)):
        tile_offset = tl.arange(0, block_n)
        logical = tile * block_n + tile_offset
        head = tl.arange(0, 16)
        dim = tl.arange(0, 64)
        live = (logical < length) & (logical < max_len)
        table_row = table_ptr + batch * table_batch_stride

        if packed:
            slot = tl.load(
                table_row + logical * table_token_stride, mask=live, other=-1
            )
            live &= slot >= 0
            if wide_address:
                slot = slot.to(tl.int64)
            key = _load_packed_key(
                cache_ptr, exponent_pairs_ptr, headers_ptr, exceptions_ptr,
                slot, live, dim, wide_address,
            )
        else:
            if use_runs:
                start = tl.load(runs_ptr + batch * tiles + tile)
            else:
                start = -1
            if start >= 0:
                if wide_address:
                    start = start.to(tl.int64)
                # A proven run permits a directly vectorized load in row-major order.
                key = tl.load(
                    cache_ptr
                    + start * cache_slot_stride
                    + tile_offset[:, None] * cache_slot_stride
                    + dim[None, :] * cache_dim_stride,
                    mask=live[:, None],
                    other=0,
                    cache_modifier=cache_modifier,
                )
            else:
                slot = tl.load(
                    table_row + logical * table_token_stride, mask=live, other=-1
                )
                live &= slot >= 0
                if wide_address:
                    slot = slot.to(tl.int64)
                key = tl.load(
                    cache_ptr + slot[:, None] * cache_slot_stride + dim[None, :] * cache_dim_stride,
                    mask=live[:, None],
                    other=0,
                    cache_modifier=cache_modifier,
                )
        query = tl.load(
            query_ptr
            + batch * query_batch_stride
            + head[:, None] * query_head_stride
            + dim[None, :] * query_dim_stride
        )
        gates = tl.load(
            weights_ptr + batch * weight_batch_stride + head * weight_head_stride
        ).to(tl.float32)

        dot = tl.dot(query, tl.trans(key), allow_tf32=False)
        dot = dot.to(tl.bfloat16).to(tl.float32)
        scores = tl.sum(tl.maximum(dot * scale, 0.0) * gates[:, None], 0)

        tl.store(
            scores_ptr + batch * score_stride + logical,
            tl.where(live, scores, -float("inf")),
            mask=logical < max_len,
        )


def _selector(lengths: torch.Tensor):
    from flashinfer.topk import top_k_ragged_transform

    contiguous_lengths = lengths.contiguous()
    offsets = torch.zeros_like(contiguous_lengths)
    return lambda scores: top_k_ragged_transform(scores, offsets, contiguous_lengths, K)


def _run_descriptor(page_table, lengths, block, runs, *, tail_only):
    batch, width = page_table.shape
    tiles = triton.cdiv(width, block)
    if tiles:
        _describe_runs[((1 if tail_only else tiles), batch)](
            page_table,
            lengths,
            runs,
            *page_table.stride(),
            lengths.stride(0),
            width,
            tiles,
            block,
            tail_only,
            num_warps=4,
        )
    return runs


def prepare_stages(q, key_cache, page_table, weights, lengths, scale):
    """Expose descriptor refresh and encoded-cache insertion benchmark stages."""
    block = 128
    batch, width = page_table.shape
    runs = torch.empty(
        (batch, triton.cdiv(width, block)), dtype=torch.int32, device=page_table.device
    )
    _run_descriptor(page_table, lengths, block, runs, tail_only=False)

    def full():
        return _run_descriptor(page_table, lengths, block, runs, tail_only=False)

    def tail():
        return _run_descriptor(page_table, lengths, block, runs, tail_only=True)

    refresh_packed_tail = None

    def packed_tail():
        nonlocal refresh_packed_tail
        if refresh_packed_tail is None:
            encoded = encode_keys(key_cache, tail_reserve=batch)
            refresh_packed_tail = _prepare_packed_tail(
                key_cache, page_table, lengths, encoded, batch
            )
        return refresh_packed_tail()

    return {
        "run_describe_full_n128": full,
        "run_describe_tail_n128": tail,
        "pack12_tail_write": packed_tail,
    }


def _make_scoring_variant(
    q,
    key_cache,
    page_table,
    weights,
    lengths,
    scale,
    block,
    warps,
    *,
    reverse_grid=True,
    use_runs=False,
    refresh_runs="full",
    skip_short_rows=True,
    cache_modifier="",
    encoded=None,
):
    batch, width = page_table.shape
    tiles = triton.cdiv(width, block)
    packed = encoded is not None
    if packed:
        cache_stride = (64, 1)
        wide_address = encoded.headers.numel() * 64 >= 2**31
    else:
        cache_stride = key_cache.stride()
        wide_address = key_cache.size(0) * cache_stride[0] >= 2**31
    # FlashInfer accepts the logical row stride as the width of its contiguous input.
    padded_width = triton.cdiv(max(width, K), block) * block
    scores = torch.empty((batch, padded_width), dtype=torch.float32, device=q.device)
    runs = torch.empty((batch, tiles), dtype=torch.int32, device=q.device) if use_runs else page_table
    dense_buffers = (key_cache, lengths, lengths, key_cache) if not packed else None

    if use_runs and refresh_runs != "full":
        _run_descriptor(page_table, lengths, block, runs, tail_only=False)

    def score(query=q, gates=weights, current_keys=encoded):
        if tiles:
            cache_buffers = current_keys if packed else dense_buffers
            if use_runs and refresh_runs:
                _run_descriptor(
                    page_table, lengths, block, runs, tail_only=(refresh_runs == "tail")
                )
            grid = (tiles, batch) if reverse_grid else (batch, tiles)
            _score_tile[grid](
                query, cache_buffers[0], page_table, gates, lengths, runs, scores,
                *cache_buffers[1:],
                *query.stride(), *cache_stride, *page_table.stride(), *gates.stride(),
                lengths.stride(0), width, padded_width, tiles, float(scale), block,
                use_runs, reverse_grid, skip_short_rows, wide_address,
                cache_modifier, packed, num_warps=warps,
            )
        return scores

    return score


def prepare(q, key_cache, page_table, weights, lengths, scale):
    """Return benchmark functions producing int32 logical top-2048 indices."""
    if q.shape[1:] != (16, 64) or key_cache.shape[1] != 64:
        raise ValueError("prototype requires the Keye 16-head, 64-dimension indexer")
    select = _selector(lengths)

    variants = {
        "opt_qk_n128w4_flashinfer": (128, 4, {}),
        "opt_qk_n64w4_flashinfer": (64, 4, {}),
        "opt_qk_n128w8_flashinfer": (128, 8, {}),
        "opt_qk_n256w4_flashinfer": (256, 4, {}),
        "opt_qk_n64w2_flashinfer": (64, 2, {}),
        "opt_qk_n128w2_flashinfer": (128, 2, {}),
        "opt_qk_n128w4cg_flashinfer": (128, 4, {"cache_modifier": ".cg"}),
        "opt_runs_n128w4_flashinfer": (128, 4, {"use_runs": True}),
        "opt_tail_runs_n128w4_flashinfer": (
            128, 4, {"use_runs": True, "refresh_runs": "tail"}
        ),
        "opt_static_runs_n128w4_flashinfer": (
            128, 4, {"use_runs": True, "refresh_runs": False}
        ),
    }
    functions = {}
    for name, (block, warps, kwargs) in variants.items():
        score = _make_scoring_variant(
            q, key_cache, page_table, weights, lengths, scale, block, warps, **kwargs
        )
        functions[name] = lambda score=score: select(score())
    packed_cache = None

    def make_packed_variant(block, warps, selector, *, refresh=False):
        score = None
        refresh_tail = None

        def apply():
            nonlocal packed_cache, score, refresh_tail
            if score is None:
                if packed_cache is None:
                    packed_cache = encode_keys(key_cache, tail_reserve=q.size(0))
                score = _make_scoring_variant(
                    q, key_cache, page_table, weights, lengths, scale, block, warps,
                    encoded=packed_cache,
                )
                if refresh:
                    refresh_tail = _prepare_packed_tail(
                        key_cache, page_table, lengths, packed_cache, q.size(0)
                    )
            if refresh:
                refresh_tail()
            return selector(score())

        return apply

    for block, warps in ((64, 4), (128, 4), (256, 4)):
        functions[f"opt_pack12_n{block}w{warps}_flashinfer"] = make_packed_variant(
            block, warps, select
        )
    functions["opt_pack12_tail_n128w4_flashinfer"] = make_packed_variant(
        128, 4, select, refresh=True
    )

    try:
        from vllm_baseline import prepare_selector
    except ModuleNotFoundError as error:
        if error.name != "vllm_baseline":
            raise
    else:
        if width := page_table.size(1):
            # Lazily initialize the optional CUDA extension when selected.
            def make_vllm_variant(score):
                selector = None

                def apply():
                    nonlocal selector
                    if selector is None:
                        selector = prepare_selector(lengths, width)
                    return selector(score())

                return apply

            for block, warps in ((64, 4), (128, 4), (256, 4)):
                score = _make_scoring_variant(
                    q, key_cache, page_table, weights, lengths, scale, block, warps
                )
                functions[f"opt_qk_n{block}w{warps}_vllm"] = make_vllm_variant(score)
            for mode, prefix in (
                ("full", "runs"), ("tail", "tail_runs"), (False, "static_runs")
            ):
                score = _make_scoring_variant(
                    q, key_cache, page_table, weights, lengths, scale, 128, 4,
                    use_runs=True, refresh_runs=mode,
                )
                functions[f"opt_{prefix}_n128w4_vllm"] = make_vllm_variant(score)

            pack_selector = None

            def select_packed(scores):
                nonlocal pack_selector
                if pack_selector is None:
                    pack_selector = prepare_selector(lengths, width)
                return pack_selector(scores)

            for block, warps in ((64, 4), (128, 4), (256, 4)):
                functions[f"opt_pack12_n{block}w{warps}_vllm"] = make_packed_variant(
                    block, warps, select_packed
                )
            functions["opt_pack12_tail_n128w4_vllm"] = make_packed_variant(
                128, 4, select_packed, refresh=True
            )
    return functions


def prepare_packed_full(
    q, key_cache, page_table, weights, lengths, scale, project, selector,
    *, reserve=None, block=128, warps=4,
):
    """Prepare the projected-key write plus full score-and-select path.

    Reserve covers new physical slots that can be written during graph replay.
    A fixed-input replay needs only one slot per request. A multi-step graph
    needs room for each distinct appended slot of its decode horizon.
    """
    batch = lengths.numel()
    reserve = batch if reserve is None else reserve
    if reserve < batch:
        raise ValueError("reserve must cover the writes in at least one decode step")
    arena = PackedKeyArena.from_cache(key_cache, reserve=reserve)
    arena.reserve_insertions(reserve)
    score = _make_scoring_variant(
        q, None, page_table, weights, lengths, scale, block, warps,
        encoded=arena.keys,
    )
    overflow = torch.empty_like(lengths, dtype=torch.int32)

    def append_projected(current_key):
        if batch:
            keys = arena.keys
            _pack_tail_keys[(batch,)](
                current_key, page_table, lengths, *keys, arena.next_exception, overflow,
                *page_table.stride(), lengths.stride(0), keys.exceptions.size(0),
                keys.headers.numel(), page_table.size(1), keys.headers.numel() * 64 >= 2**31,
                True, current_key.stride(0),
                num_warps=4,
            )
        return overflow

    def call():
        current_q, current_key, current_gates = project()
        append_projected(current_key)
        return selector(score(current_q, current_gates, arena.keys))

    call.arena = arena
    call.last_overflow = overflow
    call.append_projected = append_projected
    return call


def prepare_ablation(
    q, key_cache, page_table, weights, lengths, scale, selector,
    *, block=128, warps=4,
):
    """Isolate the effect of fusing decompression into the BF16 scorer."""
    keys = encode_keys(key_cache)
    materialized = torch.empty_like(key_cache)
    raw_score = _make_scoring_variant(
        q, key_cache, page_table, weights, lengths, scale, block, warps,
    )
    fused_score = _make_scoring_variant(
        q, None, page_table, weights, lengths, scale, block, warps, encoded=keys,
    )
    separate_score = _make_scoring_variant(
        q, materialized, page_table, weights, lengths, scale, block, warps,
    )

    def materialize_score():
        decode_keys(keys, materialized)
        return separate_score()

    return {
        "ablate_raw_bf16_same_qk_topk": lambda: selector(raw_score()),
        "ablate_pack12_fused_same_qk_topk": lambda: selector(fused_score()),
        "ablate_pack12_materialized_same_qk_topk": lambda: selector(materialize_score()),
    }


def prepare_dense_full(
    q, key_cache, page_table, weights, lengths, scale, project, selector,
    *, block=128, warps=4,
):
    """Use the same projected Q, gates and BF16 dot with the fixed-replay raw cache."""
    slots = page_table.gather(1, lengths.long()[:, None] - 1).view(-1).long()
    score = _make_scoring_variant(
        q, key_cache, page_table, weights, lengths, scale, block, warps,
    )

    def call():
        current_q, current_key, current_gates = project()
        key_cache.index_copy_(0, slots, current_key)
        return selector(score(current_q, current_gates))

    return call
