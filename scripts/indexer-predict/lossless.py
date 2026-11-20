"""Static-exponent lossless Keye Index Keys with sparse BF16 escapes."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _encode(bits, low: tl.constexpr, rows: tl.constexpr):
    exponent = ((bits >> 7) & 255).to(tl.int32)
    code = exponent - low
    extended = (code < 0) | (code >= 15)
    code = tl.where(extended, 15, code).to(tl.uint8)
    a, b = tl.split(tl.reshape(code, (rows, 32, 2)))
    payload = ((bits & 127) | ((bits >> 8) & 128)).to(tl.uint8)
    return payload, a | (b << 4), tl.sum(extended.to(tl.int32), 1) != 0


@triton.jit
def _initialize(raw, live_slots, mantissa, exponents, masks, active_counts, exception_counts,
                low: tl.constexpr):
    tile = tl.program_id(0)
    lane = tl.arange(0, 32)
    slot = tile * 32 + lane
    dim = tl.arange(0, 64)
    active = tl.load(live_slots + slot)
    key = tl.load(raw + slot[:, None] * 64 + dim[None, :], mask=active[:, None], other=0)
    bits = key.to(tl.uint16, bitcast=True).to(tl.uint32)
    ms, packed, escape = _encode(bits, low, 32)
    valid_escape = active & escape
    tl.store(mantissa + slot[:, None] * 64 + dim[None, :], ms, mask=active[:, None])
    tl.store(exponents + slot[:, None] * 32 + lane[None, :], packed, mask=active[:, None])
    mask = tl.sum(tl.where(valid_escape, 1 << lane, 0).to(tl.uint32), 0)
    tl.store(masks + tile, mask)
    tl.store(active_counts + tile, tl.sum(active.to(tl.int32), 0))
    tl.store(exception_counts + tile, tl.sum(valid_escape.to(tl.int32), 0))


@triton.jit
def _copy_exceptions(raw, masks, bases, out):
    tile = tl.program_id(0)
    bitmap = tl.load(masks + tile)
    if bitmap != 0:
        lane = tl.arange(0, 32)
        d = tl.arange(0, 64)
        valid = ((bitmap >> lane) & 1) != 0
        prior = bitmap & ((1 << lane.to(tl.uint32)) - 1)
        rank = tl.extra.cuda.libdevice.popc(prior.to(tl.int32)).to(tl.int32)
        src = tile * 32 + lane
        dst = tl.load(bases + tile) + rank
        value = tl.load(raw + src[:, None] * 64 + d[None, :], mask=valid[:, None], other=0)
        tl.store(out + dst[:, None] * 64 + d[None, :], value, mask=valid[:, None])


@triton.jit
def _append(current, slots, payload, exponents, masks, exceptions, bases, low: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    if slot >= 0:
        d = tl.arange(0, 64)
        pairs = tl.arange(0, 32)
        x = tl.load(current + row * 64 + d)[None, :]
        bits = x.to(tl.uint16, bitcast=True).to(tl.uint32)
        ms, packed, overflow = _encode(bits, low, 1)
        tl.store(payload + slot * 64 + d, tl.reshape(ms, (64,)))
        tl.store(exponents + slot * 32 + pairs, tl.reshape(packed, (32,)))
        if tl.sum(overflow.to(tl.int32), 0) != 0:
            tile = slot // 32
            lane = slot % 32
            bit = 1 << lane.to(tl.uint32)
            # Each request appends in ascending slot order. A replay of the
            # same append reuses the same rank rather than leaking a slot.
            previous = tl.atomic_or(masks + tile, bit, sem="relaxed")
            prior = previous & (bit - 1)
            rank = tl.extra.cuda.libdevice.popc(prior.to(tl.int32)).to(tl.int32)
            dst = tl.load(bases + tile) + rank
            tl.store(exceptions + dst * 64 + d, tl.reshape(x, (64,)))


@triton.jit
def _decode(mantissa, exponents, masks, exceptions, bases, slot, live, low: tl.constexpr,
            branched: tl.constexpr, cache_modifier: tl.constexpr):
    d = tl.arange(0, 64)
    p = tl.arange(0, 32)
    ms = tl.load(mantissa + slot[:, None] * 64 + d[None, :], mask=live[:, None], other=0,
                 cache_modifier=cache_modifier).to(tl.uint32)
    exp = tl.load(exponents + slot[:, None] * 32 + p[None, :], mask=live[:, None], other=0,
                  cache_modifier=cache_modifier).to(tl.uint32)
    code = tl.interleave(exp & 15, exp >> 4)
    bits = (ms & 127) | ((ms & 128) << 8) | ((code + low) << 7)
    key = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    group = slot // 32
    lane = slot % 32
    bitmap = tl.load(masks + group, mask=live, other=0)
    shifted_one = 1 << lane.to(tl.uint32)
    bad = live & ((bitmap & shifted_one) != 0)
    if not branched or tl.sum(bad.to(tl.int32), 0) != 0:
        prior = bitmap & (shifted_one - 1)
        rank = tl.extra.cuda.libdevice.popc(prior.to(tl.int32)).to(tl.int32)
        offset = tl.load(bases + group, mask=bad, other=0) + rank
        raw = tl.load(exceptions + offset[:, None] * 64 + d[None, :],
                      mask=bad[:, None], other=0)
        key = tl.where(bad[:, None], raw, key)
    return key


@triton.jit
def _restore(payload, exponents, masks, exceptions, bases, slots, out, rows: tl.constexpr, low: tl.constexpr):
    row = tl.program_id(0) * 32 + tl.arange(0, 32)
    slot = tl.load(slots + row, row < rows, other=0)
    value = _decode(payload, exponents, masks, exceptions, bases, slot, row < rows, low, True, "")
    d = tl.arange(0, 64)
    tl.store(out + row[:, None] * 64 + d[None, :], value, mask=row[:, None] < rows)


@triton.jit
def _score(q, gates, lengths, key_bases, mantissa, exponents, masks, exceptions, bases, out,
           stride_q: tl.constexpr, stride_h: tl.constexpr, stride_gate: tl.constexpr,
           width: tl.constexpr, block: tl.constexpr, low: tl.constexpr, scale: tl.constexpr,
           branched: tl.constexpr, cache_modifier: tl.constexpr):
    batch = tl.program_id(1)
    first = tl.program_id(0) * block
    length = tl.load(lengths + batch)
    j = first + tl.arange(0, block)
    if first < length:
        kbase = tl.multiple_of(tl.load(key_bases + batch), 128)
        key = _decode(mantissa, exponents, masks, exceptions, bases,
                      kbase + j, j < length, low, branched, cache_modifier)
        h = tl.arange(0, 16)
        d = tl.arange(0, 64)
        query = tl.load(q + batch * stride_q + h[:, None] * stride_h + d[None, :])
        w = tl.load(gates + batch * stride_gate + h).to(tl.float32)
        dot = tl.dot(query, tl.trans(key), allow_tf32=False).to(tl.bfloat16).to(tl.float32)
        value = tl.sum(tl.maximum(dot * scale, 0.0) * w[:, None], 0)
        tl.store(out + batch * width + j, tl.where(j < length, value, -float("inf")))
    else:
        tl.store(out + batch * width + j, -float("inf"))


class StaticExponentKeys:
    """Exact, request-local immutable history plus monotonic append slots.

    Each group of 32 slots reserves enough exception entries for every unused
    slot to be an escape. A request that outgrows its slab or is reassigned
    rebuilds its layout before the owner publishes new lengths.
    """

    def __init__(self, raw_slab, low=115):
        self.plan = raw_slab.plan
        self.low = low
        keys = raw_slab.resident
        rows = self.plan.capacity
        device = keys.device
        self.payload = torch.empty((rows, 64), dtype=torch.uint8, device=device)
        self.exponents = torch.empty((rows, 32), dtype=torch.uint8, device=device)
        self.masks = torch.empty(rows // 32, dtype=torch.uint32, device=device)

        active = torch.zeros(rows, dtype=torch.bool, device=device)
        for base, length in zip(self.plan.bases.tolist(), raw_slab.lengths.tolist(), strict=True):
            active[base:base + length] = True
        counts = torch.empty(rows // 32, dtype=torch.int32, device=device)
        escapes = torch.empty_like(counts)
        _initialize[(rows // 32,)](
            keys, active, self.payload, self.exponents, self.masks, counts, escapes,
            low, num_warps=4,
        )
        allocations = 32 - counts + escapes
        self.bases = allocations.cumsum(0, dtype=torch.int32) - allocations
        self.exception_count = int(escapes.sum())
        self.exceptions = torch.empty((int(allocations.sum()), 64), dtype=torch.bfloat16, device=device)
        _copy_exceptions[(rows // 32,)](
            keys, self.masks, self.bases, self.exceptions, num_warps=4,
        )

    @property
    def storage_bytes(self):
        return sum(t.nbytes for t in (
            self.payload, self.exponents, self.masks, self.bases, self.exceptions,
            self.plan.bases, self.plan.capacities,
        ))

    def append(self, keys, slots):
        _append[(keys.size(0),)](
            keys, slots, self.payload, self.exponents, self.masks, self.exceptions,
            self.bases, self.low, num_warps=4,
        )

    def restore(self, slots):
        out = torch.empty((slots.numel(), 64), dtype=torch.bfloat16, device=slots.device)
        _restore[(triton.cdiv(slots.numel(), 32),)](
            self.payload, self.exponents, self.masks, self.exceptions, self.bases,
            slots, out, slots.numel(), self.low, num_warps=4,
        )
        return out

    def prepare(self, lengths, scale, block, warps=4, branched=True, cache_modifier=""):
        width = triton.cdiv(self.plan.width, block) * block
        batch = lengths.numel()
        result = torch.empty((batch, width), dtype=torch.float32, device=lengths.device)

        def score(query, gate):
            _score[(width // block, batch)](
                query, gate, lengths, self.plan.bases, self.payload, self.exponents,
                self.masks, self.exceptions, self.bases, result,
                query.stride(0), query.stride(1), gate.stride(0), width, block,
                self.low, scale, branched, cache_modifier, num_warps=warps,
            )
            return result

        return score, result.nbytes
