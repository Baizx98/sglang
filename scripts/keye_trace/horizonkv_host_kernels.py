"""GPU-driven pinned-Host KV gather and bounded, request-local cache kernels.

No selected indices are copied to Python. The cache policy follows HiSparse's
hit-above-new-miss recency refinement; this is an Ampere mechanism adaptation,
not the original fused HiSparse implementation.
"""

import torch
from contextlib import nullcontext
import triton
import triton.language as tl


class HostPointer:
    """Keep a pinned allocation alive while exposing its UVA pointer to Triton."""

    def __init__(self, tensor):
        if not tensor.is_pinned():
            raise ValueError("Host KV must be allocated in pinned memory")
        self.tensor = tensor
        self.dtype = tensor.dtype

    def data_ptr(self):
        return self.tensor.data_ptr()


@triton.jit
def _gather(K, V, IDX, OUTK, OUTV, LENGTHS,
            MAXLEN: tl.constexpr, TOPK: tl.constexpr, WIDTH: tl.constexpr,
            NT: tl.constexpr, ND: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1) * NT + tl.arange(0, NT)
    d = tl.arange(0, ND)
    idx = tl.load(IDX + b * TOPK + n, n < TOPK, -1)
    length = tl.load(LENGTHS + b)
    valid = (n < TOPK) & (idx >= 0) & (idx < length)
    src = (b * MAXLEN + tl.maximum(idx, 0))[:, None] * WIDTH + d[None, :]
    dst = (b * TOPK + n)[:, None] * WIDTH + d[None, :]
    mask = valid[:, None] & (d[None, :] < WIDTH)
    kval = tl.load(K + src, mask, 0)
    vval = tl.load(V + src, mask, 0)
    tl.store(OUTK + dst, kval, (n[:, None] < TOPK) & (d[None, :] < WIDTH))
    tl.store(OUTV + dst, vval, (n[:, None] < TOPK) & (d[None, :] < WIDTH))


def gather(k, v, indices, outk, outv, lengths, maxlen):
    width = outk.shape[-2] * outk.shape[-1]
    _gather[(indices.shape[0], triton.cdiv(indices.shape[1], 8))](
        k, v, indices, outk, outv, lengths, maxlen, indices.shape[1], width,
        NT=8, ND=triton.next_power_of_2(width), num_warps=4)


@triton.jit
def _resident_gather(HK, HV, CK, CV, MAP, IDX, OUTK, OUTV, LENGTHS,
                     PREFMAP, PK, PV, MAXLEN: tl.constexpr, CAP: tl.constexpr,
                     TOPK: tl.constexpr, WIDTH: tl.constexpr, HAS_PREF: tl.constexpr,
                     NT: tl.constexpr, ND: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1) * NT + tl.arange(0, NT)
    d = tl.arange(0, ND)
    idx = tl.load(IDX + b * TOPK + n, n < TOPK, -1)
    valid = (n < TOPK) & (idx >= 0) & (idx < tl.load(LENGTHS + b))
    slot = tl.load(MAP + b * MAXLEN + tl.maximum(idx, 0), valid, -1)
    hit = valid & (slot >= 0)
    prefslot = tl.full((NT,), -1, tl.int32)
    if HAS_PREF:
        prefslot = tl.load(PREFMAP + b * MAXLEN + tl.maximum(idx, 0), valid & ~hit, -1)
    pref = valid & ~hit & (prefslot >= 0)
    miss = valid & ~hit & ~pref
    hd = (b * MAXLEN + tl.maximum(idx, 0))[:, None] * WIDTH + d[None, :]
    cd = (b * CAP + tl.maximum(slot, 0))[:, None] * WIDTH + d[None, :]
    pd = (b * TOPK + tl.maximum(prefslot, 0))[:, None] * WIDTH + d[None, :]
    dm = d[None, :] < WIDTH
    k = tl.load(CK + cd, hit[:, None] & dm, 0) + tl.load(HK + hd, miss[:, None] & dm, 0)
    v = tl.load(CV + cd, hit[:, None] & dm, 0) + tl.load(HV + hd, miss[:, None] & dm, 0)
    if HAS_PREF:
        k += tl.load(PK + pd, pref[:, None] & dm, 0)
        v += tl.load(PV + pd, pref[:, None] & dm, 0)
    out = (b * TOPK + n)[:, None] * WIDTH + d[None, :]
    tl.store(OUTK + out, k, (n[:, None] < TOPK) & dm)
    tl.store(OUTV + out, v, (n[:, None] < TOPK) & dm)


def resident_gather(cache, hostk, hostv, indices, outk, outv, lengths, maxlen, prefetch=None):
    """Diagnostic frozen-residency gather; no online replacement or metadata sort."""
    prefmap, pk, pv = prefetch if prefetch else (cache.map, cache.k, cache.v)
    width = outk.shape[-2] * outk.shape[-1]
    _resident_gather[(indices.shape[0], triton.cdiv(indices.shape[1], 8))](
        hostk, hostv, cache.k, cache.v, cache.map, indices, outk, outv, lengths,
        prefmap, pk, pv, maxlen, cache.capacity, indices.shape[1], width,
        prefetch is not None, NT=8, ND=triton.next_power_of_2(width), num_warps=4)


@triton.jit
def _write_current(K, V, NEWK, NEWV, LENGTHS,
                   MAXLEN: tl.constexpr, WIDTH: tl.constexpr,
                   SK: tl.constexpr, SV: tl.constexpr, ND: tl.constexpr):
    b = tl.program_id(0)
    d = tl.arange(0, ND)
    index = b * MAXLEN + tl.load(LENGTHS + b) - 1
    kval = tl.load(NEWK + b * SK + d, d < WIDTH, 0)
    vval = tl.load(NEWV + b * SV + d, d < WIDTH, 0)
    tl.store(K + index * WIDTH + d, kval, d < WIDTH)
    tl.store(V + index * WIDTH + d, vval, d < WIDTH)


def write_current(k, v, newk, newv, lengths, maxlen):
    width = newk.shape[-1]
    _write_current[(newk.shape[0],)](k, v, newk, newv, lengths, maxlen, width,
                                   newk.stride(0), newv.stride(0),
                                   ND=triton.next_power_of_2(width), num_warps=4)


@triton.jit
def _prefetch(K, V, IDX, MAP, PREFMAP, OUTK, OUTV, LENGTHS, COUNTS,
              MAXLEN: tl.constexpr, TOPK: tl.constexpr, WIDTH: tl.constexpr,
              CTA: tl.constexpr, NT: tl.constexpr, ND: tl.constexpr):
    b = tl.program_id(0)
    d = tl.arange(0, ND)
    count = tl.full((), 0, tl.int32)
    for start in range(tl.program_id(1) * NT, TOPK, CTA * NT):
        n = start + tl.arange(0, NT)
        idx = tl.load(IDX + b * TOPK + n, n < TOPK, -1)
        length = tl.load(LENGTHS + b)
        # The current token does not exist when an earlier layer starts.
        valid = (n < TOPK) & (idx >= 0) & (idx < length - 1)
        slot = tl.load(MAP + b * MAXLEN + tl.maximum(idx, 0), valid, -1)
        wanted = valid & (slot < 0)
        src = (b * MAXLEN + tl.maximum(idx, 0))[:, None] * WIDTH + d[None, :]
        dst = (b * TOPK + n)[:, None] * WIDTH + d[None, :]
        mask = wanted[:, None] & (d[None, :] < WIDTH)
        kval = tl.load(K + src, mask, 0)
        vval = tl.load(V + src, mask, 0)
        tl.store(OUTK + dst, kval, mask)
        tl.store(OUTV + dst, vval, mask)
        tl.store(PREFMAP + b * MAXLEN + tl.maximum(idx, 0), n, wanted)
        count += tl.sum(wanted.to(tl.int32))
    tl.atomic_add(COUNTS + b, count)


def prefetch(k, v, indices, resident, prefmap, outk, outv, lengths, counts, maxlen):
    prefmap.fill_(-1)
    counts.zero_()
    width = outk.shape[-2] * outk.shape[-1]
    # Bound copy CTAs so that earlier-layer compute can actually make progress.
    _prefetch[(indices.shape[0], 4)](
        k, v, indices, resident, prefmap, outk, outv, lengths, counts, maxlen,
        indices.shape[1], width, CTA=4, NT=8,
        ND=triton.next_power_of_2(width), num_warps=4)


@triton.jit
def _pref_current(PREFMAP, PREFK, PREFV, NEWK, NEWV, LENGTHS,
                  MAXLEN: tl.constexpr, TOPK: tl.constexpr, WIDTH: tl.constexpr,
                  SK: tl.constexpr, SV: tl.constexpr, ND: tl.constexpr):
    b = tl.program_id(0)
    d = tl.arange(0, ND)
    idx = tl.load(LENGTHS + b) - 1
    slot = tl.load(PREFMAP + b * MAXLEN + idx)
    kval = tl.load(NEWK + b * SK + d, d < WIDTH, 0)
    vval = tl.load(NEWV + b * SV + d, d < WIDTH, 0)
    tl.store(PREFK + (b * TOPK + slot) * WIDTH + d, kval, (slot >= 0) & (d < WIDTH))
    tl.store(PREFV + (b * TOPK + slot) * WIDTH + d, vval, (slot >= 0) & (d < WIDTH))


def update_prefetched_current(prefmap, prefk, prefv, newk, newv, lengths, maxlen, topk):
    width = newk.shape[-1]
    _pref_current[(newk.shape[0],)](prefmap, prefk, prefv, newk, newv, lengths,
                                   maxlen, topk, width, newk.stride(0), newv.stride(0),
                                   triton.next_power_of_2(width))


@triton.jit
def _touch(MAP, AGE, IDX, SLOTS, MISSRANK, COUNTS, LENGTHS,
           MAXLEN: tl.constexpr, CAP: tl.constexpr, TOPK: tl.constexpr,
           EPOCH: tl.constexpr, N: tl.constexpr):
    b = tl.program_id(0)
    n = tl.arange(0, N)
    idx = tl.load(IDX + b * TOPK + n, n < TOPK, -1)
    length = tl.load(LENGTHS + b)
    valid = (n < TOPK) & (idx >= 0) & (idx < length)
    slot = tl.load(MAP + b * MAXLEN + tl.maximum(idx, 0), valid, -1)
    hit = valid & (slot >= 0)
    miss = valid & ~hit
    tl.store(AGE + b * CAP + tl.maximum(slot, 0), EPOCH + TOPK + n, hit)
    rank = tl.cumsum(miss.to(tl.int32)) - 1
    tl.store(SLOTS + b * TOPK + n, slot, n < TOPK)
    tl.store(MISSRANK + b * TOPK + n, rank, n < TOPK)
    tl.store(COUNTS + b, tl.sum(miss.to(tl.int32)))


@triton.jit
def _cache_gather(K, V, CACHEK, CACHEV, MAP, TAG, AGE, IDX, SLOTS,
                  MISSRANK, VICTIM, OUTK, OUTV, LENGTHS,
                  PREFMAP, PREFK, PREFV,
                  MAXLEN: tl.constexpr, CAP: tl.constexpr, TOPK: tl.constexpr,
                  WIDTH: tl.constexpr, EPOCH: tl.constexpr,
                  USE_PREF: tl.constexpr, NT: tl.constexpr, ND: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1) * NT + tl.arange(0, NT)
    d = tl.arange(0, ND)
    idx = tl.load(IDX + b * TOPK + n, n < TOPK, -1)
    length = tl.load(LENGTHS + b)
    valid = (n < TOPK) & (idx >= 0) & (idx < length)
    oldslot = tl.load(SLOTS + b * TOPK + n, n < TOPK, -1)
    miss = valid & (oldslot < 0)
    rank = tl.load(MISSRANK + b * TOPK + n, n < TOPK, 0)
    victim = tl.load(VICTIM + b * CAP + tl.maximum(rank, 0), miss, 0)
    slot = tl.where(miss, victim, tl.maximum(oldslot, 0))
    oldtag = tl.load(TAG + b * CAP + slot, miss, -1)
    tl.store(MAP + b * MAXLEN + tl.maximum(oldtag, 0), -1, miss & (oldtag >= 0))
    tl.store(MAP + b * MAXLEN + tl.maximum(idx, 0), slot, miss)
    tl.store(TAG + b * CAP + slot, idx, miss)
    tl.store(AGE + b * CAP + slot, EPOCH + n, miss)
    prefslot = tl.full((NT,), -1, tl.int32)
    if USE_PREF:
        prefslot = tl.load(PREFMAP + b * MAXLEN + tl.maximum(idx, 0), miss, -1)
    from_pref = miss & (prefslot >= 0)
    from_host = miss & ~from_pref
    src = (b * MAXLEN + tl.maximum(idx, 0))[:, None] * WIDTH + d[None, :]
    cached = (b * CAP + slot)[:, None] * WIDTH + d[None, :]
    pref = (b * TOPK + tl.maximum(prefslot, 0))[:, None] * WIDTH + d[None, :]
    m = d[None, :] < WIDTH
    kval = tl.load(K + src, from_host[:, None] & m, 0)
    vval = tl.load(V + src, from_host[:, None] & m, 0)
    if USE_PREF:
        kval += tl.load(PREFK + pref, from_pref[:, None] & m, 0)
        vval += tl.load(PREFV + pref, from_pref[:, None] & m, 0)
    kval += tl.load(CACHEK + cached, (valid & ~miss)[:, None] & m, 0)
    vval += tl.load(CACHEV + cached, (valid & ~miss)[:, None] & m, 0)
    tl.store(CACHEK + cached, kval, miss[:, None] & m)
    tl.store(CACHEV + cached, vval, miss[:, None] & m)
    dst = (b * TOPK + n)[:, None] * WIDTH + d[None, :]
    tl.store(OUTK + dst, kval, (n[:, None] < TOPK) & m)
    tl.store(OUTV + dst, vval, (n[:, None] < TOPK) & m)


class DeviceLRU:
    def __init__(self, batch, maxlen, capacity, topk, heads=4, dim=128):
        self.batch, self.maxlen, self.capacity, self.topk = batch, maxlen, capacity, topk
        self.map = torch.full((batch, maxlen), -1, device="cuda", dtype=torch.int32)
        self.tag = torch.full((batch, capacity), -1, device="cuda", dtype=torch.int32)
        self.age = torch.full((batch, capacity), -1, device="cuda", dtype=torch.int32)
        self.k = torch.empty((batch, capacity, heads, dim), device="cuda", dtype=torch.bfloat16)
        self.v = torch.empty_like(self.k)
        self.slots = torch.empty((batch, topk), device="cuda", dtype=torch.int32)
        self.rank = torch.empty_like(self.slots)
        self.misses = torch.empty(batch, device="cuda", dtype=torch.int32)

    def resolve(self, hostk, hostv, indices, outk, outv, lengths, step, prefetch=None):
        epoch = step * (2 * self.topk + 1)
        span = getattr(self, "profile_span", None)
        with span("kv_diff") if span else nullcontext():
            _touch[(self.batch,)](self.map, self.age, indices, self.slots, self.rank,
                                 self.misses, lengths, self.maxlen, self.capacity,
                                 self.topk, epoch, N=triton.next_power_of_2(self.topk))
            victims = torch.argsort(self.age, dim=-1, stable=True)
        if prefetch is None:
            prefmap, prefk, prefv = self.map, self.k, self.v
        else:
            prefmap, prefk, prefv = prefetch
        width = self.k.shape[-2] * self.k.shape[-1]
        with span("kv_fetch") if span else nullcontext():
            _cache_gather[(self.batch, triton.cdiv(self.topk, 8))](
                hostk, hostv, self.k, self.v, self.map, self.tag, self.age,
                indices, self.slots, self.rank, victims, outk, outv, lengths,
                prefmap, prefk, prefv, self.maxlen, self.capacity, self.topk, width,
                epoch, prefetch is not None, NT=8, ND=triton.next_power_of_2(width), num_warps=4)
        return self.misses

    def clear(self):
        self.map.fill_(-1)
        self.tag.fill_(-1)
        self.age.fill_(-1)


@triton.jit
def _score_prefetch(Q, IK, TABLE, W, LENGTHS, SCORES, THRESHOLD,
                    HK, HV, MAP, PREFMAP, PREFK, PREFV, COUNT,
                    QB: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
                    WB: tl.constexpr, WH: tl.constexpr,
                    MAXLEN: tl.constexpr, TOPK: tl.constexpr,
                    SCALE: tl.constexpr, WIDTH: tl.constexpr,
                    BN: tl.constexpr, DKV: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    h = tl.arange(0, 16)
    d = tl.arange(0, 64)
    length = tl.load(LENGTHS + b)
    valid = n < length
    physical = tl.load(TABLE + b * MAXLEN + n, n < MAXLEN, 0).to(tl.int64)
    query = tl.load(Q + b * QB + h[:, None] * QH + d[None, :] * QD)
    keys = tl.load(IK + physical[:, None] * 64 + d[None, :], valid[:, None], 0)
    per_head = tl.dot(query, tl.trans(keys), allow_tf32=False).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(W + b * WB + h * WH).to(tl.float32)
    score = tl.sum(tl.maximum(per_head * SCALE, 0.) * gate[:, None], axis=0)
    score = tl.where(valid, score, -float("inf"))
    tl.store(SCORES + b * MAXLEN + n, score, n < MAXLEN)
    resident = tl.load(MAP + b * MAXLEN + n, n < MAXLEN, -1)
    wanted = valid & (n < length - 1) & (resident < 0) & (score > tl.load(THRESHOLD + b))
    count = tl.sum(wanted.to(tl.int32))
    base = tl.atomic_add(COUNT + b, count)
    slot = base + tl.cumsum(wanted.to(tl.int32)) - 1
    take = wanted & (slot < TOPK)
    kvd = tl.arange(0, DKV)
    src = (b * MAXLEN + n)[:, None] * WIDTH + kvd[None, :]
    dst = (b * TOPK + tl.maximum(slot, 0))[:, None] * WIDTH + kvd[None, :]
    mask = take[:, None] & (kvd[None, :] < WIDTH)
    kval = tl.load(HK + src, mask, 0)
    vval = tl.load(HV + src, mask, 0)
    tl.store(PREFK + dst, kval, mask)
    tl.store(PREFV + dst, vval, mask)
    tl.store(PREFMAP + b * MAXLEN + n, slot, take)


def score_prefetch(query, keys, table, weights, lengths, scale, threshold,
                   hostk, hostv, resident, prefmap, prefk, prefv, counts):
    """Ampere port of EMA threshold recall + exact final selection.

    This uses tile-local fused loads, not ECHO's Hopper TMA/warp pipeline.
    A bounded k-slot scratch pool prevents unbounded speculative residency.
    """
    prefmap.fill_(-1)
    counts.zero_()
    batch, maxlen = table.shape
    topk = prefk.shape[1]
    scores = torch.empty(batch, maxlen, device="cuda", dtype=torch.float32)
    width = prefk.shape[-2] * prefk.shape[-1]
    _score_prefetch[(batch, triton.cdiv(maxlen, 32))](
        query, keys, table, weights, lengths, scores, threshold, hostk, hostv,
        resident, prefmap, prefk, prefv, counts,
        *query.stride(), *weights.stride(), maxlen, topk, scale, width,
        BN=32, DKV=triton.next_power_of_2(width), num_warps=8)
    return scores
