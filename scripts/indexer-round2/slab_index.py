"""Request-local exact Index Key storage and a directly addressed BF16 scorer."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from optimize import PackedKeyArena, _load_packed_key, _write_packed_row


@triton.jit
def _copy_history(
    source, table, length_ptr, bases, target,
    table_stride: tl.constexpr, block: tl.constexpr,
):
    row = tl.program_id(1)
    token = tl.program_id(0) * block + tl.arange(0, block)
    length = tl.load(length_ptr + row)
    base = tl.load(bases + row)
    dim = tl.arange(0, 64)
    slot = tl.load(table + row * table_stride + token, mask=token < length, other=-1)
    live = (token < length) & (slot >= 0)
    key = tl.load(source + slot[:, None] * 64 + dim[None, :], mask=live[:, None], other=0)
    tl.store(target + (base + token[:, None]) * 64 + dim[None, :], key, mask=live[:, None])


@triton.jit
def _append_key(
    projected, bases, caps, lengths,
    cache, exponents, headers, exceptions, next_exception, overflow,
    projected_stride: tl.constexpr, exception_capacity: tl.constexpr, packed: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(lengths + row)
    capacity = tl.load(caps + row)
    if (length > 0) & (length <= capacity):
        position = tl.load(bases + row) + length - 1
        dim = tl.arange(0, 64)
        key = tl.load(projected + row * projected_stride + dim)
        if packed:
            _write_packed_row(
                row, position, key, cache, exponents, headers, exceptions,
                next_exception, overflow, exception_capacity,
            )
        else:
            tl.store(cache + position * 64 + dim, key)
            tl.store(overflow + row, 0)
    else:
        tl.store(overflow + row, (length != 0).to(tl.int32))


@triton.jit
def _score_slab(
    query, gates, bases, caps, lengths, cache, exponents, headers, exceptions, out,
    q_batch_stride: tl.constexpr, q_head_stride: tl.constexpr,
    w_batch_stride: tl.constexpr, width: tl.constexpr, scale: tl.constexpr,
    block: tl.constexpr, packed: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    logical = tile * block + tl.arange(0, block)
    length = tl.minimum(tl.load(lengths + row), tl.load(caps + row))
    if tile * block < length:
        slot = tl.multiple_of(tl.load(bases + row), 128) + logical
        dim = tl.arange(0, 64)
        head = tl.arange(0, 16)
        live = logical < length
        if packed:
            key = _load_packed_key(cache, exponents, headers, exceptions, slot, live, dim, False)
        else:
            key = tl.load(cache + slot[:, None] * 64 + dim[None, :], mask=live[:, None], other=0)
        q = tl.load(query + row * q_batch_stride + head[:, None] * q_head_stride + dim[None, :])
        w = tl.load(gates + row * w_batch_stride + head).to(tl.float32)
        dot = tl.dot(q, tl.trans(key), allow_tf32=False).to(tl.bfloat16).to(tl.float32)
        logits = tl.sum(tl.maximum(dot * scale, 0.0) * w[:, None], 0)
        tl.store(out + row * width + logical, tl.where(live, logits, -float("inf")))
    else:
        tl.store(out + row * width + logical, -float("inf"))


@dataclass
class Plan:
    bases: torch.Tensor
    capacities: torch.Tensor
    capacity: int
    width: int
    live_slots: int

    @classmethod
    def from_lengths(cls, lengths, growth):
        if growth < 1:
            raise ValueError("each request must reserve at least one future token")
        live = lengths.tolist()
        caps = [triton.cdiv(n + growth, 128) * 128 for n in live]
        cap = sum(caps)
        if cap * 64 >= 2**31:
            raise ValueError("prototype uses 32-bit word addresses")
        bases = list(itertools.accumulate(caps, initial=0))[:-1]
        return cls(torch.tensor(bases, dtype=torch.int32, device=lengths.device),
                   torch.tensor(caps, dtype=torch.int32, device=lengths.device),
                   cap, max(caps), sum(live))


class IndexSlab:
    """One Index Key copy with reserved room for each live request.

    The owner only publishes lengths within the corresponding fixed reservation.
    Reusing a row resets its length, followed by sequential repopulation.
    Overflow is observable. Growth beyond the reservation requires rebuilding.
    """

    def __init__(self, plan, raw_cache, page_table, lengths, *, packed, exception_reserve):
        self.plan = plan
        self.lengths = lengths
        self.packed = packed
        self.overflow = torch.empty_like(lengths)
        resident = torch.zeros((plan.capacity, 64), dtype=torch.bfloat16, device=raw_cache.device)
        _copy_history[(triton.cdiv(page_table.shape[1], 128), lengths.numel())](
            raw_cache, page_table, lengths, plan.bases, resident, page_table.stride(0),
            128, num_warps=4,
        )
        if packed:
            self.arena = PackedKeyArena.from_cache(resident, reserve=exception_reserve)
            self.arena.reserve_insertions(exception_reserve)
        else:
            self.resident = resident

    @property
    def storage_bytes(self):
        if self.packed:
            nbytes = self.arena.keys.storage_bytes
        else:
            nbytes = self.resident.nbytes
        return nbytes + self.plan.bases.nbytes + self.plan.capacities.nbytes

    def buffers(self):
        if self.packed:
            keys = self.arena.keys
            return (*keys, self.arena.next_exception), keys.exceptions.shape[0]
        return (self.resident,) * 5, 0

    def append(self, key, lengths=None):
        lengths = self.lengths if lengths is None else lengths
        buffers, capacity = self.buffers()
        _append_key[(lengths.numel(),)](
            key, self.plan.bases, self.plan.capacities, lengths,
            *buffers, self.overflow, key.stride(0), capacity, self.packed, num_warps=4,
        )
        return self.overflow

    def prepare(self, project, selector, scale, block, lengths=None):
        lengths = self.lengths if lengths is None else lengths
        batch = lengths.numel()
        width = triton.cdiv(self.plan.width, block) * block
        scores = torch.full((batch, width), -float("inf"), dtype=torch.float32,
                            device=self.lengths.device)

        def append_and_score(q, k, gates):
            self.append(k, lengths)
            buffers, _ = self.buffers()
            _score_slab[(triton.cdiv(width, block), batch)](
                q, gates, self.plan.bases, self.plan.capacities, lengths,
                *buffers[:4], scores, q.stride(0), q.stride(1), gates.stride(0),
                width, scale, block, self.packed, num_warps=4,
            )
            return scores

        def run():
            return selector(append_and_score(*project()))

        run.slab = self
        run.last_overflow = self.overflow
        run.scores = append_and_score
        return run
