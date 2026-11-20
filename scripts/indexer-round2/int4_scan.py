"""Approximate full scan of 4-bit codes with optional BF16 candidate scoring."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from optimize import _load_packed_key
from slab_index import _append_key


@triton.jit
def _encode_k(values):
    scale = (tl.maximum(tl.max(tl.abs(values), 1), 1.0e-20) / 7.0).to(tl.bfloat16)
    code = tl.extra.cuda.libdevice.rint(values / scale.to(tl.float32)[:, None])
    code = (tl.minimum(tl.maximum(code, -7), 7).to(tl.int32) + 8).to(tl.uint8)
    first, second = tl.split(tl.reshape(code, (code.shape[0], 32, 2)))
    return (first | (second << 4)).to(tl.uint8), scale


@triton.jit
def _encode_history(raw, codes, scales, rows: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0) * block + tl.arange(0, block)
    dim = tl.arange(0, 64)
    values = tl.load(raw + row[:, None] * 64 + dim[None, :],
                     mask=row[:, None] < rows, other=0).to(tl.float32)
    packed, scale = _encode_k(values)
    offsets = row[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(codes + offsets, packed, mask=row[:, None] < rows)
    tl.store(scales + row, scale, mask=row < rows)


@triton.jit
def _prepare_step(
    q, key, bases, capacities, lengths, q_codes, q_scales, k_codes, k_scales,
    q_batch_stride: tl.constexpr, q_head_stride: tl.constexpr, k_stride: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, 64)
    if head < 16:
        q_values = tl.load(q + row * q_batch_stride + head * q_head_stride + dim).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(q_values), 0), 1.0e-20) / 127.0
        code = tl.extra.cuda.libdevice.rint(q_values / scale)
        code = tl.minimum(tl.maximum(code, -127), 127).to(tl.int8)
        index = row * 16 + head
        tl.store(q_codes + index * 64 + dim, code)
        tl.store(q_scales + index, scale)
    else:
        length = tl.load(lengths + row)
        capacity = tl.load(capacities + row)
        if (length > 0) & (length <= capacity):
            slot = tl.load(bases + row) + length - 1
            key_values = tl.load(key + row * k_stride + dim).to(tl.float32)
            code, scale = _encode_k(key_values[None, :])
            tl.store(k_codes + slot * 32 + tl.arange(0, 32), tl.reshape(code, (32,)))
            tl.store(k_scales + slot, tl.sum(scale.to(tl.float32), 0))


@triton.jit
def _scan(
    q_codes, q_scales, k_codes, k_scales, gates, bases, lengths, out,
    gate_stride: tl.constexpr, width: tl.constexpr, scale: tl.constexpr,
    block: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    logical = tile * block + tl.arange(0, block)
    length = tl.load(lengths + row)
    if tile * block < length:
        head = tl.arange(0, 16)
        dim = tl.arange(0, 64)
        col = tl.arange(0, 32)
        slot = tl.multiple_of(tl.load(bases + row), 128) + logical
        valid = logical < length
        packed = tl.load(k_codes + slot[:, None] * 32 + col[None, :],
                         mask=valid[:, None], other=136).to(tl.int32)
        k = tl.interleave(((packed & 15) - 8).to(tl.int8), ((packed >> 4) - 8).to(tl.int8))
        q_row = row * 16 + head
        q = tl.load(q_codes + q_row[:, None] * 64 + dim[None, :])
        qs = tl.load(q_scales + q_row)
        ks = tl.load(k_scales + slot, mask=valid, other=0).to(tl.float32)
        w = tl.load(gates + row * gate_stride + head).to(tl.float32)
        dot = tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32)
        approx = (dot * qs[:, None] * ks[None, :]).to(tl.bfloat16).to(tl.float32)
        score = tl.sum(tl.maximum(approx * scale, 0.0) * w[:, None], 0)
        tl.store(out + row * width + logical, tl.where(valid, score, -float("inf")))
    else:
        tl.store(out + row * width + logical, -float("inf"))


@triton.jit
def _refine(
    q, gates, bases, lengths, candidates, cache, exponents, headers, exceptions, scores,
    q_stride: tl.constexpr, q_head_stride: tl.constexpr, gate_stride: tl.constexpr,
    width: tl.constexpr, scale: tl.constexpr, block: tl.constexpr,
):
    row = tl.program_id(1)
    col = tl.program_id(0) * block + tl.arange(0, block)
    logical = tl.load(candidates + row * width + col)
    valid = (logical >= 0) & (logical < tl.load(lengths + row))
    slot = tl.load(bases + row) + logical
    dims = tl.arange(0, 64)
    heads = tl.arange(0, 16)
    key = _load_packed_key(cache, exponents, headers, exceptions, slot, valid, dims, False)
    query = tl.load(q + q_stride * row + q_head_stride * heads[:, None] + dims[None, :])
    w = tl.load(gates + row * gate_stride + heads).to(tl.float32)
    dot = tl.dot(query, tl.trans(key), allow_tf32=False).to(tl.bfloat16).to(tl.float32)
    score = tl.sum(tl.maximum(dot * scale, 0.0) * w[:, None], 0)
    tl.store(scores + row * width + col, tl.where(valid, score, -float("inf")))


class Int4Scan:
    def __init__(self, raw_slab, packed_slab=None):
        self.plan = raw_slab.plan
        self.lengths = raw_slab.lengths
        self.packed_slab = packed_slab
        self.batch = self.lengths.numel()
        device = self.lengths.device
        self.key_codes = torch.empty((self.plan.capacity, 32), dtype=torch.uint8, device=device)
        self.key_scales = torch.empty(self.plan.capacity, dtype=torch.bfloat16, device=device)
        self.query_codes = torch.empty((self.batch, 16, 64), dtype=torch.int8, device=device)
        self.query_scales = torch.empty((self.batch, 16), dtype=torch.float32, device=device)
        self.offsets = torch.zeros_like(self.lengths)
        self.encode_history(raw_slab)

    def encode_history(self, raw_slab):
        _encode_history[(triton.cdiv(self.plan.capacity, 32),)](
            raw_slab.resident, self.key_codes, self.key_scales, self.plan.capacity, 32, num_warps=4,
        )

    @property
    def storage_bytes(self):
        if self.packed_slab is None:
            supporting_bytes = self.plan.bases.nbytes + self.plan.capacities.nbytes
        else:
            supporting_bytes = self.packed_slab.storage_bytes
        return supporting_bytes + self.key_codes.nbytes + self.key_scales.nbytes

    def prepare_coarse(self, scale):
        block = 256
        width = triton.cdiv(self.plan.width, block) * block
        scores = torch.empty((self.batch, width), dtype=torch.float32, device=self.lengths.device)

        def coarse(q, k, gates):
            _prepare_step[(self.batch, 17)](
                q, k, self.plan.bases, self.plan.capacities, self.lengths,
                self.query_codes, self.query_scales, self.key_codes, self.key_scales,
                q.stride(0), q.stride(1), k.stride(0), num_warps=2,
            )
            _scan[(triton.cdiv(width, block), self.batch)](
                self.query_codes, self.query_scales, self.key_codes, self.key_scales, gates,
                self.plan.bases, self.lengths, scores, gates.stride(0), width, scale, block,
                num_warps=4,
            )
            return scores

        return coarse, scores

    def prepare_direct(self, project, selector, scale):
        """Return logical Top-2048 from the approximate scores; keep only INT4 keys."""
        if self.packed_slab is not None:
            raise ValueError("direct selection must not retain an exact Key representation")
        coarse, scores = self.prepare_coarse(scale)

        def call():
            return selector(coarse(*project()))

        call.approx_scores = scores
        call.workspace_bytes = scores.nbytes + self.query_codes.nbytes + self.query_scales.nbytes
        return call

    def prepare(self, project, scale, candidate_k):
        from flashinfer import top_k_page_table_transform, top_k_ragged_transform

        if self.packed_slab is None:
            raise ValueError("BF16 refinement requires a persistent exact Key representation")
        if min(self.lengths.tolist()) < candidate_k:
            raise ValueError("this prototype assumes every active row has at least candidate_k tokens")
        fine = torch.empty((self.batch, candidate_k), dtype=torch.float32, device=self.lengths.device)
        fine_lens = torch.full_like(self.lengths, candidate_k)
        keys, capacity = self.packed_slab.buffers()
        coarse, _ = self.prepare_coarse(scale)

        def call():
            q, k, gates = project()
            _append_key[(self.batch,)](
                k, self.plan.bases, self.plan.capacities, self.lengths, *keys,
                self.packed_slab.overflow, k.stride(0), capacity, True, num_warps=4,
            )
            candidates = top_k_ragged_transform(coarse(q, k, gates), self.offsets,
                                                self.lengths, candidate_k)
            _refine[(candidate_k // 128, self.batch)](
                q, gates, self.plan.bases, self.lengths, candidates, *keys[:4], fine,
                q.stride(0), q.stride(1), gates.stride(0), candidate_k, scale, 128,
                num_warps=4,
            )
            return top_k_page_table_transform(fine, candidates, fine_lens, 2048)

        call.last_overflow = self.packed_slab.overflow
        return call, coarse
