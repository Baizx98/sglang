"""Read-only prediction over request-local 4- to 8-bit Index Keys."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _encode(keys, codes, planes, scales, slots,
            rows: tl.constexpr, bits: tl.constexpr, groups: tl.constexpr,
            block: tl.constexpr, append: tl.constexpr):
    r = tl.program_id(0) * block + tl.arange(0, block)
    g = tl.program_id(1)
    d = tl.arange(0, 64 // groups)
    low_bits: tl.constexpr = 8 if bits == 8 else 4
    row_bytes: tl.constexpr = 64 * low_bits // 8
    width: tl.constexpr = row_bytes // groups
    max_code: tl.constexpr = (1 << (bits - 1)) - 1
    logical = r[:, None] * 64 + g * (64 // groups) + d[None, :]
    if append:
        # Append receives physical slots reserved by the owner. It has no role
        # in prediction and is timed separately on actual target-layer K.
        dst = tl.load(slots + r, mask=r < rows, other=0)
        valid = (r < rows) & (dst >= 0)
    else:
        dst = r
        valid = r < rows
    key = tl.load(keys + logical, mask=valid[:, None], other=0).to(tl.float32)
    scale = (tl.maximum(tl.max(tl.abs(key), 1), 1e-20) / max_code).to(tl.bfloat16)
    quantized = tl.extra.cuda.libdevice.rint(tl.div_rn(key, scale.to(tl.float32)[:, None]))
    bounded = tl.minimum(tl.maximum(quantized, -max_code), max_code)
    if bits < 8:
        biased = bounded.to(tl.int32) + (1 << (bits - 1))
        lo, hi = tl.split(tl.reshape(biased & 15, (block, width, 2)))
        packed = (lo | (hi << 4)).to(tl.uint8)
        if bits > 4:
            # Keep the dense nibble stream aligned and store the remaining
            # bits separately as bytes of eight 1-bit or four 2-bit fields.
            residual_bits: tl.constexpr = bits - 4
            elements: tl.constexpr = 8 // residual_bits
            bytes_per_group: tl.constexpr = (64 // groups) // elements
            residual = tl.reshape(biased >> 4, (block, bytes_per_group, elements))
            shifts = tl.arange(0, elements) * residual_bits
            high = tl.sum(residual << shifts[None, None, :], 2).to(tl.uint8)
            offsets = dst[:, None] * (64 // elements) + g * bytes_per_group
            tl.store(planes + offsets + tl.arange(0, bytes_per_group),
                     high, mask=valid[:, None])
    else:
        packed = bounded.to(tl.int8)
    col = tl.arange(0, width)
    tl.store(codes + dst[:, None] * row_bytes + g * width + col[None, :],
             packed, mask=valid[:, None])
    tl.store(scales + dst * groups + g, scale, mask=valid)


@triton.jit
def _encode_query(q, q_codes, q_scales, stride: tl.constexpr, head_stride: tl.constexpr,
                  groups: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    group = tl.program_id(2)
    d = group * (64 // groups) + tl.arange(0, 64 // groups)
    values = tl.load(q + row * stride + head * head_stride + d).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 0), 1e-20) / 127.0
    codes = tl.extra.cuda.libdevice.rint(tl.div_rn(values, scale))
    tl.store(q_codes + (row * 16 + head) * 64 + d,
             tl.minimum(tl.maximum(codes, -127), 127).to(tl.int8))
    tl.store(q_scales + (row * 16 + head) * groups + group, scale)


@triton.jit
def _scan(q, q_scales, k, planes, k_scales, gates, bases, lengths, out,
          gate_stride: tl.constexpr, width: tl.constexpr, scale: tl.constexpr,
          block: tl.constexpr, bits: tl.constexpr, groups: tl.constexpr):
    row = tl.program_id(1)
    tile = tl.program_id(0)
    token = tile * block + tl.arange(0, block)
    length = tl.load(lengths + row)
    if tile * block < length:
        live = token < length
        slot = tl.multiple_of(tl.load(bases + row), 128) + token
        heads = row * 16 + tl.arange(0, 16)
        d = tl.arange(0, 64 // groups)
        low_bits: tl.constexpr = 8 if bits == 8 else 4
        row_bytes: tl.constexpr = 64 * low_bits // 8
        packed_width: tl.constexpr = row_bytes // groups
        residual_bits: tl.constexpr = bits - 4 if (bits > 4 and bits < 8) else 1
        elements: tl.constexpr = 8 // residual_bits
        bytes_per_group: tl.constexpr = (64 // groups) // elements
        col = tl.arange(0, packed_width)
        total = tl.full((16, block), 0.0, tl.float32)
        for g in tl.static_range(groups):
            packed = tl.load(k + slot[:, None] * row_bytes + g * packed_width + col[None, :],
                             mask=live[:, None], other=0)
            if bits < 8:
                encoded = packed.to(tl.int32)
                if bits == 4:
                    key = tl.interleave(((encoded & 15) - 8).to(tl.int8),
                                        ((encoded >> 4) - 8).to(tl.int8))
                else:
                    low = tl.interleave(encoded & 15, encoded >> 4)
                    high_offset = slot[:, None] * (64 // elements) + g * bytes_per_group
                    high = tl.load(planes + high_offset + d[None, :] // elements,
                                   mask=live[:, None], other=0).to(tl.int32)
                    residual = (high >> ((d[None, :] % elements) * residual_bits))
                    low |= (residual & ((1 << residual_bits) - 1)) << 4
                    key = (low - (1 << (bits - 1))).to(tl.int8)
            else:
                key = packed
            head = tl.load(q + heads[:, None] * 64 + g * (64 // groups) + d[None, :])
            dot = tl.dot(head, tl.trans(key), out_dtype=tl.int32).to(tl.float32)
            qs = tl.load(q_scales + heads * groups + g)
            ks = tl.load(k_scales + slot * groups + g, mask=live, other=0).to(tl.float32)
            total += dot * qs[:, None] * ks[None, :]
        w = tl.load(gates + row * gate_stride + tl.arange(0, 16)).to(tl.float32)
        approx = total.to(tl.bfloat16).to(tl.float32)
        score = tl.sum(tl.maximum(approx * scale, 0.0) * w[:, None], 0)
        tl.store(out + row * width + token, tl.where(live, score, -float("inf")))
    else:
        tl.store(out + row * width + token, -float("inf"))


class QuantizedKeys:
    def __init__(self, raw_slab, bits, groups):
        if bits not in (4, 5, 6, 8) or groups not in (1, 2):
            raise ValueError("Index Key supports 4, 5, 6 or 8 bits and 1 or 2 groups")
        self.plan = raw_slab.plan
        self.bits = bits
        self.groups = groups
        self.batch = raw_slab.lengths.numel()
        device = raw_slab.lengths.device
        low_bytes = 64 if bits == 8 else 32
        self.codes = torch.empty((self.plan.capacity, low_bytes),
                                 dtype=torch.int8 if bits == 8 else torch.uint8, device=device)
        residual_bytes = 8 * (bits - 4) if bits in (5, 6) else 0
        self.planes = torch.empty((self.plan.capacity, residual_bytes), dtype=torch.uint8, device=device)
        self.scales = torch.empty((self.plan.capacity, groups), dtype=torch.bfloat16, device=device)
        self.query_codes = torch.empty((self.batch, 16, 64), dtype=torch.int8, device=device)
        self.query_scales = torch.empty((self.batch, 16, groups), dtype=torch.float32, device=device)
        self.build(raw_slab.resident)

    @property
    def storage_bytes(self):
        return self.codes.nbytes + self.planes.nbytes + self.scales.nbytes + self.plan.bases.nbytes + self.plan.capacities.nbytes

    def build(self, keys):
        rows = keys.size(0)
        _encode[(triton.cdiv(rows, 32), self.groups)](
            keys, self.codes, self.planes, self.scales, self.plan.bases,
            rows, self.bits, self.groups, 32, False, num_warps=4,
        )

    def append(self, keys, slots):
        rows = keys.size(0)
        _encode[(rows, self.groups)](
            keys, self.codes, self.planes, self.scales, slots,
            rows, self.bits, self.groups, 1, True, num_warps=2,
        )

    def prepare(self, lengths, scale, block=256, warps=4, score_dtype=torch.float32):
        width = triton.cdiv(self.plan.width, block) * block
        scores = torch.empty((self.batch, width), dtype=score_dtype, device=lengths.device)

        def score(q, gates):
            _encode_query[(self.batch, 16, self.groups)](
                q, self.query_codes, self.query_scales, q.stride(0), q.stride(1), self.groups,
                num_warps=2,
            )
            _scan[(width // block, self.batch)](
                self.query_codes, self.query_scales, self.codes, self.planes, self.scales,
                gates, self.plan.bases, lengths, scores,
                gates.stride(0), width, scale, block, self.bits, self.groups,
                num_warps=warps, enable_fp_fusion=False,
            )
            return scores

        return score, scores.nbytes + self.query_codes.nbytes + self.query_scales.nbytes
