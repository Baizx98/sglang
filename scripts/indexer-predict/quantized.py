"""Read-only prediction over request-local 4- to 8-bit Index Keys."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from prefix_topm import _accumulate_histogram


@triton.jit
def _write_rows(key, dst, valid, codes, planes, scales, g, encode_table,
                bits: tl.constexpr, groups: tl.constexpr, block: tl.constexpr,
                fit_steps: tl.constexpr, fp4: tl.constexpr, pair_pq: tl.constexpr):
    low_bits: tl.constexpr = 8 if bits == 8 else 4
    row_bytes: tl.constexpr = 64 * low_bits // 8
    width: tl.constexpr = row_bytes // groups
    if pair_pq:
        scale = (tl.maximum(tl.max(tl.abs(key), 1), 1e-20) / 127.0).to(tl.bfloat16)
        normalized = tl.extra.cuda.libdevice.rint(tl.div_rn(key, scale.to(tl.float32)[:, None]))
        shifted = tl.minimum(tl.maximum(normalized, -127), 127).to(tl.int32) + 128
        first, second = tl.split(tl.reshape(shifted, (block, width, 2)))
        packed = tl.load(encode_table + first * 256 + second, mask=valid[:, None], other=0)
    elif fp4:
        # Scaled E2M1 values expressed in integer units. The sign lives in
        # bit 3 and the seven thresholds round to the nearest FP4 level.
        scale = (tl.maximum(tl.max(tl.abs(key), 1), 1e-20) / 96.0).to(tl.bfloat16)
        magnitude = tl.div_rn(tl.abs(key), scale.to(tl.float32)[:, None])
        code = (
            (magnitude > 4.0).to(tl.int32) + (magnitude > 12.0).to(tl.int32)
            + (magnitude > 20.0).to(tl.int32) + (magnitude > 28.0).to(tl.int32)
            + (magnitude > 40.0).to(tl.int32) + (magnitude > 56.0).to(tl.int32)
            + (magnitude > 80.0).to(tl.int32)
        )
        biased = code | tl.where(key < 0.0, 8, 0)
    else:
        max_code: tl.constexpr = (1 << (bits - 1)) - 1
        scale = (tl.maximum(tl.max(tl.abs(key), 1), 1e-20) / max_code).to(tl.bfloat16)
        for _ in tl.static_range(fit_steps):
            quantized = tl.extra.cuda.libdevice.rint(tl.div_rn(key, scale.to(tl.float32)[:, None]))
            code = tl.minimum(tl.maximum(quantized, -max_code), max_code)
            correlation = tl.sum(key * code, 1)
            power = tl.sum(code * code, 1)
            optimum = tl.maximum(tl.div_rn(correlation, tl.maximum(power, 1.0)), 1.0e-20)
            scale = tl.where(power > 0.0, optimum.to(tl.bfloat16), scale)
        quantized = tl.extra.cuda.libdevice.rint(tl.div_rn(key, scale.to(tl.float32)[:, None]))
        bounded = tl.minimum(tl.maximum(quantized, -max_code), max_code)
        biased = bounded.to(tl.int32) + (1 << (bits - 1))
    if pair_pq:
        pass
    elif bits < 8:
        # The dense nibble stream stores the four most significant bits.
        # The extra stream can then be omitted by a cheaper first-pass scan.
        lo, hi = tl.split(tl.reshape(biased >> (bits - 4), (block, width, 2)))
        packed = (lo | (hi << 4)).to(tl.uint8)
        if bits > 4:
            residual_bits: tl.constexpr = bits - 4
            elements: tl.constexpr = 8 // residual_bits
            bytes_per_group: tl.constexpr = (64 // groups) // elements
            residual = tl.reshape(
                biased & ((1 << residual_bits) - 1), (block, bytes_per_group, elements),
            )
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
def _encode(keys, codes, planes, scales, slots, encode_table,
            rows: tl.constexpr, bits: tl.constexpr, groups: tl.constexpr,
            block: tl.constexpr, append: tl.constexpr, fit_steps: tl.constexpr,
            fp4: tl.constexpr, pair_pq: tl.constexpr):
    r = tl.program_id(0) * block + tl.arange(0, block)
    g = tl.program_id(1)
    d = tl.arange(0, 64 // groups)
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
    _write_rows(key, dst, valid, codes, planes, scales, g, encode_table,
                bits, groups, block, fit_steps, fp4, pair_pq)


@triton.jit
def _encode_paged(raw, table, lengths, bases, codes, planes, scales, encode_table,
                  page_stride: tl.constexpr, page_token_stride: tl.constexpr,
                  key_stride: tl.constexpr, bits: tl.constexpr,
                  groups: tl.constexpr, fit_steps: tl.constexpr, block: tl.constexpr,
                  fp4: tl.constexpr, pair_pq: tl.constexpr):
    row = tl.program_id(1)
    group = tl.program_id(2)
    token = tl.program_id(0) * block + tl.arange(0, block)
    live = token < tl.load(lengths + row)
    physical = tl.load(table + row * page_stride + token * page_token_stride, mask=live, other=-1)
    live &= physical >= 0
    d = group * (64 // groups) + tl.arange(0, 64 // groups)
    key = tl.load(raw + physical[:, None] * key_stride + d[None, :],
                  mask=live[:, None], other=0).to(tl.float32)
    dst = tl.load(bases + row) + token
    _write_rows(key, dst, live, codes, planes, scales, group, encode_table,
                bits, groups, block, fit_steps, fp4, pair_pq)


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
def _e2m1(code):
    exp = (code & 7) >> 1
    mantissa = code & 1
    significand = tl.where(exp == 0, mantissa, mantissa + 2)
    magnitude = (significand * 8) << tl.maximum(exp - 1, 0)
    return tl.where((code & 8) == 0, magnitude, -magnitude).to(tl.int8)


@triton.jit
def _scan(q, q_scales, k, planes, k_scales, gates, bases, lengths, mapping, out, decode_table,
          histogram,
          gate_stride: tl.constexpr, width: tl.constexpr, scale: tl.constexpr,
          block: tl.constexpr, bits: tl.constexpr, groups: tl.constexpr,
          mapping_kind: tl.constexpr, coarse: tl.constexpr, fp4: tl.constexpr, pair_pq: tl.constexpr,
          hist_bins: tl.constexpr, hist_trailing: tl.constexpr, hist_shards: tl.constexpr):
    row = tl.program_id(1)
    tile = tl.program_id(0)
    token = tile * block + tl.arange(0, block)
    length = tl.load(lengths + row)
    if mapping_kind != 0 or tile * block < length:
        if mapping_kind == 2:
            segment = tl.load(mapping + row * (width // block) + tile)
            logical = segment * block + tl.arange(0, block)
        elif mapping_kind == 1:
            logical = tl.load(mapping + row * width + token)
        else:
            logical = token
        live = (logical >= 0) & (logical < length)
        slot = tl.multiple_of(tl.load(bases + row), 128) + logical
        heads = row * 16 + tl.arange(0, 16)
        d = tl.arange(0, 64 // groups)
        low_bits: tl.constexpr = 8 if bits == 8 else 4
        row_bytes: tl.constexpr = 64 * low_bits // 8
        packed_width: tl.constexpr = row_bytes // groups
        factor: tl.constexpr = 1 if bits <= 4 or bits == 8 else 1 << (bits - 4)
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
                if pair_pq:
                    pair = tl.load(decode_table + encoded).to(tl.int32)
                    key = tl.interleave(pair.to(tl.int8), (pair >> 8).to(tl.int8))
                elif fp4:
                    key = tl.interleave(_e2m1(encoded & 15), _e2m1(encoded >> 4))
                elif bits == 4 or coarse:
                    key = tl.interleave((((encoded & 15) - 8) * factor).to(tl.int8),
                                        (((encoded >> 4) - 8) * factor).to(tl.int8))
                else:
                    main = tl.interleave(encoded & 15, encoded >> 4)
                    offset = slot[:, None] * (64 // elements) + g * bytes_per_group
                    extra = tl.load(planes + offset + tl.arange(0, bytes_per_group),
                                    mask=live[:, None], other=0).to(tl.int32)
                    shifts = tl.arange(0, elements) * residual_bits
                    fields = (extra[:, :, None] >> shifts[None, None, :]) & ((1 << residual_bits) - 1)
                    residual = tl.reshape(fields, (block, 64 // groups))
                    code = (main << residual_bits) | residual
                    key = (code - (1 << (bits - 1))).to(tl.int8)
            else:
                key = packed
            head = tl.load(q + heads[:, None] * 64 + g * (64 // groups) + d[None, :])
            dot = tl.dot(head, tl.trans(key), out_dtype=tl.int32).to(tl.float32)
            if coarse:
                # Replace the missing residual by the center of its bin.
                q_total = tl.sum(head.to(tl.int32), axis=1).to(tl.float32)
                dot += q_total[:, None] * ((factor - 1) / 2)
            qs = tl.load(q_scales + heads * groups + g)
            ks = tl.load(k_scales + slot * groups + g, mask=live, other=0).to(tl.float32)
            total += dot * qs[:, None] * ks[None, :]
        w = tl.load(gates + row * gate_stride + tl.arange(0, 16)).to(tl.float32)
        approx = total.to(tl.bfloat16).to(tl.float32)
        score = tl.sum(tl.maximum(approx * scale, 0.0) * w[:, None], 0)
        tl.store(out + row * width + token, tl.where(live, score, -float("inf")))
        if hist_bins:
            _accumulate_histogram(score.to(tl.float16), live, histogram, row, tile,
                                  hist_bins, hist_trailing, hist_shards)
    else:
        tl.store(out + row * width + token, -float("inf"))


class QuantizedKeys:
    def __init__(self, raw_slab, bits, groups, fit_steps=0, fp4=False, codebook=None):
        self._allocate(raw_slab.plan, raw_slab.lengths, bits, groups, fit_steps, fp4, codebook)
        self.build(raw_slab.resident)

    @classmethod
    def from_paged(cls, plan, raw, table, lengths, bits, groups, fit_steps=0, fp4=False, codebook=None):
        state = cls.__new__(cls)
        state._allocate(plan, lengths, bits, groups, fit_steps, fp4, codebook)
        _encode_paged[(triton.cdiv(table.size(1), 32), lengths.numel(), groups)](
            raw, table, lengths, plan.bases, state.codes, state.planes, state.scales,
            state.encode_table, *table.stride(), raw.stride(0), bits, groups, fit_steps, 32,
            fp4, state.pair_pq, num_warps=4,
        )
        return state

    def _allocate(self, plan, lengths, bits, groups, fit_steps, fp4, codebook):
        if bits not in (4, 5, 6, 8) or groups not in (1, 2):
            raise ValueError("Index Key supports 4, 5, 6 or 8 bits and 1 or 2 groups")
        if fp4 and (bits != 4 or fit_steps):
            raise ValueError("E2M1 uses four bits and peak scaling")
        if codebook is not None and (bits != 4 or groups != 1 or fp4 or fit_steps):
            raise ValueError("pairwise PQ uses one four-bit group per key")
        self.fp4 = fp4
        self.plan = plan
        self.bits = bits
        self.groups = groups
        self.fit_steps = fit_steps
        self.batch = lengths.numel()
        device = lengths.device
        low_bytes = 64 if bits == 8 else 32
        self.codes = torch.empty((self.plan.capacity, low_bytes),
                                 dtype=torch.int8 if bits == 8 else torch.uint8, device=device)
        residual_bytes = 8 * (bits - 4) if bits in (5, 6) else 0
        self.planes = torch.empty((self.plan.capacity, residual_bytes), dtype=torch.uint8, device=device)
        self.pair_pq = codebook is not None
        self.encode_table, self.decode_table = codebook if self.pair_pq else (self.planes, self.planes)
        self.scales = torch.empty((self.plan.capacity, groups), dtype=torch.bfloat16, device=device)
        self.query_codes = torch.empty((self.batch, 16, 64), dtype=torch.int8, device=device)
        self.query_scales = torch.empty((self.batch, 16, groups), dtype=torch.float32, device=device)

    @property
    def storage_bytes(self):
        codebook_bytes = self.encode_table.nbytes + self.decode_table.nbytes if self.pair_pq else 0
        return (self.codes.nbytes + self.planes.nbytes + self.scales.nbytes
                + self.plan.bases.nbytes + self.plan.capacities.nbytes + codebook_bytes)

    def build(self, keys):
        rows = keys.size(0)
        _encode[(triton.cdiv(rows, 32), self.groups)](
            keys, self.codes, self.planes, self.scales, self.plan.bases, self.encode_table,
            rows, self.bits, self.groups, 32, False, self.fit_steps, self.fp4, self.pair_pq, num_warps=4,
        )

    def append(self, keys, slots):
        rows = keys.size(0)
        _encode[(rows, self.groups)](
            keys, self.codes, self.planes, self.scales, slots, self.encode_table,
            rows, self.bits, self.groups, 1, True, self.fit_steps, self.fp4, self.pair_pq, num_warps=2,
        )

    def encode_query(self, q):
        _encode_query[(self.batch, 16, self.groups)](
            q, self.query_codes, self.query_scales, q.stride(0), q.stride(1), self.groups,
            num_warps=2,
        )

    def prepare_scan(self, lengths, scale, block=256, warps=4, score_dtype=torch.float32,
                     shortlist_width=None, coarse=False, block_selection=False, prefix=None):
        if shortlist_width is not None and shortlist_width % block:
            raise ValueError("shortlist width must be divisible by the scan block size")
        if block_selection and shortlist_width is None:
            raise ValueError("block selection requires a compact shortlist width")
        if coarse and self.bits not in (5, 6):
            raise ValueError("coarse representation requires five or six bits")
        width = shortlist_width or triton.cdiv(self.plan.width, block) * block
        scores = torch.empty((self.batch, width), dtype=score_dtype, device=lengths.device)
        mapping_kind = 2 if block_selection else int(shortlist_width is not None)
        histogram = scores
        hist_config = (0, 0, 0)
        if prefix is not None:
            if mapping_kind or coarse or score_dtype != torch.float16 or prefix.width != width:
                raise ValueError("a prefix histogram needs a matching dense FP16 scan")
            histogram = prefix.histogram
            hist_config = (prefix.bins, prefix.trailing, prefix.shards)

        def scan(gates, candidates=None):
            if prefix is not None:
                histogram.zero_()
            _scan[(width // block, self.batch)](
                self.query_codes, self.query_scales, self.codes, self.planes, self.scales,
                gates, self.plan.bases, lengths, lengths if candidates is None else candidates,
                scores, self.decode_table, histogram, gates.stride(0), width, scale, block, self.bits, self.groups,
                mapping_kind, coarse, self.fp4, self.pair_pq, *hist_config, num_warps=warps, enable_fp_fusion=False,
            )
            return scores

        return scan, scores.nbytes

    def prepare(self, lengths, scale, block=256, warps=4, score_dtype=torch.float32, prefix=None):
        scan, score_bytes = self.prepare_scan(lengths, scale, block, warps, score_dtype, prefix=prefix)

        def score(q, gates):
            self.encode_query(q)
            return scan(gates)

        return score, score_bytes + self.query_codes.nbytes + self.query_scales.nbytes
