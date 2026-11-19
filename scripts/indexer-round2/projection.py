"""Fuse the Indexer projection epilogue after both native BF16 linear projections."""

import torch
import triton
import triton.language as tl


@triton.jit
def _normalize_heads(
    q_and_w, k, q_weight, k_weight, k_bias, q_out, k_out, positions, trig,
    q_stride: tl.constexpr, k_stride: tl.constexpr,
    q_eps: tl.constexpr, k_eps: tl.constexpr,
    position_stride: tl.constexpr, section_t: tl.constexpr, section_h: tl.constexpr,
    epilogue: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, 64)
    if head < 16:
        q = tl.load(q_and_w + q_stride * row + head * 64 + dim).to(tl.float32)
        inv = tl.rsqrt(tl.sum(q * q, 0) * (1.0 / 64) + q_eps)
        w = tl.load(q_weight + dim).to(tl.float32)
        values = ((q * inv) * w).to(tl.bfloat16)
    else:
        key = tl.load(k + k_stride * row + dim).to(tl.float32)
        centered = key - tl.sum(key, 0) * (1.0 / 64)
        inv = tl.rsqrt(tl.sum(centered * centered, 0) * (1.0 / 64) + k_eps)
        w = tl.load(k_weight + dim).to(tl.float32)
        bias = tl.load(k_bias + dim).to(tl.float32)
        values = (((centered * inv) * w) + bias).to(tl.bfloat16)

    if epilogue:
        column = dim % 32
        axis = tl.where(column < section_t, 0, tl.where(column < section_t + section_h, 1, 2))
        position = tl.load(positions + axis * position_stride + row)
        cos = tl.load(trig + position * 64 + column)
        sin = tl.load(trig + position * 64 + column + 32)
        swap = tl.gather(values, dim ^ 32, 0)
        primary = values.to(cos.dtype) * cos
        cross = swap.to(sin.dtype) * sin
        rotated = tl.where(dim < 32, primary - cross, primary + cross)
        butterfly = rotated.to(tl.bfloat16).to(tl.float32)
        for stage in tl.static_range(6):
            bit = 1 << stage
            partner = tl.gather(butterfly, dim ^ bit, 0)
            butterfly = tl.where((dim & bit) == 0, butterfly + partner, partner - butterfly)
        values = (butterfly * 0.125).to(tl.bfloat16)

    if head < 16:
        tl.store(q_out + (row * 16 + head) * 64 + dim, values)
    else:
        tl.store(k_out + row * 64 + dim, values)


def prepare_projection(indexer, hidden, positions, *, epilogue=False):
    from sglang.srt.layers.attention.keye_topk.keye_indexer import _rotate_activation

    batch = hidden.shape[0]
    rope = indexer.rotary_emb
    if epilogue and (rope.mrope_interleaved or rope.mrope_interleaved_glm or
                     not rope.is_neox_style or rope.rotary_dim != 64):
        raise ValueError("fused epilogue is specialized to Keye indexer's grouped NeoX MRoPE")
    q_normalized = torch.empty((batch, 1024), dtype=torch.bfloat16, device=hidden.device)
    k_normalized = torch.empty((batch, 64), dtype=torch.bfloat16, device=hidden.device)

    def call():
        q_and_w, _ = indexer.q_proj(hidden)
        k, _ = indexer.k_proj(hidden)
        _normalize_heads[(batch, 17)](
            q_and_w, k, indexer.q_norm.weight, indexer.k_norm.weight, indexer.k_norm.bias,
            q_normalized, k_normalized, positions, rope.cos_sin_cache,
            q_and_w.stride(0), k.stride(0),
            indexer.q_norm.variance_epsilon, indexer.k_norm.variance_epsilon,
            positions.stride(0), *rope.mrope_section[:2], epilogue,
            num_warps=2, enable_fp_fusion=False,
        )
        if epilogue:
            return q_normalized.view(batch, 16, 64), k_normalized, q_and_w[:, 1024:]
        q_roped, k_roped = indexer.rotary_emb(positions, q_normalized, k_normalized)
        q = _rotate_activation(q_roped.view(-1, 64)).view(batch, 16, 64)
        key = _rotate_activation(k_roped)
        return q, key, q_and_w[:, 1024:]

    return call
