"""Ampere kernels for the exact-BF16 Keye sparse-attention indexer."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _keye_indexer_score_paged(
    query_ptr,
    key_cache_ptr,
    page_table_ptr,
    weights_ptr,
    seqlens_ptr,
    scores_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kd,
    stride_pb,
    stride_pk,
    stride_wb,
    stride_wh,
    stride_sb,
    stride_sk,
    max_len,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fuse paged K gather, QK, ReLU, gate, and head reduction."""
    batch = tl.program_id(0)
    start_n = tl.program_id(1) * BLOCK_N
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    seq_len = tl.load(seqlens_ptr + batch)
    in_bounds = offs_n < max_len
    valid_n = in_bounds & (offs_n < seq_len)
    slot = tl.load(
        page_table_ptr + batch * stride_pb + offs_n * stride_pk,
        mask=valid_n,
        other=0,
    ).to(tl.int64)
    valid_n &= slot >= 0
    slot = tl.maximum(slot, 0)

    query = tl.load(
        query_ptr
        + batch * stride_qb
        + offs_h[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=(offs_h[:, None] < num_heads) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    keys = tl.load(
        key_cache_ptr + slot[:, None] * stride_ks + offs_d[None, :] * stride_kd,
        mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    per_head = tl.dot(query, tl.trans(keys), allow_tf32=False)
    # torch.matmul with BF16 inputs materializes a BF16 result before the
    # score equation converts it to FP32.
    per_head = per_head.to(tl.bfloat16).to(tl.float32)
    gate = tl.load(
        weights_ptr + batch * stride_wb + offs_h * stride_wh,
        mask=offs_h < num_heads,
        other=0.0,
    ).to(tl.float32)
    score = tl.sum(tl.maximum(per_head * softmax_scale, 0.0) * gate[:, None], axis=0)
    score = tl.where(valid_n, score, -float("inf"))
    tl.store(
        scores_ptr + batch * stride_sb + offs_n * stride_sk,
        score,
        mask=in_bounds,
    )


@torch.no_grad()
def keye_indexer_score_paged(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    weights: torch.Tensor,
    seqlens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Return exact Keye index scores in logical page-table order."""
    if query.ndim != 3 or key_cache.ndim != 2 or page_table.ndim != 2:
        raise ValueError("expected query [B,H,D], key cache [S,D], page table [B,K]")
    batch, num_heads, head_dim = query.shape
    if weights.shape != (batch, num_heads) or page_table.shape[0] != batch:
        raise ValueError("incompatible query, weight, and page-table shapes")
    if key_cache.shape[1] != head_dim:
        raise ValueError("query and key head dimensions differ")
    if query.dtype != torch.bfloat16 or key_cache.dtype != torch.bfloat16:
        raise TypeError("the Ampere indexer score kernel requires BF16 query and key")
    if num_heads > 16 or head_dim > 128:
        raise ValueError("the Ampere indexer score kernel supports H <= 16 and D <= 128")

    max_len = page_table.shape[1]
    scores = torch.empty((batch, max_len), device=query.device, dtype=torch.float32)
    block_h = triton.next_power_of_2(num_heads)
    block_d = triton.next_power_of_2(head_dim)
    block_n = 128
    _keye_indexer_score_paged[(batch, triton.cdiv(max_len, block_n))](
        query,
        key_cache,
        page_table,
        weights,
        seqlens,
        scores,
        *query.stride(),
        *key_cache.stride(),
        *page_table.stride(),
        *weights.stride(),
        *scores.stride(),
        max_len,
        num_heads=num_heads,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return scores
