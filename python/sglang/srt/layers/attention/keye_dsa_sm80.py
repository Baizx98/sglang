"""Exact-token Keye DSA attention for NVIDIA Ampere.

The Keye indexer has already applied request, padding, and causal constraints
before this module is called.  ``token_slots`` therefore contains exact
physical SGLang KV-pool slots, with ``-1`` denoting padding.  No block rounding
or token-set approximation is performed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl


@dataclass
class KeyeDSAWorkspace:
    # CUDA graphs for several batch sizes coexist. Keep every captured address
    # alive instead of replacing a single workspace when the shape changes.
    _buffers: Dict[
        Tuple[torch.device, int, int, int, int, int],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ] = field(default_factory=dict)

    def split_buffers(
        self,
        *,
        device: torch.device,
        batch: int,
        hq: int,
        q_len: int,
        num_splits: int,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (device, batch, hq, q_len, num_splits, head_dim)
        buffers = self._buffers.get(key)
        if buffers is None:
            partial_m = torch.empty(
                (batch, hq, q_len, num_splits), device=device, dtype=torch.float32
            )
            partial_l = torch.empty_like(partial_m)
            partial_out = torch.empty(
                (batch, hq, q_len, num_splits, head_dim),
                device=device,
                dtype=torch.float32,
            )
            buffers = (partial_m, partial_l, partial_out)
            self._buffers[key] = buffers
        return buffers


@triton.jit
def _load_q(
    q_ptr,
    b,
    kv_head,
    q_idx,
    stride_qb,
    stride_qh,
    stride_qq,
    stride_qd,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_head = kv_head * GROUP_SIZE + offs_m
    head_mask = offs_m < GROUP_SIZE
    q = tl.load(
        q_ptr
        + b * stride_qb
        + q_head[:, None] * stride_qh
        + q_idx * stride_qq
        + offs_d[None, :] * stride_qd,
        mask=head_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    return q, q_head, head_mask, offs_d


@triton.jit
def _load_logits(
    q,
    k_ptr,
    topk_ptr,
    b,
    kv_head,
    q_idx,
    start_n,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_tb,
    stride_tq,
    stride_tt,
    kv_len,
    topk_len,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROUND_LOGITS: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    NEG: tl.constexpr = -3.4028234663852886e38
    LOG2E: tl.constexpr = 1.4426950408889634

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    in_topk = offs_n < topk_len
    selected_idx = tl.load(
        topk_ptr + b * stride_tb + q_idx * stride_tq + offs_n * stride_tt,
        mask=in_topk,
        other=-1,
    ).to(tl.int32)
    valid = in_topk & (selected_idx >= 0) & (selected_idx < kv_len)
    safe_idx = tl.maximum(0, tl.minimum(selected_idx, kv_len - 1))
    batch_offset = b.to(tl.int64) * stride_kb
    token_offset = safe_idx.to(tl.int64) * stride_ks

    k = tl.load(
        k_ptr
        + batch_offset
        + kv_head * stride_kh
        + token_offset[:, None]
        + offs_d[None, :] * stride_kd,
        mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    logits = tl.dot(q, tl.trans(k))

    # Match eager BF16/FP16 QK materialization and post-scale materialization.
    if ROUND_LOGITS:
        if IS_BF16:
            logits = logits.to(tl.bfloat16).to(tl.float32)
        else:
            logits = logits.to(tl.float16).to(tl.float32)
    logits *= sm_scale
    if ROUND_LOGITS:
        if IS_BF16:
            logits = logits.to(tl.bfloat16).to(tl.float32)
        else:
            logits = logits.to(tl.float16).to(tl.float32)

    matrix_valid = (offs_m[:, None] < GROUP_SIZE) & valid[None, :]
    return tl.where(matrix_valid, logits * LOG2E, NEG), valid, safe_idx


@triton.jit
def _load_v(
    v_ptr,
    b,
    kv_head,
    safe_idx,
    valid,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    offs_d = tl.arange(0, BLOCK_D)
    batch_offset = b.to(tl.int64) * stride_vb
    token_offset = safe_idx.to(tl.int64) * stride_vs
    return tl.load(
        v_ptr
        + batch_offset
        + kv_head * stride_vh
        + token_offset[:, None]
        + offs_d[None, :] * stride_vd,
        mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )


@triton.jit
def _dsa_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    topk_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_tb,
    stride_tq,
    stride_tt,
    stride_ob,
    stride_oh,
    stride_oq,
    stride_od,
    q_len,
    kv_len,
    topk_len,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROUND_LOGITS: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    NEG: tl.constexpr = -3.4028234663852886e38
    bq = tl.program_id(0)
    kv_head = tl.program_id(1)
    b = bq // q_len
    q_idx = bq - b * q_len
    q, q_head, head_mask, offs_d = _load_q(
        q_ptr,
        b,
        kv_head,
        q_idx,
        stride_qb,
        stride_qh,
        stride_qq,
        stride_qd,
        HEAD_DIM,
        GROUP_SIZE,
        BLOCK_M,
        BLOCK_D,
    )

    m_i = tl.full((BLOCK_M,), NEG, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    for start_n in tl.range(0, topk_len, BLOCK_N, num_stages=2):
        logits, valid, unused_safe_idx = _load_logits(
            q,
            k_ptr,
            topk_ptr,
            b,
            kv_head,
            q_idx,
            start_n,
            stride_kb,
            stride_kh,
            stride_ks,
            stride_kd,
            stride_tb,
            stride_tq,
            stride_tt,
            kv_len,
            topk_len,
            sm_scale,
            HEAD_DIM,
            GROUP_SIZE,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            ROUND_LOGITS,
            IS_BF16,
        )
        matrix_valid = head_mask[:, None] & valid[None, :]
        block_m = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, block_m)
        p = tl.where(matrix_valid, tl.exp2(logits - new_m[:, None]), 0.0)
        l_i = l_i * tl.exp2(m_i - new_m) + tl.sum(p, axis=1)
        m_i = new_m

    inv_l = tl.where(l_i > 0.0, 1.0 / l_i, 0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for start_n in tl.range(0, topk_len, BLOCK_N, num_stages=2):
        logits, valid, safe_idx = _load_logits(
            q,
            k_ptr,
            topk_ptr,
            b,
            kv_head,
            q_idx,
            start_n,
            stride_kb,
            stride_kh,
            stride_ks,
            stride_kd,
            stride_tb,
            stride_tq,
            stride_tt,
            kv_len,
            topk_len,
            sm_scale,
            HEAD_DIM,
            GROUP_SIZE,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            ROUND_LOGITS,
            IS_BF16,
        )
        matrix_valid = head_mask[:, None] & valid[None, :]
        p = tl.where(
            matrix_valid,
            tl.exp2(logits - m_i[:, None]) * inv_l[:, None],
            0.0,
        )
        p_low = p.to(tl.bfloat16) if IS_BF16 else p.to(tl.float16)
        value = _load_v(
            v_ptr,
            b,
            kv_head,
            safe_idx,
            valid,
            stride_vb,
            stride_vh,
            stride_vs,
            stride_vd,
            HEAD_DIM,
            BLOCK_D,
        )
        acc += tl.dot(p_low, value)

    tl.store(
        out_ptr
        + b * stride_ob
        + q_head[:, None] * stride_oh
        + q_idx * stride_oq
        + offs_d[None, :] * stride_od,
        acc,
        mask=head_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
    )


@triton.jit
def _dsa_stats_split(
    q_ptr,
    k_ptr,
    topk_ptr,
    partial_m_ptr,
    partial_l_ptr,
    stride_qb,
    stride_qh,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_tb,
    stride_tq,
    stride_tt,
    stride_pmb,
    stride_pmh,
    stride_pmq,
    stride_pms,
    stride_plb,
    stride_plh,
    stride_plq,
    stride_pls,
    q_len,
    kv_len,
    topk_len,
    split_size,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROUND_LOGITS: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    NEG: tl.constexpr = -3.4028234663852886e38
    bq = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    b = bq // q_len
    q_idx = bq - b * q_len
    q, q_head, head_mask, unused_offs_d = _load_q(
        q_ptr,
        b,
        kv_head,
        q_idx,
        stride_qb,
        stride_qh,
        stride_qq,
        stride_qd,
        HEAD_DIM,
        GROUP_SIZE,
        BLOCK_M,
        BLOCK_D,
    )
    split_start = split_id * split_size
    split_end = tl.minimum(split_start + split_size, topk_len)
    m_i = tl.full((BLOCK_M,), NEG, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    for start_n in tl.range(split_start, split_end, BLOCK_N, num_stages=2):
        logits, valid, unused_safe_idx = _load_logits(
            q,
            k_ptr,
            topk_ptr,
            b,
            kv_head,
            q_idx,
            start_n,
            stride_kb,
            stride_kh,
            stride_ks,
            stride_kd,
            stride_tb,
            stride_tq,
            stride_tt,
            kv_len,
            topk_len,
            sm_scale,
            HEAD_DIM,
            GROUP_SIZE,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            ROUND_LOGITS,
            IS_BF16,
        )
        matrix_valid = head_mask[:, None] & valid[None, :]
        block_m = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, block_m)
        p = tl.where(matrix_valid, tl.exp2(logits - new_m[:, None]), 0.0)
        l_i = l_i * tl.exp2(m_i - new_m) + tl.sum(p, axis=1)
        m_i = new_m

    tl.store(
        partial_m_ptr
        + b * stride_pmb
        + q_head * stride_pmh
        + q_idx * stride_pmq
        + split_id * stride_pms,
        m_i,
        mask=head_mask,
    )
    tl.store(
        partial_l_ptr
        + b * stride_plb
        + q_head * stride_plh
        + q_idx * stride_plq
        + split_id * stride_pls,
        l_i,
        mask=head_mask,
    )


@triton.jit
def _dsa_reduce_stats(
    partial_m_ptr,
    partial_l_ptr,
    global_m_ptr,
    global_l_ptr,
    stride_pmb,
    stride_pmh,
    stride_pmq,
    stride_pms,
    stride_plb,
    stride_plh,
    stride_plq,
    stride_pls,
    stride_gmb,
    stride_gmh,
    stride_gmq,
    stride_glb,
    stride_glh,
    stride_glq,
    q_len,
    num_splits,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    NEG: tl.constexpr = -3.4028234663852886e38
    bq = tl.program_id(0)
    kv_head = tl.program_id(1)
    b = bq // q_len
    q_idx = bq - b * q_len
    offs_m = tl.arange(0, BLOCK_M)
    offs_s = tl.arange(0, BLOCK_S)
    q_head = kv_head * GROUP_SIZE + offs_m
    mask = (offs_m[:, None] < GROUP_SIZE) & (offs_s[None, :] < num_splits)
    pm = tl.load(
        partial_m_ptr
        + b * stride_pmb
        + q_head[:, None] * stride_pmh
        + q_idx * stride_pmq
        + offs_s[None, :] * stride_pms,
        mask=mask,
        other=NEG,
    )
    pl = tl.load(
        partial_l_ptr
        + b * stride_plb
        + q_head[:, None] * stride_plh
        + q_idx * stride_plq
        + offs_s[None, :] * stride_pls,
        mask=mask,
        other=0.0,
    )
    gm = tl.max(pm, axis=1)
    gl = tl.sum(tl.where(mask, pl * tl.exp2(pm - gm[:, None]), 0.0), axis=1)
    head_mask = offs_m < GROUP_SIZE
    tl.store(
        global_m_ptr + b * stride_gmb + q_head * stride_gmh + q_idx * stride_gmq,
        gm,
        mask=head_mask,
    )
    tl.store(
        global_l_ptr + b * stride_glb + q_head * stride_glh + q_idx * stride_glq,
        gl,
        mask=head_mask,
    )


@triton.jit
def _dsa_out_split(
    q_ptr,
    k_ptr,
    v_ptr,
    topk_ptr,
    global_m_ptr,
    global_l_ptr,
    partial_out_ptr,
    stride_qb,
    stride_qh,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_tb,
    stride_tq,
    stride_tt,
    stride_gmb,
    stride_gmh,
    stride_gmq,
    stride_glb,
    stride_glh,
    stride_glq,
    stride_pob,
    stride_poh,
    stride_poq,
    stride_pos,
    stride_pod,
    q_len,
    kv_len,
    topk_len,
    split_size,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROUND_LOGITS: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    bq = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    b = bq // q_len
    q_idx = bq - b * q_len
    q, q_head, head_mask, offs_d = _load_q(
        q_ptr,
        b,
        kv_head,
        q_idx,
        stride_qb,
        stride_qh,
        stride_qq,
        stride_qd,
        HEAD_DIM,
        GROUP_SIZE,
        BLOCK_M,
        BLOCK_D,
    )
    gm = tl.load(
        global_m_ptr + b * stride_gmb + q_head * stride_gmh + q_idx * stride_gmq,
        mask=head_mask,
        other=0.0,
    )
    gl = tl.load(
        global_l_ptr + b * stride_glb + q_head * stride_glh + q_idx * stride_glq,
        mask=head_mask,
        other=0.0,
    )
    inv_l = tl.where(gl > 0.0, 1.0 / gl, 0.0)
    split_start = split_id * split_size
    split_end = tl.minimum(split_start + split_size, topk_len)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for start_n in tl.range(split_start, split_end, BLOCK_N, num_stages=2):
        logits, valid, safe_idx = _load_logits(
            q,
            k_ptr,
            topk_ptr,
            b,
            kv_head,
            q_idx,
            start_n,
            stride_kb,
            stride_kh,
            stride_ks,
            stride_kd,
            stride_tb,
            stride_tq,
            stride_tt,
            kv_len,
            topk_len,
            sm_scale,
            HEAD_DIM,
            GROUP_SIZE,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            ROUND_LOGITS,
            IS_BF16,
        )
        matrix_valid = head_mask[:, None] & valid[None, :]
        p = tl.where(
            matrix_valid,
            tl.exp2(logits - gm[:, None]) * inv_l[:, None],
            0.0,
        )
        p_low = p.to(tl.bfloat16) if IS_BF16 else p.to(tl.float16)
        value = _load_v(
            v_ptr,
            b,
            kv_head,
            safe_idx,
            valid,
            stride_vb,
            stride_vh,
            stride_vs,
            stride_vd,
            HEAD_DIM,
            BLOCK_D,
        )
        acc += tl.dot(p_low, value)
    tl.store(
        partial_out_ptr
        + b * stride_pob
        + q_head[:, None] * stride_poh
        + q_idx * stride_poq
        + split_id * stride_pos
        + offs_d[None, :] * stride_pod,
        acc,
        mask=head_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
    )


@triton.jit
def _dsa_reduce_out(
    partial_out_ptr,
    out_ptr,
    stride_pob,
    stride_poh,
    stride_poq,
    stride_pos,
    stride_pod,
    stride_ob,
    stride_oh,
    stride_oq,
    stride_od,
    q_len,
    num_splits,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bq = tl.program_id(0)
    kv_head = tl.program_id(1)
    b = bq // q_len
    q_idx = bq - b * q_len
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_head = kv_head * GROUP_SIZE + offs_m
    mask = (offs_m[:, None] < GROUP_SIZE) & (offs_d[None, :] < HEAD_DIM)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for split_id in tl.range(0, num_splits, 1):
        acc += tl.load(
            partial_out_ptr
            + b * stride_pob
            + q_head[:, None] * stride_poh
            + q_idx * stride_poq
            + split_id * stride_pos
            + offs_d[None, :] * stride_pod,
            mask=mask,
            other=0.0,
        )
    tl.store(
        out_ptr
        + b * stride_ob
        + q_head[:, None] * stride_oh
        + q_idx * stride_oq
        + offs_d[None, :] * stride_od,
        acc,
        mask=mask,
    )


@triton.jit
def _dsa_split_online(
    q_ptr,
    k_ptr,
    v_ptr,
    topk_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_out_ptr,
    stride_qb,
    stride_qh,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_tb,
    stride_tq,
    stride_tt,
    stride_pmb,
    stride_pmh,
    stride_pmq,
    stride_pms,
    stride_plb,
    stride_plh,
    stride_plq,
    stride_pls,
    stride_pob,
    stride_poh,
    stride_poq,
    stride_pos,
    stride_pod,
    q_len,
    kv_len,
    topk_len,
    split_size,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROUND_LOGITS: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """Compute each split's online-softmax state and numerator in one pass."""
    NEG: tl.constexpr = -3.4028234663852886e38
    bq = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    b = bq // q_len
    q_idx = bq - b * q_len
    q, q_head, head_mask, offs_d = _load_q(
        q_ptr,
        b,
        kv_head,
        q_idx,
        stride_qb,
        stride_qh,
        stride_qq,
        stride_qd,
        HEAD_DIM,
        GROUP_SIZE,
        BLOCK_M,
        BLOCK_D,
    )
    split_start = split_id * split_size
    split_end = tl.minimum(split_start + split_size, topk_len)
    m_i = tl.full((BLOCK_M,), NEG, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for start_n in tl.range(split_start, split_end, BLOCK_N, num_stages=2):
        logits, valid, safe_idx = _load_logits(
            q,
            k_ptr,
            topk_ptr,
            b,
            kv_head,
            q_idx,
            start_n,
            stride_kb,
            stride_kh,
            stride_ks,
            stride_kd,
            stride_tb,
            stride_tq,
            stride_tt,
            kv_len,
            topk_len,
            sm_scale,
            HEAD_DIM,
            GROUP_SIZE,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            ROUND_LOGITS,
            IS_BF16,
        )
        matrix_valid = head_mask[:, None] & valid[None, :]
        block_m = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, block_m)
        alpha = tl.exp2(m_i - new_m)
        p = tl.where(matrix_valid, tl.exp2(logits - new_m[:, None]), 0.0)
        value = _load_v(
            v_ptr,
            b,
            kv_head,
            safe_idx,
            valid,
            stride_vb,
            stride_vh,
            stride_vs,
            stride_vd,
            HEAD_DIM,
            BLOCK_D,
        )
        p_low = p.to(tl.bfloat16) if IS_BF16 else p.to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(p_low, value)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = new_m

    tl.store(
        partial_m_ptr
        + b * stride_pmb
        + q_head * stride_pmh
        + q_idx * stride_pmq
        + split_id * stride_pms,
        m_i,
        mask=head_mask,
    )
    tl.store(
        partial_l_ptr
        + b * stride_plb
        + q_head * stride_plh
        + q_idx * stride_plq
        + split_id * stride_pls,
        l_i,
        mask=head_mask,
    )
    tl.store(
        partial_out_ptr
        + b * stride_pob
        + q_head[:, None] * stride_poh
        + q_idx * stride_poq
        + split_id * stride_pos
        + offs_d[None, :] * stride_pod,
        acc,
        mask=head_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
    )


@triton.jit
def _dsa_reduce_online(
    partial_m_ptr,
    partial_l_ptr,
    partial_out_ptr,
    out_ptr,
    stride_pmb,
    stride_pmh,
    stride_pmq,
    stride_pms,
    stride_plb,
    stride_plh,
    stride_plq,
    stride_pls,
    stride_pob,
    stride_poh,
    stride_poq,
    stride_pos,
    stride_pod,
    stride_ob,
    stride_oh,
    stride_oq,
    stride_od,
    q_len,
    num_splits,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Merge split online-softmax states and numerators into final output."""
    NEG: tl.constexpr = -3.4028234663852886e38
    bq = tl.program_id(0)
    kv_head = tl.program_id(1)
    b = bq // q_len
    q_idx = bq - b * q_len
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, BLOCK_S)
    q_head = kv_head * GROUP_SIZE + offs_m
    head_mask = offs_m < GROUP_SIZE
    split_mask = head_mask[:, None] & (offs_s[None, :] < num_splits)
    partial_m = tl.load(
        partial_m_ptr
        + b * stride_pmb
        + q_head[:, None] * stride_pmh
        + q_idx * stride_pmq
        + offs_s[None, :] * stride_pms,
        mask=split_mask,
        other=NEG,
    )
    partial_l = tl.load(
        partial_l_ptr
        + b * stride_plb
        + q_head[:, None] * stride_plh
        + q_idx * stride_plq
        + offs_s[None, :] * stride_pls,
        mask=split_mask,
        other=0.0,
    )
    global_m = tl.max(partial_m, axis=1)
    global_l = tl.sum(
        tl.where(
            split_mask,
            partial_l * tl.exp2(partial_m - global_m[:, None]),
            0.0,
        ),
        axis=1,
    )
    inv_l = tl.where(global_l > 0.0, 1.0 / global_l, 0.0)
    out_mask = head_mask[:, None] & (offs_d[None, :] < HEAD_DIM)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for split_id in tl.range(0, num_splits, 1):
        split_m = tl.load(
            partial_m_ptr
            + b * stride_pmb
            + q_head * stride_pmh
            + q_idx * stride_pmq
            + split_id * stride_pms,
            mask=head_mask,
            other=NEG,
        )
        weight = tl.exp2(split_m - global_m) * inv_l
        split_out = tl.load(
            partial_out_ptr
            + b * stride_pob
            + q_head[:, None] * stride_poh
            + q_idx * stride_poq
            + split_id * stride_pos
            + offs_d[None, :] * stride_pod,
            mask=out_mask,
            other=0.0,
        )
        acc += split_out * weight[:, None]
    tl.store(
        out_ptr
        + b * stride_ob
        + q_head[:, None] * stride_oh
        + q_idx * stride_oq
        + offs_d[None, :] * stride_od,
        acc,
        mask=out_mask,
    )


@torch.no_grad()
def keye_dsa_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    sm_scale: Optional[float] = None,
    round_logits_to_input_dtype: bool = True,
    split_k_threshold: int = 4,
    max_decode_splits: int = 32,
    workspace: Optional[KeyeDSAWorkspace] = None,
) -> torch.Tensor:
    if not (q.is_cuda and k.is_cuda and v.is_cuda and topk_indices.is_cuda):
        raise ValueError("q, k, v, and topk_indices must be CUDA tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or topk_indices.ndim != 3:
        raise ValueError("expected q/k/v rank 4 and topk_indices rank 3")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("only bfloat16 and float16 are supported")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must have the same dtype")

    batch, hq, q_len, head_dim = q.shape
    bk, hkv, kv_len, kd = k.shape
    if v.shape != k.shape or bk != batch or kd != head_dim:
        raise ValueError("incompatible q/k/v shapes")
    if topk_indices.shape[:2] != (batch, q_len):
        raise ValueError("topk_indices must have shape [B,Q,T]")
    topk_len = topk_indices.shape[2]
    if hq % hkv != 0:
        raise ValueError("Hq must be divisible by Hkv")
    group_size = hq // hkv
    if group_size > 16 or head_dim > 128 or head_dim % 16:
        raise ValueError("requires Hq/Hkv <= 16 and head_dim <= 128 divisible by 16")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q/k/v head dimensions must be contiguous")
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        raise RuntimeError("the kernel requires NVIDIA Ampere or newer")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        topk_indices = topk_indices.to(torch.int32)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)
    block_m, block_n = 16, 64
    block_d = triton.next_power_of_2(head_dim)
    common = dict(
        HEAD_DIM=head_dim,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        ROUND_LOGITS=round_logits_to_input_dtype,
        IS_BF16=q.dtype == torch.bfloat16,
    )

    use_split_k = q_len <= split_k_threshold and topk_len > block_n
    if not use_split_k:
        _dsa_fwd[(batch * q_len, hkv)](
            q,
            k,
            v,
            topk_indices,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *topk_indices.stride(),
            *out.stride(),
            q_len,
            kv_len,
            topk_len,
            float(sm_scale),
            **common,
            num_warps=4,
            num_stages=2,
        )
        return out

    num_topk_blocks = triton.cdiv(topk_len, block_n)
    target_splits = min(max_decode_splits, num_topk_blocks)
    split_blocks = triton.cdiv(num_topk_blocks, target_splits)
    split_size = split_blocks * block_n
    num_splits = triton.cdiv(topk_len, split_size)
    block_s = triton.next_power_of_2(num_splits)
    workspace = workspace or KeyeDSAWorkspace()
    partial_m, partial_l, partial_out = workspace.split_buffers(
        device=q.device,
        batch=batch,
        hq=hq,
        q_len=q_len,
        num_splits=num_splits,
        head_dim=head_dim,
    )
    split_grid = (batch * q_len, hkv, num_splits)
    _dsa_split_online[split_grid](
        q,
        k,
        v,
        topk_indices,
        partial_m,
        partial_l,
        partial_out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *topk_indices.stride(),
        *partial_m.stride(),
        *partial_l.stride(),
        *partial_out.stride(),
        q_len,
        kv_len,
        topk_len,
        split_size,
        float(sm_scale),
        **common,
        num_warps=4,
        num_stages=2,
    )
    reduce_grid = (batch * q_len, hkv)
    _dsa_reduce_online[reduce_grid](
        partial_m,
        partial_l,
        partial_out,
        out,
        *partial_m.stride(),
        *partial_l.stride(),
        *partial_out.stride(),
        *out.stride(),
        q_len,
        num_splits,
        HEAD_DIM=head_dim,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_S=block_s,
        num_warps=4,
        num_stages=1,
    )
    return out


@torch.no_grad()
def keye_dsa_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    *,
    num_q_heads: int,
    head_dim: int,
    sm_scale: float,
    is_decode: bool,
    workspace: Optional[KeyeDSAWorkspace] = None,
) -> torch.Tensor:
    """Adapt SGLang packed Q and physical KV-pool slots to the DSA kernel."""
    q_packed = q.view(-1, num_q_heads, head_dim)
    num_tokens = q_packed.shape[0]
    k_base = k_cache.permute(1, 0, 2).unsqueeze(0)
    v_base = v_cache.permute(1, 0, 2).unsqueeze(0)

    if is_decode:
        q_4d = q_packed.unsqueeze(2)
        k_4d = k_base.expand(num_tokens, -1, -1, -1)
        v_4d = v_base.expand(num_tokens, -1, -1, -1)
        indices_3d = token_slots.view(num_tokens, 1, -1)
        out = keye_dsa_sparse_attention(
            q_4d,
            k_4d,
            v_4d,
            indices_3d,
            sm_scale=sm_scale,
            workspace=workspace,
        )
        return out.squeeze(2).reshape(num_tokens, num_q_heads * head_dim)

    q_4d = q_packed.permute(1, 0, 2).unsqueeze(0)
    indices_3d = token_slots.unsqueeze(0)
    out = keye_dsa_sparse_attention(
        q_4d,
        k_base,
        v_base,
        indices_3d,
        sm_scale=sm_scale,
        workspace=workspace,
    )
    return out.squeeze(0).permute(1, 0, 2).reshape(
        num_tokens, num_q_heads * head_dim
    )
