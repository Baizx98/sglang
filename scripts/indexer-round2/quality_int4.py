"""C1 Keye INT4 shortlist using the same encoder and scan as the layer benchmark."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "indexer-optimization"))

from int4_scan import _encode_k, _prepare_step, _scan
from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_candidates

CANDIDATE_K = 4096


@lru_cache
def selection_ops(device):
    import flashinfer

    aot = os.getenv("INDEXER_QUALITY_TOPK_AOT")
    if aot:
        from flashinfer.jit import env as flashinfer_jit

        major, minor = torch.cuda.get_device_capability(device)
        root = Path(aot) / f"sm{major}{minor}"
        if flashinfer.__version__ != "0.7.4" or not (root / "topk" / "topk.so").is_file():
            raise ValueError("cached TopK requires the tested architecture-specific FlashInfer 0.7.4 library")
        flashinfer_jit.FLASHINFER_AOT_DIR = root
    return flashinfer.top_k_ragged_transform, flashinfer.top_k_page_table_transform


@triton.jit
def _encode_paged_history(raw, pages, lengths, codes, scales,
                          stride_k: tl.constexpr, stride_page: tl.constexpr,
                          tile: tl.constexpr):
    token = tl.program_id(0) * tile + tl.arange(0, tile)
    dim = tl.arange(0, 64)
    in_history = token < tl.load(lengths) - 1
    slot = tl.load(pages + token * stride_page, in_history, other=-1).to(tl.int64)
    value = tl.load(raw + slot[:, None] * stride_k + dim[None, :],
                    (in_history & (slot >= 0))[:, None], other=0).to(tl.float32)
    packed, scale = _encode_k(value)
    dst = token[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(codes + dst, packed, in_history[:, None])
    tl.store(scales + token, scale, in_history)


class Int4History:
    """Lazy per-layer sidecar for one active logical request.

    On the first decode step, read original historical K according to the
    current page table. Later steps encode the just-projected key directly.
    Prefill and non-INT4 requests leave the canonical cache path unchanged.
    """

    def __init__(self, key_cache):
        capacity = key_cache.shape[0]
        device = key_cache.device
        self.key_codes = torch.empty((capacity, 32), dtype=torch.uint8, device=device)
        self.key_scales = torch.empty(capacity, dtype=torch.bfloat16, device=device)
        self.query_codes = torch.empty((1, 16, 64), dtype=torch.int8, device=device)
        self.query_scales = torch.empty((1, 16), dtype=torch.float32, device=device)
        self.base = torch.zeros(1, dtype=torch.int32, device=device)
        self.capacity = torch.full_like(self.base, capacity)
        self.refine_lengths = {k: torch.full_like(self.base, k) for k in (CANDIDATE_K, 8192)}
        self.last_request = None

    def score(self, q, current_k, gates, raw, table, lengths, scale, restart):
        if restart:
            _encode_paged_history[(triton.cdiv(table.shape[1], 32),)](
                raw, table, lengths, self.key_codes, self.key_scales,
                raw.stride(0), table.stride(1), 32, num_warps=4,
            )
        _prepare_step[(1, 17)](
            q, current_k, self.base, self.capacity, lengths,
            self.query_codes, self.query_scales, self.key_codes, self.key_scales,
            q.stride(0), q.stride(1), current_k.stride(0), num_warps=2,
        )
        block = 256
        width = triton.cdiv(table.shape[1], block) * block
        scores = torch.empty((1, width), dtype=torch.float32, device=table.device)
        _scan[(width // block, 1)](
            self.query_codes, self.query_scales, self.key_codes, self.key_scales, gates,
            self.base, lengths, scores, gates.stride(0), width, scale, block,
            num_warps=4,
        )
        return scores

    def select(self, q, current_k, gates, raw, table, lengths, scale, restart,
               candidate_k=CANDIDATE_K):
        top_k_ragged_transform, top_k_page_table_transform = selection_ops(q.device)

        coarse = self.score(q, current_k, gates, raw, table, lengths, scale, restart)
        candidates = top_k_ragged_transform(coarse, self.base, lengths, candidate_k)
        if candidate_k == 2048:
            return candidates, coarse
        exact = keye_indexer_score_candidates(q, raw, table, candidates, gates, lengths, scale)
        selected = top_k_page_table_transform(exact, candidates, self.refine_lengths[candidate_k], 2048)
        return selected, exact
