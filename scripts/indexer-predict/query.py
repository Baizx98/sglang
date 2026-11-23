"""Projection shared by offline and online cross-layer predictor experiments."""

import torch

from projection import _normalize_heads
from sglang.srt.layers.attention.keye_topk.keye_indexer import _rotate_activation


class QueryProjection:
    """Prepare Q/gate from an earlier layer without producing a predicted K."""

    def __init__(self, indexer, batch, device, fused=False, full_epilogue=False):
        self.indexer = indexer
        self.batch = batch
        self.fused = fused or full_epilogue
        self.full_epilogue = full_epilogue
        self.no_keys = torch.empty((batch, 0), dtype=torch.bfloat16, device=device)
        self.normalized = torch.empty((batch, indexer.total_q_dim), dtype=torch.bfloat16, device=device)
        if full_epilogue:
            rope = indexer.rotary_emb
            if (rope.mrope_interleaved or rope.mrope_interleaved_glm
                    or not rope.is_neox_style or rope.rotary_dim != 64):
                raise ValueError("the full epilogue requires grouped 64-dimension NeoX MRoPE")
            rope._match_cos_sin_cache_dtype(self.normalized)

    def _fused_postprocess(self, query_gate, positions, out, scales=None, groups=0, packed_nibbles=False):
        indexer = self.indexer
        positions = positions if positions.ndim == 2 else positions.unsqueeze(0).expand(3, -1)
        rope = indexer.rotary_emb
        # The grid contains only query heads. The shared K branch is never used.
        _normalize_heads[(self.batch, indexer.num_heads)](
            query_gate, query_gate, indexer.q_norm.weight, indexer.k_norm.weight,
            indexer.k_norm.bias, out, self.no_keys, positions, rope.cos_sin_cache,
            query_gate.stride(0), query_gate.stride(0), indexer.q_norm.variance_epsilon,
            indexer.k_norm.variance_epsilon, positions.stride(0),
            *rope.mrope_section[:2], self.full_epilogue, self.full_epilogue, scales, groups, packed_nibbles,
            num_warps=2, enable_fp_fusion=False,
        )

    def encode_into(self, hidden, positions, codes, scales, groups, *, packed_nibbles=False):
        if not self.full_epilogue or groups not in (1, 2):
            raise ValueError("the INT8 Q output requires the full epilogue with one or two groups")
        if packed_nibbles and (groups != 1 or codes.dtype != torch.int32):
            raise ValueError("packed Q fragments require one group and an INT32 output buffer")
        query_gate, _ = self.indexer.q_proj(hidden)
        self._fused_postprocess(query_gate, positions, codes, scales, groups, packed_nibbles)
        return query_gate[:, self.indexer.total_q_dim:]

    def __call__(self, hidden, positions):
        indexer = self.indexer
        positions = positions if positions.ndim == 2 else positions.unsqueeze(0).expand(3, -1)
        rope = indexer.rotary_emb
        query_gate, _ = indexer.q_proj(hidden)
        if self.fused:
            self._fused_postprocess(query_gate, positions, self.normalized)
            query = self.normalized
        else:
            query = query_gate[:, :indexer.total_q_dim].reshape(-1, indexer.head_dim)
            query = indexer.q_norm(query).to(torch.bfloat16).view(self.batch, indexer.total_q_dim)
        if not self.full_epilogue:
            query, _ = rope(positions, query, self.no_keys)
            query = _rotate_activation(query.reshape(-1, indexer.head_dim))
        return query.view(self.batch, indexer.num_heads, indexer.head_dim), query_gate[:, indexer.total_q_dim:]


def predictor_projection(indexer, hidden, positions, fused=False, full_epilogue=False):
    prepare = QueryProjection(indexer, hidden.size(0), hidden.device, fused, full_epilogue)
    return lambda: prepare(hidden, positions)
