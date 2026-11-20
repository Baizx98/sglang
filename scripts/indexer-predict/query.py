"""Projection shared by offline and online cross-layer predictor experiments."""

import torch

from projection import _normalize_heads
from sglang.srt.layers.attention.keye_topk.keye_indexer import _rotate_activation


class QueryProjection:
    """Prepare Q/gate from an earlier layer without producing a predicted K."""

    def __init__(self, indexer, batch, device, fused=False):
        self.indexer = indexer
        self.batch = batch
        self.fused = fused
        self.no_keys = torch.empty((batch, 0), dtype=torch.bfloat16, device=device)
        self.normalized = torch.empty((batch, indexer.total_q_dim), dtype=torch.bfloat16, device=device)

    def __call__(self, hidden, positions):
        indexer = self.indexer
        positions = positions if positions.ndim == 2 else positions.unsqueeze(0).expand(3, -1)
        rope = indexer.rotary_emb
        query_gate, _ = indexer.q_proj(hidden)
        if self.fused:
            # The launch contains only query heads. The shared K branch does not run.
            _normalize_heads[(self.batch, indexer.num_heads)](
                query_gate, query_gate, indexer.q_norm.weight, indexer.k_norm.weight,
                indexer.k_norm.bias, self.normalized, self.no_keys, positions, rope.cos_sin_cache,
                query_gate.stride(0), query_gate.stride(0), indexer.q_norm.variance_epsilon,
                indexer.k_norm.variance_epsilon, positions.stride(0),
                *rope.mrope_section[:2], False, num_warps=2, enable_fp_fusion=False,
            )
            query = self.normalized
        else:
            query = query_gate[:, :indexer.total_q_dim].reshape(-1, indexer.head_dim)
            query = indexer.q_norm(query).to(torch.bfloat16).view(self.batch, indexer.total_q_dim)
        query, _ = rope(positions, query, self.no_keys)
        query = _rotate_activation(query.reshape(-1, indexer.head_dim)).view(
            self.batch, indexer.num_heads, indexer.head_dim,
        )
        return query, query_gate[:, indexer.total_q_dim:]


def predictor_projection(indexer, hidden, positions, fused=False):
    prepare = QueryProjection(indexer, hidden.size(0), hidden.device, fused)
    return lambda: prepare(hidden, positions)
