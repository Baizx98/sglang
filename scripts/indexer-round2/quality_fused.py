"""C1 INT4 direct selection with live fused norms and the vLLM selector."""
from __future__ import annotations

import torch

from quality_int4 import Int4History
from projection import _normalize_heads
from vllm_baseline import _extension
from sglang.srt.layers.attention.keye_topk.keye_indexer import _rotate_activation

# Each PP scheduler runs one CUDA device. Workspace ownership follows the stream
# that launches the selector, and every layer keeps its own output buffer.
_workspaces = {}


class FusedInt4Direct(Int4History):
    def __init__(self, key_cache):
        super().__init__(key_cache)
        device = key_cache.device
        self.q_normalized = torch.empty((1, 1024), dtype=torch.bfloat16, device=device)
        self.k_normalized = torch.empty((1, 64), dtype=torch.bfloat16, device=device)
        self.output = torch.empty((1, 2048), dtype=torch.int32, device=device)
        major, minor = torch.cuda.get_device_capability(device)
        self.launch = _extension(f"{major}{minor}")

    def project(self, indexer, hidden, positions):
        q_and_gate, _ = indexer.q_proj(hidden)
        k, _ = indexer.k_proj(hidden)
        rope = indexer.rotary_emb
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        _normalize_heads[(1, 17)](
            q_and_gate, k, indexer.q_norm.weight, indexer.k_norm.weight, indexer.k_norm.bias,
            self.q_normalized, self.k_normalized, positions, rope.cos_sin_cache,
            q_and_gate.stride(0), k.stride(0), indexer.q_norm.variance_epsilon,
            indexer.k_norm.variance_epsilon, positions.stride(0),
            *rope.mrope_section[:2], False, num_warps=2, enable_fp_fusion=False,
        )
        q_roped, k_roped = rope(positions, self.q_normalized, self.k_normalized)
        q = _rotate_activation(q_roped.view(-1, 64)).view(1, 16, 64)
        return q, _rotate_activation(k_roped), q_and_gate[:, 1024:]

    def select_direct(self, q, current_k, gates, raw, table, lengths, scale, restart,
                      live_length):
        scores = self.score(q, current_k, gates, raw, table, lengths, scale, restart)
        stream = torch.cuda.current_stream(scores.device)
        identity = (scores.device, stream.cuda_stream)
        scratch = _workspaces.get(identity)
        if scratch is None:
            scratch = _workspaces[identity] = torch.empty(
                1024 * 1024, dtype=torch.uint8, device=scores.device,
            )
        error = self.launch(
            scores.data_ptr(), lengths.data_ptr(), self.output.data_ptr(),
            scratch.data_ptr(), 1, scores.size(1), scores.stride(0),
            scratch.numel(), live_length, stream.cuda_stream,
        )
        if error:
            raise RuntimeError(error.decode())
        return self.output, scores
