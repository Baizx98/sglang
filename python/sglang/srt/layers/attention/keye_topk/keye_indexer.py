"""KeyeIndexer for KeyeTopKMask Sparse Attention

Computes sparse attention indices using a lightweight FP8 indexer network:
    I_{t,s} = sum_{j=1}^{H^I} w^I_{t,j} * ReLU(q^I_{t,j} * k^I_s)

Current implementation:
  KeyeIndexer — unified FP8 path:
      projections/norms/RoPE/Hadamard → act_quant → FP8 paged cache → deep_gemm kernels
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, ceil_align, is_cuda, is_hip

if is_cuda():
    try:
        import deep_gemm
    except ImportError as e:
        deep_gemm = e

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KeyeTokenToKVPool

_is_cuda = is_cuda()
_is_hip = is_hip()

logger = logging.getLogger(__name__)
_use_sm80_dsa = os.getenv("KEYE_SM80_DSA", "0") == "1"
_trace_counter = itertools.count()
_trace_decode_counters: dict[Tuple[str, int], int] = {}
_trace_chunk_buffers: dict[Tuple[str, int], list[dict[str, object]]] = {}
_lookahead_counter = itertools.count()
_lookahead_decode_counters: dict[Tuple[str, int], int] = {}
_lookahead_chunk_buffers: dict[Tuple[str, int], list[dict[str, object]]] = {}
_rescore_counter = itertools.count()
_rescore_decode_counters: dict[Tuple[str, int], int] = {}
_rescore_chunk_buffers: dict[Tuple[str, int], list[dict[str, object]]] = {}

DUAL_STREAM_TOKEN_THRESHOLD = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _use_exact_topk_sm80() -> bool:
    """Return whether SM80 experiments require an exact FP32 top-k."""
    raw_value = os.getenv("KEYE_SM80_EXACT_TOPK", "0").strip()
    if raw_value not in {"0", "1"}:
        raise ValueError("KEYE_SM80_EXACT_TOPK must be either '0' or '1'")
    return raw_value == "1"


def _should_trace_layer(layer_id: int) -> bool:
    """Return whether the configured top-k trace includes ``layer_id``."""
    layer_spec = os.getenv("KEYE_SM80_TRACE_LAYERS", "all").strip()
    if layer_spec.lower() == "all":
        return True
    if not layer_spec:
        raise ValueError(
            "KEYE_SM80_TRACE_LAYERS must be 'all' or a comma-separated layer list"
        )

    try:
        traced_layers = {
            int(value.strip()) for value in layer_spec.split(",") if value.strip()
        }
    except ValueError as exc:
        raise ValueError(
            "KEYE_SM80_TRACE_LAYERS must be 'all' or a comma-separated layer list"
        ) from exc
    if not traced_layers or any(layer < 0 for layer in traced_layers):
        raise ValueError("KEYE_SM80_TRACE_LAYERS must contain non-negative layer IDs")
    return layer_id in traced_layers


def _trace_mode() -> str:
    """Return the configured payload mode for SM80 trace records."""
    mode = os.getenv("KEYE_SM80_TRACE_MODE", "topk").strip().lower()
    if mode not in {"topk", "score", "both", "compact"}:
        raise ValueError(
            "KEYE_SM80_TRACE_MODE must be 'topk', 'score', 'both', or 'compact'"
        )
    return mode


def _trace_compact_k() -> int:
    """Return the ranked candidate width saved by compact schema v5."""
    raw_value = os.getenv("KEYE_SM80_TRACE_COMPACT_K", "4096").strip()
    try:
        compact_k = int(raw_value)
    except ValueError as exc:
        raise ValueError("KEYE_SM80_TRACE_COMPACT_K must be at least 2048") from exc
    if compact_k < 2048:
        raise ValueError("KEYE_SM80_TRACE_COMPACT_K must be at least 2048")
    return compact_k


def _trace_compact_ranks() -> list[int]:
    """Return score-rank thresholds retained by compact schema v5."""
    raw_value = os.getenv(
        "KEYE_SM80_TRACE_COMPACT_RANKS", "2048,2560,3072,4096"
    ).strip()
    try:
        ranks = sorted(
            {int(value.strip()) for value in raw_value.split(",") if value.strip()}
        )
    except ValueError as exc:
        raise ValueError(
            "KEYE_SM80_TRACE_COMPACT_RANKS must be comma-separated positive ints"
        ) from exc
    if not ranks or ranks[0] <= 0:
        raise ValueError(
            "KEYE_SM80_TRACE_COMPACT_RANKS must be comma-separated positive ints"
        )
    if ranks[-1] > _trace_compact_k():
        raise ValueError(
            "KEYE_SM80_TRACE_COMPACT_RANKS cannot exceed KEYE_SM80_TRACE_COMPACT_K"
        )
    return ranks


def _trace_score_block_size() -> int:
    """Return the token block size used for compact score summaries."""
    raw_value = os.getenv("KEYE_SM80_TRACE_SCORE_BLOCK_SIZE", "256").strip()
    try:
        block_size = int(raw_value)
    except ValueError as exc:
        raise ValueError("KEYE_SM80_TRACE_SCORE_BLOCK_SIZE must be positive") from exc
    if block_size <= 0:
        raise ValueError("KEYE_SM80_TRACE_SCORE_BLOCK_SIZE must be positive")
    return block_size


def _trace_full_score_layers() -> set[int]:
    """Return layers whose compact records additionally retain full scores."""
    raw_value = os.getenv("KEYE_SM80_TRACE_FULL_SCORE_LAYERS", "").strip()
    if not raw_value:
        return set()
    try:
        layers = {
            int(value.strip()) for value in raw_value.split(",") if value.strip()
        }
    except ValueError as exc:
        raise ValueError(
            "KEYE_SM80_TRACE_FULL_SCORE_LAYERS must be comma-separated layer IDs"
        ) from exc
    if any(layer < 0 for layer in layers):
        raise ValueError("KEYE_SM80_TRACE_FULL_SCORE_LAYERS must be non-negative")
    return layers


def _should_trace_full_score(request_id: str, layer_id: int) -> bool:
    """Return whether a compact row belongs to the explicit full-score sample."""
    if layer_id not in _trace_full_score_layers():
        return False
    raw_prefixes = os.getenv("KEYE_SM80_TRACE_FULL_SCORE_RID_PREFIXES", "").strip()
    if not raw_prefixes:
        return True
    prefixes = [value.strip() for value in raw_prefixes.split(",") if value.strip()]
    return any(request_id.startswith(prefix) for prefix in prefixes)


def _trace_decode_step_limit() -> int:
    """Return the per-request, per-layer decode-step limit (0 means unlimited)."""
    raw_value = os.getenv("KEYE_SM80_TRACE_DECODE_STEPS", "0").strip()
    try:
        limit = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "KEYE_SM80_TRACE_DECODE_STEPS must be a non-negative integer"
        ) from exc
    if limit < 0:
        raise ValueError("KEYE_SM80_TRACE_DECODE_STEPS must be a non-negative integer")
    return limit


def _trace_chunk_steps() -> int:
    """Return the number of decode rows stored in one trace chunk."""
    raw_value = os.getenv("KEYE_SM80_TRACE_CHUNK_STEPS", "0").strip()
    try:
        chunk_steps = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "KEYE_SM80_TRACE_CHUNK_STEPS must be a non-negative integer"
        ) from exc
    if chunk_steps < 0:
        raise ValueError(
            "KEYE_SM80_TRACE_CHUNK_STEPS must be a non-negative integer"
        )
    return chunk_steps


def _trace_rid_prefix() -> str:
    """Return the optional request-id prefix used to exclude health/warmup calls."""
    return os.getenv("KEYE_SM80_TRACE_RID_PREFIX", "").strip()


def _lookahead_trace_dir() -> Optional[Path]:
    value = os.getenv("KEYE_LOOKAHEAD_TRACE_DIR", "").strip()
    return Path(value) if value else None


def _lookahead_decode_steps() -> int:
    raw_value = os.getenv("KEYE_LOOKAHEAD_DECODE_STEPS", "32").strip()
    try:
        steps = int(raw_value)
    except ValueError as exc:
        raise ValueError("KEYE_LOOKAHEAD_DECODE_STEPS must be positive") from exc
    if steps <= 0:
        raise ValueError("KEYE_LOOKAHEAD_DECODE_STEPS must be positive")
    return steps


def _lookahead_max_k() -> int:
    raw_value = os.getenv("KEYE_LOOKAHEAD_MAX_K", "3072").strip()
    try:
        max_k = int(raw_value)
    except ValueError as exc:
        raise ValueError("KEYE_LOOKAHEAD_MAX_K must be at least 2048") from exc
    if max_k < 2048:
        raise ValueError("KEYE_LOOKAHEAD_MAX_K must be at least 2048")
    return max_k


def _lookahead_rid_prefix() -> str:
    return os.getenv("KEYE_LOOKAHEAD_RID_PREFIX", "").strip()


def _lookahead_use_layers() -> set[int]:
    raw = os.getenv("KEYE_LOOKAHEAD_USE_LAYERS", "").strip()
    if not raw:
        return set()
    try:
        layers = {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError as exc:
        raise ValueError("KEYE_LOOKAHEAD_USE_LAYERS must be comma-separated integers") from exc
    invalid = sorted(layer for layer in layers if layer < 1)
    if invalid:
        raise ValueError(f"lookahead target layers must be positive: {invalid}")
    return layers


def _rescore_candidate_k(layer_id: Optional[int] = None) -> int:
    raw_value = os.getenv("KEYE_RESCORE_CANDIDATE_K", "3072").strip()
    try:
        candidate_k = int(raw_value)
    except ValueError as exc:
        raise ValueError("KEYE_RESCORE_CANDIDATE_K must be at least 2048") from exc
    if candidate_k < 2048:
        raise ValueError("KEYE_RESCORE_CANDIDATE_K must be at least 2048")

    raw_overrides = os.getenv("KEYE_RESCORE_CANDIDATE_K_BY_LAYER", "").strip()
    if raw_overrides:
        try:
            overrides = {
                int(layer.strip()): int(value.strip())
                for item in raw_overrides.split(",")
                if item.strip()
                for layer, value in [item.split(":", 1)]
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "KEYE_RESCORE_CANDIDATE_K_BY_LAYER must use layer:k pairs"
            ) from exc
        invalid = {
            layer: value
            for layer, value in overrides.items()
            if layer < 1 or value < 2048
        }
        if invalid:
            raise ValueError(
                "rescore per-layer candidate K requires layer >= 1 and K >= 2048: "
                f"{invalid}"
            )
        if layer_id is not None:
            candidate_k = overrides.get(layer_id, candidate_k)
    return candidate_k


def _rescore_use_layers() -> set[int]:
    raw = os.getenv("KEYE_RESCORE_USE_LAYERS", "").strip()
    if not raw:
        return set()
    try:
        layers = {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError as exc:
        raise ValueError("KEYE_RESCORE_USE_LAYERS must be comma-separated integers") from exc
    invalid = sorted(layer for layer in layers if layer < 1)
    if invalid:
        raise ValueError(f"rescore target layers must be positive: {invalid}")
    return layers


def _rescore_enabled() -> bool:
    """Allow request-boundary exact/rescore A/B tests in one loaded server."""
    control_file = os.getenv("KEYE_RESCORE_ENABLE_FILE", "").strip()
    return not control_file or Path(control_file).is_file()


def _rescore_exact_rid_prefix() -> str:
    """Return the request prefix whose rows use full exact top-k as controls."""
    return os.getenv("KEYE_RESCORE_EXACT_RID_PREFIX", "").strip()


def _rescore_trace_dir() -> Optional[Path]:
    value = os.getenv("KEYE_RESCORE_TRACE_DIR", "").strip()
    return Path(value) if value else None


def _rescore_decode_steps() -> int:
    raw_value = os.getenv("KEYE_RESCORE_DECODE_STEPS", "32").strip()
    try:
        steps = int(raw_value)
    except ValueError as exc:
        raise ValueError("KEYE_RESCORE_DECODE_STEPS must be positive") from exc
    if steps <= 0:
        raise ValueError("KEYE_RESCORE_DECODE_STEPS must be positive")
    return steps


def _rescore_rid_prefix() -> str:
    return os.getenv("KEYE_RESCORE_RID_PREFIX", "").strip()


def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Hadamard transform."""
    # Upstream moved hadamard_transform out of sgl_kernel; mirror nsa_indexer.rotate_activation:
    # HIP -> fast_hadamard_transform (AMD), CUDA -> sglang.jit_kernel.hadamard.
    if _is_hip:
        from fast_hadamard_transform import hadamard_transform
    else:
        from sglang.jit_kernel.hadamard import hadamard_transform

    hidden_size = x.size(-1)
    assert (hidden_size & (hidden_size - 1)) == 0, (
        f"Hidden size ({hidden_size}) must be a power of 2 for Hadamard transform."
    )
    assert x.dtype == torch.bfloat16
    return hadamard_transform(x, scale=hidden_size**-0.5)


# ---------------------------------------------------------------------------
# KeyeIndexer — unified FP8 indexer implementation
# ---------------------------------------------------------------------------


class KeyeIndexer(MultiPlatformOp):
    """FP8-accelerated Keye Top-K sparse attention indexer (MQA).

    Responsibilities:
      - Build lightweight indexer Q/K/gate activations.
      - Apply norm + MRoPE + Hadamard transform.
      - Quantise Q/K to FP8 and write K + scale to the index cache.
      - Compute top-k indices for prefill/decode via deep_gemm kernels.
    """

    _fp8_logged: bool = False

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        topk: int,
        mrope_section: List[int],
        rope_theta: float = 8000000.0,
        main_head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        scale_fmt: Optional[str] = "ue8m0",
        block_size: int = 128,
        layer_id: int = 0,
        alt_stream: Optional[torch.cuda.Stream] = None,
        quant_config: Optional[object] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.topk = topk
        self.layer_id = layer_id
        self.total_q_dim = num_heads * head_dim
        self.softmax_scale = head_dim**-0.5
        self.alt_stream = alt_stream
        self._last_sm80_scores: Optional[torch.Tensor] = None
        self._last_rescore_candidate_indices: Optional[torch.Tensor] = None
        self._last_rescore_exact_indices: Optional[torch.Tensor] = None

        assert main_head_dim is not None, (
            "main_head_dim is required for mrope_section scaling"
        )
        scaling_factor = self.head_dim / main_head_dim
        self.indexer_mrope_section = [int(x * scaling_factor) for x in mrope_section]
        assert sum(self.indexer_mrope_section) * 2 == head_dim, (
            f"Indexer mrope section {self.indexer_mrope_section} sum*2 != head_dim {head_dim}"
        )

        # Use MRotaryEmbedding (via get_rope_wrapper) to align with SGLang framework style.
        # Provides a pre-computed cos_sin_cache (O(N) lookup vs recompute each forward)
        # and dispatches to the Triton MRoPE kernel on CUDA.
        self.rotary_emb = get_rope_wrapper(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            is_neox_style=True,
            rope_scaling={
                "rope_type": "default",
                "mrope_section": self.indexer_mrope_section,
            },
            dtype=torch.bfloat16,
            device=get_global_server_args().device,
        )

        self.scale_fmt = scale_fmt
        self.block_size = block_size

        self.q_proj = ReplicatedLinear(
            hidden_size,
            self.total_q_dim + num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ReplicatedLinear(
            hidden_size,
            head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )

        from sglang.srt.layers.layernorm import LayerNorm, RMSNorm

        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6, dtype=torch.float32)

        if not _is_cuda:
            raise RuntimeError("KeyeIndexer requires CUDA")
        if not _use_sm80_dsa and isinstance(deep_gemm, Exception):
            raise RuntimeError(
                "KeyeIndexer requires CUDA and deep_gemm. "
                "Make sure deep_gemm is installed."
            )

        self.sm_count = 0 if _use_sm80_dsa else deep_gemm.get_num_sms()
        self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)

        assert self.head_dim in (32, 64, 128), (
            f"KeyeIndexer head_dim={self.head_dim} must be 32, 64, or 128"
        )
        self.block_size = min(self.head_dim, self.block_size)

        if _use_sm80_dsa:
            logger.info(
                "KeyeIndexer: using exact BF16 Ampere path "
                f"(head_dim={self.head_dim}, num_heads={self.num_heads}, topk={self.topk})"
            )
        elif not KeyeIndexer._fp8_logged:
            KeyeIndexer._fp8_logged = True
            logger.info(
                "KeyeIndexer: using FP8 path "
                f"(head_dim={self.head_dim}, num_heads={self.num_heads}, topk={self.topk}, "
                f"block_size={self.block_size}, scale_fmt={self.scale_fmt!r}, "
                f"sm_count={self.sm_count})"
            )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute only bf16 K (skip Q and gate-w). Used by the skip_topk fast path."""
        key, _ = self.k_proj(x)
        key = self.k_norm(key)

        # MRotaryEmbedding.forward expects positions [3, N] and tensors [N, D].
        # Use key as a dummy query; only the returned key is used.
        if positions.dim() == 1:
            positions_2d = positions.unsqueeze(0).expand(3, -1)  # [3, N]
        else:
            positions_2d = positions  # already [3, N]

        if key.dtype != torch.bfloat16:
            key = key.to(torch.bfloat16)
        _, key = self.rotary_emb(positions_2d, key, key)
        return _rotate_activation(key)

    def _get_q_k_w_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute bf16 Q [N, H, D], K [N, D] (single MQA head), and gate-w [N, H].

        Steps: linear projection → reshape + norm → MRoPE → Hadamard transform.
        """

        total_tokens = x.shape[0]
        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_and_w, _ = self.q_proj(x)
            query = q_and_w[..., : self.total_q_dim]
            w_raw = q_and_w[..., self.total_q_dim :]
            query = query.view(total_tokens, self.num_heads, self.head_dim)
            query = self.q_norm(query.reshape(-1, self.head_dim)).view(
                total_tokens, self.num_heads, self.head_dim
            )
            with torch.cuda.stream(self.alt_stream):
                key, _ = self.k_proj(x)
                key = self.k_norm(key)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_and_w, _ = self.q_proj(x)
            key, _ = self.k_proj(x)
            query = q_and_w[..., : self.total_q_dim]
            w_raw = q_and_w[..., self.total_q_dim :]
            query = query.view(total_tokens, self.num_heads, self.head_dim)
            query = self.q_norm(query.reshape(-1, self.head_dim)).view(
                total_tokens, self.num_heads, self.head_dim
            )
            key = self.k_norm(key)

        # MRotaryEmbedding.forward expects:
        #   positions: [3, N]  (2D MRoPE)
        #   query:     [N, num_heads * head_dim]  (flattened)
        #   key:       [N, head_dim]
        # and returns the already-rotated (query, key) in the same shapes.
        if positions.dim() == 1:
            positions_2d = positions.unsqueeze(0).expand(3, -1)  # [3, N]
        else:
            positions_2d = positions  # already [3, N]

        if query.dtype != torch.bfloat16:
            query = query.to(torch.bfloat16)
        if key.dtype != torch.bfloat16:
            key = key.to(torch.bfloat16)

        q_2d = query.view(total_tokens, self.total_q_dim)  # [N, H*D]
        q_2d, key = self.rotary_emb(positions_2d, q_2d, key)
        query = q_2d.view(total_tokens, self.num_heads, self.head_dim)  # [N, H, D]

        # Hadamard transform (applied after RoPE, same as before)
        q_flat = query.reshape(total_tokens * self.num_heads, self.head_dim)

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = _rotate_activation(q_flat).view(
                total_tokens, self.num_heads, self.head_dim
            )
            with torch.cuda.stream(self.alt_stream):
                key = _rotate_activation(key)
            current_stream.wait_stream(self.alt_stream)
        else:
            query = _rotate_activation(q_flat).view(
                total_tokens, self.num_heads, self.head_dim
            )
            key = _rotate_activation(key)

        return query, key, w_raw

    def _should_chunk_mqa_logits(
        self, num_q: int, num_k: int, device: torch.device
    ) -> Tuple[bool, int]:
        """Return (need_chunk, free_mem_bytes). Chunks large logits to avoid OOM."""
        if num_q * num_k < 8_000_000:
            return False, 0
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        logits_bytes = num_q * num_k * 4  # float32
        need_chunk = (logits_bytes * 2 > free_mem) or (logits_bytes > total_mem * 0.3)
        return need_chunk, free_mem

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(self, w_raw: torch.Tensor, q_scale: torch.Tensor):
        """Fold per-head FP8 scale into gate weights: weights[t,h] = w[t,h] * q_scale[t,h] * softmax_scale."""
        w = w_raw.float()
        weights = w.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights.squeeze(-1)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward_cuda(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        """Run the unified FP8 indexer and return top-k indices."""
        if not return_indices:
            return None
        if not hasattr(forward_batch.token_to_kv_pool, "index_head_dim"):
            return None
        if x.shape[0] == 0:
            return torch.full((0, self.topk), -1, dtype=torch.int32, device=x.device)

        metadata: Optional[BaseIndexerMetadata] = (
            forward_batch.attn_backend.get_indexer_metadata(layer_id, forward_batch)
        )
        if metadata is None:
            return None

        if _use_sm80_dsa:
            result, trace_scores = self._forward_sm80(
                x, positions, forward_batch, layer_id, metadata
            )
            self._last_sm80_scores = (
                trace_scores if _lookahead_trace_dir() is not None else None
            )
            self._dump_topk_sm80(
                result, trace_scores, positions, forward_batch, layer_id
            )
            return result

        total_tokens = x.shape[0]
        pool = forward_batch.token_to_kv_pool
        from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        loc = forward_batch.out_cache_loc
        if not loc.is_contiguous():
            loc = loc.contiguous()
        if loc.dtype != torch.int64:
            loc = loc.to(torch.int64)

        # Fast path: when max_kv_len <= topk every position is selected; skip logits.
        if (
            forward_batch.forward_mode.is_extend_without_speculative()
            and forward_batch.seq_lens_cpu is not None
            and len(forward_batch.seq_lens_cpu) > 0
        ):
            max_kv_len = int(forward_batch.seq_lens_cpu.max().item())
            if max_kv_len <= self.topk:
                key = self._get_k_bf16(x, positions)
                k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
                pool.set_index_k_scale_buffer(
                    layer_id=layer_id,
                    loc=loc,
                    index_k=k_fp8,
                    index_k_scale=k_scale,
                )

                device = x.device
                seq_lens_cpu = forward_batch.seq_lens_cpu
                extend_lens_cpu = forward_batch.extend_seq_lens_cpu
                result = torch.full(
                    (total_tokens, self.topk), -1, dtype=torch.int32, device=device
                )
                q_offset = 0
                for i in range(forward_batch.batch_size):
                    seq_len = int(seq_lens_cpu[i].item())
                    extend_len = int(extend_lens_cpu[i])
                    history = seq_len - extend_len
                    if extend_len > 0:
                        col_idx = torch.arange(
                            max_kv_len, dtype=torch.int32, device=device
                        )
                        valid_counts = torch.arange(
                            history + 1,
                            history + extend_len + 1,
                            dtype=torch.int32,
                            device=device,
                        )
                        mask = col_idx.unsqueeze(0) < valid_counts.unsqueeze(1)
                        result[q_offset : q_offset + extend_len, :max_kv_len] = (
                            torch.where(
                                mask,
                                col_idx.unsqueeze(0).expand(extend_len, -1),
                                torch.full_like(
                                    col_idx.unsqueeze(0).expand(extend_len, -1), -1
                                ),
                            )
                        )
                    q_offset += extend_len
                return result

        enable_dual_stream = (
            self.alt_stream is not None
            and get_is_capture_mode()
            and 0 < x.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        if enable_dual_stream and forward_batch.forward_mode.is_decode_or_idle():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query, key, w_raw = self._get_q_k_w_bf16(x, positions, enable_dual_stream)
            q_fp8_2d, q_scale_2d = act_quant(
                query.reshape(-1, self.head_dim), self.block_size, self.scale_fmt
            )
            with torch.cuda.stream(self.alt_stream):
                k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
                pool.set_index_k_scale_buffer(
                    layer_id=layer_id,
                    loc=loc,
                    index_k=k_fp8,
                    index_k_scale=k_scale,
                )
            current_stream.wait_stream(self.alt_stream)
        else:
            query, key, w_raw = self._get_q_k_w_bf16(x, positions, enable_dual_stream)
            q_fp8_2d, q_scale_2d = act_quant(
                query.reshape(-1, self.head_dim), self.block_size, self.scale_fmt
            )
            k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
            pool.set_index_k_scale_buffer(
                layer_id=layer_id,
                loc=loc,
                index_k=k_fp8,
                index_k_scale=k_scale,
            )

        q_fp8 = q_fp8_2d.view(total_tokens, self.num_heads, self.head_dim)
        q_scale = q_scale_2d.view(total_tokens, self.num_heads, 1)
        weights = self._get_logits_head_gate(w_raw, q_scale)

        if forward_batch.forward_mode.is_extend_without_speculative():
            return self._get_topk_ragged(
                forward_batch, layer_id, q_fp8, weights, metadata
            )
        return self._get_topk_paged(forward_batch, layer_id, q_fp8, weights, metadata)

    def _all_causal_indices_sm80(
        self, forward_batch: ForwardBatch, total_tokens: int, max_kv_len: int
    ) -> torch.Tensor:
        """Return every valid logical token when the sequence fits in top-k."""
        result = torch.full(
            (total_tokens, self.topk),
            -1,
            dtype=torch.int32,
            device=forward_batch.out_cache_loc.device,
        )
        cols = torch.arange(max_kv_len, dtype=torch.int32, device=result.device)
        q_offset = 0
        for i in range(forward_batch.batch_size):
            seq_len = int(forward_batch.seq_lens_cpu[i].item())
            extend_len = int(forward_batch.extend_seq_lens_cpu[i])
            history = seq_len - extend_len
            if extend_len:
                valid_counts = torch.arange(
                    history + 1,
                    history + extend_len + 1,
                    dtype=torch.int32,
                    device=result.device,
                )
                rows = cols.unsqueeze(0).expand(extend_len, -1)
                result[q_offset : q_offset + extend_len, :max_kv_len] = torch.where(
                    rows < valid_counts.unsqueeze(1), rows, -1
                )
            q_offset += extend_len
        return result

    def _compact_trace_row_sm80(
        self,
        scores: torch.Tensor,
        score_valid_count: int,
    ) -> dict[str, object]:
        """Build the bounded schema-v5 payload for one decode row.

        Threshold-relative block counts preserve the score distribution around
        the candidate cutoffs without writing the full context-length vector.
        The canonical top-2048 is stored separately because a generic
        ``torch.topk`` need not make the same choice at exact score ties as the
        serving kernel.
        """
        compact_k = _trace_compact_k()
        threshold_ranks = _trace_compact_ranks()
        block_size = _trace_score_block_size()
        valid_scores = scores[:score_valid_count].float()
        actual_k = min(compact_k, score_valid_count)
        candidate_scores, candidate_indices = torch.topk(
            valid_scores, actual_k, dim=0, sorted=True
        )

        padded_indices = torch.full(
            (compact_k,), -1, dtype=torch.int32, device=scores.device
        )
        padded_scores = torch.full(
            (compact_k,), -float("inf"), dtype=torch.float32, device=scores.device
        )
        padded_indices[:actual_k] = candidate_indices.to(torch.int32)
        padded_scores[:actual_k] = candidate_scores

        thresholds = torch.full(
            (len(threshold_ranks),),
            float("nan"),
            dtype=torch.float32,
            device=scores.device,
        )
        for threshold_index, rank in enumerate(threshold_ranks):
            if rank <= actual_k:
                thresholds[threshold_index] = candidate_scores[rank - 1]

        block_count = (score_valid_count + block_size - 1) // block_size
        padded_width = block_count * block_size
        padded = torch.zeros(padded_width, dtype=torch.float32, device=scores.device)
        padded[:score_valid_count] = valid_scores
        blocks = padded.view(block_count, block_size)
        counts = torch.full(
            (block_count,), block_size, dtype=torch.int32, device=scores.device
        )
        counts[-1] = score_valid_count - (block_count - 1) * block_size
        valid_mask = (
            torch.arange(block_size, device=scores.device).unsqueeze(0)
            < counts.unsqueeze(1)
        )
        count_float = counts.float()
        block_mean = blocks.sum(dim=1) / count_float
        centered = torch.where(valid_mask, blocks - block_mean.unsqueeze(1), 0.0)
        block_std = torch.sqrt((centered * centered).sum(dim=1) / count_float)
        block_min = blocks.masked_fill(~valid_mask, float("inf")).min(dim=1).values
        block_max = blocks.masked_fill(~valid_mask, -float("inf")).max(dim=1).values
        threshold_counts = torch.zeros(
            (block_count, len(threshold_ranks)),
            dtype=torch.int32,
            device=scores.device,
        )
        for threshold_index, threshold in enumerate(thresholds):
            if torch.isfinite(threshold):
                threshold_counts[:, threshold_index] = (
                    (blocks >= threshold) & valid_mask
                ).sum(dim=1, dtype=torch.int32)

        return {
            "candidate_indices": padded_indices.detach().cpu(),
            "candidate_scores": padded_scores.detach().cpu(),
            "score_thresholds": thresholds.detach().cpu(),
            "block_valid_counts": counts.detach().cpu(),
            "block_score_mean": block_mean.detach().cpu(),
            "block_score_std": block_std.detach().cpu(),
            "block_score_min": block_min.detach().cpu(),
            "block_score_max": block_max.detach().cpu(),
            "block_threshold_counts": threshold_counts.detach().cpu(),
        }

    def _dump_topk_sm80(
        self,
        indices: torch.Tensor,
        scores: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> None:
        """Persist optional snapshots and per-token trace shards on TP rank 0."""
        if torch.cuda.current_device() != 0:
            return

        path = os.getenv("KEYE_SM80_DUMP_TOPK")
        if path and layer_id == 0:
            row = indices[-1].detach().cpu()
            torch.save(
                {
                    "indices": row,
                    "valid_count": int((row >= 0).sum().item()),
                    "query_row": indices.shape[0] - 1,
                    "forward_mode": str(forward_batch.forward_mode),
                    "seq_lens": (
                        forward_batch.seq_lens.detach().cpu()
                        if forward_batch.seq_lens is not None
                        else None
                    ),
                },
                path,
            )

        trace_dir_value = os.getenv("KEYE_SM80_TRACE_DIR")
        if not trace_dir_value:
            return
        if not _should_trace_layer(layer_id):
            return

        trace_mode = _trace_mode()
        decode_step_limit = _trace_decode_step_limit()
        is_decode = forward_batch.forward_mode.is_decode()
        if trace_mode in {"score", "both", "compact"} and scores is None:
            # Score traces intentionally cover decode steps only. Prefill can
            # have thousands of query rows and would dominate both I/O and
            # storage without helping the planned cross-step comparison.
            return

        row_indices = list(range(indices.shape[0]))
        decode_step_ids: Optional[List[int]] = None
        request_ids = list(getattr(forward_batch, "rids", None) or [])
        rid_prefix = _trace_rid_prefix()
        if is_decode and len(request_ids) == indices.shape[0]:
            selected_rows = []
            decode_step_ids = []
            for row_index, request_id in enumerate(request_ids):
                if rid_prefix and not request_id.startswith(rid_prefix):
                    continue
                counter_key = (request_id, layer_id)
                decode_step = _trace_decode_counters.get(counter_key, 0)
                _trace_decode_counters[counter_key] = decode_step + 1
                if not decode_step_limit or decode_step < decode_step_limit:
                    selected_rows.append(row_index)
                    decode_step_ids.append(decode_step)
            if not selected_rows:
                return
            row_indices = selected_rows
        elif decode_step_limit or rid_prefix:
            # Formal trace collection is decode-only and requires a stable
            # row-to-request mapping.
            if not is_decode or len(request_ids) != indices.shape[0]:
                return

        row_selector = torch.tensor(
            row_indices, dtype=torch.long, device=indices.device
        )
        indices = indices.index_select(0, row_selector)
        if scores is not None:
            scores = scores.index_select(0, row_selector)
        if positions.shape[-1] == len(request_ids):
            positions = positions.index_select(
                positions.dim() - 1,
                row_selector.to(device=positions.device),
            )
        selected_request_ids = (
            [request_ids[row_index] for row_index in row_indices]
            if len(request_ids) >= max(row_indices, default=-1) + 1
            else request_ids
        )

        trace_dir = Path(trace_dir_value)
        trace_dir.mkdir(parents=True, exist_ok=True)

        def optional_cpu(name: str):
            value = getattr(forward_batch, name, None)
            return value.detach().cpu() if isinstance(value, torch.Tensor) else value

        indices_cpu = indices.detach().to(device="cpu", dtype=torch.int32)
        positions_cpu = positions.detach().to(device="cpu")
        valid_counts = (indices_cpu >= 0).sum(dim=-1).to(torch.int32)
        score_valid_counts = optional_cpu("seq_lens")
        if isinstance(score_valid_counts, torch.Tensor) and score_valid_counts.shape[
            0
        ] == len(request_ids):
            score_valid_counts = score_valid_counts.index_select(
                0, torch.tensor(row_indices, dtype=torch.long)
            ).to(torch.int32)
        input_ids_cpu = optional_cpu("input_ids")
        if isinstance(input_ids_cpu, torch.Tensor) and input_ids_cpu.shape[0] == len(
            request_ids
        ):
            input_ids_cpu = input_ids_cpu.index_select(
                0, torch.tensor(row_indices, dtype=torch.long)
            )
        req_pool_indices_cpu = optional_cpu("req_pool_indices")
        if isinstance(
            req_pool_indices_cpu, torch.Tensor
        ) and req_pool_indices_cpu.shape[0] == len(request_ids):
            req_pool_indices_cpu = req_pool_indices_cpu.index_select(
                0, torch.tensor(row_indices, dtype=torch.long)
            )

        # SGLang packs prefill tokens request-by-request, while decode has one
        # row per request.  Persist an explicit row mapping so concurrent
        # benchmark requests can be reconstructed without relying on ephemeral
        # request-pool slots.
        token_req_pool_indices = None
        token_request_ids = None
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if isinstance(extend_lens, torch.Tensor):
            extend_lens = extend_lens.tolist()
        if (
            extend_lens is not None
            and sum(int(length) for length in extend_lens) == indices_cpu.shape[0]
        ):
            repeats = torch.tensor(extend_lens, dtype=torch.long)
            if isinstance(req_pool_indices_cpu, torch.Tensor):
                token_req_pool_indices = torch.repeat_interleave(
                    req_pool_indices_cpu[: len(extend_lens)], repeats
                )
            if len(selected_request_ids) >= len(extend_lens):
                token_request_ids = [
                    selected_request_ids[request_index]
                    for request_index, length in enumerate(extend_lens)
                    for _ in range(int(length))
                ]
        elif (
            isinstance(req_pool_indices_cpu, torch.Tensor)
            and req_pool_indices_cpu.numel() == indices_cpu.shape[0]
        ):
            token_req_pool_indices = req_pool_indices_cpu.reshape(-1).clone()
            if len(selected_request_ids) == indices_cpu.shape[0]:
                token_request_ids = selected_request_ids.copy()

        chunk_steps = _trace_chunk_steps()
        if chunk_steps:
            if not is_decode or decode_step_ids is None:
                return
            if decode_step_limit and chunk_steps > decode_step_limit:
                raise ValueError(
                    "KEYE_SM80_TRACE_CHUNK_STEPS cannot exceed "
                    "KEYE_SM80_TRACE_DECODE_STEPS"
                )
            for output_row, request_id in enumerate(selected_request_ids):
                score_valid_count = (
                    int(score_valid_counts[output_row].item())
                    if isinstance(score_valid_counts, torch.Tensor)
                    else int(scores.shape[1])
                )
                keep_full_score = trace_mode == "compact" and _should_trace_full_score(
                    request_id, layer_id
                )
                row_score = (
                    scores[output_row, :score_valid_count]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    if scores is not None
                    and (trace_mode in {"score", "both"} or keep_full_score)
                    else None
                )
                compact_payload = (
                    self._compact_trace_row_sm80(
                        scores[output_row],
                        score_valid_count,
                    )
                    if scores is not None and trace_mode == "compact"
                    else None
                )
                row_input_id = (
                    input_ids_cpu[output_row].clone()
                    if isinstance(input_ids_cpu, torch.Tensor)
                    and input_ids_cpu.numel() > output_row
                    else None
                )
                row_position = (
                    positions_cpu[..., output_row].clone()
                    if positions_cpu.shape[-1] == len(selected_request_ids)
                    else positions_cpu.clone()
                )
                self._append_trace_chunk_row(
                    trace_dir=trace_dir,
                    trace_mode=trace_mode,
                    request_id=request_id,
                    layer_id=layer_id,
                    decode_step_id=decode_step_ids[output_row],
                    indices=(
                        indices_cpu[output_row].clone()
                        if trace_mode in {"topk", "both", "compact"}
                        else None
                    ),
                    score=row_score,
                    compact_payload=compact_payload,
                    keep_full_score=keep_full_score,
                    valid_count=int(valid_counts[output_row].item()),
                    score_valid_count=score_valid_count,
                    input_id=row_input_id,
                    position=row_position,
                    chunk_steps=chunk_steps,
                )
            return

        if trace_mode == "compact":
            raise ValueError(
                "KEYE_SM80_TRACE_MODE=compact requires "
                "KEYE_SM80_TRACE_CHUNK_STEPS > 0"
            )

        event_id = next(_trace_counter)
        timestamp_ns = time.time_ns()
        file_name = f"event_{event_id:09d}_{timestamp_ns}_layer_{layer_id:02d}.pt"
        final_path = trace_dir / file_name
        temp_path = trace_dir / f".{file_name}.tmp"

        record = {
            "schema_version": 3,
            "event_id": event_id,
            "timestamp_ns": timestamp_ns,
            "layer_id": layer_id,
            "forward_mode": int(forward_batch.forward_mode),
            "trace_mode": trace_mode,
            "topk_backend": "torch_exact" if _use_exact_topk_sm80() else "fast_topk_v2",
            "indices": indices_cpu if trace_mode in {"topk", "both"} else None,
            "scores": (
                scores.detach().to(device="cpu", dtype=torch.float32)
                if scores is not None and trace_mode in {"score", "both"}
                else None
            ),
            "valid_counts": valid_counts,
            "score_valid_counts": score_valid_counts,
            "decode_step_ids": decode_step_ids,
            "input_ids": input_ids_cpu,
            "positions": positions_cpu,
            "seq_lens": optional_cpu("seq_lens"),
            "extend_seq_lens": optional_cpu("extend_seq_lens"),
            "extend_seq_lens_cpu": getattr(forward_batch, "extend_seq_lens_cpu", None),
            "request_ids": selected_request_ids,
            "req_pool_indices": req_pool_indices_cpu,
            "token_request_ids": token_request_ids,
            "token_req_pool_indices": token_req_pool_indices,
            "out_cache_loc": optional_cpu("out_cache_loc"),
        }
        torch.save(record, temp_path)
        os.replace(temp_path, final_path)

        manifest_record = {
            "event_id": event_id,
            "timestamp_ns": timestamp_ns,
            "file": file_name,
            "layer_id": layer_id,
            "forward_mode": int(forward_batch.forward_mode),
            "num_tokens": indices_cpu.shape[0],
            "topk_width": indices_cpu.shape[1],
            "score_width": scores.shape[1] if scores is not None else None,
            "valid_min": int(valid_counts.min().item()),
            "valid_max": int(valid_counts.max().item()),
            "decode_step_ids": decode_step_ids,
            "request_ids": selected_request_ids,
        }
        with (trace_dir / "manifest.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(manifest_record, separators=(",", ":")) + "\n")

    def _append_trace_chunk_row(
        self,
        *,
        trace_dir: Path,
        trace_mode: str,
        request_id: str,
        layer_id: int,
        decode_step_id: int,
        indices: Optional[torch.Tensor],
        score: Optional[torch.Tensor],
        compact_payload: Optional[dict[str, object]],
        keep_full_score: bool,
        valid_count: int,
        score_valid_count: int,
        input_id: Optional[torch.Tensor],
        position: torch.Tensor,
        chunk_steps: int,
    ) -> None:
        """Buffer one decode row and atomically persist a schema-v4/v5 chunk."""
        buffer_key = (request_id, layer_id)
        rows = _trace_chunk_buffers.setdefault(buffer_key, [])
        rows.append(
            {
                "decode_step_id": decode_step_id,
                "indices": indices,
                "score": score,
                "compact_payload": compact_payload,
                "keep_full_score": keep_full_score,
                "valid_count": valid_count,
                "score_valid_count": score_valid_count,
                "input_id": input_id,
                "position": position,
            }
        )
        if len(rows) < chunk_steps:
            return
        if len(rows) > chunk_steps:
            raise RuntimeError(f"Trace chunk overflow for {buffer_key}: {len(rows)}")

        rows.sort(key=lambda row: int(row["decode_step_id"]))
        decode_step_ids = [int(row["decode_step_id"]) for row in rows]
        if len(set(decode_step_ids)) != len(decode_step_ids):
            raise RuntimeError(f"Duplicate decode steps for trace chunk {buffer_key}")

        score_valid_counts = torch.tensor(
            [int(row["score_valid_count"]) for row in rows], dtype=torch.int32
        )
        max_score_width = int(score_valid_counts.max().item())
        scores_cpu = None
        if trace_mode in {"score", "both"} or (
            trace_mode == "compact"
            and all(bool(row["keep_full_score"]) for row in rows)
        ):
            scores_cpu = torch.zeros(
                (chunk_steps, max_score_width), dtype=torch.float32
            )
            for row_index, row in enumerate(rows):
                row_score = row["score"]
                if not isinstance(row_score, torch.Tensor):
                    raise RuntimeError(f"Missing score row for {buffer_key}")
                scores_cpu[row_index, : row_score.numel()] = row_score

        indices_cpu = None
        if trace_mode in {"topk", "both", "compact"}:
            index_rows = [row["indices"] for row in rows]
            if not all(isinstance(row, torch.Tensor) for row in index_rows):
                raise RuntimeError(f"Missing top-k row for {buffer_key}")
            indices_cpu = torch.stack(index_rows).to(torch.int32)

        compact_record: dict[str, object] = {}
        if trace_mode == "compact":
            payloads = [row["compact_payload"] for row in rows]
            if not all(isinstance(payload, dict) for payload in payloads):
                raise RuntimeError(f"Missing compact payload for {buffer_key}")
            typed_payloads = [
                payload for payload in payloads if isinstance(payload, dict)
            ]
            fixed_fields = [
                "candidate_indices",
                "candidate_scores",
                "score_thresholds",
            ]
            for field in fixed_fields:
                values = [payload[field] for payload in typed_payloads]
                if not all(isinstance(value, torch.Tensor) for value in values):
                    raise RuntimeError(f"Missing compact field {field} for {buffer_key}")
                compact_record[field] = torch.stack(values)

            block_fields = [
                "block_valid_counts",
                "block_score_mean",
                "block_score_std",
                "block_score_min",
                "block_score_max",
            ]
            max_blocks = max(
                int(payload["block_valid_counts"].numel())
                for payload in typed_payloads
            )
            compact_record["score_block_counts"] = torch.tensor(
                [
                    int(payload["block_valid_counts"].numel())
                    for payload in typed_payloads
                ],
                dtype=torch.int32,
            )
            for field in block_fields:
                first = typed_payloads[0][field]
                if not isinstance(first, torch.Tensor):
                    raise RuntimeError(f"Missing compact field {field} for {buffer_key}")
                fill_value = 0 if field == "block_valid_counts" else float("nan")
                output = torch.full(
                    (chunk_steps, max_blocks), fill_value, dtype=first.dtype
                )
                for row_index, payload in enumerate(typed_payloads):
                    value = payload[field]
                    if not isinstance(value, torch.Tensor):
                        raise RuntimeError(
                            f"Missing compact field {field} for {buffer_key}"
                        )
                    output[row_index, : value.numel()] = value
                compact_record[field] = output

            threshold_count_rows = [
                payload["block_threshold_counts"] for payload in typed_payloads
            ]
            if not all(
                isinstance(value, torch.Tensor) for value in threshold_count_rows
            ):
                raise RuntimeError(
                    f"Missing compact block thresholds for {buffer_key}"
                )
            num_thresholds = int(threshold_count_rows[0].shape[1])
            block_threshold_counts = torch.zeros(
                (chunk_steps, max_blocks, num_thresholds), dtype=torch.int32
            )
            for row_index, value in enumerate(threshold_count_rows):
                block_threshold_counts[row_index, : value.shape[0]] = value
            compact_record["block_threshold_counts"] = block_threshold_counts
            compact_record.update(
                {
                    "compact_k": _trace_compact_k(),
                    "score_threshold_ranks": torch.tensor(
                        _trace_compact_ranks(), dtype=torch.int32
                    ),
                    "score_block_size": _trace_score_block_size(),
                    "full_scores_retained": scores_cpu is not None,
                }
            )

        input_ids = [row["input_id"] for row in rows]
        input_ids_cpu = (
            torch.stack(input_ids)
            if all(isinstance(value, torch.Tensor) for value in input_ids)
            else None
        )
        positions_cpu = torch.stack(
            [row["position"] for row in rows]
        )
        valid_counts = torch.tensor(
            [int(row["valid_count"]) for row in rows], dtype=torch.int32
        )

        event_id = next(_trace_counter)
        timestamp_ns = time.time_ns()
        safe_request_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_id)[:96]
        file_name = (
            f"chunk_{event_id:09d}_{safe_request_id}_"
            f"layer_{layer_id:02d}_steps_{decode_step_ids[0]:03d}-"
            f"{decode_step_ids[-1]:03d}.pt"
        )
        final_path = trace_dir / file_name
        temp_path = trace_dir / f".{file_name}.tmp"
        record = {
            "schema_version": 5 if trace_mode == "compact" else 4,
            "event_id": event_id,
            "timestamp_ns": timestamp_ns,
            "layer_id": layer_id,
            "forward_mode": 2,
            "trace_mode": trace_mode,
            "request_id": request_id,
            "request_ids": [request_id] * chunk_steps,
            "topk_backend": (
                "torch_exact" if _use_exact_topk_sm80() else "fast_topk_v2"
            ),
            "decode_step_ids": decode_step_ids,
            "indices": indices_cpu,
            "scores": scores_cpu,
            "valid_counts": valid_counts,
            "score_valid_counts": score_valid_counts,
            "input_ids": input_ids_cpu,
            "positions": positions_cpu,
            **compact_record,
        }
        torch.save(record, temp_path)
        os.replace(temp_path, final_path)

        manifest_record = {
            "schema_version": 5 if trace_mode == "compact" else 4,
            "event_id": event_id,
            "timestamp_ns": timestamp_ns,
            "file": file_name,
            "layer_id": layer_id,
            "request_id": request_id,
            "request_ids": [request_id],
            "topk_backend": "torch_exact" if _use_exact_topk_sm80() else "fast_topk_v2",
            "num_steps": chunk_steps,
            "decode_step_ids": decode_step_ids,
            "topk_width": indices_cpu.shape[1] if indices_cpu is not None else None,
            "score_width": scores_cpu.shape[1] if scores_cpu is not None else None,
            "compact_k": _trace_compact_k() if trace_mode == "compact" else None,
            "score_threshold_ranks": (
                _trace_compact_ranks() if trace_mode == "compact" else None
            ),
            "score_block_size": (
                _trace_score_block_size() if trace_mode == "compact" else None
            ),
            "full_scores_retained": (
                scores_cpu is not None if trace_mode == "compact" else None
            ),
            "valid_min": int(valid_counts.min().item()),
            "valid_max": int(valid_counts.max().item()),
            "score_valid_min": int(score_valid_counts.min().item()),
            "score_valid_max": int(score_valid_counts.max().item()),
            "bytes": final_path.stat().st_size,
        }
        with (trace_dir / "manifest.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(manifest_record, separators=(",", ":")) + "\n")
        del _trace_chunk_buffers[buffer_key]

    def _score_sm80(
        self, query: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the original Keye BF16 indexer score equation."""
        keys_t = keys.transpose(0, 1).unsqueeze(0).expand(query.shape[0], -1, -1)
        per_head = torch.bmm(query, keys_t)
        per_head = torch.relu(per_head.float() * self.softmax_scale)
        return (per_head * weights.float().unsqueeze(-1)).sum(dim=1)

    def _topk_sm80(self, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Select top-k with the optimized DSA kernel when its contract applies."""
        if _use_exact_topk_sm80():
            # Correctness/reference path for experiments.  The optimized radix
            # kernel can make boundary substitutions on highly clustered DSA
            # score distributions, so it must not define the trace ground truth.
            return self._topk_with_lengths(scores, lengths, self.topk)
        if self.topk == 2048 and scores.shape[1] >= self.topk:
            from sgl_kernel import fast_topk_v2

            if lengths.dtype != torch.int32:
                lengths = lengths.to(torch.int32)
            return fast_topk_v2(scores, lengths, self.topk)

        actual_topk = min(self.topk, scores.shape[1])
        cols = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
        scores.masked_fill_(cols >= lengths.unsqueeze(1), -float("inf"))
        values, indices = torch.topk(scores, actual_topk, dim=-1)
        indices = indices.to(torch.int32).masked_fill(~values.isfinite(), -1)
        if actual_topk == self.topk:
            return indices
        result = torch.full(
            (scores.shape[0], self.topk),
            -1,
            dtype=torch.int32,
            device=scores.device,
        )
        result[:, :actual_topk] = indices
        return result

    def _forward_sm80(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        metadata: BaseIndexerMetadata,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Exact BF16 index selection for GPUs without FP8/DeepGEMM."""
        pool = forward_batch.token_to_kv_pool
        loc = forward_batch.out_cache_loc
        if loc.dtype != torch.long:
            loc = loc.to(dtype=torch.long)
        is_extend = forward_batch.forward_mode.is_extend_without_speculative()

        if is_extend and len(forward_batch.seq_lens_cpu) > 0:
            max_kv_len = int(forward_batch.seq_lens_cpu.max().item())
            if max_kv_len <= self.topk:
                key = self._get_k_bf16(x, positions)
                pool.set_index_k_bf16_buffer(layer_id, loc, key)
                return (
                    self._all_causal_indices_sm80(
                        forward_batch, x.shape[0], max_kv_len
                    ),
                    None,
                )

        query, key, weights = self._get_q_k_w_bf16(
            x, positions, enable_dual_stream=False
        )
        pool.set_index_k_bf16_buffer(layer_id, loc, key)
        key_cache = pool.get_index_k_bf16_buffer(layer_id)
        page_table = metadata.get_page_table_1()

        if not is_extend:
            from sglang.srt.layers.attention.keye_topk.ampere_indexer import (
                keye_indexer_score_paged,
            )

            seqlens = metadata.get_seqlens_int32()
            scores = keye_indexer_score_paged(
                query,
                key_cache,
                page_table,
                weights,
                seqlens,
                self.softmax_scale,
            )
            return self._topk_sm80(scores, seqlens), scores

        result = torch.full(
            (x.shape[0], self.topk), -1, dtype=torch.int32, device=x.device
        )
        q_offset = 0
        chunk_size = 64
        for i in range(forward_batch.batch_size):
            seq_len = int(forward_batch.seq_lens_cpu[i].item())
            extend_len = int(forward_batch.extend_seq_lens_cpu[i])
            history = seq_len - extend_len
            slots = page_table[i, :seq_len].to(dtype=torch.long)
            keys = key_cache[slots]
            for start in range(0, extend_len, chunk_size):
                end = min(start + chunk_size, extend_len)
                scores = self._score_sm80(
                    query[q_offset + start : q_offset + end],
                    keys,
                    weights[q_offset + start : q_offset + end],
                )
                valid = history + torch.arange(
                    start + 1,
                    end + 1,
                    device=x.device,
                    dtype=torch.int32,
                )
                indices = self._topk_sm80(scores, valid)
                result[q_offset + start : q_offset + end] = indices
            q_offset += extend_len
        return result, None

    def _topk_with_lengths(
        self, scores: torch.Tensor, lengths: torch.Tensor, topk: int
    ) -> torch.Tensor:
        """Return exact score-ranked indices with -1 padding to ``topk``."""
        width = min(topk, scores.shape[1])
        cols = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
        masked_scores = scores.masked_fill(cols >= lengths.unsqueeze(1), -float("inf"))
        values, indices = torch.topk(masked_scores, width, dim=-1)
        indices = indices.to(torch.int32).masked_fill(~values.isfinite(), -1)
        if width == topk:
            return indices
        output = torch.full(
            (scores.shape[0], topk), -1, dtype=torch.int32, device=scores.device
        )
        output[:, :width] = indices
        return output

    def forward_cross_layer_rescore_sm80(
        self,
        *,
        previous_x: torch.Tensor,
        current_x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        """Synchronously build lookahead candidates and exactly rescore them."""
        if (
            not _use_sm80_dsa
            or not _rescore_enabled()
            or layer_id not in _rescore_use_layers()
        ):
            return None
        if layer_id <= 0 or not forward_batch.forward_mode.is_decode():
            return None
        if previous_x.shape != current_x.shape or current_x.shape[0] == 0:
            return None
        if get_is_capture_mode():
            # The experiment path has not yet registered fixed-shape graph buffers.
            return None

        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )
        if metadata is None:
            return None

        pool = forward_batch.token_to_kv_pool
        if not hasattr(pool, "index_head_dim"):
            return None
        key_cache = pool.get_index_k_bf16_buffer(layer_id)
        page_table = metadata.get_page_table_1()
        seqlens = metadata.get_seqlens_int32()
        candidate_k = _rescore_candidate_k(layer_id)

        from sglang.srt.layers.attention.keye_topk.ampere_indexer import (
            keye_indexer_score_candidates,
            keye_indexer_score_paged,
        )

        # Candidate generation intentionally uses only historical target-layer K.
        lookahead_query, _, lookahead_weights = self._get_q_k_w_bf16(
            previous_x, positions, enable_dual_stream=False
        )
        lookahead_scores = keye_indexer_score_paged(
            lookahead_query,
            key_cache,
            page_table,
            lookahead_weights,
            torch.clamp(seqlens - 1, min=0),
            self.softmax_scale,
        )
        history_lengths = torch.clamp(seqlens - 1, min=0)
        history_indices = self._topk_with_lengths(
            lookahead_scores, history_lengths, candidate_k - 1
        )
        self_indices = (seqlens - 1).to(torch.int32).unsqueeze(1)
        candidate_indices = torch.cat((self_indices, history_indices), dim=1)

        # Measure candidate-ready -> target attention separately from the
        # previous-step same-layer deadline. This adds events only when the
        # explicit deadline trace environment is enabled.
        from sglang.srt.layers.attention.keye_topk.deadline_trace import (
            record_cross_layer_candidate_ready,
        )

        record_cross_layer_candidate_ready(forward_batch, layer_id)

        # Only exact current-layer K is allowed to mutate the canonical cache.
        exact_query, exact_key, exact_weights = self._get_q_k_w_bf16(
            current_x, positions, enable_dual_stream=False
        )
        loc = forward_batch.out_cache_loc
        if loc.dtype != torch.long:
            loc = loc.to(dtype=torch.long)
        pool.set_index_k_bf16_buffer(layer_id, loc, exact_key)

        candidate_scores = keye_indexer_score_candidates(
            exact_query,
            key_cache,
            page_table,
            candidate_indices,
            exact_weights,
            seqlens,
            self.softmax_scale,
        )
        candidate_lengths = torch.clamp(seqlens, max=candidate_k).to(torch.int32)
        candidate_offsets = self._topk_sm80(candidate_scores, candidate_lengths)
        safe_offsets = candidate_offsets.clamp_min(0).to(torch.long)
        final_indices = candidate_indices.gather(1, safe_offsets)
        final_indices = final_indices.masked_fill(candidate_offsets < 0, -1)

        trace_dir = _rescore_trace_dir()
        exact_rid_prefix = _rescore_exact_rid_prefix()
        exact_control_mask = None
        if exact_rid_prefix:
            request_ids = list(getattr(forward_batch, "rids", None) or [])
            if len(request_ids) != final_indices.shape[0]:
                raise RuntimeError(
                    "KEYE_RESCORE_EXACT_RID_PREFIX requires one request id per "
                    f"decode row, got {len(request_ids)} ids for "
                    f"{final_indices.shape[0]} rows"
                )
            exact_control_mask = torch.tensor(
                [rid.startswith(exact_rid_prefix) for rid in request_ids],
                dtype=torch.bool,
                device=final_indices.device,
            )
        verify_full = (
            trace_dir is not None
            or os.getenv("KEYE_RESCORE_VERIFY_FULL", "0") == "1"
            or exact_control_mask is not None
            or bool(os.getenv("KEYE_RESCORE_ATTN_SHADOW_DIR", "").strip())
        )
        if verify_full:
            full_scores = keye_indexer_score_paged(
                exact_query,
                key_cache,
                page_table,
                exact_weights,
                seqlens,
                self.softmax_scale,
            )
            exact_indices = self._topk_sm80(full_scores, seqlens)
            safe_candidates = candidate_indices.clamp_min(0).to(torch.long)
            gathered_full_scores = full_scores.gather(1, safe_candidates)
            valid_candidates = candidate_indices >= 0
            score_difference = torch.where(
                valid_candidates,
                (candidate_scores - gathered_full_scores).abs(),
                torch.zeros_like(candidate_scores),
            )
            max_score_difference = score_difference.max(dim=1).values
            if trace_dir is not None:
                self._trace_cross_layer_rescore_sm80(
                    trace_dir=trace_dir,
                    candidate_indices=candidate_indices,
                    final_indices=final_indices,
                    exact_indices=exact_indices,
                    max_score_difference=max_score_difference,
                    seqlens=seqlens,
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    candidate_k=candidate_k,
                )
            if not torch.all(max_score_difference == 0):
                raise RuntimeError(
                    "restricted candidate scores differ from the full exact scorer: "
                    f"max_abs={float(max_score_difference.max().item())}"
                )

            self._last_rescore_candidate_indices = final_indices.detach()
            self._last_rescore_exact_indices = exact_indices.detach()

            # Paired quality experiments place identical exact and candidate
            # requests in the same scheduler batch. Selecting per row removes
            # cross-request launch and MoE/TP drift from the comparison while
            # keeping both variants in one model forward.
            if exact_control_mask is not None:
                final_indices = torch.where(
                    exact_control_mask.unsqueeze(1), exact_indices, final_indices
                )

        return final_indices

    def _trace_cross_layer_rescore_sm80(
        self,
        *,
        trace_dir: Path,
        candidate_indices: torch.Tensor,
        final_indices: torch.Tensor,
        exact_indices: torch.Tensor,
        max_score_difference: torch.Tensor,
        seqlens: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        candidate_k: int,
    ) -> None:
        """Persist compact per-request fidelity records for restricted rescoring."""
        if torch.cuda.current_device() != 0:
            return
        request_ids = list(getattr(forward_batch, "rids", None) or [])
        if len(request_ids) != candidate_indices.shape[0]:
            return

        rid_prefix = _rescore_rid_prefix()
        step_limit = _rescore_decode_steps()
        trace_dir.mkdir(parents=True, exist_ok=True)
        for row_index, request_id in enumerate(request_ids):
            if rid_prefix and not request_id.startswith(rid_prefix):
                continue
            counter_key = (request_id, layer_id)
            decode_step = _rescore_decode_counters.get(counter_key, 0)
            _rescore_decode_counters[counter_key] = decode_step + 1
            if decode_step >= step_limit:
                continue
            self._append_rescore_chunk_row(
                trace_dir=trace_dir,
                request_id=request_id,
                target_layer_id=layer_id,
                candidate_k=candidate_k,
                chunk_steps=step_limit,
                row={
                    "decode_step_id": decode_step,
                    "valid_count": int(seqlens[row_index].item()),
                    "candidate_indices": candidate_indices[row_index]
                    .detach()
                    .to(device="cpu", dtype=torch.int32),
                    "final_indices": final_indices[row_index]
                    .detach()
                    .to(device="cpu", dtype=torch.int32),
                    "exact_indices": exact_indices[row_index]
                    .detach()
                    .to(device="cpu", dtype=torch.int32),
                    "candidate_score_max_abs": float(
                        max_score_difference[row_index].item()
                    ),
                },
            )

    def _append_rescore_chunk_row(
        self,
        *,
        trace_dir: Path,
        request_id: str,
        target_layer_id: int,
        candidate_k: int,
        chunk_steps: int,
        row: dict[str, object],
    ) -> None:
        buffer_key = (request_id, target_layer_id)
        rows = _rescore_chunk_buffers.setdefault(buffer_key, [])
        rows.append(row)
        if len(rows) < chunk_steps:
            return
        if len(rows) > chunk_steps:
            raise RuntimeError(f"Rescore trace chunk overflow for {buffer_key}")

        rows.sort(key=lambda value: int(value["decode_step_id"]))
        decode_step_ids = [int(value["decode_step_id"]) for value in rows]
        if decode_step_ids != list(range(chunk_steps)):
            raise RuntimeError(
                f"Non-contiguous rescore steps for {buffer_key}: {decode_step_ids}"
            )

        event_id = next(_rescore_counter)
        timestamp_ns = time.time_ns()
        safe_request_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_id)[:96]
        file_name = (
            f"chunk_{event_id:09d}_{safe_request_id}_target_layer_"
            f"{target_layer_id:02d}_steps_000-{chunk_steps - 1:03d}.pt"
        )
        final_path = trace_dir / file_name
        temp_path = trace_dir / f".{file_name}.tmp"
        valid_counts = torch.tensor(
            [int(value["valid_count"]) for value in rows], dtype=torch.int32
        )
        record = {
            "schema_version": 1,
            "event_id": event_id,
            "timestamp_ns": timestamp_ns,
            "request_id": request_id,
            "source_layer_id": target_layer_id - 1,
            "target_layer_id": target_layer_id,
            "decode_step_ids": decode_step_ids,
            "valid_counts": valid_counts,
            "candidate_k": candidate_k,
            "final_topk": self.topk,
            "candidate_indices": torch.stack(
                [value["candidate_indices"] for value in rows]
            ).to(torch.int32),
            "final_indices": torch.stack(
                [value["final_indices"] for value in rows]
            ).to(torch.int32),
            "exact_indices": torch.stack(
                [value["exact_indices"] for value in rows]
            ).to(torch.int32),
            "candidate_score_max_abs": torch.tensor(
                [value["candidate_score_max_abs"] for value in rows],
                dtype=torch.float32,
            ),
            "self_token_forced_into_candidate": True,
            "final_self_token_forced": False,
            "canonical_exact_k_cache": True,
            "synchronous": True,
        }
        torch.save(record, temp_path)
        os.replace(temp_path, final_path)
        manifest_record = {
            "schema_version": 1,
            "event_id": event_id,
            "timestamp_ns": timestamp_ns,
            "file": file_name,
            "request_id": request_id,
            "source_layer_id": target_layer_id - 1,
            "target_layer_id": target_layer_id,
            "num_steps": chunk_steps,
            "candidate_k": candidate_k,
            "final_topk": self.topk,
            "valid_min": int(valid_counts.min().item()),
            "valid_max": int(valid_counts.max().item()),
            "max_candidate_score_abs": max(
                float(value["candidate_score_max_abs"]) for value in rows
            ),
            "bytes": final_path.stat().st_size,
        }
        with (trace_dir / "manifest.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(manifest_record, separators=(",", ":")) + "\n")
        del _rescore_chunk_buffers[buffer_key]

    def trace_cross_layer_lookahead_sm80(
        self,
        *,
        previous_x: torch.Tensor,
        current_x: torch.Tensor,
        previous_indices: torch.Tensor,
        exact_indices: torch.Tensor,
        exact_scores: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        """Record shadow scores and optionally return top-k for selected layers."""
        trace_dir = _lookahead_trace_dir()
        use_for_attention = layer_id in _lookahead_use_layers()
        record_enabled = trace_dir is not None and torch.cuda.current_device() == 0
        if (not record_enabled and not use_for_attention) or not _use_sm80_dsa:
            return None
        if layer_id <= 0 or not forward_batch.forward_mode.is_decode():
            return None
        if previous_x.shape != current_x.shape:
            return None
        if record_enabled and exact_scores is None:
            return None

        request_ids = list(getattr(forward_batch, "rids", None) or [])
        if len(request_ids) != exact_indices.shape[0]:
            return None
        rid_prefix = _lookahead_rid_prefix()
        step_limit = _lookahead_decode_steps()
        selected_rows: list[int] = []
        decode_steps: list[int] = []
        selected_request_ids: list[str] = []
        if record_enabled:
            for row_index, request_id in enumerate(request_ids):
                if rid_prefix and not request_id.startswith(rid_prefix):
                    continue
                counter_key = (request_id, layer_id)
                decode_step = _lookahead_decode_counters.get(counter_key, 0)
                _lookahead_decode_counters[counter_key] = decode_step + 1
                if decode_step >= step_limit:
                    continue
                selected_rows.append(row_index)
                decode_steps.append(decode_step)
                selected_request_ids.append(request_id)
        if not selected_rows and not use_for_attention:
            return None

        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )
        if metadata is None:
            return None
        query, _, weights = self._get_q_k_w_bf16(
            previous_x, positions, enable_dual_stream=False
        )
        pool = forward_batch.token_to_kv_pool
        key_cache = pool.get_index_k_bf16_buffer(layer_id)
        page_table = metadata.get_page_table_1()
        seqlens = metadata.get_seqlens_int32()
        from sglang.srt.layers.attention.keye_topk.ampere_indexer import (
            keye_indexer_score_paged,
        )

        lookahead_scores = keye_indexer_score_paged(
            query,
            key_cache,
            page_table,
            weights,
            seqlens,
            self.softmax_scale,
        )
        max_k = max(_lookahead_max_k(), self.topk)
        history_lengths = torch.clamp(seqlens - 1, min=0)
        history_indices = self._topk_with_lengths(
            lookahead_scores, history_lengths, max_k - 1
        )
        self_indices = (seqlens - 1).to(torch.int32).unsqueeze(1)
        lookahead_indices = torch.cat((self_indices, history_indices), dim=1)

        selfcheck_max_abs: Optional[torch.Tensor] = None
        selfcheck_topk_equal: Optional[torch.Tensor] = None
        if record_enabled and os.getenv("KEYE_LOOKAHEAD_SELFCHECK", "0") == "1" and any(
            step == 0 for step in decode_steps
        ):
            exact_query, _, exact_weights = self._get_q_k_w_bf16(
                current_x, positions, enable_dual_stream=False
            )
            recomputed_scores = keye_indexer_score_paged(
                exact_query,
                key_cache,
                page_table,
                exact_weights,
                seqlens,
                self.softmax_scale,
            )
            valid_mask = torch.arange(
                exact_scores.shape[1], device=exact_scores.device
            ).unsqueeze(0) < seqlens.unsqueeze(1)
            difference = torch.where(
                valid_mask,
                (recomputed_scores - exact_scores).abs(),
                torch.zeros_like(exact_scores),
            )
            selfcheck_max_abs = difference.max(dim=1).values
            recomputed_topk = self._topk_sm80(recomputed_scores, seqlens)
            selfcheck_topk_equal = torch.sort(recomputed_topk, dim=1).values.eq(
                torch.sort(exact_indices, dim=1).values
            ).all(dim=1)

        if trace_dir is not None and selected_rows:
            trace_dir.mkdir(parents=True, exist_ok=True)
        for list_index, row_index in enumerate(selected_rows):
            valid_count = int(seqlens[row_index].item())
            row = {
                "decode_step_id": decode_steps[list_index],
                "valid_count": valid_count,
                "exact_indices": exact_indices[row_index].detach().to(
                    device="cpu", dtype=torch.int32
                ),
                "lookahead_indices": lookahead_indices[row_index].detach().to(
                    device="cpu", dtype=torch.int32
                ),
                "direct_reuse_indices": previous_indices[row_index].detach().to(
                    device="cpu", dtype=torch.int32
                ),
                "exact_scores": exact_scores[row_index, :valid_count].detach().to(
                    device="cpu", dtype=torch.float32
                ),
                "lookahead_scores": lookahead_scores[
                    row_index, :valid_count
                ].detach().to(device="cpu", dtype=torch.float32),
                "selfcheck_max_abs": (
                    float(selfcheck_max_abs[row_index].item())
                    if selfcheck_max_abs is not None and decode_steps[list_index] == 0
                    else None
                ),
                "selfcheck_topk_equal": (
                    bool(selfcheck_topk_equal[row_index].item())
                    if selfcheck_topk_equal is not None
                    and decode_steps[list_index] == 0
                    else None
                ),
            }
            self._append_lookahead_chunk_row(
                # selected_rows is populated only when trace_dir is non-null.
                trace_dir=trace_dir,
                request_id=selected_request_ids[list_index],
                target_layer_id=layer_id,
                max_k=max_k,
                chunk_steps=step_limit,
                row=row,
            )
        if use_for_attention:
            return lookahead_indices[:, : self.topk]
        return None

    def _append_lookahead_chunk_row(
        self,
        *,
        trace_dir: Path,
        request_id: str,
        target_layer_id: int,
        max_k: int,
        chunk_steps: int,
        row: dict[str, object],
    ) -> None:
        buffer_key = (request_id, target_layer_id)
        rows = _lookahead_chunk_buffers.setdefault(buffer_key, [])
        rows.append(row)
        if len(rows) < chunk_steps:
            return
        if len(rows) > chunk_steps:
            raise RuntimeError(f"Lookahead chunk overflow for {buffer_key}")
        rows.sort(key=lambda value: int(value["decode_step_id"]))
        decode_step_ids = [int(value["decode_step_id"]) for value in rows]
        if decode_step_ids != list(range(chunk_steps)):
            raise RuntimeError(
                f"Non-contiguous lookahead steps for {buffer_key}: {decode_step_ids}"
            )

        valid_counts = torch.tensor(
            [int(value["valid_count"]) for value in rows], dtype=torch.int32
        )
        score_width = int(valid_counts.max().item())
        exact_scores = torch.full(
            (chunk_steps, score_width), float("nan"), dtype=torch.float32
        )
        lookahead_scores = torch.full_like(exact_scores, float("nan"))
        for row_index, value in enumerate(rows):
            valid_count = int(value["valid_count"])
            exact_scores[row_index, :valid_count] = value["exact_scores"]
            lookahead_scores[row_index, :valid_count] = value["lookahead_scores"]

        event_id = next(_lookahead_counter)
        timestamp_ns = time.time_ns()
        safe_request_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_id)[:96]
        file_name = (
            f"chunk_{event_id:09d}_{safe_request_id}_target_layer_"
            f"{target_layer_id:02d}_steps_000-{chunk_steps - 1:03d}.pt"
        )
        final_path = trace_dir / file_name
        temp_path = trace_dir / f".{file_name}.tmp"
        record = {
            "schema_version": 1,
            "event_id": event_id,
            "timestamp_ns": timestamp_ns,
            "request_id": request_id,
            "source_layer_id": target_layer_id - 1,
            "target_layer_id": target_layer_id,
            "decode_step_ids": decode_step_ids,
            "valid_counts": valid_counts,
            "exact_indices": torch.stack(
                [value["exact_indices"] for value in rows]
            ).to(torch.int32),
            "lookahead_indices": torch.stack(
                [value["lookahead_indices"] for value in rows]
            ).to(torch.int32),
            "direct_reuse_indices": torch.stack(
                [value["direct_reuse_indices"] for value in rows]
            ).to(torch.int32),
            "exact_scores": exact_scores,
            "lookahead_scores": lookahead_scores,
            "selfcheck_max_abs": [value["selfcheck_max_abs"] for value in rows],
            "selfcheck_topk_equal": [
                value["selfcheck_topk_equal"] for value in rows
            ],
            "max_k": max_k,
            "self_token_forced": True,
            "canonical_historical_k_cache": True,
        }
        torch.save(record, temp_path)
        os.replace(temp_path, final_path)
        manifest_record = {
            "schema_version": 1,
            "event_id": event_id,
            "timestamp_ns": timestamp_ns,
            "file": file_name,
            "request_id": request_id,
            "source_layer_id": target_layer_id - 1,
            "target_layer_id": target_layer_id,
            "num_steps": chunk_steps,
            "score_width": score_width,
            "exact_topk": int(record["exact_indices"].shape[1]),
            "lookahead_max_k": max_k,
            "valid_min": int(valid_counts.min().item()),
            "valid_max": int(valid_counts.max().item()),
            "bytes": final_path.stat().st_size,
        }
        with (trace_dir / "manifest.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(manifest_record, separators=(",", ":")) + "\n")
        del _lookahead_chunk_buffers[buffer_key]

    def _get_topk_ragged(
        self,
        forward_batch,
        layer_id,
        q_fp8,
        weights,
        metadata: Optional[BaseIndexerMetadata] = None,
    ):
        """Prefill top-k via deep_gemm.fp8_mqa_logits (ragged batch)."""
        assert forward_batch.forward_mode.is_extend_without_speculative()
        assert forward_batch.seq_lens_cpu is not None
        assert forward_batch.extend_seq_lens_cpu is not None

        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, KeyeTokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page_size=64"

        block_tables_raw = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :
        ]
        strided = torch.arange(0, block_tables_raw.shape[-1], page_size, device="cuda")
        block_tables = block_tables_raw[:, strided] // page_size

        k_fp8_list, k_scale_list, ks_list, ke_list = [], [], [], []
        q_offset = 0
        k_offset = 0

        for i in range(forward_batch.batch_size):
            seq_len = int(forward_batch.seq_lens_cpu[i].item())
            extend_len = int(forward_batch.extend_seq_lens_cpu[i])
            history_len = seq_len - extend_len

            k_fp8_i, k_scale_i = (
                forward_batch.token_to_kv_pool.get_index_k_scale_buffer(
                    layer_id, seq_len, block_tables[i]
                )
            )
            k_fp8_list.append(k_fp8_i)
            k_scale_list.append(k_scale_i)

            ks_list.append(
                torch.full((extend_len,), k_offset, dtype=torch.int32, device="cuda")
            )
            ke_list.append(
                k_offset
                + history_len
                + torch.arange(1, extend_len + 1, dtype=torch.int32, device="cuda")
            )
            q_offset += extend_len
            k_offset += seq_len

        k_fp8_cat = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
        k_scale_cat = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8_cat, k_scale_cat)
        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)

        total_tokens = q_fp8.shape[0]
        device = q_fp8.device

        block_q = 128 // self.num_heads
        padded_q_offset = (q_offset + block_q - 1) // block_q * block_q
        need_pad = padded_q_offset > q_offset

        def _pad_to(t: torch.Tensor, target_len: int) -> torch.Tensor:
            if t.shape[0] >= target_len:
                return t
            pad_shape = (target_len - t.shape[0],) + t.shape[1:]
            return torch.cat(
                [
                    t,
                    torch.zeros(pad_shape, dtype=t.dtype, device=t.device),
                ],
                dim=0,
            )

        need_chunk, free_mem = self._should_chunk_mqa_logits(
            padded_q_offset, k_offset, device
        )

        if not need_chunk:
            q_in = (
                _pad_to(q_fp8[:q_offset], padded_q_offset)
                if need_pad
                else q_fp8[:q_offset]
            )
            w_in = (
                _pad_to(weights[:q_offset], padded_q_offset)
                if need_pad
                else weights[:q_offset]
            )
            ks_in = _pad_to(ks, padded_q_offset) if need_pad else ks
            ke_in = _pad_to(ke, padded_q_offset) if need_pad else ke
            logits = deep_gemm.fp8_mqa_logits(
                q_in, kv_fp8, w_in, ks_in, ke_in, clean_logits=False
            )
            raw_result = metadata.topk_transform(
                logits[:q_offset],
                self.topk,
                forward_batch,
                ks=ks,
                context_length=k_offset,
            )
            return raw_result

        # Chunk to avoid OOM on large sequences
        bytes_per_row = k_offset * 4
        max_rows = max(block_q, int((free_mem * 0.5) // max(bytes_per_row, 1)))
        max_rows = (min(max_rows, padded_q_offset) // block_q) * block_q
        max_rows = max(max_rows, block_q)

        topk_result = torch.full(
            (total_tokens, self.topk), -1, device=device, dtype=torch.int32
        )
        start = 0
        while start < q_offset:
            end_real = min(start + max_rows, q_offset)
            end_pad = (end_real - start + block_q - 1) // block_q * block_q + start
            chunk_need_pad = end_pad > end_real

            q_chunk = (
                _pad_to(q_fp8[start:end_real], end_pad - start)
                if chunk_need_pad
                else q_fp8[start:end_real]
            )
            w_chunk = (
                _pad_to(weights[start:end_real], end_pad - start)
                if chunk_need_pad
                else weights[start:end_real]
            )
            ks_chunk = (
                _pad_to(ks[start:end_real], end_pad - start)
                if chunk_need_pad
                else ks[start:end_real]
            )
            ke_chunk = (
                _pad_to(ke[start:end_real], end_pad - start)
                if chunk_need_pad
                else ke[start:end_real]
            )

            logits_chunk = deep_gemm.fp8_mqa_logits(
                q_chunk,
                kv_fp8,
                w_chunk,
                ks_chunk,
                ke_chunk,
                clean_logits=False,
            )
            real_len = end_real - start
            logits_real = logits_chunk[:real_len]
            topk_result[start:end_real] = metadata.topk_transform(
                logits_real,
                self.topk,
                forward_batch,
                ks=ks[start:end_real],
                ke_offset=ke[start:end_real] - ks[start:end_real],
                context_length=k_offset,
            )
            start = end_real

        return topk_result

    def _get_topk_paged(
        self,
        forward_batch,
        layer_id,
        q_fp8,
        weights,
        metadata: Optional[BaseIndexerMetadata],
    ):
        """Decode top-k via deep_gemm.fp8_paged_mqa_logits."""
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, KeyeTokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page_size=64"

        seqlens_32 = metadata.get_seqlens_int32()
        block_tables = metadata.get_page_table_64()

        max_seq_len = block_tables.shape[1] * page_size
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        schedule_metadata = getattr(metadata, "paged_mqa_schedule_metadata", None)
        if schedule_metadata is None:
            schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                seqlens_32, page_size, self.sm_count
            )

        head_dim_with_sf = self.head_dim + self.head_dim // self.block_size * 4
        q_4d = q_fp8.unsqueeze(1)
        kv_4d = kv_cache_fp8.view(kv_cache_fp8.shape[0], page_size, 1, head_dim_with_sf)

        logits = deep_gemm.fp8_paged_mqa_logits(
            q_4d,
            kv_4d,
            weights,
            seqlens_32,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )

        return metadata.topk_transform(
            logits, self.topk, forward_batch, ks=None, context_length=max_seq_len
        )
