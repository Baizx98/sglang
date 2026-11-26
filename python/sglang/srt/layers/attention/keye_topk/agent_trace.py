"""Record only the arrays needed to reconstruct per-token Agent activations."""

import os
from pathlib import Path

import numpy as np
import torch

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.attention.keye_topk.ampere_indexer import (
    keye_indexer_score_paged,
)

_pending = {}
_offsets = {}
_CHUNK_STEPS = 32


def predict(indexer, previous_x, positions, batch, layer):
    """Rank historical rows with the target indexer and reserve current slot."""
    metadata = batch.attn_backend.get_indexer_metadata(layer, batch)
    history = (metadata.get_seqlens_int32() - 1).clamp_min(0)
    pages = metadata.get_page_table_1()
    keys = batch.token_to_kv_pool.get_index_k_bf16_buffer(layer)
    query, _, weights = indexer._get_q_k_w_bf16(
        previous_x, positions, enable_dual_stream=False
    )
    scores = keye_indexer_score_paged(
        query, keys, pages, weights, history, indexer.softmax_scale
    )
    older = indexer._topk_with_lengths(scores, history, 3071)
    return torch.cat((history[:, None], older), dim=1)


def _write(key):
    rid, layer = key
    rows = _pending.pop(key)
    start = _offsets.get(key, 0)
    root = Path(os.environ["KEYE_AGENT_TRACE_DIR"])
    output = root / f"{rid}__layer{layer:02d}__{start:04d}.npz"
    fields = {
        "step": np.arange(start, start + len(rows), dtype=np.int32),
        "context_len": np.asarray([row[0] for row in rows], dtype=np.int32),
        "true_topk": np.stack([row[1] for row in rows]),
    }
    if layer:
        fields["predicted_topm"] = np.stack([row[2] for row in rows])
    with output.with_suffix(".tmp").open("xb") as handle:
        np.savez_compressed(handle, **fields)
    if output.exists():
        raise FileExistsError(output)
    output.with_suffix(".tmp").rename(output)
    _offsets[key] = start + len(rows)


def flush():
    if _pending:
        for key in list(_pending):
            _write(key)
    _offsets.clear()


def record_selection(indexer, previous_x, exact, positions, batch, layer):
    if get_tensor_model_parallel_rank() != 0:
        return
    rids = batch.rids or ()
    selected = [i for i, rid in enumerate(rids) if rid.startswith("agenttrace__")]
    if not selected:
        return
    if layer and previous_x is None:
        raise RuntimeError(f"Missing preceding-layer state for layer {layer}")
    predicted = predict(indexer, previous_x, positions, batch, layer) if layer else None
    exact_cpu = exact.detach().cpu().numpy()
    predicted_cpu = predicted.detach().cpu().numpy() if predicted is not None else None
    lengths = batch.seq_lens_cpu
    for i in selected:
        key = (rids[i], layer)
        rows = _pending.setdefault(key, [])
        rows.append((int(lengths[i]), exact_cpu[i], predicted_cpu[i] if layer else None))
        if len(rows) == _CHUNK_STEPS:
            _write(key)
