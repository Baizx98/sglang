"""Run an isolated Keye PP2 server with request-scoped candidate reuse."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys

import torch

from sglang.srt.layers.attention.keye_topk.ampere_indexer import (
    keye_indexer_score_candidates,
    keye_indexer_score_paged,
)
from sglang.srt.layers.attention.keye_topk.keye_indexer import KeyeIndexer

logger = logging.getLogger(__name__)
native_forward = KeyeIndexer._forward_sm80
logged_routes = set()
audit_path = os.getenv("INDEXER_ROUND2_QUALITY_AUDIT")
audit_layers = {1, 12, 25, 36, 47}
int4_modes = {"int4direct": 2048, "int4fused": 2048, "int4": 4096, "int4wide": 8192}

def auditing(batch, layer):
    return (audit_path is not None and layer in audit_layers
            and batch.rids[0].split("__", 2)[-1].startswith("audit_"))


def refine(indexer, x, positions, batch, layer, metadata, previous):
    pool = batch.token_to_kv_pool
    query, key, weights = indexer._get_q_k_w_bf16(x, positions, enable_dual_stream=False)
    pool.set_index_k_bf16_buffer(layer, batch.out_cache_loc.long(), key)
    lengths = metadata.get_seqlens_int32()
    newest = (lengths - 1).unsqueeze(1)
    candidate = torch.cat((previous.masked_fill(previous == newest, -1), newest), dim=1)
    scores = keye_indexer_score_candidates(
        query, pool.get_index_k_bf16_buffer(layer), metadata.get_page_table_1(),
        candidate, weights, lengths, indexer.softmax_scale,
    )
    top = indexer._topk_sm80(scores, torch.full_like(lengths, candidate.shape[1]))
    # The selector masks -inf positions. Retain the logical indices for main Attention.
    output = candidate.gather(1, top.clamp_min(0).long()).masked_fill(top < 0, -1)
    if auditing(batch, layer):
        audit_selection(indexer, batch, layer, metadata, query, weights, output)
    return output, scores


def audit_selection(indexer, batch, layer, metadata, query, weights, output,
                    key_bitwise_equal=None, current_write_equal=None, native_key_equal=None):
    lengths = metadata.get_seqlens_int32()
    keys = batch.token_to_kv_pool.get_index_k_bf16_buffer(layer)
    scores = keye_indexer_score_paged(
        query, keys, metadata.get_page_table_1(), weights, lengths, indexer.softmax_scale,
    )
    exact = indexer._topk_sm80(scores, lengths)
    cutoff = scores.gather(1, exact.long()).amin(1, keepdim=True)
    valid = (output >= 0) & (output < lengths[:, None])
    safe = output.clamp(0, scores.shape[1] - 1).long()
    chosen = scores.gather(1, safe).masked_fill(~valid, -float("inf"))
    ordered = output.sort(dim=1).values
    duplicates = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
    row = {
        "request_id": batch.rids[0],
        "length": int(batch.seq_lens_cpu[0]),
        "layer": layer,
        "overlap": int(torch.isin(output, exact).sum()),
        "selected": int(valid.sum()),
        "invalid_or_missing": int((~valid).sum()),
        "duplicate_indices": int(duplicates.sum()),
        "native_selects_current": bool((exact == lengths[:, None] - 1).any()),
        "result_selects_current": bool((output == lengths[:, None] - 1).any()),
        "strictly_higher_missing": int(((scores > cutoff).sum() - (chosen > cutoff).sum())),
        "selected_below_threshold": int(((chosen < cutoff) & valid).sum()),
        "k_only_matches_qkw": key_bitwise_equal,
        "current_write_matches_projection": current_write_equal,
        "fused_key_matches_native": native_key_equal,
    }
    with (Path(audit_path) / f"candidate-shadow-pp{layer // 24}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def candidate_forward(self, x, positions, batch, layer, metadata):
    request_ids = batch.rids or ()
    if len(request_ids) != 1 or not request_ids[0].startswith("quality__"):
        return native_forward(self, x, positions, batch, layer, metadata)
    request = request_ids[0]
    mode = request.split("__", 2)[1]
    if not batch.forward_mode.is_decode():
        if mode in int4_modes:
            state = getattr(self, int4_state_name(mode), None)
            if state is not None:
                state.last_request = None
        if mode == "skip4":
            self._round2_skip_anchor = None
        return native_forward(self, x, positions, batch, layer, metadata)
    if mode not in {"step", "layer", *int4_modes, "skip4"} or x.shape[0] != 1 or batch.seq_lens_cpu is None:
        return native_forward(self, x, positions, batch, layer, metadata)

    length = int(batch.seq_lens_cpu[0])
    if length <= self.topk:
        return native_forward(self, x, positions, batch, layer, metadata)
    if mode == "skip4":
        return skip_forward(self, x, positions, batch, layer, metadata, request, length)
    if mode in int4_modes:
        if length < int4_modes[mode]:
            return native_forward(self, x, positions, batch, layer, metadata)
        return int4_forward(self, x, positions, batch, layer, metadata, request, length, mode)

    if mode == "step":
        anchor = getattr(self, "_round2_step_anchor", None)
        if anchor is not None and anchor[0] == request and anchor[1] + 1 == length:
            self._round2_step_anchor = None
            output = refine(self, x, positions, batch, layer, metadata, anchor[2])
            log_route(mode, layer, length, True)
            return output
        output = native_forward(self, x, positions, batch, layer, metadata)
        self._round2_step_anchor = (request, length, output[0])
        log_route(mode, layer, length, False)
        return output

    anchor = getattr(batch, "_round2_layer_anchor", None)
    if layer % 2 == 1 and anchor is not None and anchor[0] + 1 == layer:
        output = refine(self, x, positions, batch, layer, metadata, anchor[1])
        log_route(mode, layer, length, True)
        return output
    output = native_forward(self, x, positions, batch, layer, metadata)
    batch._round2_layer_anchor = (layer, output[0])
    log_route(mode, layer, length, False)
    return output


def skip_forward(indexer, x, positions, batch, layer, metadata, request, length):
    anchor = getattr(indexer, "_round2_skip_anchor", None)
    if (anchor is None or anchor[0] != request or anchor[1] + 1 != length
            or length - anchor[2] >= 4):
        result = native_forward(indexer, x, positions, batch, layer, metadata)
        indexer._round2_skip_anchor = (request, length, length, result[0])
        log_route("skip4", layer, length, False, 0)
        return result
    key = indexer._get_k_bf16(x, positions)
    batch.token_to_kv_pool.set_index_k_bf16_buffer(layer, batch.out_cache_loc.long(), key)
    indexer._round2_skip_anchor = (request, length, anchor[2], anchor[3])
    log_route("skip4", layer, length, True, length - anchor[2])
    if auditing(batch, layer):
        query, full_key, weights = indexer._get_q_k_w_bf16(x, positions, enable_dual_stream=False)
        key_equal = torch.equal(key.view(torch.int16), full_key.view(torch.int16))
        audit_selection(indexer, batch, layer, metadata, query, weights, anchor[3], key_equal)
    return anchor[3], None


def int4_forward(indexer, x, positions, batch, layer, metadata, request, length, mode):
    from quality_int4 import Int4History

    pool = batch.token_to_kv_pool
    raw = pool.get_index_k_bf16_buffer(layer)
    name = int4_state_name(mode)
    state = getattr(indexer, name, None)
    if state is None:
        if mode == "int4fused":
            from quality_fused import FusedInt4Direct
            state = FusedInt4Direct(raw)
        else:
            state = Int4History(raw)
        setattr(indexer, name, state)

    if mode == "int4fused":
        q, key, gates = state.project(indexer, x, positions)
    else:
        q, key, gates = indexer._get_q_k_w_bf16(x, positions, enable_dual_stream=False)
    # Preserve the actual BF16 page-table cache as well as the live INT4 sidecar.
    pool.set_index_k_bf16_buffer(layer, batch.out_cache_loc.long(), key)
    restart = state.last_request != (request, length - 1)
    args = (
        q, key, gates, raw, metadata.get_page_table_1(), metadata.get_seqlens_int32(),
        indexer.softmax_scale, restart,
    )
    if mode == "int4fused":
        output, scores = state.select_direct(*args, live_length=length)
    else:
        output, scores = state.select(*args, candidate_k=int4_modes[mode])
    state.last_request = (request, length)
    log_route(mode, layer, length, True)
    if auditing(batch, layer):
        written = raw[batch.out_cache_loc.long()]
        actual_write = torch.equal(written.view(torch.int16), key.view(torch.int16))
        native_key_match = None
        if mode == "int4fused":
            native = indexer._get_k_bf16(x, positions)
            native_key_match = torch.equal(native.view(torch.int16), key.view(torch.int16))
        audit_selection(indexer, batch, layer, metadata, q, gates, output,
                        current_write_equal=actual_write, native_key_equal=native_key_match)
    return output, scores


def int4_state_name(mode):
    return "_round2_int4_fused" if mode == "int4fused" else "_round2_int4"


def log_route(mode, layer, length, approximate, cycle_offset=None):
    group = (mode, approximate, layer // 24, cycle_offset)
    if group not in logged_routes:
        logged_routes.add(group)
        logger.warning("INDEXER_ROUND2 route=%s layer=%d length=%d approximate=%s cycle_offset=%s",
                       mode, layer, length, approximate, cycle_offset)


# Spawned schedulers import this launcher as __mp_main__ and install the same hook.
KeyeIndexer._forward_sm80 = candidate_forward


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    options = prepare_server_args(sys.argv[1:])
    try:
        run_server(options)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
