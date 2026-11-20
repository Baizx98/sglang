"""Capture compact real Keye prediction inputs without changing attention choices."""

from collections import defaultdict
import os
from pathlib import Path

import torch

from sglang.srt.layers.attention.keye_topk.keye_indexer import KeyeIndexer


CAPTURE_ROOT = Path(os.environ["KEYE_LOOKAHEAD_TRACE_DIR"])
LAYERS = {int(layer) for layer in os.environ["KEYE_PREDICT_CAPTURE_LAYERS"].split(",")}
STEPS = int(os.environ["KEYE_PREDICT_CAPTURE_STEPS"])
COUNTERS = defaultdict(int)


def capture(
    self, *, previous_x, current_x, previous_indices, exact_indices, exact_scores,
    positions, forward_batch, layer_id,
):
    if layer_id not in LAYERS or not forward_batch.forward_mode.is_decode():
        return
    rids = forward_batch.rids
    if len(rids) != 1 or not rids[0].startswith("predict-full-"):
        return
    rid = rids[0]
    name = (rid, layer_id)
    step = COUNTERS[name]
    if step >= STEPS:
        return
    metadata = forward_batch.attn_backend.get_indexer_metadata(layer_id, forward_batch)
    length = int(metadata.get_seqlens_int32()[0])
    pages = metadata.get_page_table_1()[0, :length]
    raw = forward_batch.token_to_kv_pool.get_index_k_bf16_buffer(layer_id)
    if step == 0:
        keys = raw.index_select(0, pages.long()).cpu()
        torch.save(keys, CAPTURE_ROOT / f"{rid}-l{layer_id:02}-keys.pt")
    torch.save(
        dict(
            request=rid, layer=layer_id, step=step, length=length,
            source_hidden=previous_x.cpu(), target_hidden=current_x.cpu(),
            position=positions.cpu(), target_topk=exact_indices.cpu(),
            target_key=raw.index_select(0, pages[-1:].long()).cpu(),
        ),
        CAPTURE_ROOT / f"{rid}-l{layer_id:02}-s{step:02}.pt",
    )
    COUNTERS[name] += 1


KeyeIndexer.trace_cross_layer_lookahead_sm80 = capture


if __name__ == "__main__":
    import sys

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
