"""Opt-in C1 task-quality server for BF16 and low-bit cross-layer prediction."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("scripts/indexer-round2", "scripts/indexer-optimization")]

from flashinfer import top_k_ragged_transform
from progressive import prepare_progressive
from quantized import QuantizedKeys
from query import QueryProjection
from slab_index import IndexSlab, Plan
from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged
from sglang.srt.layers.attention.keye_topk.keye_indexer import KeyeIndexer


POLICY = json.loads(Path(os.environ["KEYE_PREDICT_QUALITY_POLICY"]).read_text())
M = int(POLICY["candidate_width"])
AUDIT = Path(os.environ["KEYE_LOOKAHEAD_TRACE_DIR"])


class CompactState:
    def __init__(self, indexer, rid, layer, buffer, table, history, mode):
        self.rid = rid
        self.layer = layer
        self.last_length = None
        self.lengths = history.clone()
        self.plan = Plan.from_lengths(self.lengths, POLICY["append_reserve"])
        self.reservation = int(self.plan.capacities[0])
        raw = IndexSlab(self.plan, buffer, table, self.lengths, packed=False, exception_reserve=0)
        self.key = QuantizedKeys(raw, int(mode["bits"]), int(mode.get("groups", 1)), int(mode.get("fit_steps", 0)))
        self.project = QueryProjection(indexer, 1, buffer.device)
        self.score_dtype = torch.float16 if mode.get("half_scores", False) else torch.float32
        self.offsets = torch.zeros_like(self.lengths)
        self.width = int(mode.get("width", M))
        self.oversample = float(mode.get("oversample", 0))
        self.scan, _ = self.key.prepare(
            self.lengths, indexer.softmax_scale, block=int(mode.get("tile", 128)),
            score_dtype=self.score_dtype,
        )
        if self.oversample:
            # Query input changes each Decode step. Scoring, buffers and the
            # physical layout are prepared once per request/layer reservation.
            self.inputs = None
            self.progressive = prepare_progressive(
                self.key, lambda: self.project(*self.inputs), self.lengths, indexer.softmax_scale,
                self.width, self.oversample, self.score_dtype, block=int(mode.get("tile", 128)),
            )

    def predict(self, hidden, positions):
        if self.oversample:
            self.inputs = (hidden, positions)
            output = self.progressive.call()
            self.inputs = None
            return output
        current = self.lengths[:, None]
        scores = self.scan(*self.project(hidden, positions))
        shortlist = top_k_ragged_transform(scores, self.offsets, self.lengths, self.width - 1)
        return torch.cat((current, shortlist), dim=1)

    def append(self, new_key, count):
        self.key.append(new_key, self.plan.bases + self.lengths)
        self.last_length = count


def prediction(self, *, previous_x, current_x, previous_indices, exact_indices,
               exact_scores, positions, forward_batch, layer_id):
    if not forward_batch.forward_mode.is_decode():
        return
    rids = forward_batch.rids or ()
    if len(rids) != 1 or not rids[0].startswith("predict-eval__"):
        return
    rid = rids[0]
    mode = rid.split("__", 2)[1]
    if mode == "native":
        self._predict_quality_state = None
        return
    route = POLICY.get("layers", {}).get(str(layer_id), POLICY["default"])
    if route["method"] == "native" or exact_scores is None:
        return
    meta = forward_batch.attn_backend.get_indexer_metadata(layer_id, forward_batch)
    lengths = meta.get_seqlens_int32()
    count = int(forward_batch.seq_lens_cpu[0])
    width = int(route.get("width", M))
    oversample = float(route.get("oversample", 0))
    if count <= max(width, int(width * oversample) + 128):
        return

    table = meta.get_page_table_1()
    raw = forward_batch.token_to_kv_pool.get_index_k_bf16_buffer(layer_id)
    prefix = lengths - 1

    if mode == "bf16" or route["method"] == "bf16":
        self._predict_quality_state = None
        projector = getattr(self, "_predict_quality_projector", None)
        if projector is None:
            projector = self._predict_quality_projector = QueryProjection(self, 1, previous_x.device)
        q, gate = projector(previous_x, positions)
        past = keye_indexer_score_paged(q, raw, table, gate, prefix, self.softmax_scale)
        older = past.topk(width - 1, sorted=True).indices.int()
        ids = torch.cat((prefix[:, None], older), dim=1)
    elif mode == "compact" and route["method"] == "quantized":
        state = getattr(self, "_predict_quality_state", None)
        if state is None or state.rid != rid or state.last_length != count-1 or count > state.reservation:
            self._predict_quality_state = None
            state = self._predict_quality_state = CompactState(self, rid, layer_id, raw, table, prefix, route)
        else:
            state.lengths.copy_(prefix)
        ids = state.predict(previous_x, positions)
        # The native layer has already written its real current K. Publish
        # only that value to the predictor for use during the next Decode step.
        new_slot = table.gather(1, prefix[:, None].long()).flatten().long()
        state.append(raw.index_select(0, new_slot), count)
    else:
        raise ValueError(f"Unsupported quality mode {mode}/{route['method']}")

    # Use the same target-layer BF16 scores. This shadow server still performs
    # the native dense computation above, so serve timings are not used.
    within = exact_scores.gather(1, ids.long())
    selected = ids.gather(1, within.topk(2048, sorted=True).indices)
    if layer_id in POLICY.get("audit_layers", []):
        ref = exact_indices.sort(dim=1).values.contiguous()
        index = torch.searchsorted(ref, ids.contiguous()).clamp_max(2047)
        entry = dict(rid=rid, layer=layer_id, context=count, width=width,
                     true_intersection=int((ref.gather(1, index) == ids).sum()),
                     unique=int(ids.unique().numel()))
        with (AUDIT / f"shadow-layer{layer_id:02}.jsonl").open("a") as fp:
            fp.write(json.dumps(entry) + "\n")
    return selected


KeyeIndexer.trace_cross_layer_lookahead_sm80 = prediction


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
