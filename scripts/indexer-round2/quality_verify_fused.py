"""Exercise the service fused projector, current K write and vLLM selector."""
from __future__ import annotations

import json

import torch

from quality_fused import FusedInt4Direct
from bench import TRACE, Trace, load_indexer
from horizonkv_layer_replay import initialize
from projection import prepare_projection


@torch.inference_mode()
def main():
    initialize()
    trace = Trace(TRACE)
    layer = 12
    indexer = load_indexer(layer)
    case_ids = trace.ids(32768, 1, 8)
    raw = None
    for step in range(3):
        case = trace.load(case_ids, layer, step, indexer, "merged")
        _, canonical, pages, _, lens, scale = case.args()
        rid = case_ids[0]
        frame = trace.read(trace.frames[(rid, layer)][step])
        x = frame["attention_input"].cuda()
        p = frame["positions"].cuda()
        if raw is None:
            raw = torch.empty((canonical.shape[0] + 16, canonical.shape[1]),
                              dtype=canonical.dtype, device=canonical.device)
            raw[:canonical.shape[0]].copy_(canonical)
            state = FusedInt4Direct(raw)
        q, key, gates = state.project(indexer, x, p)
        benchmark = prepare_projection(indexer, x, p)()
        native = indexer._get_q_k_w_bf16(x, p, enable_dual_stream=False)
        equivalent = all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                         for a, b in zip((q, key, gates), benchmark, strict=True))
        count = int(lens[0])
        loc = pages[:, count - 1].long()
        raw[loc] = key
        actual_append = torch.equal(raw[loc].view(torch.int16), key.view(torch.int16))
        out, scores = state.select_direct(q, key, gates, raw, pages, lens, scale,
                                          restart=step == 0, live_length=count)
        chosen = scores.gather(1, out.long()).sort(dim=1).values
        best = scores.topk(2048).values.flip(1)
        optimal = torch.equal(chosen, best) and out.unique().numel() == 2048
        native_difference = [int((a.view(torch.int16) != b.view(torch.int16)).sum())
                             for a, b in zip((q, key, gates), native, strict=True)]
        print(json.dumps(dict(request_id=rid, step=step, layer=layer, live_index_keys=count,
                              matches_benchmark_fused_projection=equivalent,
                              key_write_bitwise=actual_append, top2048_score_optimal=optimal,
                              native_projection_bit_diffs=native_difference)), flush=True)
        if not (equivalent and actual_append and optimal):
            raise AssertionError("fused sidecar implementation mismatch")


if __name__ == "__main__":
    main()
