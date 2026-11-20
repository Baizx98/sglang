"""Verify the INT4-only history is updated in a repeated CUDA Graph."""

from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "indexer-optimization")]

from bench import assess, synthetic_inputs, torch_select
from int4_scan import Int4Scan
from slab_index import IndexSlab, Plan


@torch.inference_mode()
def main():
    inputs = synthetic_inputs(2, 4096, "full", 20261014)
    plan = Plan.from_lengths(inputs.lengths, growth=64)
    slab = IndexSlab(plan, inputs.key_cache, inputs.page_table, inputs.lengths,
                     packed=False, exception_reserve=0)
    state = Int4Scan(slab)
    assert state.storage_bytes == 34 * plan.capacity + 8 * len(inputs.lengths)
    assert state.packed_slab is None

    current = torch.ones((2, 64), device="cuda", dtype=torch.bfloat16)
    projected = lambda: (inputs.q, current, inputs.weights)
    call = state.prepare_direct(projected, torch_select, inputs.scale)
    call()
    captured_return = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_return = call()

    current[:, 0] = 4.0
    current[:, 1] = -2.0
    graph.replay()
    gold = torch_select(call.approx_scores)
    assert assess(captured_return, call.approx_scores, gold)["valid"]

    tail_slots = (plan.bases + inputs.lengths - 1).long()
    slab.resident.index_copy_(0, tail_slots, current)
    rebuilt = Int4Scan(slab)
    assert torch.equal(state.key_codes, rebuilt.key_codes)
    assert torch.equal(state.key_scales, rebuilt.key_scales)
    print("Graph replay encoded the current Key; only 34 bytes/Key persist; returned coarse Top-2048 is valid")


if __name__ == "__main__":
    main()
