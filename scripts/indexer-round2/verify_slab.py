"""Exercise Index Slab growth inside a reservation, retirement, and slot reuse."""

from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "indexer-optimization")]

from bench import page_layout, shuffle_pages, synthetic_inputs, torch_select
from optimize import decode_keys
from slab_index import IndexSlab, Plan


@torch.inference_mode()
def main():
    inputs = shuffle_pages(synthetic_inputs(2, 127, "full", 20261014), 1, 20261014)
    lengths = inputs.lengths.clone()
    plan = Plan.from_lengths(lengths, growth=8)
    caps = plan.capacities.tolist()
    bases = plan.bases.tolist()
    expected = torch.zeros((plan.capacity, 64), dtype=torch.bfloat16, device="cuda")
    for row in range(2):
        expected[bases[row]:bases[row] + 127] = inputs.key_cache[inputs.page_table[row].long()]

    for packed in (False, True):
        state = IndexSlab(plan, inputs.key_cache, inputs.page_table, inputs.lengths.clone(),
                          packed=packed, exception_reserve=16)
        current = torch.ones((2, 64), dtype=torch.bfloat16, device="cuda")
        # Each newly inserted key needs a full escape row.
        current[:, 0] = 2.0 ** -40
        project = lambda: (inputs.q, current, inputs.weights)
        call = state.prepare(project, torch_select, inputs.scale, 128)
        desired = expected.clone()

        for pair in ((128, 128), (129, 129), (0, 130), (1, 131), (2, 0)):
            state.lengths.copy_(torch.tensor(pair, dtype=torch.int32, device="cuda"))
            ids = call()
            assert not state.overflow.any()
            resident = decode_keys(state.arena.keys) if packed else state.resident
            for row, length in enumerate(pair):
                if length > 0:
                    desired[bases[row] + length - 1] = current[row]
                active = slice(bases[row], bases[row] + length)
                assert torch.equal(resident[active].view(torch.int16), desired[active].view(torch.int16))
                assert int((ids[row] >= 0).sum()) == length
                assert int(ids[row].max()) < length

        state.lengths.copy_(torch.tensor([caps[0] + 1, 0], dtype=torch.int32, device="cuda"))
        call()
        assert state.overflow.tolist() == [1, 0]
        print(("lossless" if packed else "BF16"), "append, retirement, reuse, overflow passed")


if __name__ == "__main__":
    main()
