"""Check logical INT4 sidecar against the benchmark slab encoder and selector."""
from __future__ import annotations

import json

import torch

from quality_int4 import CANDIDATE_K, Int4History, selection_ops
from bench import shuffle_pages, synthetic_inputs
from int4_scan import Int4Scan
from slab_index import IndexSlab, Plan


@torch.inference_mode()
def main():
    selection_ops(torch.device("cuda", 0))
    for width in (8193, 32769):
        for shuffle in (0, 128):
            case = synthetic_inputs(1, width, "full", 20261119)
            if shuffle:
                case = shuffle_pages(case, shuffle, 20261119)
            q, raw, pages, gates, lens, scale = case.args()
            lens.fill_(width - 1)
            plan = Plan.from_lengths(lens, 4)
            raw_slab = IndexSlab(plan, raw, pages, lens, packed=False, exception_reserve=4)
            packed_slab = IndexSlab(plan, raw, pages, lens, packed=True, exception_reserve=4)
            slab = Int4Scan(raw_slab, packed_slab)
            state = Int4History(raw)
            for restart, length in ((True, width - 1), (False, width)):
                lens.fill_(length)
                current = raw[pages[0, length - 1].long()].unsqueeze(0)
                old_output, old_coarse = slab.prepare(lambda: (q, current, gates), scale, CANDIDATE_K)
                expected = old_coarse(q, current, gates)[:, :length]
                actual = state.score(q, current, gates, raw, pages, lens, scale, restart)[:, :length]
                exact_scores = torch.equal(actual.view(torch.int32), expected.view(torch.int32))
                old_indices = old_output()
                new_indices, _ = state.select(q, current, gates, raw, pages, lens, scale, False)
                equal_indices = torch.equal(old_indices.sort(dim=1).values, new_indices.sort(dim=1).values)
                direct_ids, direct_scores = state.select(
                    q, current, gates, raw, pages, lens, scale, False, candidate_k=2048,
                )
                direct_values = direct_scores.gather(1, direct_ids.long())
                optimal = torch.equal(direct_values.sort(dim=1).values,
                                      direct_scores.topk(2048).values.flip(1))
                distinct = direct_ids.unique().numel() == 2048
                print(json.dumps(dict(length=length, physical_shuffle_block=shuffle, restart=restart,
                                      identical_coarse_scores=exact_scores,
                                      identical_refined_set=equal_indices,
                                      direct_top2048_score_optimal=optimal and distinct)), flush=True)
                if not (exact_scores and equal_indices and optimal and distinct):
                    raise AssertionError("INT4 sidecar or selection semantics diverged")


if __name__ == "__main__":
    main()
