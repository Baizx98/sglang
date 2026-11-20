"""Check exact scorer bits and logical IDs across ties, ragged inputs, and graphs."""

from __future__ import annotations

import importlib.util

import torch

from refine_kernels import K, ROOT, prepare_logical_radix, prepare_scorer


def original_scorer():
    source = ROOT / "python/sglang/srt/layers/attention/keye_topk/ampere_indexer.py"
    spec = importlib.util.spec_from_file_location("refine_reference_ampere", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.keye_indexer_score_candidates


def verify_selection(result, scores, ids, lengths):
    rows, width = scores.shape
    order = ids.argsort(dim=1)
    logical = result.long() - torch.arange(rows, device="cuda")[:, None] * width
    active = result >= 0
    offset = order.gather(1, logical.clamp(0, width - 1))
    column = torch.arange(width, device="cuda")[None, :]
    reference = scores.masked_fill(column >= lengths[:, None], -torch.inf).topk(K, sorted=True).values
    selected = scores.gather(1, offset).masked_fill(~active, -torch.inf)
    assert torch.equal(selected.sort(descending=True).values, reference)
    assert (logical[active] >= 0).all() and (logical[active] < width).all()
    assert torch.equal(active.sum(1).int(), lengths.clamp_max(K))
    assert (offset[active] < lengths[:, None].expand_as(offset)[active]).all()
    for row in result:
        valid = row[row >= 0]
        assert valid.unique().numel() == valid.numel()


def check_selectors(width):
    rows = 6
    lengths = torch.tensor([width, width-1, K, K-1, 7, 0], device="cuda", dtype=torch.int32)
    scores = torch.randn(rows, width, device="cuda", dtype=torch.float32)
    ids = torch.stack([
        torch.randperm(width, device="cuda", dtype=torch.int32) + row * width
        for row in range(rows)
    ])
    for threads, wide_digits in ((256, False), (512, False), (512, True), (1024, True)):
        selector = prepare_logical_radix(lengths, width, threads=threads, wide_digits=wide_digits)
        verify_selection(selector(scores, ids), scores, ids, lengths)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = selector(scores, ids)
        for mode in ("wide exponent", "all ties", "boundary ties", "signed zero"):
            if mode == "wide exponent":
                values = torch.randn_like(scores)
                exponent = torch.randint(-124, 120, values.shape, dtype=torch.int32, device="cuda")
                scores.copy_(torch.ldexp(values, exponent))
            elif mode == "all ties":
                scores.fill_(-16.0)
            elif mode == "boundary ties":
                scores.fill_(-1.0)
                scores[:, :1024].fill_(3.0)
                scores[:, 1024:2777].fill_(0.5)
            else:
                scores.zero_()
                scores[:, ::2].mul_(-1)
            graph.replay()
            verify_selection(output, scores, ids, lengths)
        print(f"radix{threads}, wide={wide_digits}: M={width}, ragged rows and four graph replays passed")


def check_scores(width):
    rows, context, dim = 4, width * 2 + 109, 64
    q = torch.randn(rows, 16, dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(rows * context, dim, device="cuda", dtype=torch.bfloat16)
    gates = torch.randn(rows, 16, device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(rows * context, device="cuda", dtype=torch.int32).reshape(rows, context)
    lengths = torch.full((rows,), context, device="cuda", dtype=torch.int32)
    ids = torch.stack([
        torch.randperm(context, device="cuda", dtype=torch.int32)[:width] for _ in range(rows)
    ])
    gold = original_scorer()(q, key, table, ids, gates, lengths, 0.125)
    for tile, warps, sorted_tile in (
        (64, 4, False), (64, 2, False), (128, 4, False), (64, 4, True), (128, 4, True),
    ):
        variant = prepare_scorer(
            q, key, table, gates, lengths, ids, 0.125, tile, warps, sort_local=sorted_tile,
        )
        scores, logical = variant()
        expected_order = ids.argsort(dim=1)
        order = logical.argsort(dim=1)
        assert torch.equal(logical.gather(1, order), ids.gather(1, expected_order))
        assert torch.equal(scores.gather(1, order), gold.gather(1, expected_order))
    print(f"candidate score: M={width}, all tile configurations bit-identical")


if __name__ == "__main__":
    torch.manual_seed(20261120)
    with torch.inference_mode():
        for width in (3072, 8192):
            check_selectors(width)
            check_scores(width)
