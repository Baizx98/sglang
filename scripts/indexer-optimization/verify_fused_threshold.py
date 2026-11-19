"""Verify cutoff success and both fallback directions with deterministic cases."""

import torch

from fused_threshold import _CAPACITY, _SAMPLE, prepare
from verify_packed_cache import _reference


def test_case(mode, reference):
    width = 32769
    batch = 2
    dtype = torch.bfloat16
    scale = 0.125
    q = torch.randn(batch, 16, 64, dtype=dtype, device="cuda")
    key = torch.randn(batch * width, 64, dtype=dtype, device="cuda")
    weights = torch.randn(batch, 16, dtype=dtype, device="cuda")
    lengths = torch.tensor([width, 2043], dtype=torch.int32, device="cuda")
    table = torch.randperm(
        batch * width, dtype=torch.int32, device="cuda"
    ).reshape(batch, width)
    table.masked_fill_(torch.arange(width, device="cuda") >= lengths[:, None], -1)
    if mode == "flat":
        weights.zero_()
    elif mode == "sample-only high":
        q.zero_()
        weights.zero_()
        q[:, 0, 0] = 1
        weights[:, 0] = 1
        key.zero_()
        sym = torch.arange(_SAMPLE // 2, device="cuda")
        sampled = ((2 * sym + 1) * width) // (2 * _SAMPLE)
        key[table[0, sampled].long(), 0] = 1

    fn = prepare(q, key, table, weights, lengths, scale)["fused_threshold_exact_flashinfer"]
    eager = fn().clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    graph.replay()
    scores = reference(q, key, table, weights, lengths, scale)
    reference_scores = scores.topk(2048).values
    output = scores.gather(1, result.clamp_min(0).long()).masked_fill(result < 0, -float("inf"))
    assert torch.equal(output.sort(descending=True).values, reference_scores)
    eager_scores = scores.gather(1, eager.clamp_min(0).long()).masked_fill(eager < 0, -float("inf"))
    assert torch.equal(eager_scores.sort(descending=True).values, reference_scores)
    assert int((result[1] >= 0).sum()) == 2043
    for row in result:
        live = row[row >= 0]
        assert live.unique().numel() == live.numel()
    count = int(fn.candidate_counts[0])
    fallback = int(fn.fallback_lengths[0])
    if mode == "flat":
        assert count > _CAPACITY and fallback == width
    elif mode == "sample-only high":
        assert count == _SAMPLE // 2 and fallback == width
    else:
        assert 2048 <= count <= _CAPACITY and fallback == 0
    return count, fallback


if __name__ == "__main__":
    torch.manual_seed(20261118)
    with torch.inference_mode():
        score = _reference()
        for case in ("normal", "flat", "sample-only high"):
            observed = test_case(case, score)
            print(f"{case}: candidates={observed[0]}, fallback_length={observed[1]}, exact Top-K")
