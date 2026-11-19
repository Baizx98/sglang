"""Audit exact BF16 encoding and multi-step packed Keye decode with slot reuse."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from optimize import (
    K,
    PackedKeyArena,
    _make_scoring_variant,
    decode_keys,
    encode_keys,
    prepare_packed_full,
)


def _bits(tensor):
    return tensor.view(torch.int16)


def _bf16(shape):
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16)


def _exception_rows(rows):
    values = _bf16((rows, 64))
    values[:, 0] = 32
    values[:, 1] = -1e-12
    return values


def _reference():
    path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/keye_topk/ampere_indexer.py"
    )
    spec = importlib.util.spec_from_file_location("original_keye_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.keye_indexer_score_paged


def check_encoding():
    all_bits = torch.arange(1 << 16, device="cuda", dtype=torch.int32)
    all_bits = all_bits.to(torch.int16).reshape(-1, 64)
    mixed = torch.randint(-32768, 32768, (1025, 64), device="cuda", dtype=torch.int16)
    near_zero = torch.zeros((64, 64), device="cuda", dtype=torch.int16)
    near_zero[::3, ::5] = -32768
    near_zero[::7, ::4] = 1
    near_zero[::5, ::6] = -32767
    for values in (all_bits, mixed, near_zero):
        packed = encode_keys(values.view(torch.bfloat16))
        assert torch.equal(_bits(decode_keys(packed)), values)
    print("BF16 bit patterns: 65,536 + mixed exponents + signed zeros/subnormals")


def _check_scores(original_scorer, packed_scorer, raw, arena, q, table, weights, lengths):
    live = torch.arange(table.size(1), device=table.device) < lengths[:, None]
    ref = original_scorer(q, raw, table, weights, lengths, 0.125)
    got = packed_scorer(q, weights, arena.keys)[:, : table.size(1)]
    assert torch.equal(ref[live].view(torch.int32), got[live].view(torch.int32))


def check_arena(original_scorer):
    batch, width = 3, 2183
    initial = 2058
    stored = batch * initial
    raw = _bf16((stored + 32, 64))
    raw[1:5] = _exception_rows(4)
    arena = PackedKeyArena.from_cache(
        raw[:stored], reserve=0, slot_capacity=raw.size(0)
    )
    table = torch.full((batch, width), -1, dtype=torch.int32, device="cuda")
    table[:, :initial] = torch.arange(stored, device="cuda").reshape(batch, initial)
    lengths = torch.full((batch,), initial, dtype=torch.int32, device="cuda")
    q = _bf16((batch, 16, 64))
    weights = _bf16((batch, 16))
    score = _make_scoring_variant(
        q, None, table, weights, lengths, 0.125, 128, 4, encoded=arena.keys
    )
    _check_scores(original_scorer, score, raw, arena, q, table, weights, lengths)

    for call in range(5):
        slots = torch.arange(stored + batch * call, stored + batch * (call + 1), device="cuda")
        values = _exception_rows(batch)
        assert not arena.write(values, slots).any()
        raw.index_copy_(0, slots, values)
        table[:, initial + call] = slots.int()
        lengths.add_(1)
        _check_scores(original_scorer, score, raw, arena, q, table, weights, lengths)
        mapped = table[:, : initial + call + 1].flatten().long()
        assert torch.equal(_bits(decode_keys(arena.keys)[mapped]), _bits(raw[mapped]))

    # Retire original request zero, reorder the survivors, and reuse its physical
    # slots for a new request. An exception reservation survives a normal rewrite.
    last_count = int(arena.next_exception)
    order = torch.tensor([2, 1, 0], device="cuda")
    table.copy_(table[order])
    lengths.copy_(lengths[order])
    table[2].fill_(-1)
    table[2, :initial] = torch.arange(initial, device="cuda", dtype=torch.int32)
    lengths[2] = initial
    recycled = torch.tensor([1, 2], device="cuda")
    normal = torch.ones((2, 64), device="cuda", dtype=torch.bfloat16)
    assert not arena.write(normal, recycled).any()
    raw.index_copy_(0, recycled, normal)
    _check_scores(original_scorer, score, raw, arena, q, table, weights, lengths)
    replacement = _exception_rows(recycled.numel())
    assert not arena.write(replacement, recycled).any()
    raw.index_copy_(0, recycled, replacement)
    assert int(arena.next_exception) == last_count
    _check_scores(original_scorer, score, raw, arena, q, table, weights, lengths)

    invalid_slots = torch.tensor([-1, raw.size(0)], device="cuda")
    before = decode_keys(arena.keys)
    assert arena.write(normal, invalid_slots).tolist() == [0, 1]
    assert torch.equal(_bits(before), _bits(decode_keys(arena.keys)))
    print(
        f"arena: 5 steps, 15 consecutive exceptional writes, retire/reorder/reuse, "
        f"{last_count} allocated exception rows, {arena.keys.storage_bytes} B reserved"
    )


def check_overflow_retry():
    values = torch.ones((7, 64), device="cuda", dtype=torch.bfloat16)
    arena = PackedKeyArena.from_cache(values)
    slots = torch.tensor([2, 5], device="cuda", dtype=torch.int32)
    rejected = torch.empty_like(slots)
    exception = _exception_rows(slots.numel())
    assert arena.write_reserved(exception, slots, rejected).tolist() == [1, 1]
    assert int(arena.next_exception) == 0
    assert torch.equal(_bits(decode_keys(arena.keys)), _bits(values))
    assert not arena.write(exception, slots).any()
    values.index_copy_(0, slots.long(), exception)
    assert torch.equal(_bits(decode_keys(arena.keys)), _bits(values))
    print("arena overflow: old keys preserved, counter recovered, retry succeeded")


def check_graph(original_scorer):
    batch, width, initial, steps = 3, 2105, 2082, 4
    stored = batch * initial
    raw = torch.ones((stored + batch * steps, 64), device="cuda", dtype=torch.bfloat16)
    expected_cache = raw.clone()
    table = torch.full((batch, width), -1, dtype=torch.int32, device="cuda")
    table[:, :initial] = torch.arange(stored, device="cuda").reshape(batch, initial)
    lengths = torch.full((batch,), initial, dtype=torch.int32, device="cuda")
    q = _bf16((batch, 16, 64))
    weights = _bf16((batch, 16))
    current = torch.ones((batch, 64), device="cuda", dtype=torch.bfloat16)

    def project():
        return q, current, weights

    def select(scores):
        return torch.topk(scores[:, :width], K, sorted=False).indices.int()

    full = prepare_packed_full(
        q, raw, table, weights, lengths, 0.125, project, select, reserve=batch * steps,
    )
    full()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indices = full()

    for step in range(steps):
        slots = torch.arange(stored + batch * step, stored + batch * (step + 1), device="cuda")
        current.copy_(_exception_rows(batch))
        expected_cache.index_copy_(0, slots, current)
        table[:, initial + step] = slots.int()
        lengths.add_(1)
        graph.replay()
        assert not full.last_overflow.any()
        scores = original_scorer(q, expected_cache, table, weights, lengths, 0.125)
        selected = scores.gather(1, indices.long()).sort(descending=True).values
        expected = torch.topk(scores, K, sorted=True).values
        assert torch.equal(selected, expected)
        assert (indices.sort().values.diff() > 0).all()
        mapped = table[:, : initial + step + 1].flatten().long()
        assert torch.equal(
            _bits(decode_keys(full.arena.keys)[mapped]), _bits(expected_cache[mapped])
        )
    assert torch.equal(raw, torch.ones_like(raw))
    print(f"CUDA Graph: {steps} distinct append steps, {batch * steps} new exceptions, exact Top-K")


if __name__ == "__main__":
    torch.manual_seed(20261118)
    with torch.inference_mode():
        original = _reference()
        check_encoding()
        check_arena(original)
        check_overflow_retry()
        check_graph(original)
