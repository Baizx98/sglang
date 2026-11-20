"""Rescore current queries on blocks ranked by their prior-step scores."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

from flashinfer import top_k_ragged_transform
import torch
import triton


@dataclass
class Temporal:
    partial: Callable
    debug: Callable
    refresh: Callable
    prior: torch.Tensor
    next_state: torch.Tensor
    shortlist_tokens: int
    workspace_bytes: int


def prepare_temporal(keys, project, previous_q, previous_gate, lengths, scale,
                     budget, segment, candidates, score_dtype):
    batch = lengths.numel()
    active_segments = (lengths + segment - 1) // segment
    keep = max(math.ceil(int(active_segments.max()) * budget), math.ceil((candidates-1)/segment) + 1)
    scores0, _ = keys.prepare(lengths - 1, scale, block=segment, score_dtype=score_dtype)
    previous = scores0(previous_q, previous_gate).view(batch, -1, segment).amax(dim=2)
    width = triton.cdiv(keys.plan.width, segment)
    if previous.size(1) != width or keep > width:
        raise ValueError("temporal budget exceeds the prepared history")
    state = torch.empty_like(previous)
    last = (active_segments - 1)[:, None].long()
    live_groups = torch.arange(width, device=lengths.device)[None, :] < active_segments[:, None]
    shortlist = keep * segment
    shortlist_lens = torch.full_like(lengths, shortlist)
    offsets = torch.zeros_like(lengths)
    scan, compact_memory = keys.prepare_scan(
        lengths, scale, block=segment, score_dtype=score_dtype,
        shortlist_width=shortlist, block_selection=True,
    )
    scan_full, refresh_memory = keys.prepare_scan(lengths, scale, block=segment, score_dtype=score_dtype)

    def debug():
        q, gate = project()
        ranks = previous.masked_fill(~live_groups, -float("inf"))
        ranks[:, 0] = float("inf")
        ranks.scatter_(1, last, float("inf"))
        picked = ranks.topk(keep, sorted=False).indices
        keys.encode_query(q)
        values = scan(gate, picked.int())
        local = top_k_ragged_transform(values, offsets, shortlist_lens, candidates - 1)
        logical = picked.gather(1, (local // segment).long()).int() * segment + local % segment
        state.copy_(previous)
        state.scatter_(1, picked, values.view(batch, keep, segment).amax(dim=2))
        return torch.cat((lengths[:, None], logical), dim=1), picked, values

    def partial():
        return debug()[0]

    def refresh():
        q, gate = project()
        keys.encode_query(q)
        scores = scan_full(gate)
        state.copy_(scores.view(batch, width, segment).amax(dim=2))
        ids = top_k_ragged_transform(scores, offsets, lengths, candidates - 1)
        return torch.cat((lengths[:, None], ids), dim=1)

    workspace = (
        previous.nbytes + state.nbytes + compact_memory + refresh_memory
        + keys.query_scales.nbytes + keys.query_codes.nbytes
    )
    return Temporal(partial, debug, refresh, previous, state, shortlist, workspace)


def validate_temporal(runtime, reference, query, gate, lengths, segment, m):
    selected, kept, values = runtime.debug()
    offsets = torch.arange(segment, device=kept.device)[None, None, :]
    logical = (kept[..., None] * segment + offsets).flatten(1)
    live = logical < lengths[:, None]
    truth = reference(query, gate)
    expect = truth.gather(1, logical.clamp_max(truth.size(1) - 1)).masked_fill(~live, -float("inf"))
    if not torch.equal(expect.to(values.dtype), values):
        raise RuntimeError("block rescore disagrees with the dense quantized scores")
    if not (selected[:, 0] == lengths).all():
        raise RuntimeError("temporal selection omitted the current position")
    hist = selected[:, 1:]
    if not ((hist >= 0) & (hist < lengths[:, None])).all() or (hist.sort(dim=1).values.diff() == 0).any():
        raise RuntimeError("invalid or duplicate temporal output ID")
    sorted_segments = kept.sort(dim=1).values.contiguous()
    chosen_segments = (hist // segment).long().contiguous()
    at = torch.searchsorted(sorted_segments, chosen_segments).clamp_max(kept.size(1) - 1)
    if not (sorted_segments.gather(1, at) == chosen_segments).all():
        raise RuntimeError("temporal output includes an unscored block")
    cutoff = values.topk(m - 1).values[:, -1:]
    picked = truth.gather(1, hist.long()).to(values.dtype)
    if (picked < cutoff).any() or not torch.equal(
        (values > cutoff).sum(dim=1), (picked > cutoff).sum(dim=1),
    ):
        raise RuntimeError("temporal selector missed a strictly higher score in rescored blocks")
    return selected, live.sum(dim=1).tolist()
