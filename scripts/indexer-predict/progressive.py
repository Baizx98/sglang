"""Form a Top-M using a four-bit scan followed by a compact six-bit rescore."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

from flashinfer import top_k_page_table_transform, top_k_ragged_transform
import torch


@dataclass
class Progressive:
    call: Callable
    shortlist: int
    graph_workspace_bytes: int
    logical_key_bytes: int
    debug: Callable


def prepare_progressive(keys, project, lengths, scale, candidates, multiplier, dtype, block=128):
    keep = math.ceil(candidates * multiplier / block) * block
    batch = lengths.numel()
    if keep > int(lengths.min()):
        raise ValueError("the shortlist must fit in every active request history")
    current = lengths[:, None]
    offsets = torch.zeros_like(lengths)
    short_lens = torch.full_like(lengths, keep)
    scan, scan_memory = keys.prepare_scan(lengths, scale, block=block, score_dtype=dtype, coarse=True)
    refine, refine_memory = keys.prepare_scan(
        lengths, scale, block=block, score_dtype=dtype, shortlist_width=keep,
    )

    def debug():
        query, gate = project()
        keys.encode_query(query)
        coarse_scores = scan(gate)
        shortlist = top_k_ragged_transform(coarse_scores, offsets, lengths, keep)
        fine = refine(gate, shortlist)
        ids = top_k_page_table_transform(fine, shortlist, short_lens, candidates - 1)
        return torch.cat((current, ids), dim=1), shortlist, coarse_scores, fine

    def call():
        return debug()[0]

    unique_reads = int(lengths.sum()) * (32 + 2 * keys.groups)
    unique_reads += batch * keep * (8 * keys.bits + 2 * keys.groups)
    workspace = scan_memory + refine_memory + keys.query_scales.nbytes + keys.query_codes.nbytes
    workspace += batch * keep * 4
    return Progressive(call, keep, workspace, unique_reads, debug)


def validate(method, dense, q, gate, lengths, candidates):
    ids, shortlist, _, fine = method.debug()
    expected = dense(q, gate).gather(1, shortlist.long()).to(fine.dtype)
    if not torch.equal(fine, expected):
        raise RuntimeError("sparse rescore differs from the dense quantized arithmetic")
    valid = (ids >= 0) & (ids <= lengths[:, None])
    if not valid.all() or (ids.sort(dim=1).values.diff(dim=1) <= 0).any():
        raise RuntimeError("invalid or duplicate IDs in the progressive output")
    if not (ids[:, :1] == lengths[:, None]).all():
        raise RuntimeError("reserved current position missing from progressive output")
    cutoff = fine.topk(candidates - 1).values[:, -1:]
    original_ids = ids[:, 1:].contiguous().sort(dim=1).values
    insertion = torch.searchsorted(original_ids, shortlist.contiguous()).clamp_max(candidates - 2)
    chosen = original_ids.gather(1, insertion) == shortlist
    if (chosen.sum(dim=1) != candidates - 1).any() or (chosen & (fine < cutoff)).any():
        raise RuntimeError("progressive final selector disagrees with the shortlist score")
    return ids
