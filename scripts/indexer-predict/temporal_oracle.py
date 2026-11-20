"""Measure previous-step threshold and block-ranking diagnostics on real traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn.functional import pad

from benchmark import (
    CapturedInputs, initialize, intersections, keye_indexer_score_paged,
    load_indexer, page_layout, predictor_projection,
)


def methods(current, prior, lens, target, candidate_width):
    k = candidate_width - 1
    base = torch.cat((lens[:, None], current.topk(k).indices.int()), dim=1)
    hit0 = intersections(base, target)
    for factor in (1, 1.25, 1.5, 2):
        cutoff = prior.topk(math.ceil(k * factor)).values[:, -1:]
        live = (current >= cutoff).sum(dim=1)
        yield dict(
            kind="previous_threshold", factor=factor, kept=(live / lens).tolist(),
            exact_safe=(live >= k).tolist(), live_tokens=live.tolist(),
        )

    for size in (32, 64, 128):
        width = current.shape[1]
        chunks = math.ceil(width / size)
        padding = chunks * size - width
        prior_max = pad(prior, (0, padding), value=-float("inf")).view(
            lens.numel(), chunks, size,
        ).amax(dim=2)
        # Border blocks contain both the sink and the new history entry that
        # was still the current token in the previous step.
        prior_max[:, 0] = float("inf")
        prior_max.scatter_(1, ((lens - 1) // size)[:, None].long(), float("inf"))
        seg_order = prior_max.argsort(dim=1, descending=True)
        current_pad = pad(current, (0, padding), value=-float("inf"))
        chunk_sizes = (
            lens[:, None] - torch.arange(chunks, device=lens.device)[None, :] * size
        ).clamp(0, size)
        for factor in (0.125, 0.25, 0.5):
            budget = ((lens + size - 1) // size * factor).ceil().int()
            budget = budget.clamp_min(math.ceil(k/size) + 1)
            choice = torch.arange(chunks, device=lens.device)[None, :] < budget[:, None]
            kept = torch.zeros_like(prior_max, dtype=torch.bool).scatter_(1, seg_order, choice)
            filtered = current_pad.masked_fill(~kept.repeat_interleave(size, 1), -float("inf"))
            val, loc = filtered.topk(k)
            selected = torch.cat((lens[:, None], loc.masked_fill(~val.isfinite(), -1)), dim=1)
            yield dict(
                kind="previous_block_max", factor=factor, size=size,
                kept=((chunk_sizes * kept).sum(dim=1) / lens).tolist(),
                original_hit=hit0.tolist(), observed_hit=intersections(selected, target).tolist(),
            )


def rolling(frames, m, size, fraction, refresh_period):
    """Score only selected blocks after the first refresh in a real sequence."""
    last = frames[max(frames)][0]
    groups = math.ceil(last.size(1) / size)
    device = last.device
    slots = torch.arange(groups, device=device)[None, :]
    estimates = None
    for step, (score, lens, target) in sorted(frames.items()):
        counts = (lens[:, None] - slots * size).clamp(0, size)
        data = pad(score, (0, groups * size - score.size(1)), value=-float("inf"))
        block_truth = data.view(lens.numel(), groups, size).amax(dim=2)
        full = estimates is None or step % refresh_period == 0
        if full:
            estimates = block_truth.clone()
            chosen = counts > 0
        else:
            eligible = estimates.masked_fill(counts == 0, -float("inf"))
            eligible[:, 0] = float("inf")
            eligible.scatter_(1, ((lens - 1) // size)[:, None].long(), float("inf"))
            order = eligible.argsort(dim=1, descending=True)
            block_budget = ((lens + size - 1) // size * fraction).ceil().int()
            block_budget = block_budget.clamp_min(math.ceil((m - 1) / size) + 1)
            chosen = torch.zeros_like(estimates, dtype=torch.bool).scatter_(
                1, order, slots < block_budget[:, None],
            )
            estimates[chosen] = block_truth[chosen]
        given = data.masked_fill(~chosen.repeat_interleave(size, dim=1), -float("inf"))
        values, indices = given.topk(m - 1)
        output = torch.cat((lens[:, None], indices.masked_fill(~values.isfinite(), -1)), dim=1)
        exact = torch.cat((lens[:, None], data.topk(m - 1).indices), dim=1)
        yield dict(
            kind="rolling_block_max", step=step, M=m, size=size,
            factor=fraction, refresh=full, refresh_period=refresh_period,
            kept=((counts * chosen).sum(dim=1) / lens).tolist(),
            original_hit=intersections(exact, target).tolist(),
            observed_hit=intersections(output, target).tolist(),
        )


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    initialize()
    data = CapturedInputs(args.capture)
    if min(args.steps) <= 0:
        raise ValueError("previous-step diagnostics require positive steps")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for layer in args.layers:
            indexer = load_indexer(layer)
            frames = {}
            for step in args.steps:
                replay = data.load(args.length, args.split, args.batch, layer, step, 0)
                previous = data.load(args.length, args.split, args.batch, layer, step - 1, 0)
                table, lengths = page_layout(replay.lengths, max(replay.lengths))
                if [n + 1 for n in previous.lengths] != replay.lengths:
                    raise RuntimeError("non-consecutive history lengths")
                current_keys = replay.keys.split(replay.lengths)
                old_keys = previous.keys.split(previous.lengths)
                if not all(torch.equal(a[:-1], b) for a,b in zip(current_keys, old_keys, strict=True)):
                    raise RuntimeError("previous history changed in a later layer step")
                q, g = predictor_projection(indexer, replay.hidden, replay.positions)()
                pq, pg = predictor_projection(indexer, previous.hidden, previous.positions)()
                scale = indexer.softmax_scale
                current = keye_indexer_score_paged(q, replay.keys, table, g, lengths, scale)
                prior = keye_indexer_score_paged(pq, replay.keys, table, pg, lengths - 1, scale)
                frames[step] = (current, lengths, replay.target_ids)
                frames.setdefault(step - 1, (prior, lengths - 1, previous.target_ids))
                for m in args.candidates:
                    for metrics in methods(current, prior, lengths, replay.target_ids, m):
                        stream.write(json.dumps(dict(
                            bucket=args.length, batch=args.batch, layer=layer, step=step,
                            M=m, rids=replay.request_ids, **metrics,
                        )) + "\n")
                        stream.flush()
            if args.rolling:
                if sorted(frames) != list(range(min(frames), max(frames) + 1)):
                    raise ValueError("rolling diagnostics require consecutive steps")
                for m in args.candidates:
                    for size in (32, 64, 128):
                        for fraction in (0.25, 0.5):
                            for period in (2, 4):
                                for item in rolling(frames, m, size, fraction, period):
                                    stream.write(json.dumps(dict(
                                        bucket=args.length, batch=args.batch, layer=layer,
                                        rids=replay.request_ids, **item,
                                    )) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 12, 24, 47])
    parser.add_argument("--candidates", type=int, nargs="+", default=[3072, 8192])
    run(parser.parse_args())
