"""Real-trace viability test for signed-score block certificates and screening.

All reported budget sweeps reuse exact scores inside retained blocks. They
measure the best recall possible with the chosen prefilter, not GPU latency.
"""

from __future__ import annotations

import math

import torch


def grouped_history(keys, lengths, segment):
    batch = len(lengths)
    width = math.ceil(max(lengths) / segment) * segment
    groups = width // segment
    device = keys.device
    lens = torch.tensor(lengths, device=device)
    base = torch.tensor([0, *lengths[:-1]], device=device).cumsum(0)
    token = torch.arange(width, device=device)[None, :]
    valid = token.reshape(1, groups, segment) < lens[:, None, None]
    address = base[:, None] + torch.minimum(token, lens[:, None] - 1)
    values = keys[address].reshape(batch, groups, segment, 64).double()
    count = valid.sum(2)
    center = values.masked_fill(~valid[..., None], 0).sum(2) / count.clamp_min(1)[..., None]
    maximum = values.masked_fill(~valid[..., None], -float("inf")).amax(2)
    minimum = values.masked_fill(~valid[..., None], float("inf")).amin(2)
    maximum = maximum.masked_fill((count == 0)[..., None], 0)
    minimum = minimum.masked_fill((count == 0)[..., None], 0)
    radius = (values - center.unsqueeze(2)).square().sum(3).sqrt().masked_fill(~valid, 0).amax(2)
    return center, radius, minimum, maximum, count


def upper_bounds(q, gate, center, radius, minimum, maximum):
    """Construct FP64 candidate bounds and add margins for the native score."""
    # Shapes: queries [B,H,D], metadata [B,S,D], per-head bounds [B,S,H].
    mid = (minimum + maximum) / 2
    span = (maximum - minimum) / 2
    unit = q.transpose(1, 2)
    dot = center @ unit
    spherical = radius[..., None] * q.norm(dim=2)[:, None, :]
    boxed = span @ unit.abs()
    box_dot = mid @ unit
    lo = torch.maximum(dot - spherical, box_dot - boxed)
    hi = torch.minimum(dot + spherical, box_dot + boxed)
    w = gate.unsqueeze(1)
    positive = w >= 0
    by_head = torch.where(positive, hi, lo).relu().mul(w).sum(2)

    # ReLU secant upper bounds positive heads. For negative heads use one
    # of the global affine lower bounds relu(x)>=0 and relu(x)>=x.
    crossing = (lo < 0) & (hi > 0)
    pos_slope = torch.where(crossing, hi / (hi - lo), (lo >= 0).double())
    pos_bias = torch.where(crossing, -lo * pos_slope, 0)
    slope = torch.where(positive, pos_slope, (dot > 0).double())
    direction = (slope * w) @ q
    bias = (pos_bias * w.clamp_min(0)).sum(2)
    combined = torch.minimum(
        (direction * center).sum(2) + direction.norm(dim=2) * radius,
        (direction * mid + direction.abs() * span).sum(2),
    ) + bias

    product_bound = torch.maximum(minimum.abs(), maximum.abs()) @ unit.abs()
    fp32_dot_slack = product_bound * (2 ** -17)
    bf16_slack = (torch.maximum(lo.abs(), hi.abs()) + fp32_dot_slack) * (2 ** -8)
    fast_reduce_slack = (w.abs() * product_bound).sum(2) * (2 ** -18)
    slack = ((fp32_dot_slack + bf16_slack) * w.abs()).sum(2) + fast_reduce_slack
    return by_head + slack, combined + slack, (dot.relu() * w).sum(2)


def membership(ids, ref):
    ordered = ref.sort(dim=1).values.contiguous()
    at = torch.searchsorted(ordered, ids.contiguous()).clamp_max(ref.size(1) - 1)
    return ((ordered.gather(1, at) == ids) & (ids >= 0)).sum(1).tolist()


def audit(keys, query, gates, lengths, scores, exact_ids, true_ids, m, scale, sizes):
    outputs = []
    q, w = query.double(), gates.double()
    for size in sizes:
        center, radius, lo, hi, count = grouped_history(keys, lengths, size)
        naive, affine, mean = upper_bounds(q, w, center, radius, lo, hi)
        max_cols = size * count.size(1)
        valid_scores = scores[:, :max_cols]
        if max_cols > valid_scores.size(1):
            valid_scores = torch.nn.functional.pad(
                valid_scores, (0, max_cols - valid_scores.size(1)), value=-float("inf"),
            )
        truth_block = valid_scores.reshape(len(lengths), -1, size).amax(2)
        gold_cutoff = valid_scores.topk(m - 1).values[:, -1:].double()
        tokens = count.sum(1)
        for name, bound in (("per_head_interval", naive), ("combined_affine", torch.minimum(naive, affine))):
            bound = bound * scale
            if (bound[count > 0] < truth_block[count > 0]).any():
                raise RuntimeError(f"observed native score above the {name} bound")
            skip = (bound < gold_cutoff) & (count > 0)
            outputs.append(dict(
                kind="bound_oracle_threshold", method=name, segment=size,
                skipped_token_fraction=((count * skip).sum(1) / tokens).tolist(),
                minimum_slack=float((bound[count > 0] - truth_block[count > 0]).min()),
            ))

        # Compare to a selection budget after giving the filter perfect
        # BF16 scores within each chosen segment. Clamp ragged rows by their
        # live blocks when selecting the segment budget.
        rank = mean.masked_fill(count == 0, -float("inf")).argsort(1, descending=True)
        for fraction in (0.125, 0.25, 0.5):
            budget = (count.gt(0).sum(1) * fraction).ceil().int().clamp_min(math.ceil((m - 1) / size) + 1)
            via = torch.arange(count.size(1), device=keys.device)[None, :] < budget[:, None]
            keep = torch.zeros_like(count, dtype=torch.bool).scatter_(1, rank, via)
            filtered = valid_scores.masked_fill(~keep.repeat_interleave(size, dim=1), -float("inf"))
            values, ids = filtered.topk(m - 1)
            ids = torch.cat((tokens[:, None], ids.masked_fill(~values.isfinite(), -1)), dim=1)
            outputs.append(dict(
                kind="centroid_budget", segment=size, fraction=fraction,
                retained_token_fraction=((count * keep).sum(1) / tokens).tolist(),
                predicted_set_intersection=membership(ids, exact_ids),
                true_top2048_intersection=membership(ids, true_ids),
            ))
    return outputs
