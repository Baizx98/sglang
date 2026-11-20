"""Test static INT4/INT8 promotions on real predictor states without GPU timings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from benchmark import (
    CapturedInputs, Plan, QuantizedKeys, initialize, intersections, keye_indexer_score_paged,
    load_indexer, page_layout, predictor_projection, reference_topm,
)


def promotion_features(keys):
    x = keys.float()
    peak = x.abs().amax(-1)
    scale = (peak.clamp_min(1e-20) / 7).bfloat16().float()
    reconstructed = (x / scale[:, None]).round().clamp(-7, 7) * scale[:, None]
    norm_sq = x.square().sum(-1)
    residual_sq = (x - reconstructed).square().sum(-1)
    return dict(peak=peak, norm=norm_sq, residual=residual_sq,
                relative_residual=residual_sq / norm_sq.clamp_min(1e-20))


def promote(features, histories, step, fraction, width):
    mask = torch.zeros((len(histories), width), dtype=torch.bool, device=features.device)
    for row, values in enumerate(features.split(histories)):
        # The threshold is decided at bootstrap using the prompt history.
        # Later keys meet that threshold or remain in the low-bit tier.
        prefix = len(values) - step
        wanted = max(1, math.ceil(prefix * fraction))
        cutoff, selected = values[:prefix].topk(wanted)
        mask[row].scatter_(0, selected, True)
        mask[row, prefix:len(values)] = values[prefix:] > cutoff[-1]
    return mask


def audit(replay, indexer, case, budgets, candidates, stream):
    width = max(replay.lengths) + 1
    pages, lengths = page_layout(replay.lengths, width)
    plan = Plan.from_lengths(lengths, growth=64)
    query, gates = predictor_projection(indexer, replay.hidden, replay.positions)()
    scale = indexer.softmax_scale
    gold_scores = keye_indexer_score_paged(query, replay.keys, pages, gates, lengths, scale)[:, :width]
    base = {}
    for m in candidates:
        ids, _ = reference_topm(gold_scores, lengths, m)
        base[m] = intersections(ids, replay.target_ids).tolist()

    scored = {}
    for bits, groups in ((4, 1), (5, 2), (6, 1), (8, 1)):
        keys = QuantizedKeys.from_paged(plan, replay.keys, pages, lengths, bits, groups)
        score, _ = keys.prepare(lengths, scale, 128, score_dtype=torch.float16)
        scored[bits] = score(query, gates)[:, :width]

    def report(method, scores, feature=None, target_fraction=None, keep=None, bytes_low=34, bytes_high=66):
        promoted = keep.sum(-1).tolist() if keep is not None else [0] * len(replay.lengths)
        key_bytes = [n * bytes_low + h * (bytes_high - bytes_low)
                     for n, h in zip(replay.lengths, promoted, strict=True)]
        for m, initial in base.items():
            ids, _ = reference_topm(scores, lengths, m)
            record = dict(
                case, M=m, method=method, promotion_feature=feature, target_fraction=target_fraction,
                requests=replay.request_ids, history_tokens=replay.lengths, promoted_tokens=promoted,
                ideal_key_payload_bytes=key_bytes, exact_intersection=initial,
                method_intersection=intersections(ids, replay.target_ids).tolist(),
            )
            print(json.dumps(record), file=stream, flush=True)

    for bits, size in ((4, 34), (5, 44), (6, 50), (8, 66)):
        report(f"int{bits}", scored[bits], bytes_low=size)
    reference_half = gold_scores.half()
    for criterion, features in promotion_features(replay.keys).items():
        for fraction in budgets:
            promoted = promote(features, replay.lengths, case["step"], fraction, width)
            report("int4_int8", torch.where(promoted, scored[8], scored[4]),
                   criterion, fraction, promoted)
            report("int4_bf16_oracle", torch.where(promoted, reference_half, scored[4]),
                   criterion, fraction, promoted, bytes_high=128)


def run(args, stream):
    initialize()
    inputs = CapturedInputs(args.capture)
    for layer in args.layers:
        indexer = load_indexer(layer)
        for bucket, batch in zip(args.length, args.batch, strict=True):
            for step in args.steps:
                replay = inputs.load(bucket, "calibration", batch, layer, step, 0)
                audit(replay, indexer, dict(bucket=bucket, batch=batch, layer=layer, step=step),
                      args.budgets, args.candidates, stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--length", required=True, type=int, nargs="+")
    parser.add_argument("--batch", required=True, type=int, nargs="+")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 12, 24, 47])
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--candidates", type=int, nargs="+", default=[3072, 8192])
    parser.add_argument("--budgets", type=float, nargs="+", default=[.125, .25, .5, .75])
    torch.set_num_threads(2)
    args = parser.parse_args()
    if any(not 0 < fraction < 1 for fraction in args.budgets):
        parser.error("promotion fractions must fall in (0, 1)")
    with torch.inference_mode(), args.output.open("x") as out:
        run(args, out)
