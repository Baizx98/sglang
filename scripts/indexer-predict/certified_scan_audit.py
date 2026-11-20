"""Audit rowwise score intervals around a low-bit scan, without GPU timing."""

import argparse
import json
import math
from pathlib import Path

import torch

from benchmark import (
    CapturedInputs, Plan, QuantizedKeys, initialize, keye_indexer_score_paged,
    load_indexer, page_layout, predictor_projection,
)
from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_candidates
from verify import read_format


ROUNDING_MARGIN = 2 ** -8 + 2 ** -13


def row_statistics(state, keys, row, query, chunk=8192):
    values, errors, dot_parts = [], [], []
    base = state.plan.bases[row].long()
    for start in range(0, keys.size(0), chunk):
        stop = min(start + chunk, keys.size(0))
        slots = torch.arange(start, stop, device=keys.device) + base
        encoded = read_format(state, slots).double()
        decoded = encoded * state.scales[slots].double()
        values.append(decoded.norm(dim=1))
        errors.append((keys[start:stop].double() - decoded).norm(dim=1))
        dot_parts.append(query @ decoded.T)
    metadata = torch.stack((torch.cat(values), torch.cat(errors)))
    # Two FP16 values can represent conservative upper bounds on the
    # norms. Round upwards at construction so the compact metadata is safe.
    fp16 = metadata.half()
    outward = torch.where(fp16.double() < metadata,
                          torch.nextafter(fp16, torch.full_like(fp16, float("inf"))), fp16)
    reconstructed_norm, reconstruction_error = outward.double().unbind()
    return reconstructed_norm, reconstruction_error, torch.cat(dot_parts, dim=1)


def score_error_budget(query, approximated, gate, key_norm, key_error, softmax_scale):
    # Cauchy-Schwarz applied to q·(k-k') + (q-q')·k'. ReLU is
    # 1-Lipschitz, so both positive and negative gate weights use abs(w).
    # The extra term bounds rounding against the products of the norms.
    weighted_norm = (query.norm(dim=1) * gate.abs()).sum()
    weighted_approx = (approximated.norm(dim=1) * gate.abs()).sum()
    weighted_diff = ((query - approximated).norm(dim=1) * gate.abs()).sum()
    key_term = (1 + ROUNDING_MARGIN) * weighted_norm * key_error
    round_term = ROUNDING_MARGIN * (weighted_norm + weighted_approx)
    return (key_term + (weighted_diff + round_term) * key_norm) * abs(softmax_scale)


def headwise_bounds(query, approx_query, gate, key_norm, key_error, dot, scale):
    native_norm = query.norm(dim=1)
    projection_error = (query - approx_query).norm(dim=1)
    uncertainty = ((1 + ROUNDING_MARGIN) * native_norm)[:, None] * key_error
    uncertainty += (projection_error + ROUNDING_MARGIN * native_norm)[:, None] * key_norm
    a = (dot - uncertainty).relu() * gate[:, None]
    b = (dot + uncertainty).relu() * gate[:, None]
    # Covers the remaining FP32 scale, gate product and head reduction.
    scale_slack = torch.maximum(a.abs(), b.abs()).sum(dim=0) * (2 ** -16)
    lower = (torch.minimum(a, b).sum(dim=0) - scale_slack) * scale
    upper = (torch.maximum(a, b).sum(dim=0) + scale_slack) * scale
    return lower, upper


def report(case, method, args, stream, bits, request, native, estimated, lower, upper,
           score_subset=None):
    minimum_slack = torch.minimum(native - lower, upper - native)
    if not torch.isfinite(minimum_slack).all() or (minimum_slack < 0).any():
        raise RuntimeError(f"observed score outside {method} interval in {case}, bits {bits}")
    length = native.numel()
    initial_bytes = 8 * bits + 2 + 4
    for width_m in args.candidates:
        k = width_m - 1
        chosen = native.topk(k)
        true_cutoff = chosen.values[-1]
        lower_cutoff = lower.topk(k).values[-1]
        if lower_cutoff > true_cutoff:
            raise RuntimeError("lower-bound threshold exceeds the native threshold")
        retained = upper >= lower_cutoff
        at_cutoff = native >= true_cutoff
        if not retained[at_cutoff].all():
            raise RuntimeError("interval screening removed a native top candidate or tie")
        count = int(retained.sum())
        seed = estimated.topk(min(length, math.ceil(k * 1.25))).indices
        seed_scores = native[seed] if score_subset is None else score_subset(seed)
        if not torch.equal(seed_scores, native[seed]):
            raise RuntimeError("sparse seed scores differ from the original BF16 scorer")
        seed_cutoff = torch.maximum(seed_scores.topk(k).values[-1], lower_cutoff)
        seeded_retained = upper >= seed_cutoff
        if seed_cutoff > true_cutoff or not seeded_retained[at_cutoff].all():
            raise RuntimeError("seeded threshold removed a native top candidate or tie")
        rechecked = seeded_retained.clone()
        rechecked[seed] = True
        reread = int(rechecked.sum())
        if score_subset is None:
            final_scores = native.masked_fill(~seeded_retained, -float("inf"))
        else:
            remaining = seeded_retained.clone()
            remaining[seed] = False
            rest = remaining.nonzero().flatten()
            rest_scores = score_subset(rest) if rest.numel() else native[:0]
            if not torch.equal(rest_scores, native[rest]):
                raise RuntimeError("sparse refine scores differ from the original BF16 scorer")
            final_scores = torch.full_like(native, -float("inf"))
            final_scores[seed] = seed_scores.masked_fill(~seeded_retained[seed], -float("inf"))
            final_scores[rest] = rest_scores
        reranked = final_scores.topk(k).indices
        matches_native = torch.equal(chosen.indices.sort().values, reranked.sort().values)
        record = dict(
            case, request=request, history_tokens=length, M=width_m,
            bits=bits, bound=method, lower_cutoff=float(lower_cutoff), native_cutoff=float(true_cutoff),
            native_threshold_retained=int((upper >= true_cutoff).sum()),
            bounded_threshold_retained=count, minimum_slack=float(minimum_slack.min()),
            seeded_cutoff=float(seed_cutoff), seeded_threshold_retained=int(seeded_retained.sum()),
            seeds=int(seed.numel()), seeded_exact_reads=reread, seeded_native_set=matches_native,
            median_half_interval=float(((upper - lower) / 2).median()),
            max_actual_error=float((native - estimated).abs().max()),
            optimistic_logical_bytes=initial_bytes * length + 100 * count,
            seeded_optimistic_logical_bytes=initial_bytes * length + 100 * reread,
            seeded_raw_bf16_logical_bytes=initial_bytes * length + 128 * reread,
            sparse_scoring_checked=score_subset is not None,
            metadata_bytes_per_token=4, scoring_key_bytes_per_token=8 * bits + 2,
        )
        print(json.dumps(record), file=stream, flush=True)


def audit(replay, indexer, case, args, stream):
    width = max(replay.lengths) + 1
    pages, lengths = page_layout(replay.lengths, width)
    plan = Plan.from_lengths(lengths, growth=64)
    q, gate = predictor_projection(indexer, replay.hidden, replay.positions)()
    softmax_scale = indexer.softmax_scale
    if softmax_scale <= 0:
        raise ValueError("the score interval assumes the positive Keye scale")
    native = keye_indexer_score_paged(q, replay.keys, pages, gate, lengths, softmax_scale)
    native_q = q.double()
    gates = gate.double()
    for bits in args.bits:
        state = QuantizedKeys.from_paged(plan, replay.keys, pages, lengths, bits, 1)
        score, _ = state.prepare(lengths, softmax_scale, 128, score_dtype=torch.float32)
        approx = score(q, gate)
        approx_q = state.query_codes.double() * state.query_scales.double()
        for row, (key, length) in enumerate(zip(replay.keys.split(replay.lengths), replay.lengths, strict=True)):
            compact_norm, compact_err, dots = row_statistics(state, key, row, approx_q[row])
            e = score_error_budget(native_q[row], approx_q[row], gates[row],
                                   compact_norm, compact_err, softmax_scale)
            exact = native[row, :length].double()
            estimated = approx[row, :length].double()
            request = replay.request_ids[row]
            report(case, "score_norm", args, stream, bits, request, exact, estimated,
                   estimated - e, estimated + e)
            lower, upper = headwise_bounds(native_q[row], approx_q[row], gates[row],
                                           compact_norm, compact_err, dots, softmax_scale)

            def score_subset(ids):
                return keye_indexer_score_candidates(
                    q[row:row + 1], replay.keys, pages[row:row + 1], ids[None, :],
                    gate[row:row + 1], lengths[row:row + 1], softmax_scale,
                )[0].double()

            report(case, "headwise_signed", args, stream, bits, request, exact, estimated,
                   lower, upper, score_subset if args.sparse_check else None)


def run(args, stream):
    initialize()
    inputs = CapturedInputs(args.capture)
    for layer in args.layers:
        indexer = load_indexer(layer)
        for bucket, batch in zip(args.length, args.batch, strict=True):
            for step in args.steps:
                trace = inputs.load(bucket, "calibration", batch, layer, step, 0)
                case = dict(length_bucket=bucket, batch=batch, layer=layer, step=step)
                audit(trace, indexer, case, args, stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--length", required=True, type=int, nargs="+")
    parser.add_argument("--batch", required=True, type=int, nargs="+")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 12, 24, 47])
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--bits", type=int, choices=(4, 5, 6, 8), nargs="+", default=[4, 5, 6, 8])
    parser.add_argument("--candidates", type=int, nargs="+", default=[3072, 8192])
    parser.add_argument("--sparse-check", action="store_true",
                        help="score the seeded and surviving headwise candidates with the BF16 kernel")
    torch.set_num_threads(2)
    args = parser.parse_args()
    with torch.inference_mode(), args.output.open("x") as out:
        run(args, out)
