"""Report INT8 score errors, BF16 key counts, and online summary append checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bench import TRACE, Trace, assess, load_indexer, synthetic_inputs, torch_select
from segments import TOPK, _prepare
from sglang.srt.layers.attention.keye_topk.ampere_indexer import keye_indexer_score_paged


@torch.inference_mode()
def diagnose(case):
    calls, stages = _prepare(*case.args())
    base = keye_indexer_score_paged(*case.args())
    valid = base.isfinite()
    coarse = calls["int8_approx4096_bf16_refine"].scores()
    error = (coarse[valid] - base[valid]).abs()
    bound = calls["int8_bounded_bf16_refine"].scores()
    scored = bound.isfinite()
    chosen = torch_select(base)
    threshold = base.gather(1, chosen.clamp_min(0).long()).masked_fill(chosen < 0, -float("inf")).amin(1)
    retained_error = (bound[scored] - base[scored]).abs()
    slot_count = case.key_cache.shape[0]

    data = {
        "input": case.metadata,
        "physical_key_slots": slot_count,
        "raw_key_bytes": case.key_cache.nbytes,
        "int8_codes_and_summary_bytes": 80 * slot_count,
        "coarse_score_abs_max": float(error.max()) if error.numel() else 0.0,
        "coarse_score_abs_mean": float(error.mean()) if error.numel() else 0.0,
        "coarse_score_abs_p99": float(torch.quantile(error, .99)) if error.numel() else 0.0,
        "valid_keys_by_row": valid.sum(dim=1).tolist(),
        "bounded": {
            "scored_bf16_keys_by_row": scored.sum(dim=1).tolist(),
            "score_error_abs_max_on_retained": float(retained_error.max()) if retained_error.numel() else 0.0,
            "missing_at_or_above_threshold_by_row": (
                valid & ~scored & (base >= threshold[:, None])
            ).sum(dim=1).tolist(),
        },
        "refined_tokens": {"int8_approx4096_bf16_refine": valid.sum(1).clamp(max=4096).tolist()},
    }
    for name, run in calls.items():
        if not hasattr(run, "candidates"):
            continue
        candidate = run.candidates()
        accepted = valid.gather(1, candidate.clamp_min(0).long()) & (candidate >= 0)
        data["refined_tokens"][name] = accepted.sum(1).tolist()

    if case.metadata["source"] == "synthetic" and case.metadata["variant"] == "quantization_alias":
        data["selection"] = {
            name: assess(calls[name](), base, chosen)
            for name in ("int8_approx4096_bf16_refine", "int8_bounded_bf16_refine")
        }

    # Simulate the canonical BF16 slot being refreshed, then compare online
    # summary maintenance to a full historical recomputation.
    append = stages["int8_key_summary_append"]
    loc = case.page_table.gather(1, case.lengths[:, None].long() - 1).flatten().long()
    original = case.key_cache[loc]
    case.key_cache[loc] = -original * 0.5
    try:
        append()
        codes = append.quantized[loc]
        summaries = append.summary[loc]
        stages["int8_key_summary_precompute"]()
        data["append_codes_bit_exact"] = bool(torch.equal(codes, append.quantized[loc]))
        data["append_summary_bit_exact"] = bool(torch.equal(
            summaries.view(torch.int32), append.summary[loc].view(torch.int32)
        ))
    finally:
        case.key_cache[loc] = original
    return data


def quantization_alias(batch, width):
    """Create strictly ordered BF16 groups that share an INT8 code."""
    case = synthetic_inputs(batch, width, "full", 20261014)
    case.metadata["variant"] = "quantization_alias"
    case.q.zero_()
    case.q[:, 0, 0] = 1
    case.weights.zero_()
    case.weights[:, 0] = 1
    cache = case.key_cache.view(batch, width, -1)
    cache.zero_()
    cache[:, :, -1] = 4
    cache[:, :, 0] = 0.984375
    generator = torch.Generator(device="cuda").manual_seed(20261014)
    for row in cache:
        selected = torch.randperm(width, device="cuda", generator=generator)[:TOPK]
        row[selected[:TOPK * 3 // 4], 0] = 1.015625
        row[selected[TOPK * 3 // 4:], 0] = 1.0
    return case


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("real", "synthetic", "quantization-alias"), default="real")
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--batch", type=int, nargs="+", default=(1, 16))
    parser.add_argument("--length", type=int, nargs="+", default=(32768, 131072))
    parser.add_argument("--layer", type=int, nargs="+", default=(12,))
    parser.add_argument("--step", type=int, nargs="+", default=(0,))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        if args.source == "quantization-alias":
            for width in args.length:
                for batch in args.batch:
                    case = quantization_alias(batch, width)
                    output.write(json.dumps(diagnose(case)) + "\n")
            return
        if args.source == "synthetic":
            for width in args.length:
                for batch in args.batch:
                    for variant in ("full", "ragged", "all_ties", "boundary_ties"):
                        case = synthetic_inputs(batch, width, variant, 20261014)
                        output.write(json.dumps(diagnose(case)) + "\n")
            return

        from horizonkv_layer_replay import initialize
        initialize()
        trace = Trace(args.trace.resolve())
        for layer in args.layer:
            indexer = load_indexer(layer)
            for step in args.step:
                for width in args.length:
                    for batch in args.batch:
                        case = trace.load(trace.ids(width, batch, 0), layer, step, indexer, "merged")
                        case.metadata["length_bucket"] = width
                        output.write(json.dumps(diagnose(case)) + "\n")
                        output.flush()


if __name__ == "__main__":
    main()
