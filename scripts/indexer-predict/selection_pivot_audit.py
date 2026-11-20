"""Check candidate compaction pivots for the frozen low-bit prediction routes."""

import argparse
import json
import math
from pathlib import Path

import torch

from benchmark import CapturedInputs, Plan, QuantizedKeys, initialize, load_indexer, page_layout, predictor_projection
from quality_cases import layer_policy


def pivot_counts(current, previous, m, samples):
    k = m - 1
    prior = previous.topk(2 * k)
    anchors = current[prior.indices[:k]]
    anchor_values = anchors.sort(descending=True).values
    locations = (2 * torch.arange(samples, device=current.device) + 1) * current.numel() // (2 * samples)
    sample_values = current[locations].sort(descending=True).values
    pivots = {
        "stale_k": prior.values[k - 1],
        "stale_1.5k": prior.values[math.ceil(1.5 * k) - 1],
        "current_sample_1.25k": sample_values[math.ceil(1.25 * k * samples / current.numel()) - 1],
        "current_sample_1.5k": sample_values[math.ceil(1.5 * k * samples / current.numel()) - 1],
        "anchor_0.5k": anchor_values[math.ceil(.5 * k) - 1],
        "anchor_0.75k": anchor_values[math.ceil(.75 * k) - 1],
        "anchor_k": anchor_values[k - 1],
    }
    native_cutoff = current.topk(k).values[-1]
    tied_at_cutoff = bool((current >= native_cutoff).sum() > k)
    for name, threshold in pivots.items():
        count = int((current >= threshold).sum())
        if name == "anchor_k" and count < k:
            raise RuntimeError("the anchor cutoff discarded a previous member")
        yield name, count, tied_at_cutoff


def run(args, stream):
    initialize()
    inputs = CapturedInputs(args.capture)
    rows = json.loads(args.policy.read_text())["routes"]
    for route in rows:
        bucket, layer, m = route["length_bucket"], route["layer"], route["M"]
        scheme = layer_policy(route, args.codebooks)
        if scheme["method"] != "quantized" or scheme.get("oversample"):
            raise ValueError(f"the pivot audit needs a dense compact scorer for {route['method']}")
        indexer = load_indexer(layer)
        codebook = None
        if path := scheme.get("codebook"):
            trained = torch.load(path, weights_only=True)
            codebook = trained["mapping"].cuda(), trained["packed"].cuda()
        for step in args.steps:
            current = inputs.load(bucket, "calibration", 4, layer, step, 0)
            previous = inputs.load(bucket, "calibration", 4, layer, step - 1, 0)
            if [n + 1 for n in previous.lengths] != current.lengths or previous.request_ids != current.request_ids:
                raise RuntimeError("prior state is not the previous step of these requests")
            pages, lengths = page_layout(current.lengths, max(current.lengths) + 1)
            state = QuantizedKeys.from_paged(
                Plan.from_lengths(lengths, 64), current.keys, pages, lengths, scheme["bits"],
                scheme["groups"], scheme["fit_steps"], scheme["fp4"], codebook,
            )
            dtype = torch.float16 if scheme["half_scores"] else torch.float32
            score, _ = state.prepare(lengths, indexer.softmax_scale, scheme["tile"], score_dtype=dtype)
            old_score, _ = state.prepare(lengths - 1, indexer.softmax_scale, scheme["tile"], score_dtype=dtype)
            now = score(*predictor_projection(indexer, current.hidden, current.positions)())
            before = old_score(*predictor_projection(indexer, previous.hidden, previous.positions)())
            for i, (rid, length) in enumerate(zip(current.request_ids, current.lengths, strict=True)):
                for pivot, count, ties in pivot_counts(now[i, :length], before[i, :length - 1], m, args.samples):
                    record = dict(length_bucket=bucket, layer=layer, M=m, step=step, request=rid,
                                  history_tokens=length, method=route["method"], pivot=pivot,
                                  candidates=count, enough=(count >= m - 1), cutoff_tie=ties)
                    print(json.dumps(record), file=stream, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--codebooks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--samples", type=int, default=2048)
    torch.set_num_threads(2)
    args = parser.parse_args()
    with torch.inference_mode(), args.output.open("x") as out:
        run(args, out)
