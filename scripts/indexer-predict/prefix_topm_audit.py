"""Verify prefix compaction on the calibrated FP16 low-bit score routes."""

import argparse
import json
from pathlib import Path

import torch
from flashinfer import top_k_ragged_transform

from benchmark import (
    CapturedInputs, Plan, QuantizedKeys, initialize, intersections,
    load_indexer, page_layout, predictor_projection,
)
from certified_device_audit import graph_result
from prefix_topm import PrefixTopM
from quality_cases import layer_policy


def audit(args, stream):
    initialize()
    inputs = CapturedInputs(args.capture)
    routes = json.loads(args.policy.read_text())["routes"]
    for route in routes:
        mode = layer_policy(route, args.codebooks)
        if mode["method"] != "quantized" or mode.get("oversample") or not mode["half_scores"]:
            continue
        bucket, layer, m = route["length_bucket"], route["layer"], route["M"]
        if args.layers and layer not in args.layers:
            continue
        indexer = load_indexer(layer)
        codebook = None
        if path := mode.get("codebook"):
            trained = torch.load(path, weights_only=True)
            codebook = trained["mapping"].cuda(), trained["packed"].cuda()
        for step in args.steps:
            replay = inputs.load(bucket, "calibration", 4, layer, step, 0)
            pages, lengths = page_layout(replay.lengths, max(replay.lengths) + 1)
            state = QuantizedKeys.from_paged(
                Plan.from_lengths(lengths, 64), replay.keys, pages, lengths, mode["bits"],
                mode["groups"], mode["fit_steps"], mode["fp4"], codebook,
            )
            scorer, _ = state.prepare(lengths, indexer.softmax_scale, mode["tile"], score_dtype=torch.float16)
            query, gates = predictor_projection(indexer, replay.hidden, replay.positions)()
            scores = scorer(query, gates)
            current = lengths[:, None]
            k = m - 1
            baseline = top_k_ragged_transform(scores, torch.zeros_like(lengths), lengths, k)
            reference = scores.topk(k, sorted=True).values
            cutoff = reference[:, -1:]
            baseline_hits = intersections(torch.cat((current, baseline), dim=1), replay.target_ids.int())
            for bits in args.prefix_bits:
                select = PrefixTopM(lengths, scores.size(1), m, bits, args.capacity_factor)
                if args.score_histogram:
                    fused, _ = state.prepare(lengths, indexer.softmax_scale, mode["tile"],
                                             score_dtype=torch.float16, prefix=select)
                    if not torch.equal(fused(query, gates), scores):
                        raise RuntimeError("fusing the prefix histogram changed low-bit scores")

                    def pick():
                        return select.select_from_histogram(fused(query, gates))
                else:
                    def pick():
                        return select(scores)
                result = graph_result(pick) if args.check_graph else pick()
                if not torch.equal(select.histogram.sum((1, 2)).int(), lengths):
                    raise RuntimeError("histogram lost a live score")
                if not torch.equal(select.offsets, select.compact_lengths):
                    raise RuntimeError("prefix histogram and compaction counts differ")
                sorted_ids = result.sort(1).values
                if ((sorted_ids < 0) | (sorted_ids >= current)).any() or (sorted_ids.diff(dim=1) <= 0).any():
                    raise RuntimeError("prefix selector returned invalid or duplicate logical IDs")
                ranked = scores.gather(1, result.long()).sort(1, descending=True).values
                if not torch.equal(reference, ranked):
                    raise RuntimeError("prefix selector missed a greater score")
                hits = intersections(torch.cat((current, result), dim=1), replay.target_ids.int())
                overlap = intersections(result, baseline)
                cutoff_tie = (scores == cutoff).sum(1) > (reference == cutoff).sum(1)
                info = zip(replay.request_ids, replay.lengths, select.observed.tolist(),
                           select.fallback_lengths.tolist(), cutoff_tie.tolist(),
                           hits.tolist(), baseline_hits.tolist(), overlap.tolist(), strict=True)
                for rid, history, count, fallback, tied, got, same, common in info:
                    record = dict(
                        length_bucket=bucket, batch=4, layer=layer, step=step, M=m,
                        method=route["method"], prefix_bits=bits, capacity=select.capacity,
                        request=rid, history_tokens=history, retained_scores=count,
                        dense_fallback=bool(fallback), cutoff_tie=tied,
                        hit_prefix=got, hit_dense_fp16=same, dense_set_intersection=common,
                        graph_checked=args.check_graph, fused_histogram=args.score_histogram,
                    )
                    print(json.dumps(record), file=stream, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--codebooks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--prefix-bits", type=int, nargs="+", default=[7, 8, 9])
    parser.add_argument("--capacity-factor", type=float, default=2)
    parser.add_argument("--check-graph", action="store_true")
    parser.add_argument("--score-histogram", action="store_true")
    torch.set_num_threads(2)
    args = parser.parse_args()
    with torch.inference_mode(), args.output.open("x") as out:
        audit(args, out)
