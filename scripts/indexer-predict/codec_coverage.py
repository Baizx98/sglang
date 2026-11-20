"""Compare real same-state candidate coverage, without GPU timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark import (
    CapturedInputs, Plan, QuantizedKeys, initialize, intersections, keye_indexer_score_paged,
    load_indexer, page_layout, predictor_projection, reference_topm,
)


def run(args, stream):
    initialize()
    fixtures = CapturedInputs(args.capture)
    for layer in args.layers:
        indexer = load_indexer(layer)
        scale = indexer.softmax_scale
        trained = torch.load(args.pair_pq / f"l{layer:02}.pt", weights_only=True)
        book = trained["mapping"].cuda(), trained["packed"].cuda()
        for bucket, batch in zip(args.length, args.batch, strict=True):
            for step in args.steps:
                replay = fixtures.load(bucket, args.split, batch, layer, step, 0)
                width = max(replay.lengths) + 1
                pages, lengths = page_layout(replay.lengths, width)
                plan = Plan.from_lengths(lengths, growth=64)
                query, gate = predictor_projection(indexer, replay.hidden, replay.positions)()
                scores = keye_indexer_score_paged(query, replay.keys, pages, gate, lengths, scale)
                base = {m: reference_topm(scores[:, :width], lengths, m)[0] for m in args.candidates}
                formats = (
                    ("int4_g1", 4, 1, 0, False, None),
                    ("int4fit2_g2", 4, 2, 2, False, None),
                    ("fp4_g1", 4, 1, 0, True, None),
                    ("pq4_g1", 4, 1, 0, False, book),
                    ("int5_g1", 5, 1, 0, False, None),
                    ("int5_g2", 5, 2, 0, False, None),
                    ("int6_g1", 6, 1, 0, False, None),
                    ("int8_g1", 8, 1, 0, False, None),
                )
                for method, bits, groups, fit, fp4, codebook in formats:
                    compact = QuantizedKeys.from_paged(
                        plan, replay.keys, pages, lengths, bits, groups, fit, fp4, codebook,
                    )
                    score, _ = compact.prepare(lengths, scale, 128, score_dtype=torch.float16)
                    predicted = score(query, gate)
                    for m, gold in base.items():
                        topm, _ = reference_topm(predicted, lengths, m)
                        data = dict(
                            bucket=bucket, batch=batch, layer=layer, step=step, M=m, method=method,
                            requests=replay.request_ids, history_lengths=replay.lengths,
                            bf16=intersections(gold, replay.target_ids).tolist(),
                            observed=intersections(topm, replay.target_ids).tolist(),
                        )
                        print(json.dumps(data), file=stream, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--pair-pq", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, nargs="+", required=True)
    parser.add_argument("--batch", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 12, 24, 47])
    parser.add_argument("--steps", type=int, nargs="+", default=[2])
    parser.add_argument("--candidates", type=int, nargs="+", default=[3072, 8192])
    parser.add_argument("--split", choices=("calibration", "test"), default="calibration")
    torch.set_num_threads(2)
    args = parser.parse_args()
    with torch.inference_mode(), args.output.open("x") as stream:
        run(args, stream)
