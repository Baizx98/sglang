"""Measure the real adjacent-layer ordered predictor and serial sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from refine_bench import K, TRACE, Trace, load_indexer, measure_graph, reconstruct_layer0_input


def run(args):
    from flashinfer import top_k_page_table_transform
    from horizonkv_layer_replay import initialize
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import (
        keye_indexer_score_candidates, keye_indexer_score_paged,
    )

    config = initialize()
    indexer = load_indexer(1)
    trace = Trace(args.trace)
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    raw = args.output / "predictor.jsonl"
    with raw.open("x") as stream:
        for context in args.length:
            for batch in args.batch:
                request_ids = trace.ids(context, batch, args.offset)
                inputs = trace.load(request_ids, 1, args.step, indexer, "merged")
                frames = [trace.read(trace.frames[rid, 1][args.step]) for rid in request_ids]
                tokens = [int(frame["input_ids"].item()) for frame in frames]
                old_input = reconstruct_layer0_input(tokens, config.rms_norm_eps)
                positions = torch.cat([frame["positions"] for frame in frames], dim=-1).cuda()
                history = inputs.lengths - 1
                current = history[:, None]
                current_slots = inputs.page_table.gather(1, current.long()).view(-1).long()
                _, new_key, _ = inputs.project()
                inputs.key_cache.index_copy_(0, current_slots, new_key)

                def predictor_scores():
                    predicted_q, _, predicted_gate = indexer._get_q_k_w_bf16(old_input, positions, False)
                    return keye_indexer_score_paged(
                        predicted_q, inputs.key_cache, inputs.page_table, predicted_gate,
                        history, inputs.scale,
                    )

                existing = predictor_scores()
                for width in args.width:
                    lengths = torch.full_like(history, width)

                    def order(scores=existing):
                        ordered = torch.topk(scores, width - 1, dim=1, sorted=True).indices.int()
                        return torch.cat((current, ordered), dim=1)

                    def predict():
                        return order(predictor_scores())

                    def exact(candidates, query=inputs.q, gates=inputs.weights):
                        scores = keye_indexer_score_candidates(
                            query, inputs.key_cache, inputs.page_table, candidates,
                            gates, inputs.lengths, inputs.scale,
                        )
                        return top_k_page_table_transform(scores, candidates, lengths, K)

                    ordered = predict()
                    observed_scores = existing.gather(1, ordered[:, 1:].long())
                    assert ordered[:, :1].equal(current)
                    assert observed_scores.diff(dim=1).le(0).all()
                    assert torch.equal(observed_scores, existing.topk(width - 1, sorted=True).values)
                    assert ordered.sort(dim=1).values.diff(dim=1).gt(0).all()

                    def serial_exact():
                        return exact(predict())

                    def serial_exact_with_target_write():
                        candidates = predict()
                        query, key, gates = inputs.project()
                        inputs.key_cache.index_copy_(0, current_slots, key)
                        return exact(candidates, query, gates)

                    def serial_prefix():
                        return predict()[:, :K].contiguous()

                    methods = {
                        "projection_score": predictor_scores,
                        "order_and_emit": order,
                        "ordered_predictor": predict,
                        "canonical_refine_only": lambda: exact(ordered),
                        "ordered_predictor_then_canonical_refine": serial_exact,
                        "ordered_predictor_then_prefix2048": serial_prefix,
                        "ordered_predictor_target_write_then_canonical_refine": serial_exact_with_target_write,
                    }
                    latencies = measure_graph(methods, args.trials, args.iterations, args.seed, args.warmups)
                    point = {
                        "length_bucket": context, "batch": batch, "candidate_width": width,
                        "layer": 1, "step": args.step, "request_offset": args.offset, "layout": "contiguous",
                    }
                    record = {**point, **{name: item["median_us"] for name, item in latencies.items()}}
                    records.append(record)
                    stream.write(json.dumps({**point, "request_ids": request_ids, "timings": latencies}) + "\n")
                    stream.flush()
                    print(record, flush=True)
    with (args.output / "predictor.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "predictor-environment.json").write_text(json.dumps({
        "gpu": str(torch.cuda.get_device_properties()),
        "torch": torch.__version__,
        "flashinfer": __import__("flashinfer").__version__,
        "timing": "same-graph actual serial chain; cached prerequisite h0 and committed historical K",
        "scopes": "target Indexer prediction projection as implemented computes and discards the unused predicted K; top-M uses PyTorch FP32 sorted output and a current-token prepend; exact refine uses native candidates and FlashInfer mapped output",
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--batch", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--width", type=int, nargs="+", default=[3072, 8192])
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--offset", type=int, default=8)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(options.seed)
    with torch.inference_mode():
        run(options)
