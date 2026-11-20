"""Validate device score bounds and masked BF16 reads on real Keye histories."""

import argparse
import json
import math
from pathlib import Path

import torch

from benchmark import (
    CapturedInputs, Plan, QuantizedKeys, initialize, keye_indexer_score_paged,
    load_indexer, page_layout, predictor_projection, reference_topm,
)
from certified_device import CertifiedRows
from verify import read_format


def graph_result(predict):
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        predict()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = predict()
    current.wait_stream(stream)
    graph.replay()
    return output.clone()


def audit(replay, indexer, case, args, stream):
    width = max(replay.lengths) + 1
    pages, lengths = page_layout(replay.lengths, width)
    plan = Plan.from_lengths(lengths, growth=64)
    q, gates = predictor_projection(indexer, replay.hidden, replay.positions)()
    exact = keye_indexer_score_paged(q, replay.keys, pages, gates, lengths, indexer.softmax_scale)
    live = torch.arange(width, device=exact.device)[None, :] < lengths[:, None]
    block_live = torch.nn.functional.pad(live, (0, -width % 128)).view(lengths.numel(), -1, 128)
    live_tiles = block_live.any(-1).sum(1)
    for bits in args.bits:
        compact = QuantizedKeys.from_paged(plan, replay.keys, pages, lengths, bits, 1)
        certified = CertifiedRows(compact, replay.keys, pages, lengths, indexer.softmax_scale)
        append_slots = plan.bases + lengths
        certified.append(replay.current_keys, append_slots)
        key_scales = compact.scales[append_slots.long()].double()
        restored = read_format(compact, append_slots).double() * key_scales
        exact_norms = torch.stack((restored.norm(dim=1),
                                  (replay.current_keys.double() - restored).norm(dim=1)), dim=1)
        append_upper = certified.k_norms[append_slots.long()].double()
        if not (append_upper >= exact_norms).all():
            raise RuntimeError("appended compact metadata underestimates the exact key norms")
        approx, upper = certified.score_and_bound(q, gates)
        expected_scorer, _ = compact.prepare(lengths, indexer.softmax_scale, block=128)
        expected = expected_scorer(q, gates)[:, :width]
        score_diff = (approx != expected) & live
        actual_slack = (upper - exact).masked_fill(~live, float("inf"))
        if torch.isnan(actual_slack).any() or (actual_slack < 0).any():
            raise RuntimeError(f"device bound drops a native score, {case}, INT{bits}")
        for m in args.candidates:
            k = m - 1
            gold, cutoff_ties = reference_topm(exact, lengths, m)
            seed = approx.topk(math.ceil(k * 1.25)).indices
            seed_scores = certified.exact_subset(q, gates, seed)
            if not torch.equal(seed_scores, exact.gather(1, seed)):
                raise RuntimeError("the seed scorer changed native BF16 scores")
            threshold = seed_scores.topk(k).values[:, -1:]
            present = live & (upper >= threshold)
            if not present.gather(1, gold[:, 1:].long()).all():
                raise RuntimeError("device screening pruned the original Top-M")
            rest = present.scatter(1, seed, False)
            capacity = min(width, math.ceil(k * args.capacity_factor))
            compacted, counts, overflow = certified.compact_ids(rest, capacity)
            if not torch.equal(counts, rest.sum(1)):
                raise RuntimeError("device compaction counted different candidates")
            valid = compacted >= 0
            if not torch.equal(valid.sum(1), counts.masked_fill(overflow, 0)):
                raise RuntimeError("unexpected number of candidates in the compact buffer")
            remaining_scores = certified.exact_subset(q, gates, compacted)
            expected = exact.gather(1, compacted.clamp_min(0).long()).masked_fill(~valid, -float("inf"))
            if not torch.equal(remaining_scores, expected):
                raise RuntimeError("compacted BF16 scores differ from the paged full score")
            masked = certified.final_scores.fill_(-float("inf"))
            seed_values = seed_scores.masked_fill(~present.gather(1, seed), -float("inf"))
            certified.scatter_scores(seed, seed_values)
            certified.scatter_scores(compacted, remaining_scores)
            if not torch.equal(masked[~overflow], exact.masked_fill(~present, -float("inf"))[~overflow]):
                raise RuntimeError("some screened BF16 scores were lost during scatter")
            if overflow.any():
                fallback = keye_indexer_score_paged(
                    q, replay.keys, pages, gates, torch.where(overflow, lengths, 0), indexer.softmax_scale,
                )
                masked.copy_(torch.where(overflow[:, None], fallback, masked))
            chosen, observed_ties = reference_topm(masked, lengths, m)
            if not torch.equal(observed_ties, cutoff_ties):
                raise RuntimeError("the masked scan lost a native cutoff tie")
            predicted_eq = (chosen.sort(1).values == gold.sort(1).values).all(1)
            if (~predicted_eq & ~observed_ties).any():
                raise RuntimeError("the masked selector changed a unique BF16 cutoff")
            # A fixed-shape eager replay of the fallback. Rows with native
            # cutoff ties can keep the original dense selection's tie order.
            tie_fallback = observed_ties & ~overflow
            if tie_fallback.any():
                fallback_lens = torch.where(tie_fallback, lengths, 0)
                fallback = keye_indexer_score_paged(q, replay.keys, pages, gates,
                                                    fallback_lens, indexer.softmax_scale)
                recovered, _ = reference_topm(torch.where(tie_fallback[:, None], fallback, masked), lengths, m)
                if not torch.equal(recovered.sort(1).values, gold.sort(1).values):
                    raise RuntimeError("fallback failed to recover the original native set")
            predict = certified.prepare(lambda: (q, gates), m, args.capacity_factor)
            routed = graph_result(predict) if args.check_graph else predict()
            if not torch.equal(routed.sort(1).values, gold.sort(1).values):
                raise RuntimeError("the device-only prediction changed the native candidate set")
            slots = torch.nn.functional.pad(rest, (0, -width % 128)).view(lengths.numel(), -1, 128)
            active_tiles = slots.any(-1).sum(1)
            exact_reads = seed.size(1) + valid.sum(1)
            logical_bytes = lengths * (8 * bits + 2 + 4) + (exact_reads + (overflow | tie_fallback) * lengths) * 128
            names = (
                ("min_bound_slack", actual_slack.min(1).values),
                ("compact_score_differences", score_diff.sum(1)),
                ("active_exact_tiles", active_tiles), ("live_tiles", live_tiles),
                ("exact_seed_and_compact_reads", exact_reads), ("cutoff_tie", cutoff_ties),
                ("pre_tie_fallback_set_equal", predicted_eq), ("overflow", overflow),
                ("remainder_candidates", counts), ("logical_key_read_bytes", logical_bytes),
            )
            values = {name: value.tolist() for name, value in names}
            values["key_norm_reserved_bytes"] = certified.k_norms.nbytes
            values["candidate_capacity"] = capacity
            for i, (rid, n) in enumerate(zip(replay.request_ids, replay.lengths, strict=True)):
                out = dict(case, M=m, request=rid, history_tokens=n, bits=bits,
                           device_route_set_equal=True, graph_checked=args.check_graph,
                           append_norm_upper=True,
                           **{name: result[i] if isinstance(result, list) else result
                              for name, result in values.items()})
                print(json.dumps(out), file=stream, flush=True)


def run(args, stream):
    initialize()
    fixtures = CapturedInputs(args.capture)
    for layer in args.layers:
        indexer = load_indexer(layer)
        for bucket, batch in zip(args.length, args.batch, strict=True):
            for step in args.steps:
                trace = fixtures.load(bucket, "calibration", batch, layer, step, 0)
                audit(trace, indexer, dict(length_bucket=bucket, batch=batch, layer=layer, step=step),
                      args, stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, nargs="+", required=True)
    parser.add_argument("--batch", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 12, 24, 47])
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--bits", type=int, choices=[6, 8], nargs="+", default=[6, 8])
    parser.add_argument("--candidates", type=int, nargs="+", default=[3072, 8192])
    parser.add_argument("--capacity-factor", type=float, default=4)
    parser.add_argument("--check-graph", action="store_true")
    torch.set_num_threads(2)
    args = parser.parse_args()
    if not 0 < args.capacity_factor:
        parser.error("--capacity-factor must be positive")
    with torch.inference_mode(), args.output.open("x") as out:
        run(args, out)
