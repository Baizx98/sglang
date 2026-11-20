"""Compare native nibble MMA with the original INT8 MMA on calibrated inputs."""

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
from nibble_scan import prepare_scan
from quality_cases import layer_policy


def synthetic():
    torch.manual_seed(8554)
    lengths = torch.tensor([0, 129, 173, 511], dtype=torch.int32, device="cuda")
    plan = Plan.from_lengths(lengths, 64)
    state = QuantizedKeys.__new__(QuantizedKeys)
    state._allocate(plan, lengths, 4, 1, 0, False, None)
    keys = torch.randn((plan.capacity, 64), device="cuda", dtype=torch.bfloat16)
    state.build(keys)
    query = torch.randn((4, 16, 64), device="cuda", dtype=torch.bfloat16)
    state.encode_query(query)
    old, _ = state.prepare_scan(lengths, 0.125, 128)
    new, _ = prepare_scan(state, lengths, 0.125, torch.float32)
    gates = torch.zeros((4, 16), device="cuda", dtype=torch.bfloat16)
    for head in range(16):
        gates.zero_()
        gates[:, head] = 1
        if not torch.equal(old(gates), new(gates)):
            raise RuntimeError(f"nibble dot disagrees with INT8 MMA at head {head}")


def audit(args, out):
    initialize()
    synthetic()
    inputs = CapturedInputs(args.capture)
    routes = json.loads(args.policy.read_text())["routes"]
    for route in routes:
        mode = layer_policy(route, args.codebooks)
        if (mode["method"] != "quantized" or mode["bits"] != 4 or mode["groups"] != 1
                or mode["fp4"] or mode.get("codebook")):
            continue
        bucket, layer, m = route["length_bucket"], route["layer"], route["M"]
        indexer = load_indexer(layer)
        for step in (2, 4):
            replay = inputs.load(bucket, "calibration", 4, layer, step, 0)
            pages, lengths = page_layout(replay.lengths, max(replay.lengths) + 1)
            state = QuantizedKeys.from_paged(
                Plan.from_lengths(lengths, 64), replay.keys, pages, lengths, 4, 1, mode["fit_steps"],
            )
            query, gates = predictor_projection(indexer, replay.hidden, replay.positions)()
            state.encode_query(query)
            scale = indexer.softmax_scale
            for dtype in (torch.float32, torch.float16):
                old, _ = state.prepare_scan(lengths, scale, 128, score_dtype=dtype)
                new, workspace = prepare_scan(state, lengths, scale, dtype)
                reference = old(gates)
                new(gates)
                candidate = graph_result(lambda: new(gates))
                live = torch.arange(reference.size(1), device="cuda")[None, :] < lengths[:, None]
                if not torch.isfinite(candidate[live]).all() or (candidate[~live] != -float("inf")).any():
                    raise RuntimeError("native nibble scan changed live positions or padding")
                view = torch.int16 if dtype == torch.float16 else torch.int32
                if not torch.equal(reference.view(view), candidate.view(view)):
                    raise RuntimeError("native nibble scan changed the INT4 score bits")
                empty = torch.zeros_like(lengths)
                prior_ids = top_k_ragged_transform(reference, empty, lengths, m - 1)
                new_ids = top_k_ragged_transform(candidate, empty, lengths, m - 1)
                repeat_ids = top_k_ragged_transform(reference, empty, lengths, m - 1)
                exact_values = reference.topk(m - 1, dim=1).values
                for ids in (prior_ids, new_ids, repeat_ids):
                    valid = (ids >= 0) & (ids < lengths[:, None])
                    if not valid.all() or (ids.sort(1).values.diff(dim=1) <= 0).any():
                        raise RuntimeError("FlashInfer returned invalid or duplicate logical IDs")
                    if not torch.equal(reference.gather(1, ids.long()).sort(1, descending=True).values, exact_values):
                        raise RuntimeError("FlashInfer dropped a higher INT4 score")
                targets = replay.target_ids.int()
                current = lengths[:, None]
                common = intersections(prior_ids, new_ids).tolist()
                repeat_common = intersections(prior_ids, repeat_ids).tolist()
                prior_hits = intersections(torch.cat((current, prior_ids), dim=1), targets).tolist()
                new_hits = intersections(torch.cat((current, new_ids), dim=1), targets).tolist()
                changed = (reference != candidate).sum(1).tolist()
                errors = (reference.float() - candidate.float()).abs().masked_fill(~live, 0).amax(1).tolist()
                for i, request in enumerate(replay.request_ids):
                    record = dict(length_bucket=bucket, layer=layer, step=step, M=m, request=request,
                                  history_tokens=replay.lengths[i], score_dtype=str(dtype),
                                  fit_steps=mode["fit_steps"], changed_scores=changed[i],
                                  max_abs_error=errors[i], set_intersection=common[i],
                                  selector_repeat_intersection=repeat_common[i],
                                  prior_hits=prior_hits[i], new_hits=new_hits[i], graph_checked=True,
                                  batch_workspace_bytes=workspace)
                    print(json.dumps(record), file=out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--codebooks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with torch.inference_mode(), args.output.open("x") as stream:
        audit(args, stream)
