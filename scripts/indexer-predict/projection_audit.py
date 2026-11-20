"""Check the fused prediction Q projection bitwise on real Keye inputs."""

import argparse
import json
from pathlib import Path

import torch

from benchmark import CapturedInputs, initialize, load_indexer, predictor_projection
from quantized import _encode_query
from query import QueryProjection


def audit(args, stream):
    initialize()
    fixtures = CapturedInputs(args.capture)
    for layer in args.layers:
        indexer = load_indexer(layer)
        for bucket, batch in zip(args.length, args.batch, strict=True):
            for step in args.steps:
                r = fixtures.load(bucket, "calibration", batch, layer, step, 0)
                query, _, gate = indexer._get_q_k_w_bf16(r.hidden, r.positions, False)
                full = QueryProjection(indexer, batch, r.hidden.device, full_epilogue=True)
                routes = {
                    "native": predictor_projection(indexer, r.hidden, r.positions),
                    "normalization": predictor_projection(indexer, r.hidden, r.positions, fused=True),
                    "full_epilogue": lambda: full(r.hidden, r.positions),
                }
                case = dict(length_bucket=bucket, batch=batch, layer=layer, step=step,
                            request_ids=r.request_ids, history_lengths=r.lengths)
                for mode, call in routes.items():
                    variant_query, variant_gate = call()
                    q_bits = (query.view(torch.int16) != variant_query.view(torch.int16)).sum()
                    w_bits = (gate.view(torch.int16) != variant_gate.view(torch.int16)).sum()
                    print(json.dumps(dict(
                        case, variant=mode, query_values=query.numel(), query_mismatches=int(q_bits),
                        gate_mismatches=int(w_bits),
                    )), file=stream, flush=True)
                    if int(q_bits) or int(w_bits):
                        raise RuntimeError(f"projection differs in layer {layer}, bucket {bucket}, step {step}")
                for groups in (1, 2):
                    codes = torch.empty_like(query, dtype=torch.int8)
                    scales = torch.empty((*query.shape[:2], groups), device=query.device, dtype=torch.float32)
                    reference_codes = torch.empty_like(codes)
                    reference_scales = torch.empty_like(scales)
                    _encode_query[(batch, 16, groups)](
                        query, reference_codes, reference_scales, query.stride(0), query.stride(1),
                        groups, num_warps=2,
                    )
                    fused_gate = full.encode_into(r.hidden, r.positions, codes, scales, groups)
                    q_bits = (codes != reference_codes).sum()
                    s_bits = (scales.view(torch.int32) != reference_scales.view(torch.int32)).sum()
                    w_bits = (gate.view(torch.int16) != fused_gate.view(torch.int16)).sum()
                    print(json.dumps(dict(
                        case, variant=f"full_epilogue_int8_g{groups}", query_values=codes.numel(),
                        query_mismatches=int(q_bits), scale_mismatches=int(s_bits), gate_mismatches=int(w_bits),
                    )), file=stream, flush=True)
                    if int(q_bits) or int(s_bits) or int(w_bits):
                        raise RuntimeError(f"quantized projection differs in layer {layer}, bucket {bucket}, step {step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, nargs="+", required=True)
    parser.add_argument("--batch", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 12, 24, 47])
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 4])
    torch.set_num_threads(2)
    args = parser.parse_args()
    with torch.inference_mode(), args.output.open("x") as out:
        audit(args, out)
