"""Check the fused prediction Q projection bitwise on real Keye inputs."""

import argparse
import json
from pathlib import Path

import torch

from benchmark import CapturedInputs, initialize, load_indexer, predictor_projection


def audit(args, stream):
    initialize()
    fixtures = CapturedInputs(args.capture)
    for layer in args.layers:
        indexer = load_indexer(layer)
        for bucket, batch in zip(args.length, args.batch, strict=True):
            for step in args.steps:
                r = fixtures.load(bucket, "calibration", batch, layer, step, 0)
                query, _, gate = indexer._get_q_k_w_bf16(r.hidden, r.positions, False)
                for mode in ("native", "normalization", "full_epilogue"):
                    call = predictor_projection(
                        indexer, r.hidden, r.positions,
                        fused=mode == "normalization", full_epilogue=mode == "full_epilogue",
                    )
                    variant_query, variant_gate = call()
                    q_bits = (query.view(torch.int16) != variant_query.view(torch.int16)).sum()
                    w_bits = (gate.view(torch.int16) != variant_gate.view(torch.int16)).sum()
                    print(json.dumps(dict(
                        length_bucket=bucket, batch=batch, layer=layer, step=step, variant=mode,
                        request_ids=r.request_ids, history_lengths=r.lengths, query_values=query.numel(),
                        query_mismatches=int(q_bits), gate_mismatches=int(w_bits),
                    )), file=stream, flush=True)
                    if int(q_bits) or int(w_bits):
                        raise RuntimeError(f"projection differs in layer {layer}, bucket {bucket}, step {step}")


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
