"""Audit fused projection bitwise on captured hidden states without loading K histories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from projection import prepare_projection
from refine_bench import TRACE, Trace, load_indexer
from refine_projection_bench import bitwise_projection


def run(args):
    from horizonkv_layer_replay import initialize

    initialize()
    trace = Trace(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("layer", "step", "length_bucket", "batch", "q_bits", "k_bits", "gate_bits")
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for layer in args.layer:
            indexer = load_indexer(layer)
            for length in args.length:
                for batch in args.batch:
                    ids = trace.ids(length, batch, args.offset)
                    for step in args.step:
                        hidden, positions, _, _ = trace.append_frames(ids, layer, step)
                        native = indexer._get_q_k_w_bf16(hidden, positions, False)
                        fused = prepare_projection(indexer, hidden, positions)()
                        diff = bitwise_projection(native, fused)
                        record = {
                            "layer": layer, "step": step, "length_bucket": length, "batch": batch,
                            **{name + "_bits": count for name, count in diff.items()},
                        }
                        writer.writerow(record)
                        if any(diff.values()):
                            print(record, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, nargs="+", default=[1, 12, 36])
    parser.add_argument("--step", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--length", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--batch", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--offset", type=int, default=8)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--output", type=Path, required=True)
    torch.set_num_threads(4)
    with torch.inference_mode():
        run(parser.parse_args())
