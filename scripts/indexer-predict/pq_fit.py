"""Fit a 256-entry shared product-quantization codebook on held-in Index Keys."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from ivf_oracle import fit, nearest


def codes(keys):
    peak = keys.float().abs().amax(1, keepdim=True)
    scale = (peak.clamp_min(1e-20) / 127).bfloat16().float()
    return (keys.float() / scale).round().clamp(-127, 127).int()


@torch.inference_mode()
def run(args):
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    rids = [json.loads(line)["rid"] for line in (args.capture / "completed.jsonl").read_text().splitlines()]
    if any("-calibration-" not in rid for rid in rids):
        raise ValueError("product-quantization codebooks use only calibration requests")
    for layer in args.layers:
        start = time.monotonic()
        rng = torch.Generator(device="cpu").manual_seed(args.seed)
        fitting = []
        paths = []
        for rid in rids:
            p = args.capture / "fixtures" / f"{rid}-l{layer:02}-keys.pt"
            raw = torch.load(p, weights_only=True, mmap=True)[:-1]
            sample = raw[torch.randperm(len(raw), generator=rng)[:args.per_request]]
            fitting.append(codes(sample).view(-1, 2))
            paths.append(str(p))
        X = torch.cat(fitting)
        centers = fit(X, 256, args.fit_points, args.iters, args.seed).round().clamp(-127, 127).int()
        coords = torch.arange(-128, 128)
        mapping = nearest(torch.cartesian_prod(coords, coords).float(), centers.float()).byte()
        packed = ((centers[:, 0] & 255) | ((centers[:, 1] & 255) << 8)).to(torch.int16)
        assigned = nearest(X.float(), centers.float())
        err = (X - centers[assigned]).float().square().mean().item()
        duration = time.monotonic() - start
        metadata = dict(
            layer=layer, training_requests=rids, sample_rows_per_request=args.per_request,
            sample_points=min(args.fit_points, X.size(0)), seed=args.seed, iterations=args.iters,
            l2_training_error_quantization_units=err, duration_cpu_s=duration,
            input_files=paths,
            mapping_sha256=hashlib.sha256(mapping.numpy().tobytes()).hexdigest(),
        )
        torch.save(dict(mapping=mapping, packed=packed, metadata=metadata),
                   args.output / f"l{layer:02}.pt")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 12, 24, 47])
    parser.add_argument("--per-request", type=int, default=2048)
    parser.add_argument("--fit-points", type=int, default=32768)
    parser.add_argument("--iters", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
