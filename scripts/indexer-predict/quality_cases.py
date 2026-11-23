"""Prepare a deterministic RULER generation check for a calibrated route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def layer_policy(row, codebooks):
    if row["exact"]:
        return {"method": "bf16"}
    name, _ = row["method"].split(":")
    native_mma = 0
    for suffix, bits in (("_native4_fused_q", 4), ("_native8", 8)):
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
            native_mma = bits
            break
    progressive = re.fullmatch(r"int6_progressive_([\d.]+)", name)
    if progressive:
        return dict(method="quantized", bits=6, groups=1, tile=128,
                    half_scores=True, oversample=float(progressive[1]))
    match = re.fullmatch(r"(int|fp|pq)(\d)(?:fit(\d+))?_g(\d)_b(\d+)(_f16)?", name)
    if not match:
        raise ValueError(f"no quality replay for {name}")
    category, bits, fit, groups, tile, half = match.groups()
    route = dict(method="quantized", bits=int(bits), groups=int(groups), tile=int(tile),
                 fit_steps=int(fit or 0), half_scores=bool(half), fp4=category == "fp")
    if native_mma:
        if category != "int" or route["bits"] != native_mma or route["groups"] != 1 or route["tile"] != 128:
            raise ValueError(f"unsupported native MMA format {row['method']}")
        route["native_mma"] = native_mma
    if category == "pq":
        route["codebook"] = str((codebooks / f"l{row['layer']:02}.pt").resolve(strict=True))
    return route


def run(args):
    frozen = json.loads(args.policy.read_text())
    known = set(frozen["calibration_request_ids"])
    policies = {}
    for row in frozen["routes"]:
        bucket, layer, width = row["length_bucket"], row["layer"], row["M"]
        if width == args.candidates:
            policies.setdefault(bucket, {})[layer] = layer_policy(row, args.codebooks)
    samples = {}
    for path in args.requests:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if (row["split"] == "test" and row["rid"] not in known and row["dataset"] == "RULER"
                    and row["length_bucket_upper"] in policies and row.get("expected_answer")):
                samples.setdefault(row["length_bucket_upper"], []).append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    for bucket, routes in sorted(policies.items()):
        heldout = sorted(samples[bucket], key=lambda row: row["input_ids_sha256"])[:args.per_bucket]
        if len(heldout) != args.per_bucket or len({r["input_ids_sha256"] for r in heldout}) != len(heldout):
            raise ValueError(f"need {args.per_bucket} distinct held-out RULER inputs for {bucket}")
        # Quality replay isolates scorer precision. The compact runtime uses
        # FlashInfer Top-M, so an alternate timed selector may break score ties
        # differently. Both modes change exactly the configured target layers.
        policy = dict(candidate_width=args.candidates, append_reserve=args.append_reserve,
                      default={"method": "native"}, layers=routes, audit_layers=sorted(routes))
        (args.output / f"policy-{bucket}.json").write_text(json.dumps(policy, indent=2) + "\n")
        (args.output / f"requests-{bucket}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in heldout),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--codebooks", type=Path, required=True)
    parser.add_argument("--requests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=3072)
    parser.add_argument("--append-reserve", type=int, default=64)
    parser.add_argument("--per-bucket", type=int, default=1)
    run(parser.parse_args())
