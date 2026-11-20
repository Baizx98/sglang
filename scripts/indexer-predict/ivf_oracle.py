"""Measure IVF screening quality with fitted real-history clusters and exact rescoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch


def nearest(points, centroids):
    # The key norm is constant across candidate centroids.
    offsets = centroids.square().sum(dim=1)
    return torch.cat([
        (2 * (chunk @ centroids.T) - offsets[None, :]).argmax(dim=1)
        for chunk in points.split(8192)
    ])


def fit(keys, count, sample_count, steps, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    picks = torch.randperm(keys.size(0), generator=generator)[:sample_count]
    examples = keys[picks].float()
    initial = torch.randperm(examples.size(0), generator=generator)[:count]
    centers = examples[initial].clone()
    for _ in range(steps):
        assignment = nearest(examples, centers)
        active = torch.bincount(assignment, minlength=count)
        total = torch.zeros_like(centers).index_add_(0, assignment, examples)
        means = total / active[:, None].clamp_min(1)
        centers = torch.where(active[:, None] > 0, means, centers)
    return centers


def shortlists(scores, labels, counts, truth, full, m, fractions):
    from benchmark import intersections

    length = labels.size(1)
    base = torch.cat((torch.full((1, 1), length, device=full.device),
                      full[:, :length].topk(m - 1).indices), dim=1)
    overlap = int(intersections(base, truth))
    order = scores.argsort(dim=1, descending=True)
    cluster_sizes = counts.gather(1, order).cumsum(dim=1)
    # Include the current token as a reserved logical position in all methods.
    extra = torch.full((1, 1), length, device=full.device, dtype=torch.int64)
    for fraction in fractions:
        budget = max(m - 1, math.ceil(length * fraction))
        take = cluster_sizes - counts.gather(1, order) < budget
        chosen = torch.zeros_like(take).scatter_(1, order, take)
        region = chosen.gather(1, labels)
        selected = full[:, :length].masked_fill(~region, -float("inf")).topk(m - 1)
        ids = torch.cat((extra, selected.indices.masked_fill(~selected.values.isfinite(), -1)), dim=1)
        hit = int(intersections(ids, truth))
        yield dict(M=m, fraction=fraction, selected_tokens=int(region.sum()),
                   keep_ratio=float(region.float().mean()), exact_hits=overlap, observed_hits=hit,
                   additional_loss_pp=100 * (overlap - hit) / 2048)


@torch.inference_mode()
def prepare(args):
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    requests = [json.loads(line)["rid"] for line in (args.capture / "completed.jsonl").read_text().splitlines()]
    rid = requests[args.request_offset]
    for layer in args.layers:
        base = args.capture / "fixtures" / f"{rid}-l{layer:02}"
        keys = torch.load(f"{base}-keys.pt", mmap=True, weights_only=True)[:-1]
        for count in args.centroids:
            start = time.perf_counter()
            centers = fit(keys, count, args.sample, args.iterations, args.seed)
            labels = nearest(keys.float(), centers)
            wall = time.perf_counter() - start
            torch.save(dict(request=rid, layer=layer, centers=centers, labels=labels.int(),
                            prefix_tokens=len(keys), seed=args.seed, duration_cpu_s=wall),
                       args.output / f"l{layer:02}-c{count}.pt")
            print(dict(layer=layer, centroids=count, tokens=len(keys), duration_cpu_s=wall), flush=True)


@torch.inference_mode()
def assess(args):
    from benchmark import (CapturedInputs, initialize, keye_indexer_score_paged,
                           load_indexer, page_layout, predictor_projection)

    initialize()
    replay = CapturedInputs([args.capture])
    with args.output.open("x") as dst:
        for path in sorted(args.index.glob("l*-c*.pt")):
            record = torch.load(path, mmap=True, weights_only=True)
            layer = record["layer"]
            ix = load_indexer(layer)
            for step in args.steps:
                actual = replay.load(args.bucket, "calibration", 1, layer, step, args.request_offset)
                assert actual.request_ids == [record["request"]]
                centers = record["centers"].cuda()
                labels = record["labels"].cuda().long()
                prefix = record["prefix_tokens"]
                assert labels.numel() == prefix <= actual.lengths[0]
                if actual.lengths[0] > prefix:
                    tail = nearest(actual.keys[prefix:].float(), centers)
                    labels = torch.cat((labels, tail))
                labels = labels[None, :]
                counts = labels.flatten().bincount(minlength=len(centers))[None, :]
                pages, lens = page_layout(actual.lengths, max(actual.lengths))
                q, weights = predictor_projection(ix, actual.hidden, actual.positions)()
                full = keye_indexer_score_paged(q, actual.keys, pages, weights, lens, ix.softmax_scale)
                dots = (q[0] @ centers.bfloat16().T).float()
                mean_key_score = ((dots * ix.softmax_scale).relu() * weights[0, :, None].float()).sum(0)
                for m in args.candidates:
                    for test in shortlists(mean_key_score[None, :], labels, counts, actual.target_ids,
                                           full, m, (.125, .25, .5)):
                        dst.write(json.dumps(dict(
                            request=record["request"], length=actual.lengths[0], layer=layer, step=step,
                            cluster_count=len(centers), initial_fit_cpu_s=record["duration_cpu_s"], **test,
                        )) + "\n")
                        dst.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("fit")
    a.add_argument("--centroids", type=int, nargs="+", default=[64, 256])
    a.add_argument("--layers", type=int, nargs="+", default=[12, 24, 47])
    a.add_argument("--sample", type=int, default=16384)
    a.add_argument("--iterations", type=int, default=6)
    a.add_argument("--threads", type=int, default=4)
    a.add_argument("--seed", type=int, default=20261120)
    b = sub.add_parser("assess")
    b.add_argument("--index", type=Path, required=True)
    b.add_argument("--bucket", type=int, default=262144)
    b.add_argument("--steps", type=int, nargs="+", default=[2, 4])
    b.add_argument("--candidates", type=int, nargs="+", default=[3072, 8192])
    for command in (a, b):
        command.add_argument("--capture", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--request-offset", type=int, default=0)
    args = parser.parse_args()
    (prepare if args.command == "fit" else assess)(args)
