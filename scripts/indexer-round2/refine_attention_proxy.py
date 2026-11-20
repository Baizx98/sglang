"""CPU same-state attention proxy for the approximate predicted prefix."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from refine_bench import K, TRACE, Trace, write_csv


FIELDS = (
    "request_id", "length_bucket", "actual_context", "layer", "step", "prefix_overlap_exact_full",
    "prefix_cosine", "prefix_relative_l2", "prefix_max_abs",
    "cpu_truth_vs_captured_cosine", "cpu_truth_vs_captured_relative_l2",
)


def selected_cache(trace, rid, rank, frames, logical):
    prefix = f"{rid}_layer01_rank{rank:02d}"
    history = trace.read(prefix + "_history.pt")
    init = int(history["length"])
    keys = history["main_k"].index_select(0, logical.clamp_max(init - 1))
    values = history["main_v"].index_select(0, logical.clamp_max(init - 1))
    pending = logical >= init
    for step in sorted(frames):
        row = frames[step]
        position = int(row["seq_lens"][0]) - 1
        match = logical == position
        if match.any():
            delta = row["deltas"][0]
            keys[match] = delta["main_k"]
            values[match] = delta["main_v"]
            pending &= ~match
    assert not pending.any(), (rid, rank, pending.sum())
    return keys, values


def attend(query, keys, values):
    q_heads, head_dim = query.shape[-2:]
    kv_heads = keys.size(1)
    q = query.float().reshape(kv_heads, q_heads // kv_heads, head_dim)
    k, v = keys.transpose(0, 1).float(), values.transpose(0, 1).float()
    score = (q @ k.transpose(1, 2)) * head_dim ** -0.5
    return (torch.softmax(score, dim=-1) @ v).flatten()


def stats(output, truth):
    error = output - truth
    return (
        float(torch.nn.functional.cosine_similarity(output, truth, dim=0)),
        float(error.norm() / truth.norm().clamp_min(1e-30)),
        float(error.abs().max()),
    )


def run(args):
    trace = Trace(args.trace)
    checked = []
    for bucket in (32768, 65536, 131072):
        data = torch.load(args.candidates / f"candidates-{bucket}-b16.pt", weights_only=True)
        prefix = data["candidates"][8192][:, :K].long()
        for i, rid in enumerate(data["requests"]):
            query = prefix[i]
            actual, exact_out, approx_out = [], [], []
            frame_names = trace.frames[rid, 1]
            for rank in (0, 1):
                frames = {
                    step: trace.read(name.replace("rank00", f"rank{rank:02d}"))
                    for step, name in frame_names.items() if step <= args.step
                }
                frame = frames[args.step]
                exact = frame["topk"].reshape(-1).long()
                context = int(frame["seq_lens"][0])
                assert context == int(data["lengths"][i])
                assert exact.numel() == K and exact.unique().numel() == K
                assert query.numel() == K and query.unique().numel() == K
                assert query.min() >= 0 and query.max() < context
                cache_data = selected_cache(trace, rid, rank, frames, torch.cat((exact, query)))
                key, val = (tensor.split(K) for tensor in cache_data)
                exact_out.append(attend(frame["query"], key[0], val[0]))
                approx_out.append(attend(frame["query"], key[1], val[1]))
                actual.append(frame["attention_output"].float().flatten())
            exact_out, approx_out, actual = (torch.cat(rows) for rows in (exact_out, approx_out, actual))
            cosine, l2, maxabs = stats(approx_out, exact_out)
            true_cosine, true_l2, _ = stats(exact_out, actual)
            checked.append({
                "request_id": rid, "length_bucket": bucket, "actual_context": context, "layer": 1, "step": args.step,
                "prefix_overlap_exact_full": float(torch.isin(query, exact).float().mean()),
                "prefix_cosine": cosine, "prefix_relative_l2": l2, "prefix_max_abs": maxabs,
                "cpu_truth_vs_captured_cosine": true_cosine, "cpu_truth_vs_captured_relative_l2": true_l2,
            })
        print(f"{bucket}: {len(data['requests'])} native and prefix attention rows checked", flush=True)
    write_csv(args.output, checked, FIELDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=2)
    torch.set_num_threads(2)
    with torch.inference_mode():
        run(parser.parse_args())
