#!/usr/bin/env python3
"""Measure causal turn-start retention priorities on real BFCL Keye traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analyze_prompt_segment_selection import STEPS, build_lookup, load_chunk, read_jsonl


EARLY = 4
HISTORY_EDGES = (1, 2, 4, 8, 16, 24, 32)
COHORT = "prompt-prefix retention diagnostic on teacher-forced turn prompts"


@dataclass
class Transition:
    group: int
    trajectory: str
    category: str
    round: int
    layer: int
    segment: np.ndarray
    previous: np.ndarray
    prior_mean: np.ndarray
    early: np.ndarray
    whole: np.ndarray
    all_early: int
    all_whole: int


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def common_prefix(previous, current):
    return next((i for i, pair in enumerate(zip(previous, current))
                 if pair[0] != pair[1]), min(len(previous), len(current)))


def segment_features(request, segments, type_ids):
    ids = request["input_ids"]
    if len(ids) != request["prompt_len"] or token_digest(ids) != request["prompt_sha256"]:
        raise ValueError(f"prompt hash or length mismatch on {request['rid']}")
    features = np.empty(len(ids), dtype=np.int32)
    cursor = 0
    for segment in sorted(segments, key=lambda s: s["token_start"]):
        start, end = segment["token_start"], segment["token_end"]
        if not (cursor == start < end <= len(ids)) or token_digest(ids[start:end]) != segment["token_ids_sha256"]:
            raise ValueError(f"segment mismatch on {request['rid']}")
        if (request["trajectory_id"], request["round_id"]) != (segment["trajectory_id"], segment["round_id"]):
            raise ValueError("segment must belong to this trajectory and turn")
        age = segment["round_age"]
        if age is not None and age < 0:
            raise ValueError("prompt includes a future segment")
        age_bin = 0 if age is None else 1 + min(age, 3)
        features[start:end] = 5 * type_ids[segment["segment_type"]] + age_bin
        cursor = end
    if cursor != len(ids):
        raise ValueError(f"incomplete segment labels on {request['rid']}")
    return features


def counts(chunk, prompt_len):
    indices = chunk["indices"]
    counts_by_step = chunk["valid_counts"]
    ctx = chunk["score_lens"]
    if (not np.array_equal(ctx, prompt_len + 1 + np.arange(STEPS))
            or not np.array_equal((indices >= 0).sum(axis=1), counts_by_step)
            or np.any(indices >= ctx[:, None])):
        raise ValueError("invalid context or indices")
    ordered = np.sort(indices, axis=1)
    if np.any((np.diff(ordered, axis=1) == 0) & (ordered[:, 1:] >= 0)):
        raise ValueError("duplicate true Top-k IDs")
    def count_valid(xs):
        return np.bincount(xs[(xs >= 0) & (xs < prompt_len)], minlength=prompt_len).astype(np.int16)
    return (count_valid(indices[:EARLY]), count_valid(indices),
            int(counts_by_step[:EARLY].sum()), int(counts_by_step.sum()))


def collect(source, layers):
    requests = read_jsonl(source / "prepared_requests.jsonl")
    segments = defaultdict(list)
    all_segments = read_jsonl(source / "segments.jsonl")
    types = sorted({x["segment_type"] for x in all_segments})
    type_ids = {t: i for i, t in enumerate(types)}
    for entry in all_segments:
        segments[entry["rid"]].append(entry)
    paths = build_lookup(source, requests)
    trajectories = defaultdict(list)
    for entry in requests:
        trajectories[entry["trajectory_id"]].append(entry)
    if len(trajectories) < 2:
        raise ValueError("held-out trajectory evaluation requires multiple trajectories")

    transitions, prefix_rows = [], []
    for group, (trajectory, turns) in enumerate(sorted(trajectories.items())):
        turns.sort(key=lambda r: r["round_id"])
        if [t["round_id"] for t in turns] != list(range(len(turns))):
            raise ValueError(f"missing or duplicate turns in {trajectory}")
        previous = None
        last_counts, cumulative = {}, {}
        for current in turns:
            seg = segment_features(current, segments[current["rid"]], type_ids)
            plen = current["prompt_len"]
            if previous is None:
                prefix = 0
            else:
                prefix = common_prefix(previous["input_ids"], current["input_ids"])
                prefix_rows.append(dict(trajectory=trajectory, category=current["category"],
                                        round=current["round_id"], old_prompt=previous["prompt_len"],
                                        new_prompt=plen, common_prefix=prefix))
            for layer in layers:
                chunk = load_chunk(paths[(current["rid"], layer)])
                early, whole, all_early, all_whole = counts(chunk, plen)
                historical_counts = whole.astype(np.int32)
                exposures = np.ones(plen, dtype=np.int16)
                if prefix:
                    sums, prev_exposures = cumulative[layer]
                    historical_counts[:prefix] += sums[:prefix]
                    exposures[:prefix] += prev_exposures[:prefix]
                    transitions.append(Transition(
                        group, trajectory, current["category"], current["round_id"], layer,
                        seg[:prefix].copy(), last_counts[layer][:prefix].copy(),
                        sums[:prefix] / prev_exposures[:prefix], early[:prefix].copy(),
                        whole[:prefix].copy(), all_early, all_whole,
                    ))
                last_counts[layer] = whole
                cumulative[layer] = historical_counts, exposures
            previous = current
    return transitions, prefix_rows, types, len(trajectories), len(requests)


def base_rates(numerators, counts):
    global_rate = numerators.sum() / counts.sum()
    return (numerators + 256 * global_rate) / (counts + 256)


def rankings(rows, types, layers, groups, budgets):
    bins = len(HISTORY_EDGES) + 1
    nseg = 5 * len(types)
    layer_id = {layer: idx for idx, layer in enumerate(layers)}
    shape = (groups, len(layers), nseg, bins)
    exposure = np.zeros(shape)
    hits = np.zeros(shape)
    for row in rows:
        hist_bin = np.searchsorted(HISTORY_EDGES, row.previous, side="right")
        joint = row.segment * bins + hist_bin
        bucket = (row.group, layer_id[row.layer])
        exposure[bucket] += np.bincount(joint, minlength=nseg*bins).reshape(nseg, bins)
        hits[bucket] += np.bincount(joint, weights=row.early, minlength=nseg*bins).reshape(nseg, bins)
    result, stratified = [], []
    for row in rows:
        li = layer_id[row.layer]
        # Only other trajectories supply labels for these learned priorities.
        train_hits = hits[:, li].sum(axis=0) - hits[row.group, li]
        train_exp = exposure[:, li].sum(axis=0) - exposure[row.group, li]
        segment_rate = base_rates(train_hits.sum(axis=1), train_exp.sum(axis=1))
        history_rate = base_rates(train_hits.sum(axis=0), train_exp.sum(axis=0))
        joint_rate = (train_hits + 256 * history_rate) / (train_exp + 256)
        hist_bin = np.searchsorted(HISTORY_EDGES, row.previous, side="right")
        pos = np.arange(row.segment.size)
        priorities = {
            "recency": pos, "previous_count": row.previous, "prior_turn_mean": row.prior_mean,
            "segment_age": segment_rate[row.segment], "history_calibrated": history_rate[hist_bin],
            "history_segment_age": joint_rate[row.segment, hist_bin],
        }
        for semantic in np.unique(row.segment // 5):
            mask = row.segment // 5 == semantic
            for past_used in (False, True):
                sel = mask & ((row.previous > 0) == past_used)
                if not sel.any():
                    continue
                stratified.append(dict(
                    trajectory=row.trajectory, category=row.category, round=row.round,
                    layer=row.layer, segment=types[semantic], previous_used=past_used,
                    tokens=int(sel.sum()), future_active=int((row.early[sel] > 0).sum()),
                    next_4_hits=int(row.early[sel].sum()), next_32_hits=int(row.whole[sel].sum()),
                ))
        for pool, eligible in (("common_prefix", pos), ("previously_consumed", pos[row.previous > 0])):
            eligible_early = int(row.early[eligible].sum())
            eligible_all = int(row.whole[eligible].sum())
            for name, values in priorities.items():
                order = eligible[np.lexsort((-eligible, -values[eligible]))]
                for budget in budgets:
                    selected = order[:budget]
                    result.append(dict(
                        trajectory=row.trajectory, category=row.category, round=row.round,
                        layer=row.layer, pool=pool, method=name, budget=budget, retained=len(selected),
                        early_hits=int(row.early[selected].sum()), early_eligible=eligible_early,
                        full_hits=int(row.whole[selected].sum()), full_eligible=eligible_all,
                        early_all=row.all_early, full_all=row.all_whole,
                    ))
    return result, stratified


def write_csv(path, rows):
    with path.open("x") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 1, 12, 24, 36, 47])
    parser.add_argument("--budgets", nargs="+", type=int, default=[512, 1024, 2048])
    args = parser.parse_args()
    layers = sorted(set(args.layers))
    budgets = sorted(set(args.budgets))
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, prefix, types, ntraj, nturns = collect(args.source, layers)
    ranks, strata = rankings(rows, types, layers, ntraj, budgets)
    write_csv(args.output / "prefix.csv", prefix)
    write_csv(args.output / "retention.csv", ranks)
    write_csv(args.output / "segment-reuse.csv", strata)
    sources = ["prepared_requests.jsonl", "segments.jsonl", "events/manifest.jsonl"]
    meta = dict(
        scope=COHORT, source=str(args.source), layers=layers, budgets=budgets,
        trajectories=ntraj, turns=nturns, transitions=len(prefix), transition_layers=len(rows),
        actual_trace_steps=STEPS, early_label_steps=EARLY, split="leave one trajectory out",
        metadata_sha256={p: hashlib.sha256((args.source / p).read_bytes()).hexdigest() for p in sources},
    )
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
