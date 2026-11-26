"""Join terminal BFCL paired traces to causally available previous-turn priors."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import random

import numpy as np
import torch

from horizonkv_eviction_inputs import CANDIDATE_WIDTH, K, file_sha256
from horizonkv_turn_reuse import STEPS, base_rates, collect, read_jsonl


DATASET = "BFCL-agent-terminal"


def terminal_inventory(root, old_root):
    selection = json.loads((root / "selection.json").read_text())
    if selection["source_prepared_sha256"] != file_sha256(old_root / "prepared_requests.jsonl"):
        raise ValueError("paired cohort refers to different prepared prompts")
    snapshots = [json.loads((p / "server_snapshot.json").read_text())["model_info"]
                 for p in (root, old_root)]
    identifiers = ("model_path", "tokenizer_path", "weight_version")
    if any(snapshots[0][name] != snapshots[1][name] for name in identifiers):
        raise ValueError("mismatched model or tokenizer across the paired corpora")
    old = defaultdict(dict)
    for row in read_jsonl(old_root / "prepared_requests.jsonl"):
        group = old[row["trajectory_id"]]
        if row["round_id"] in group:
            raise ValueError("duplicate old trajectory/turn")
        group[row["round_id"]] = row
    served = {
        row["prepared_rid"]: row
        for row in read_jsonl(root / "requests.jsonl")
    }
    selected = {}
    for row in read_jsonl(root / "prepared_requests.jsonl"):
        tid, turn, rid = row["trajectory_id"], row["round_id"], row["rid"]
        if tid in selected or tid not in old or turn != max(old[tid]):
            raise ValueError(f"paired cohort lacks a unique terminal prompt on {tid}")
        if (turn <= 0 or row["split"] not in ("calibration", "test")
                or any(row[k] != old[tid][turn][k] for k in ("input_ids", "prompt_sha256", "rid"))
                or any(row[k] != served[rid][k] for k in ("trajectory_id", "round_id", "prompt_sha256"))):
            raise ValueError(f"unaligned terminal prompt on {tid}")
        selected[tid] = row, served[rid]["rid"]
    if set(selected) != set(old):
        raise ValueError("paired terminal and previous-turn cohorts differ")
    return selected


@lru_cache(maxsize=2)
def terminal_priors(root, old_root, layers):
    selected = terminal_inventory(root, old_root)
    model_records, _, types, _, _ = collect(old_root, layers)
    category_bins = 5 * len(types)
    calibrations = defaultdict(lambda: (np.zeros(category_bins), np.zeros(category_bins)))
    terminals = {}
    for row in model_records:
        terminal, _ = selected[row.trajectory]
        if row.round == terminal["round_id"]:
            terminals[row.trajectory, row.layer] = row
        if terminal["split"] == "calibration":
            hits, population = calibrations[row.trajectory, row.layer]
            hits += np.bincount(row.segment, weights=row.whole, minlength=category_bins)
            population += np.bincount(row.segment, minlength=category_bins)
    priors = {}
    for key, entry in terminals.items():
        traj, layer = key
        terminal, _ = selected[traj]
        hits = np.zeros(category_bins)
        counts = np.zeros(category_bins)
        for (other, level), (h, n) in calibrations.items():
            if level == layer and other != traj:
                hits += h
                counts += n
        if not counts.sum():
            raise ValueError("no calibration support for segment-age prior")
        prefix = entry.segment.size
        length = terminal["prompt_len"] + STEPS
        turn_prior = np.zeros(length, dtype=np.float32)
        role_prior = np.zeros(length, dtype=np.float32)
        turn_prior[:prefix] = entry.prior_mean / STEPS
        role_prior[:prefix] = base_rates(hits, counts)[entry.segment] / STEPS
        priors[key] = turn_prior, role_prior, prefix
    if set(priors) != {(tid, layer) for tid in selected for layer in layers}:
        raise ValueError("missing earlier turn state in paired cohort")
    return selected, priors


@lru_cache(maxsize=192)
def read_terminal(path, actual_rid, layer, prompt_len):
    trace = torch.load(path, map_location="cpu", weights_only=True)
    current = prompt_len + np.arange(STEPS, dtype=np.int32)
    if (trace["schema_version"] != 1 or trace["request_id"] != actual_rid
            or trace["source_layer_id"] != layer - 1 or trace["target_layer_id"] != layer
            or not trace["self_token_forced"] or not trace["canonical_historical_k_cache"]
            or trace["decode_step_ids"] != list(range(STEPS))
            or not np.array_equal(trace["valid_counts"].numpy(), current + 1)):
        raise ValueError(f"not a canonical true/prediction pair: {path}")
    exact = trace["exact_indices"].numpy()
    lookahead = trace["lookahead_indices"].numpy()[:, :CANDIDATE_WIDTH]
    if exact.shape != (STEPS, K) or lookahead.shape != (STEPS, CANDIDATE_WIDTH):
        raise ValueError(f"wrong exact or prediction width: {path}")
    demands, predictions, used = [], [], []
    for pos, d, p in zip(current, exact, lookahead, strict=True):
        if (pos != p[0] or min(d.min(), p.min()) < 0 or max(d.max(), p.max()) > pos
                or len(np.unique(d)) != len(d) or len(np.unique(p)) != len(p)):
            raise ValueError(f"invalid true or predicted positions: {path}")
        demands.append(d[d < pos].astype(np.int32))
        predictions.append(p[1:].astype(np.int32))
        used.append(bool(np.any(d == pos)))
    return dict(
        steps=list(range(STEPS)), current=current, demands=demands, predictions=predictions,
        used_current=used, source=str(path), source_sha256=file_sha256(path),
    )


def load_terminal_sequences(root, old_root, layers, split, seed):
    """Return fixed-split BFCL terminal streams with forecast-independent long priors."""
    if split not in (None, "all", "calibration", "test"):
        raise ValueError(f"unknown split {split}")
    root = Path(root).resolve()
    old_root = Path(old_root).resolve()
    layers = tuple(sorted(set(layers)))
    selected, priors = terminal_priors(root, old_root, layers)
    inventory = {}
    for item in read_jsonl(root / "events/manifest.jsonl"):
        key = item["request_id"], item["target_layer_id"]
        if key in inventory:
            raise ValueError(f"duplicate paired trace descriptor: {key}")
        inventory[key] = root / "events" / item["file"]

    order = sorted(selected)
    random.Random(seed).shuffle(order)
    groups = defaultdict(list)
    for tid in order:
        row, actual_rid = selected[tid]
        if split not in (None, "all", row["split"]):
            continue
        bucket = 8192
        if row["prompt_len"] + STEPS > bucket:
            raise ValueError("paired BFCL exceeds the documented 8K bucket")
        for layer in layers:
            turn, role, prefix = priors[tid, layer]
            data = read_terminal(inventory[actual_rid, layer], actual_rid, layer, row["prompt_len"])
            groups[DATASET, bucket, layer].append(dict(
                **data, rid=actual_rid, dataset=DATASET, length_bucket=bucket, layer=layer,
                split=row["split"], task=row["category"], category=row["category"], trajectory=tid,
                prompt_tokens=row["prompt_len"], turn_prior=turn, role_prior=role, turn_known=prefix,
            ))
    return dict(groups)
