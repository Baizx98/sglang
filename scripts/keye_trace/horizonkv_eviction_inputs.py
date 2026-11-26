"""Read provenanced Keye demand/prediction pairs and examine near-step reuse."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from analyze_horizonkv_predictive import load_chunk


K = 2048
CANDIDATE_WIDTH = 3072
HISTORY_NAMES = ("outside_recent_window", "recent_except_previous", "previous_step")


def jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_sequence(path, request, layer, steps):
    chunk = load_chunk(path)
    if (
        chunk["schema_version"] != 2
        or not chunk["exact_main"]
        or chunk["prediction_uses_current_key"]
        or chunk["request_id"] != request["rid"]
        or chunk["target_layer_id"] != layer
    ):
        raise ValueError(f"Not a same-state native/prediction pair: {path}")
    rows = chunk["rows"]
    if [row["step"] for row in rows] != list(range(steps)):
        raise ValueError(f"Incomplete decode sequence: {path}")

    demands, predictions, current, used_current = [], [], [], []
    for row in rows:
        position = request["prompt_tokens"] + row["step"]
        if row["context_len"] != position + 1:
            raise ValueError(f"Context/token position mismatch: {path}, step={row['step']}")
        demand = np.asarray(row["exact_indices"], dtype=np.int32)
        prediction = np.asarray(row["candidates"][:CANDIDATE_WIDTH], dtype=np.int32)
        prediction = prediction[prediction >= 0]
        if (
            len(demand) != K
            or len(prediction) != min(CANDIDATE_WIDTH, position + 1)
            or prediction[0] != position
            or np.unique(demand).size != len(demand)
            or np.unique(prediction).size != len(prediction)
            or demand.min() < 0
            or demand.max() > position
            or prediction.max() > position
        ):
            raise ValueError(f"Invalid logical positions: {path}, step={row['step']}")
        # The just-produced KV is locally available, unlike historical H2D
        # transfers. Generated positions may enter the demand on later steps.
        demands.append(demand[demand < position])
        predictions.append(prediction[1:])
        current.append(position)
        used_current.append(bool(np.any(demand == position)))

    return {
        "rid": request["rid"],
        "dataset": request["dataset"],
        "split": request.get("split"),
        "task": request.get("task"),
        "length_bucket": request["length_bucket_upper"],
        "prompt_tokens": request["prompt_tokens"],
        "layer": layer,
        "steps": list(range(steps)),
        "demands": demands,
        "predictions": predictions,
        "current": np.asarray(current, dtype=np.int32),
        "used_current": used_current,
        "source": str(path),
        "source_sha256": file_sha256(path),
    }


def load_sequences(root, layers, split, max_per_cell, seed, group_by="dataset"):
    """Return sequences grouped by (chosen request metadata field, length, layer).

    group_by="length" pools datasets within each length bucket under "all".
    A request is sampled once per group/length to keep layer samples paired.
    max_per_cell=None or 0 retains all requests in the requested split.
    """
    root = Path(root).resolve()
    config = json.loads((root / "run-config.json").read_text())
    manifest = {}
    for row in jsonl(root / "events/manifest.jsonl"):
        key = row["request_id"], row["target_layer_id"]
        if key in manifest:
            raise ValueError(f"Duplicate request/layer in manifest: {key}")
        manifest[key] = row["file"]

    cohorts = defaultdict(list)
    seen = set()
    for request in jsonl(root / "requests.jsonl"):
        if request["rid"] in seen:
            raise ValueError(f"Duplicate request metadata: {request['rid']}")
        seen.add(request["rid"])
        if split in (None, "all", request.get("split")):
            group = "all" if group_by == "length" else request[group_by]
            key = group, request["length_bucket_upper"]
            cohorts[key].append(request)
    if not cohorts or (max_per_cell is not None and max_per_cell < 0):
        raise ValueError(f"No matching requests or invalid sample cap: {split}, {max_per_cell}")

    rng = random.Random(seed)
    groups = {}
    for key, cohort in sorted(cohorts.items()):
        cohort.sort(key=lambda request: request["rid"])
        if max_per_cell and len(cohort) > max_per_cell:
            cohort = rng.sample(cohort, max_per_cell)
            cohort.sort(key=lambda request: request["rid"])
        for layer in sorted(set(layers)):
            sequences = []
            for request in cohort:
                path = root / "events" / manifest[request["rid"], layer]
                sequences.append(read_sequence(path, request, layer, config["steps"]))
            groups[(*key, layer)] = sequences
    return groups


def reuse_observations(sequence, horizons, history_steps):
    """Compare prediction membership within strata of *earlier* real access.

    For prediction at t, H=1 targets the consumption at t. Only full future
    observation windows are accumulated. Future truth is solely an outcome.
    """
    demands = sequence["demands"]
    predictions = sequence["predictions"]
    current = sequence["current"]
    steps = len(demands)
    next_use = np.full(int(current[-1]), steps, dtype=np.int32)
    for t in range(steps - 1, history_steps - 1, -1):
        position = int(current[t])
        next_use[demands[t]] = t
        recent = np.unique(np.concatenate(demands[t - history_steps:t]))
        history = np.zeros(position, dtype=np.int8)
        history[recent] = 1
        history[demands[t - 1]] = 2
        predicted_ids = predictions[t]
        predicted_history = history[predicted_ids]
        all_counts = np.bincount(history, minlength=len(HISTORY_NAMES))
        positive_counts = np.bincount(predicted_history, minlength=len(HISTORY_NAMES))
        for h in horizons:
            if t + h > steps:
                continue
            hit = next_use[:position] < t + h
            all_hits = np.bincount(history[hit], minlength=len(HISTORY_NAMES))
            pos_hits = np.bincount(predicted_history[hit[predicted_ids]], minlength=len(HISTORY_NAMES))
            for i, stratum in enumerate(HISTORY_NAMES):
                yield h, stratum, True, int(pos_hits[i]), int(positive_counts[i])
                yield h, stratum, False, int(all_hits[i] - pos_hits[i]), int(all_counts[i] - positive_counts[i])


def summarize(groups, horizons=(1, 4, 16), history_steps=8):
    overlap, future = [], []
    for (group, length, layer), sequences in groups.items():
        common = dict(group=group, length_bucket=length, layer=layer)
        overlap_counts = defaultdict(lambda: [0, 0, 0])
        future_counts = defaultdict(lambda: [0, 0, 0])
        for sequence in sequences:
            demands = sequence["demands"]
            for lag in horizons:
                counts = overlap_counts[lag]
                for i in range(lag, len(demands)):
                    counts[0] += np.intersect1d(demands[i - lag], demands[i], assume_unique=True).size
                    counts[1] += len(demands[i])
                    counts[2] += 1
            for h, stratum, predicted, hits, population in reuse_observations(sequence, horizons, history_steps):
                counts = future_counts[h, stratum, predicted]
                counts[0] += hits
                counts[1] += population
                counts[2] += 1

        for lag, (hits, demand, observed) in sorted(overlap_counts.items()):
            overlap.append(dict(common, lag=lag, requests=len(sequences), steps=observed,
                                intersection_tokens=hits, target_tokens=demand,
                                overlap=hits / demand if demand else None))
        for (horizon, history, predicted), (hits, population, observed) in sorted(future_counts.items()):
            future.append(dict(common, horizon=horizon, history=history, predicted=predicted,
                               requests=len(sequences), steps=observed, future_hits=hits,
                               historical_positions=population,
                               probability=hits / population if population else None))
    return overlap, future


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No observations for {path.name}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=(1, 12, 24, 47))
    parser.add_argument("--split", choices=("calibration", "test", "all"), default="test")
    parser.add_argument("--max-per-cell", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--group-by", choices=("dataset", "group", "task", "length"), default="dataset")
    args = parser.parse_args()
    groups = load_sequences(args.source, args.layers, args.split, args.max_per_cell, args.seed, args.group_by)
    overlap, future = summarize(groups)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "lagged-overlap.csv", overlap)
    write_csv(args.output / "conditional-reuse.csv", future)
    with (args.output / "inputs.json").open("x") as stream:
        json.dump({
            "config": {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "candidate_width_including_current": CANDIDATE_WIDTH,
            "history_steps": 8,
            "source_indices_are_logical": True,
            "inputs": [
                {k: s[k] for k in ("rid", "split", "dataset", "length_bucket", "layer", "source", "source_sha256")}
                for sequences in groups.values()
                for s in sequences
            ],
        }, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
