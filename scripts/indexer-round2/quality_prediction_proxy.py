"""Measure selection-only alternatives on saved exact-trajectory evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "keye_trace"))
from analyze_horizonkv_predictive import load_chunk


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dataset_proxy(aggregates):
    values = defaultdict(list)
    for row in csv.DictReader(aggregates.open()):
        width = int(row["candidate_k"])
        if width not in {2048, 3072, 4096, 8192}:
            continue
        group = (row["dataset"], row["length_bucket"], row["split"], width)
        values[group].append((float(row["rerank_recall"]), float(row["missing_native_tokens"])))
    return [
        dict(dataset=dataset, length_bucket=bucket, split=split, width=width, requests=len(results),
             recall_mean=float(np.mean(array[:, 0])), recall_request_p05=float(np.quantile(array[:, 0], .05)),
             recall_request_min=float(array[:, 0].min()),
             missing_native_tokens_mean=float(np.mean(array[:, 1])))
        for (dataset, bucket, split, width), results in sorted(values.items())
        for array in [np.asarray(results)]
    ]


def protected_selections(candidates, scores, topk, width, prefix):
    remainder = scores[:, prefix:width]
    count = topk - prefix
    # Native tie handling can legally differ. Preserve candidate order at equal scores.
    best = np.argsort(-remainder, axis=1, kind="stable")[:, :count] + prefix
    kept = np.take_along_axis(candidates, best, axis=1)
    return np.concatenate((candidates[:, :prefix], kept), axis=1)


def trace_proxy(events, requests):
    selected = {row["rid"]: row for row in requests}
    observed = defaultdict(int)
    values = defaultdict(list)
    newest_native = defaultdict(list)
    newest_selected = defaultdict(list)
    for line in (events / "manifest.jsonl").open():
        record = json.loads(line)
        request = record["request_id"]
        if request not in selected:
            continue
        chunk = load_chunk(events / record["file"])
        records = chunk["rows"]
        topk = chunk["topk"]
        native = np.stack([row["exact_indices"].numpy() for row in records])
        candidates = np.stack([row["candidates"].numpy() for row in records])
        scores = np.stack([row["candidate_scores"].numpy() for row in records])
        truth = np.zeros((len(records), max(row["context_len"] for row in records)), dtype=np.bool_)
        batch_indices = np.arange(len(records))[:, None]
        truth[batch_indices, native] = True
        last = np.array([row["context_len"] - 1 for row in records])
        native_contains_last = truth[batch_indices[:, 0], last]
        anchors = (np.arange(len(records)) // 4) * 4
        alternatives = {
            ("predicted_top2048", 2048, 2048): candidates[:, :topk],
            ("skip4_anchor", 2048, 0): native[anchors],
        }
        for width in (3072, 4096):
            for prefix in (0, 1024, 1536):
                alternatives[("protected_prefix" if prefix else "full_rerank", width, prefix)] = (
                    protected_selections(candidates, scores, topk, width, prefix)
                )
        for (mode, width, prefix), indices in alternatives.items():
            recall = ((indices >= 0) & truth[batch_indices, np.maximum(indices, 0)]).sum(1) / topk
            selected_contains_last = (indices == last[:, None]).any(1)
            include = np.ones(len(records), dtype=np.bool_)
            if mode == "skip4_anchor":
                include = np.arange(recall.size) % 4 != 0
            key = (request, mode, width, prefix)
            values[key].extend(recall[include].tolist())
            newest_native[key].extend(native_contains_last[include].tolist())
            newest_selected[key].extend(selected_contains_last[include].tolist())
        observed[request] += 1

    if observed != {row["rid"]: 47 for row in requests}:
        raise ValueError(f"expected all 47 target layers for each held-out case, got {observed}")
    return [
        dict(request_id=request, task=selected[request]["task"],
             length_bucket=selected[request]["length_bucket_upper"], mode=mode, width=width,
             protected=prefix, observations=len(results), recall_mean=float(np.mean(results)),
             recall_p05=float(np.quantile(results, .05)), recall_min=min(results),
             newest_native_rate=float(np.mean(newest_native[key])),
             newest_selected_rate=float(np.mean(newest_selected[key])))
        for key, results in sorted(values.items())
        for request, mode, width, prefix in [key]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--aggregates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    requests = [json.loads(line) for line in args.requests.read_text().splitlines()]
    write_csv(args.output / "h-prediction-all330-by-group.csv", dataset_proxy(args.aggregates))
    write_csv(args.output / "h-prediction-protected-prefix-six.csv", trace_proxy(args.events, requests))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
