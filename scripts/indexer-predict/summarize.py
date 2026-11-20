"""Export paired predictor latency, candidate coverage and logical byte counts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics


K = 2048
FIELDS = (
    "length_bucket", "batch", "layer", "step", "split", "M", "method",
    "status", "error", "exact", "baseline", "median_us", "mad_us", "same_layout_baseline",
    "same_layout_baseline_us", "all_baseline", "all_baseline_us", "speedup_same_layout",
    "speedup_all", "mean_true_recall", "mean_delta_pp", "worst_delta_pp",
    "mean_bf16_prediction_overlap", "index_key_bytes", "raw_logical_history_bytes",
    "workspace_bytes_before_selector", "known_cute_workspace_bytes", "initialize_wall_us",
    "total_valid_history_tokens", "logical_key_bytes", "logical_score_write_bytes", "any_bf16_cutoff_tie", "source_file", "source_line",
)
REQUEST_FIELDS = (
    "length_bucket", "batch", "layer", "step", "split", "M", "method", "request",
    "history_tokens", "exact_overlap_count", "observed_overlap_count", "extra_miss_count",
    "topm_bf16_overlap_count", "source_file", "source_line",
)


def rows(paths):
    for source in paths:
        with source.open() as stream:
            for line, payload in enumerate(stream, 1):
                record = json.loads(payload)
                prefix = dict(
                    length_bucket=record["bucket"], batch=record["batch"],
                    layer=record["layer"], step=record["step"], split=record["split"],
                    M=record["width"], source_file=str(source), source_line=line,
                )
                timings = record["timings"]
                same = record["fastest_equal_layout_bf16"]
                all_layout = record["fastest_bf16"]
                equal_time = timings[same]["median_us"]
                full_time = timings[all_layout]["median_us"]
                gold = record["exact_candidate_intersection"]
                total_tokens = sum(record["history_lengths"])
                for name, method in record["methods"].items():
                    header = dict(
                        prefix, method=name, same_layout_baseline=same,
                        same_layout_baseline_us=equal_time, all_baseline=all_layout,
                        all_baseline_us=full_time, total_valid_history_tokens=total_tokens,
                        raw_logical_history_bytes=128 * total_tokens,
                        any_bf16_cutoff_tie=any(record["exact_topm_ties"]),
                    )
                    if "failed" in method:
                        yield dict(header, status="failed", error=method["failed"]), []
                        continue
                    observed = method["true_top2048_intersection"]
                    matched = method["predicted_set_intersection"]
                    deltas = [100 * (a - b) / K for a, b in zip(gold, observed, strict=True)]
                    if method["exact"] and (any(deltas) or min(matched) != record["width"]):
                        raise ValueError(f"failed exact prediction {prefix} {name}")
                    time = timings[name]["median_us"]
                    points = dict(
                        header, status="valid", exact=method["exact"], baseline=method.get("baseline", method["exact"]),
                        median_us=time, mad_us=timings[name]["mad_us"],
                        speedup_same_layout=equal_time/time, speedup_all=full_time/time,
                        mean_true_recall=statistics.mean(observed)/K,
                        mean_delta_pp=statistics.mean(deltas), worst_delta_pp=max(deltas),
                        mean_bf16_prediction_overlap=statistics.mean(matched)/record["width"],
                        **{key: method.get(key) for key in (
                            "index_key_bytes", "workspace_bytes_before_selector",
                            "known_cute_workspace_bytes", "initialize_wall_us",
                            "logical_key_bytes", "logical_score_write_bytes",
                        )},
                    )
                    requests = [
                        dict(prefix, method=name, request=rid, history_tokens=length,
                             exact_overlap_count=a, observed_overlap_count=b, extra_miss_count=a-b,
                             topm_bf16_overlap_count=common)
                        for rid, length, a, b, common in zip(
                            record["request_ids"], record["history_lengths"], gold,
                            observed, matched, strict=True,
                        )
                    ]
                    yield points, requests


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    methods = defaultdict(list)
    with (args.output / "cells.csv").open("x") as cell_file, (args.output / "requests.csv").open("x") as req_file:
        cell_writer = csv.DictWriter(cell_file, FIELDS)
        req_writer = csv.DictWriter(req_file, REQUEST_FIELDS)
        cell_writer.writeheader()
        req_writer.writeheader()
        for row, requests in rows(args.inputs):
            cell_writer.writerow(row)
            req_writer.writerows(requests)
            if requests:
                methods[row["split"], row["M"], row["method"]].append(row)
    with (args.output / "stages.csv").open("x") as stream:
        stage_fields = ("length_bucket", "batch", "layer", "step", "split", "M",
                        "stage", "median_us", "mad_us", "source_file", "source_line")
        writer = csv.DictWriter(stream, stage_fields)
        writer.writeheader()
        for source in args.inputs:
            for line, text in enumerate(source.read_text().splitlines(), 1):
                record = json.loads(text)
                for stage, timing in record["timings"].items():
                    if stage in record["methods"]:
                        continue
                    writer.writerow(dict(
                        length_bucket=record["bucket"], batch=record["batch"], layer=record["layer"],
                        step=record["step"], split=record["split"], M=record["width"],
                        stage=stage, median_us=timing["median_us"], mad_us=timing["mad_us"],
                        source_file=str(source), source_line=line,
                    ))
    groups = [
        dict(split=split, M=m, method=name, cells=len(results),
             speedup_same_layout=statistics.geometric_mean(row["speedup_same_layout"] for row in results),
             speedup_all=statistics.geometric_mean(row["speedup_all"] for row in results),
             mean_delta_pp=statistics.mean(row["mean_delta_pp"] for row in results),
             worst_delta_pp=max(row["worst_delta_pp"] for row in results))
        for (split, m, name), results in sorted(methods.items())
    ]
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.inputs}
    report = dict(sources=provenance, groups=groups,
                  aggregate="per condition geometric-mean latency ratio; separate split and M; only validated calls")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
