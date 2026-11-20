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
    "panel", "length_bucket", "batch", "layer", "step", "split", "M", "method",
    "status", "error", "exact", "baseline", "median_us", "mad_us", "same_layout_baseline",
    "same_layout_baseline_us", "all_baseline", "all_baseline_us", "speedup_same_layout",
    "speedup_all", "mean_true_recall", "mean_delta_pp", "worst_delta_pp",
    "mean_bf16_prediction_overlap", "index_key_bytes", "raw_logical_history_bytes",
    "workspace_bytes_before_selector", "known_cute_workspace_bytes", "initialize_wall_us",
    "total_valid_history_tokens", "logical_key_bytes", "logical_score_write_bytes", "any_bf16_cutoff_tie", "source_file", "source_line",
)
REQUEST_FIELDS = (
    "panel", "length_bucket", "batch", "layer", "step", "split", "M", "method", "request",
    "history_tokens", "exact_overlap_count", "observed_overlap_count", "extra_miss_count",
    "topm_bf16_overlap_count", "source_file", "source_line",
)


def rows(paths):
    for source in paths:
        with source.open() as stream:
            for line, payload in enumerate(stream, 1):
                record = json.loads(payload)
                prefix = dict(
                    panel=source.parent.name, length_bucket=record["bucket"], batch=record["batch"],
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
        cell_writer = csv.DictWriter(cell_file, FIELDS, lineterminator="\n")
        req_writer = csv.DictWriter(req_file, REQUEST_FIELDS, lineterminator="\n")
        cell_writer.writeheader()
        req_writer.writeheader()
        for row, requests in rows(args.inputs):
            cell_writer.writerow(row)
            req_writer.writerows(requests)
            if requests:
                methods[row["panel"], row["split"], row["M"], row["method"]].append(row)
    with (args.output / "stages.csv").open("x") as stream:
        stage_fields = ("panel", "length_bucket", "batch", "layer", "step", "split", "M",
                        "stage", "median_us", "mad_us", "source_file", "source_line")
        writer = csv.DictWriter(stream, stage_fields, lineterminator="\n")
        writer.writeheader()
        for source in args.inputs:
            for line, text in enumerate(source.read_text().splitlines(), 1):
                record = json.loads(text)
                for stage, timing in record["timings"].items():
                    if stage in record["methods"]:
                        continue
                    writer.writerow(dict(
                        panel=source.parent.name, length_bucket=record["bucket"], batch=record["batch"], layer=record["layer"],
                        step=record["step"], split=record["split"], M=record["width"],
                        stage=stage, median_us=timing["median_us"], mad_us=timing["mad_us"],
                        source_file=str(source), source_line=line,
                    ))
    with (args.output / "segments.csv").open("x") as stream:
        segment_fields = (
            "panel", "length_bucket", "batch", "layer", "step", "split", "M",
            "kind", "bound", "segment", "budget_fraction", "mean_skipped_token_fraction",
            "mean_retained_token_fraction", "minimum_slack", "mean_true_recall",
            "mean_extra_miss_pp", "source_file", "source_line",
        )
        writer = csv.DictWriter(stream, segment_fields, lineterminator="\n")
        writer.writeheader()
        for source in args.inputs:
            with source.open() as results:
                for line, text in enumerate(results, 1):
                    record = json.loads(text)
                    for outcome in record.get("segment_audit", []):
                        entry = dict(
                            panel=source.parent.name, length_bucket=record["bucket"],
                            batch=record["batch"], layer=record["layer"], step=record["step"],
                            split=record["split"], M=record["width"], kind=outcome["kind"],
                            segment=outcome["segment"], source_file=str(source), source_line=line,
                        )
                        if outcome["kind"] == "bound_oracle_threshold":
                            entry.update(
                                bound=outcome["method"], minimum_slack=outcome["minimum_slack"],
                                mean_skipped_token_fraction=statistics.mean(outcome["skipped_token_fraction"]),
                            )
                        else:
                            overlap = statistics.mean(outcome["true_top2048_intersection"])
                            reference = statistics.mean(record["exact_candidate_intersection"])
                            entry.update(
                                budget_fraction=outcome["fraction"],
                                mean_retained_token_fraction=statistics.mean(outcome["retained_token_fraction"]),
                                mean_true_recall=overlap / K, mean_extra_miss_pp=(reference - overlap) * 100 / K,
                            )
                        writer.writerow(entry)
    groups = [
        dict(panel=panel, split=split, M=m, method=name, cells=len(results),
             speedup_same_layout=statistics.geometric_mean(row["speedup_same_layout"] for row in results),
             speedup_all=statistics.geometric_mean(row["speedup_all"] for row in results),
             mean_delta_pp=statistics.mean(row["mean_delta_pp"] for row in results),
             worst_delta_pp=max(row["worst_delta_pp"] for row in results))
        for (panel, split, m, name), results in sorted(methods.items())
    ]
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.inputs}
    report = dict(sources=provenance, groups=groups,
                  aggregate="per condition geometric-mean latency ratio; separate panel, split and M; only validated calls")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
