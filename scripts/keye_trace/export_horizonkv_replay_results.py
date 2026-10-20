"""Audit replay artifacts and export compact, directly plottable tables."""

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

from horizonkv_trace_work import kernel_work_from_trace


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((args.source / "results").glob("*.json"))
    records = {p.stem: json.loads(p.read_text()) for p in files}
    latency, components = [], []
    summaries = []
    missing_profiles = []
    for name, value in records.items():
        assert value["output_matches"] and len(value["elapsed_us"]) == 5
        assert all(v > 0 for v in value["elapsed_us"])
        assert 0 <= value["logical_miss_tokens"] <= value["selected_tokens"]
        med = statistics.median(value["elapsed_us"])
        key = name.rsplit("-" + value["mode"], 1)[0]
        baseline = records.get(key + "-l-cached")
        base_us = statistics.median(baseline["elapsed_us"]) if baseline else None
        common = {"case": name, "window": value["window"], "wave": value["wave"],
                  "layer": value["layer"], "snapshot": value["snapshot"], "mode": value["mode"],
                  "experiment_cache_tokens_per_request": value["capacity"],
                  "capacity_per_request_tokens": value.get("cache_tokens_per_request"),
                  "b_over_k": value.get("cache_per_request_topk_multiple"),
                  "B_over_k": value.get("cache_layer_total_topk_multiple"),
                  "concurrency": len(value["request_ids"]), "request_count": len(value["request_ids"]),
                  "input_min_tokens": min(value["lengths"]), "input_max_tokens": max(value["lengths"]),
                  "sampled_decode_step_min": min(value["decode_steps"]), "sampled_decode_step_max": max(value["decode_steps"])}
        row = dict(common, median_us=med, mean_us=statistics.mean(value["elapsed_us"]),
                   min_us=min(value["elapsed_us"]), max_us=max(value["elapsed_us"]),
                   probe_elapsed_us=value["probe_elapsed_us"], probe_overhead_ratio=value["probe_elapsed_us"] / med,
                   delta_vs_lru_us=base_us - med if base_us else None,
                   speedup_vs_lru=base_us / med if base_us else None,
                   selected_tokens=value["selected_tokens"], logical_miss_tokens=value["logical_miss_tokens"],
                   hit_ratio=1-value["logical_miss_tokens"]/value["selected_tokens"] if value["mode"] not in {"h-native","h-staged","d-staged"} else None,
                   logical_miss_bytes=value["logical_miss_bytes"],
                   actual_host_read_payload_bytes=value["actual_host_read_payload_bytes"],
                   actual_host_write_payload_bytes=value["actual_host_write_payload_bytes"],
                   prefetched_tokens=value["prefetched_tokens"], useful_prefetched_tokens=value["useful_prefetched_tokens"],
                   gpu_index_k_bytes=value.get("gpu_index_k_bytes"),
                   gpu_persistent_main_kv_bytes=value.get("gpu_persistent_main_kv_bytes"),
                   benchmark_scratch_bytes=value.get("benchmark_stage_and_prefetch_scratch_bytes"),
                   reset_snapshot_bytes=value.get("measurement_reset_snapshot_bytes"))
        for i, t in enumerate(value["elapsed_us"]):
            row[f"repeat_{i+1}_us"] = t
        latency.append(row)
        trace = args.source / "timelines" / f"{name}.json"
        work = kernel_work_from_trace(trace)
        if not work["gpu_work_total_us"]:
            missing_profiles.append(name)
        span_by_key = {f"layer{s['layer']}/{s['component']}": s for s in value["probe_component_spans"]}
        for component in sorted(set(work["component_gpu_work_us"]) | set(span_by_key)):
            span = span_by_key.get(component, {})
            parsed = re.match(r"layer(\d+)/(.*)", component)
            components.append(dict(common, component=component,
                                   component_layer=int(parsed[1]) if parsed else None,
                                   component_name=parsed[2] if parsed else component,
                                   actual_gpu_work_us=work["component_gpu_work_us"].get(component, 0) if work["gpu_work_total_us"] else None,
                                   gpu_activity_count=work["activity_counts"].get(component, 0),
                                   probe_stream_span_us=span.get("stream_span_us"),
                                   probe_start_us=span.get("start_us"), probe_end_us=span.get("end_us")))
        if value["window"] == "two_layers" and value["snapshot"] == "after32" and value["mode"] == "ahead":
            q = records[key + "-query-ready"]
            noio = records[key + "-no-io"]
            echo = records[key + "-echo"]
            q_us = statistics.median(q["elapsed_us"])
            summaries.append(dict(common, lru_us=base_us, ahead_us=med,
                                  query_perfect_us=q_us, noio_us=statistics.median(noio["elapsed_us"]),
                                  echo_style_us=statistics.median(echo["elapsed_us"]),
                                  ahead_reduction_vs_lru_pct=100*(base_us-med)/base_us,
                                  additional_layer_window_reduction_pct=100*(q_us-med)/q_us,
                                  baseline_hit_ratio=1-baseline["logical_miss_tokens"]/baseline["selected_tokens"]))
    single = [r for r in latency if r["window"] == "single_layer"]
    pairs = [r for r in latency if r["window"] == "two_layers"]
    write_csv(out / "single-layer-latency.csv", single)
    write_csv(out / "two-layer-latency.csv", pairs)
    write_csv(out / "component-work-and-spans.csv", components)
    if summaries:
        write_csv(out / "after32-oracle-summary.csv", summaries)
    audit = {"cases": len(records), "single_layer_cases": len(single), "two_layer_cases": len(pairs),
             "requests": len({r for v in records.values() for r in v["request_ids"]}),
             "source": str(args.source.resolve()), "environment": json.loads((args.source / "measurement-environment.json").read_text()),
             "missing_gpu_activity_profiles": missing_profiles,
             "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
             "component_rule": "GPU activity durations correlated to CPU launches; no record-function device totals",
             "sampling_rule": "first C4 and one after32 snapshot; five repeats are timing repeats, not independent requests",
             "oracle_rule": "pair elapsed includes both real layers; prediction cost omitted; negative gains retained",
             "paper_ready": False,
             "limits": ["mechanism ports, not full HiSparse/ECHO", "SM80 normalization uses native fallback", "default untuned MoE configuration", "4 captured layers, selected target layers12/24", "32K C4 pilot; other lengths and concurrency not measured"]}
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print({k:v for k,v in audit.items() if k not in {"source_sha256", "environment"}})
    for multiple in (1,2,4):
        selected = [r for r in summaries if r["b_over_k"] == multiple]
        if selected:
            print('b/k',multiple,'points',len(selected),'median_hit',statistics.median(r['baseline_hit_ratio'] for r in selected),
                  'median_ahead_reduction_pct',statistics.median(r['ahead_reduction_vs_lru_pct'] for r in selected),
                  'median_extra_window_reduction_pct',statistics.median(r['additional_layer_window_reduction_pct'] for r in selected))


if __name__ == "__main__":
    main()
