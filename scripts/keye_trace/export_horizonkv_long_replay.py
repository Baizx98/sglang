"""Export every long-context measurement, dependency span and paired oracle."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from horizonkv_trace_work import kernel_work_from_trace


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def export(root):
    root = Path(root)
    source, out = root / "long-replay-v01", root / "processed-long-replay-v01"
    out.mkdir(exist_ok=True)
    files = sorted((source / "results").glob("*.json"))
    values = {f.stem: json.loads(f.read_text()) for f in files}
    rows, components = [], []
    query = ("indexer_query_prepare", "indexer_score", "indexer_topk")
    pipeline = query + ("kv_writeback", "kv_resolve_gather")
    for name, d in values.items():
        assert d["usable_comparison"] or d["mode"] == "native-unsorted"
        assert len(d["elapsed_us"]) == 5 and min(d["elapsed_us"]) > 0
        trace = kernel_work_from_trace(source / "timelines" / f"{name}.json")
        assert trace["gpu_work_total_us"] > 0, name
        by_component = {k.split("/",1)[-1]:v for k,v in trace["component_gpu_work_us"].items()}
        spans = {s["component"]:s for s in d["probe_component_spans"]}
        med = statistics.median(d["elapsed_us"])
        prefix = name.removesuffix(d["mode"])
        native = values[prefix + "native"]
        baseline = values.get(prefix + "lru-b" + str(d["capacity"]), native)
        if d["mode"].startswith(("native", "h-staged", "dram")):
            baseline = native
        base_us = statistics.median(baseline["elapsed_us"])
        qwork = sum(by_component.get(k,0) for k in query)
        pwork = sum(by_component.get(k,0) for k in pipeline)
        row = dict(case=name, group=d["group"], wave=d["wave"], layer=d["layer"], concurrency=d["concurrency"],
                   snapshot=d["snapshot"], mode=d["mode"], usable=d["usable_comparison"],
                   input_min_tokens=min(d["lengths"]), input_max_tokens=max(d["lengths"]),
                   decode_step_min=min(d["decode_steps"]), decode_step_max=max(d["decode_steps"]),
                   physical_cache_tokens_per_request=d["cache_tokens_per_request"],
                   b_over_k=d["cache_per_request_topk_multiple"], B_over_k=d["cache_layer_total_topk_multiple"],
                   elapsed_median_us=med, elapsed_min_us=min(d["elapsed_us"]), elapsed_max_us=max(d["elapsed_us"]),
                   baseline_mode=baseline["mode"], delta_vs_baseline_us=base_us-med,
                   reduction_vs_baseline_pct=100*(base_us-med)/base_us,
                   probe_elapsed_us=d["probe_elapsed_us"], probe_overhead_ratio=d["probe_elapsed_us"]/med,
                   gpu_work_total_us=trace["gpu_work_total_us"], indexer_gpu_work_us=qwork,
                   indexer_gpu_work_ratio=qwork/trace["gpu_work_total_us"],
                   demand_pipeline_gpu_work_us=pwork, demand_pipeline_gpu_work_ratio=pwork/trace["gpu_work_total_us"],
                   pre_attention_probe_span_us=spans["attention"]["start_us"]-spans["indexer_query_prepare"]["start_us"],
                   attention_probe_start_us=spans["attention"]["start_us"], moe_probe_start_us=spans["moe_ffn"]["start_us"],
                   selected_tokens=d["selected_tokens"], logical_miss_tokens=d["logical_miss_tokens"],
                   logical_miss_bytes=d["logical_miss_bytes"], host_read_payload_bytes=d["actual_host_read_payload_bytes"],
                   host_write_payload_bytes=d["actual_host_write_payload_bytes"],
                   persistent_main_kv_bytes=d["gpu_persistent_main_kv_bytes"], index_k_bytes=d["gpu_index_k_bytes"],
                   layer_parameter_bytes=d["layer_parameter_bytes"], gpu_cache_metadata_bytes=d["gpu_cache_metadata_bytes"],
                   benchmark_scratch_bytes=d["benchmark_stage_and_prefetch_scratch_bytes"],
                   reset_snapshot_bytes=d["measurement_reset_snapshot_bytes"],
                   unique_experts=d["routing"]["unique_experts"],
                   unique_expert_weight_bytes=d["routing"]["unique_expert_weight_bytes"],
                   max_abs_output_error=max(d["output_max_abs_errors"]),
                   output_bitwise_equal=d["output_matches"], selected_set_equal=d["selected_set_matches"])
        row.update({f"repeat_{i+1}_us":v for i,v in enumerate(d["elapsed_us"])})
        rows.append(row)
        for key in sorted(set(by_component) | set(spans)):
            s = spans.get(key,{})
            components.append(dict(case=name, group=d["group"], layer=d["layer"], concurrency=d["concurrency"],
                                   mode=d["mode"], usable=d["usable_comparison"], component=key,
                                   actual_gpu_work_us=by_component.get(key,0),
                                   probe_span_us=s.get("stream_span_us"),
                                   probe_start_us=s.get("start_us"), probe_end_us=s.get("end_us")))
    write_csv(out / "latency-and-dependencies.csv", rows)
    write_csv(out / "component-work.csv", components)
    audit = dict(cases=len(rows), component_rows=len(components),
                 usable_cases=sum(bool(d["usable_comparison"]) for d in values.values()),
                 unusable_cases=[name for name,d in values.items() if not d["usable_comparison"]],
                 environment=json.loads((source/"environment.json").read_text()),
                 case_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                 rules="All predeclared lengths and modes retained; graph timing vs GPU work vs probe spans separate; oracles and frozen residency are attribution controls.",
                 paper_ready=False)
    save_path=out / "audit.json"
    save_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
    print({k:v for k,v in audit.items() if k not in {"environment","case_sha256"}})
    return audit


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    export(parser.parse_args().run_dir)
