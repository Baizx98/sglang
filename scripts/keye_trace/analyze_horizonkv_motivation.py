"""Audit a completed exact-Keye pilot and export diagnostic tables (CPU only).

CUDA intervals include host idle. Exports are pilot observations, not optimized
HiSparse/ECHO measurements or a claim that PCIe is the critical path.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import subprocess
from array import array
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def mean(values):
    return statistics.mean(values) if values else None


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def export(path, records):
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def partition(model, spans, check):
    """Subtract nested parents so all reported categories sum to model_total."""
    model_ms = model["cuda_stream_span_ms"]
    result = Counter()
    by_layer = defaultdict(dict)
    for value in spans:
        by_layer[value["layer_id"]][value["component"]] = value
    check(set(by_layer) == set(range(48)), "timing_layer_completeness")
    layers_ms = 0
    for layer in by_layer.values():
        for required in (
            "layer_total",
            "self_attention_total",
            "indexer_total",
            "indexer_score",
            "indexer_topk",
            "attention_compute",
        ):
            check(required in layer, "missing_component")
        if not all(
            c in layer for c in ("layer_total", "self_attention_total", "indexer_total")
        ):
            continue
        for component, value in layer.items():
            parent_name = (
                "indexer_total"
                if component in ("indexer_score", "indexer_topk")
                else "self_attention_total"
                if component != "layer_total"
                else None
            )
            if component == "self_attention_total":
                parent_name = "layer_total"
            parent = layer[parent_name] if parent_name else model
            check(
                value["start_from_model_ms"] >= parent["start_from_model_ms"] - 0.05
                and value["end_from_model_ms"] <= parent["end_from_model_ms"] + 0.05,
                "interval_outside_parent",
            )
        duration = {key: value["cuda_stream_span_ms"] for key, value in layer.items()}
        layers_ms += duration["layer_total"]
        for component, value in duration.items():
            if component not in (
                "layer_total",
                "self_attention_total",
                "indexer_total",
            ):
                result[component] += value
        result["indexer_other"] += duration["indexer_total"] - sum(
            duration.get(c, 0) for c in ("indexer_score", "indexer_topk")
        )
        # Everything inside self-attention except the indexer and measured KV/
        # attention spans: projections, normalization, dispatch and host gaps.
        direct = sum(
            v
            for c, v in duration.items()
            if c
            not in (
                "layer_total",
                "self_attention_total",
                "indexer_score",
                "indexer_topk",
            )
        )
        result["self_attention_other"] += duration["self_attention_total"] - direct
        result["layer_other_including_moe"] += (
            duration["layer_total"] - duration["self_attention_total"]
        )
    result["model_other"] = model_ms - layers_ms
    check(
        all(value >= -0.05 for value in result.values()), "negative_exclusive_component"
    )
    check(abs(sum(result.values()) - model_ms) <= 0.1, "partition_does_not_sum")
    result["model_total"] = model_ms
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    run, out = args.run_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan, launch = load(run / "plan.json"), load(run / "launch.json")
    errors, error_examples = Counter(), []

    def check(condition, name, detail=None):
        if not condition:
            errors[name] += 1
            if len(error_examples) < 30:
                error_examples.append({"error": name, "detail": detail})

    check(load(run / "state.json")["status"] == "completed", "run_not_completed")
    request_path = Path(launch["command"][launch["command"].index("--requests") + 1])
    with request_path.open("rb") as stream:
        frozen_request_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    check(frozen_request_sha256 == plan["requests_sha256"], "frozen_request_hash")
    all_records, requests, latency = [], {}, []
    for config in plan["configs"]:
        current = list(rows(run / config / "requests.jsonl"))
        phases = (
            ["control", "timing", "selection"]
            if config.startswith("lru")
            else ["control", "timing"]
        )
        expected = {
            (config, wave, repetition, phase, case)
            for wave, cases in enumerate(plan["waves"])
            for case in cases
            for repetition in range(plan["repetitions"])
            for phase in phases
        }
        observed = Counter(
            (config, v["wave_index"], v["repetition"], v["phase"], v["case_id"])
            for v in current
        )
        check(set(observed) == expected, "request_matrix", config)
        check(
            all(n == 1 for n in observed.values()),
            "duplicate_request_execution",
            config,
        )
        check(
            load(run / config / "startup.json")["source_commit"]
            == plan["source_commit"],
            "source_commit",
            config,
        )
        for record in current:
            rid, response = record["rid"], record["response"]
            check(rid not in requests, "duplicate_rid", rid)
            requests[rid] = record
            tokens, meta = response["output_ids"], response["meta_info"]
            check(len(tokens) == meta["completion_tokens"], "output_token_count", rid)
            check(
                meta["id"] == rid and meta["prompt_tokens"] == record["prompt_tokens"],
                "response_identity",
                rid,
            )
            check(meta.get("total_retractions", 0) == 0, "retraction", rid)
            check(record["output_ids_equal_hbm"] is True, "recorded_output_gate", rid)
            arrivals = record["stream_arrivals"]
            check(
                [v["completion_tokens"] for v in arrivals]
                == list(range(1, len(tokens) + 1)),
                "stream_sequence",
                rid,
            )
            gaps = [
                1000 * (b["elapsed_s"] - a["elapsed_s"])
                for a, b in itertools.pairwise(arrivals)
            ]
            check(all(v >= 0 for v in gaps), "negative_stream_interval", rid)
            latency.append(
                {
                    "config": config,
                    "phase": record["phase"],
                    "wave_index": record["wave_index"],
                    "repetition": record["repetition"],
                    "case_id": record["case_id"],
                    "workload": record["workload"],
                    "task": record["task"],
                    "requested_concurrency": record["requested_concurrency"],
                    "prompt_tokens": record["prompt_tokens"],
                    "output_tokens": len(tokens),
                    "finish_type": meta["finish_reason"]["type"],
                    "ttft_ms": arrivals[0]["elapsed_s"] * 1000,
                    "e2e_ms": meta["e2e_latency"] * 1000,
                    "tpot_ms": mean(gaps),
                    "itl_p50_ms": quantile(gaps, 0.5),
                    "itl_p95_ms": quantile(gaps, 0.95),
                    "last16_itl_mean_ms": mean(gaps[-16:]) if len(gaps) >= 16 else None,
                    "last16_aggregate_token_rate_estimate": (
                        record["requested_concurrency"] * 1000 / mean(gaps[-16:])
                        if len(gaps) >= 16
                        else None
                    ),
                    "output_equal_hbm": True,
                }
            )
        all_records.extend(current)
    reference = {
        (v["wave_index"], v["repetition"], v["case_id"]): v["response"]["output_ids"]
        for v in all_records
        if v["config"] == "hbm" and v["phase"] == "control"
    }
    for v in all_records:
        key = (v["wave_index"], v["repetition"], v["case_id"])
        check(
            v["response"]["output_ids"] == reference[key],
            "independent_output_mismatch",
            v["rid"],
        )
    by_case = defaultdict(list)
    for value in all_records:
        if value["config"] == "hbm" and value["phase"] == "control":
            by_case[value["case_id"]].append(value)
    cross_concurrency = {}
    for case, values in by_case.items():
        reference_ids = min(values, key=lambda v: v["requested_concurrency"])[
            "response"
        ]["output_ids"]
        variants = {tuple(v["response"]["output_ids"]) for v in values}
        if len(variants) > 1:
            cross_concurrency[case] = {
                "distinct_output_sequences": len(variants),
                "first_difference_zero_based_by_concurrency": {
                    v["requested_concurrency"]: next(
                        (
                            i
                            for i, (a, b) in enumerate(
                                zip(reference_ids, v["response"]["output_ids"])
                            )
                            if a != b
                        ),
                        None,
                    )
                    for v in values
                },
            }
    paired_latency = {
        (v["config"], v["wave_index"], v["repetition"], v["case_id"], v["phase"]): v
        for v in latency
    }
    for v in latency:
        baseline = paired_latency[
            (v["config"], v["wave_index"], v["repetition"], v["case_id"], "control")
        ]
        v["last16_ratio_to_control"] = ratio(
            v["last16_itl_mean_ms"], baseline["last16_itl_mean_ms"]
        )

    timing, selection, event_audit, hashes = [], [], {}, {}
    cross_rank = {}
    for config in plan["configs"]:
        selection_digests = []
        rank0_selection = {}
        rank0_sets = {}
        rank_differences = Counter()
        rank_set_overlap = []
        for rank in (0, 1):
            path = run / config / "events" / f"probe_rank_{rank:02d}.jsonl"
            kinds, components, ignored_tail = Counter(), Counter(), Counter()
            timing_groups, selection_groups = defaultdict(list), defaultdict(list)
            model_steps, selection_steps = Counter(), Counter()
            allocation, buffer, previous_sets = {}, [], {}
            selection_hash = hashlib.sha256()
            for event in rows(path):
                kinds[event["kind"]] += 1
                if event["kind"] == "allocation":
                    allocation = event
                    continue
                if event["kind"] == "span":
                    component = event["component"]
                    components[component] += 1
                    if component != "model_total":
                        buffer.append(event)
                        continue
                    ids, steps = event["request_ids"], event["decode_steps"]
                    check(
                        all(
                            rid in requests and requests[rid]["phase"] == "timing"
                            for rid in ids
                        ),
                        "unknown_timing_request",
                    )
                    valid = [
                        0 <= step < len(requests[rid]["response"]["output_ids"]) - 1
                        for rid, step in zip(ids, steps)
                    ]
                    for rid, step, context in zip(ids, steps, event["context_tokens"]):
                        check(
                            context == requests[rid]["prompt_tokens"] + step + 1,
                            "timing_context_position",
                        )
                    if not any(valid):
                        ignored_tail["model_batches"] += 1
                        buffer.clear()
                        continue
                    check(all(valid), "mixed_valid_and_tail_batch")
                    check(
                        all(
                            v["request_ids"] == ids and v["decode_steps"] == steps
                            for v in buffer
                        ),
                        "cross_step_span",
                    )
                    check(
                        len({(v["layer_id"], v["component"]) for v in buffer})
                        == len(buffer),
                        "duplicate_component",
                    )
                    for rid, step in zip(ids, steps):
                        model_steps[(rid, step)] += 1
                    values = partition(event, buffer, check)
                    values.update(
                        {
                            k: event[k]
                            for k in (
                                "cuda_allocated_bytes",
                                "cuda_reserved_bytes",
                                "cuda_peak_allocated_bytes_since_startup",
                            )
                        }
                    )
                    values["h2d_bytes"] = sum(
                        v.get("bytes", 0) for v in buffer if v["component"] == "kv_h2d"
                    )
                    values["d2h_bytes"] = sum(
                        v.get("bytes", 0)
                        for v in buffer
                        if v["component"] == "kv_d2h_writeback"
                    )
                    values["active_batch"] = event["active_batch"]
                    record = requests[ids[0]]
                    timing_groups[(record["wave_index"], record["repetition"])].append(
                        values
                    )
                    buffer.clear()
                    continue
                check(event["kind"] == "selection", "unknown_event_kind")
                rid, step, layer = (
                    event["request_id"],
                    event["decode_step"],
                    event["layer_id"],
                )
                record = requests[rid]
                check(record["phase"] == "selection", "selection_request_phase")
                check(
                    event["context_tokens"] == record["prompt_tokens"] + step + 1,
                    "selection_context_position",
                )
                if step >= len(record["response"]["output_ids"]) - 1:
                    ignored_tail["selection_layers"] += 1
                    continue
                selection_steps[(rid, step, layer)] += 1
                check(
                    0 <= event["miss_count"] <= event["selected_count"] == 2048,
                    "selection_counts",
                )
                check(
                    event["hit_count"] + event["miss_count"] == event["selected_count"],
                    "hit_miss_sum",
                )
                check(event["miss_bytes"] == event["miss_count"] * 1024, "miss_bytes")
                if "selected" in event:
                    s, d = set(event["selected"]), set(event["miss"])
                    hint = set(event["hint"]) if event["hint"] is not None else None
                    check(
                        len(s) == event["selected_count"]
                        and len(d) == event["miss_count"]
                        and d <= s,
                        "saved_set_count",
                    )
                    check(
                        min(s) >= 0 and max(s) < event["context_tokens"],
                        "saved_set_positions",
                    )
                    if layer > 0:
                        check(
                            hint == previous_sets.get((rid, step)),
                            "saved_hint_previous_layer",
                        )
                        check(
                            len(s & hint) == event["overlap_numerator"]
                            and len(d & hint) == event["miss_coverage_numerator"],
                            "saved_set_numerators",
                        )
                    previous_sets[(rid, step)] = s
                if layer > 0:
                    check(
                        0 <= event["miss_coverage_numerator"] <= event["miss_count"],
                        "coverage_range",
                    )
                    check(
                        0 <= event["overlap_numerator"] <= event["selected_count"],
                        "overlap_range",
                    )
                selection_hash.update(
                    json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
                )
                key = (rid, step, layer)
                scalar = tuple(
                    event[k]
                    for k in (
                        "selected_count",
                        "miss_count",
                        "overlap_numerator",
                        "miss_coverage_numerator",
                        "hint_resident_count",
                    )
                )
                if rank == 0:
                    rank0_selection[key] = scalar
                    if "selected" in event:
                        rank0_sets[key] = array("i", event["selected"])
                else:
                    check(key in rank0_selection, "unpaired_rank_selection")
                    rank_differences["paired_layer_observations"] += 1
                    rank_differences["different_scalar_observations"] += (
                        scalar != rank0_selection.get(key)
                    )
                    if "selected" in event:
                        selected0 = set(rank0_sets[key])
                        selected1 = set(event["selected"])
                        rank_set_overlap.append(
                            len(selected0 & selected1) / len(selected0)
                        )
                        rank_differences["different_saved_sets"] += (
                            selected0 != selected1
                        )
                for view in ("all_valid", "exclude_first_decode"):
                    if view == "exclude_first_decode" and step == 0:
                        continue
                    selection_groups[
                        (record["wave_index"], record["repetition"], view)
                    ].append(event)
            check(not buffer, "unclosed_model_span")
            expected_models = {
                (rid, step)
                for rid, rec in requests.items()
                if rec["config"] == config and rec["phase"] == "timing"
                for step in range(len(rec["response"]["output_ids"]) - 1)
            }
            check(
                set(model_steps) == expected_models
                and all(n == 1 for n in model_steps.values()),
                "model_step_completeness",
                (config, rank),
            )
            expected_selections = {
                (rid, step, layer)
                for rid, rec in requests.items()
                if rec["config"] == config and rec["phase"] == "selection"
                for step in range(len(rec["response"]["output_ids"]) - 1)
                for layer in range(48)
            }
            check(
                set(selection_steps) == expected_selections
                and all(n == 1 for n in selection_steps.values()),
                "selection_completeness",
                (config, rank),
            )
            event_audit[f"{config}/rank{rank}"] = {
                "kinds": dict(kinds),
                "components": dict(components),
                "excluded_inflight_tail": dict(ignored_tail),
                "allocation": allocation,
                "valid_model_request_steps": len(model_steps),
                "valid_selection_layers": len(selection_steps),
            }
            selection_digests.append(selection_hash.hexdigest())
            for (wave, repetition), values in sorted(timing_groups.items()):
                record = requests[
                    next(
                        rid
                        for rid, rec in requests.items()
                        if rec["config"] == config
                        and rec["phase"] == "timing"
                        and rec["wave_index"] == wave
                        and rec["repetition"] == repetition
                    )
                ]
                row = {
                    "config": config,
                    "rank": rank,
                    "wave_index": wave,
                    "repetition": repetition,
                    "workload": record["workload"],
                    "task": record["task"],
                    "requested_concurrency": record["requested_concurrency"],
                    "valid_model_batches": len(values),
                    "actual_batch_min": min(v["active_batch"] for v in values),
                    "actual_batch_max": max(v["active_batch"] for v in values),
                }
                for key in values[0]:
                    if key != "active_batch":
                        row[
                            key
                            + (
                                "_mean_ms"
                                if key not in ("h2d_bytes", "d2h_bytes")
                                and not key.startswith("cuda_")
                                else "_mean"
                            )
                        ] = mean([v[key] for v in values])
                timing.append(row)
            for (wave, repetition, view), values in sorted(selection_groups.items()):
                paired = [v for v in values if v["layer_id"] > 0]
                nonempty = [v for v in paired if v["miss_count"] > 0]
                coverage = [
                    v["miss_coverage_numerator"] / v["miss_count"] for v in nonempty
                ]
                record = requests[values[0]["request_id"]]
                selection.append(
                    {
                        "config": config,
                        "rank": rank,
                        "wave_index": wave,
                        "repetition": repetition,
                        "workload": record["workload"],
                        "task": record["task"],
                        "view": view,
                        "requested_concurrency": record["requested_concurrency"],
                        "layer_observations": len(values),
                        "paired_observations": len(paired),
                        "selected_tokens": sum(v["selected_count"] for v in values),
                        "miss_tokens": sum(v["miss_count"] for v in values),
                        "hit_ratio": 1
                        - sum(v["miss_count"] for v in values)
                        / sum(v["selected_count"] for v in values),
                        "miss_bytes": sum(v["miss_bytes"] for v in values),
                        "overlap_tokens": sum(v["overlap_numerator"] for v in paired),
                        "paired_selected_tokens": sum(
                            v["selected_count"] for v in paired
                        ),
                        "paired_miss_tokens": sum(v["miss_count"] for v in paired),
                        "covered_miss_tokens": sum(
                            v["miss_coverage_numerator"] for v in paired
                        ),
                        "overlap_ratio": ratio(
                            sum(v["overlap_numerator"] for v in paired),
                            sum(v["selected_count"] for v in paired),
                        ),
                        "miss_weighted_coverage": ratio(
                            sum(v["miss_coverage_numerator"] for v in paired),
                            sum(v["miss_count"] for v in paired),
                        ),
                        "coverage_p50": quantile(coverage, 0.5),
                        "coverage_p95": quantile(coverage, 0.95),
                        "zero_miss_observations": len(paired) - len(nonempty),
                        "hint_resident_tokens": sum(
                            v["hint_resident_count"] for v in paired
                        ),
                    }
                )
            print(f"audited {config} rank {rank}", flush=True)
        # Different GPU arithmetic can change selections. Preserve and quantify
        # this finding; output equivalence does not prove intermediate equality.
        cross_rank[config] = {
            **dict(rank_differences),
            "identical_selection_digest": selection_digests[0] == selection_digests[1],
            "saved_set_pairs": len(rank_set_overlap),
            "saved_set_overlap_mean": mean(rank_set_overlap),
            "saved_set_overlap_min": min(rank_set_overlap)
            if rank_set_overlap
            else None,
        }
    for path in sorted(run.rglob("*")):
        if path.is_file() and (
            path.name
            in (
                "requests.jsonl",
                "plan.json",
                "launch.json",
                "state.json",
                "startup.json",
            )
            or path.name.startswith("probe_")
        ):
            with path.open("rb") as stream:
                hashes[str(path.relative_to(run))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    audit = {
        "run_dir": str(run),
        "run_source_commit": plan["source_commit"],
        "analysis_worktree_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "analysis_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "launch": launch,
        "terminal_state": load(run / "state.json"),
        "request_executions": len(all_records),
        "distinct_cases": len({v["case_id"] for v in all_records}),
        "expected_request_executions": sum(len(w) for w in plan["waves"])
        * plan["repetitions"]
        * sum(3 if c.startswith("lru") else 2 for c in plan["configs"]),
        "finish_types": dict(
            Counter(
                v["response"]["meta_info"]["finish_reason"]["type"] for v in all_records
            )
        ),
        "errors": dict(errors),
        "error_examples": error_examples,
        "events": event_audit,
        "cross_rank_selection": cross_rank,
        "cross_concurrency_output_changes": cross_concurrency,
        "frozen_request_sha256": frozen_request_sha256,
        "warnings": [
            "Timing/control perturbation must be assessed before using breakdown as performance evidence.",
            "Mixed-GPU rank selections differ; preserve ranks separately and investigate numerics before formal runs.",
            "CPU/Python LRU management overhead must not be attributed to PCIe.",
            "HBM outputs can differ across batch sizes; per-wave equivalence does not establish identical trajectories across concurrency.",
        ],
        "input_sha256": hashes,
        "contract": "pilot diagnostics only; one repetition; stream TPOT may include prefill; last16 is a window, not proof of steady state; ranks are not independent samples",
    }
    export(out / f"{args.prefix}_request-metrics_v01_reviewed.csv", latency)
    export(out / f"{args.prefix}_selection-summary_v01_reviewed.csv", selection)
    export(out / f"{args.prefix}_breakdown-diagnostics_v01_reviewed.csv", timing)
    (run / "audit-results.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {
                "request_executions": len(all_records),
                "errors": dict(errors),
                "error_examples": error_examples,
                "exported_rows": {
                    "requests": len(latency),
                    "selection": len(selection),
                    "timing": len(timing),
                },
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit("Audit failed; exported diagnostics require investigation")


if __name__ == "__main__":
    main()
