"""Summarize audited layer replays without pooling protocols or trace views."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run_dir, args.processed
    wave_task, requests = {}, []
    for path in sorted(root.glob("input-wave*.json")):
        batch = json.loads(path.read_text())
        assert len(batch) == 4 and len({r["task"] for r in batch}) == 1
        wave_task[int(path.stem.removeprefix("input-wave"))] = batch[0]["task"]
        requests.extend(batch)
    workload = {}
    for task in sorted(set(wave_task.values())):
        batch = [r for r in requests if r["task"] == task]
        workload[task] = dict(requests=len(batch),
                              input_min_tokens=min(r["prompt_tokens"] for r in batch),
                              input_max_tokens=max(r["prompt_tokens"] for r in batch),
                              output_min_tokens=min(len(r["response"]["output_ids"]) for r in batch),
                              output_max_tokens=max(len(r["response"]["output_ids"]) for r in batch))
    single = read_csv(out / "single-layer-latency.csv")
    pairs = read_csv(out / "two-layer-latency.csv")
    latency = single + pairs
    assert len({r["case"] for r in latency}) == len(latency)
    for row in latency:
        row["task"] = wave_task[int(row["wave"])]
        samples = [float(row[f"repeat_{i}_us"]) for i in range(1, 6)]
        assert min(samples) > 0 and statistics.median(samples) == float(row["median_us"])
    records = {r["case"]: r for r in latency}
    summary = []

    def summarize(section, task, metric, values, unit, budget="", mode="", snapshot="after32"):
        assert values
        summary.append(dict(section=section, task=task, snapshot=snapshot,
                            b_over_k=budget, mode=mode, metric=metric, unit=unit,
                            observations=len(values), median=statistics.median(values),
                            minimum=min(values), maximum=max(values)))

    for task in ["ALL", *sorted(workload)]:
        selected = [r for r in single if r["snapshot"] == "after32"
                    and (task == "ALL" or r["task"] == task)]
        for mode in sorted({r["mode"] for r in selected}):
            summarize("single_layer", task, "elapsed", [float(r["median_us"]) for r in selected
                      if r["mode"] == mode], "us", mode=mode)
        for row in selected:
            if row["mode"] != "l-cached":
                continue
            prefix = row["case"].removesuffix("l-cached")
            for mode in ("h-native", "h-staged", "d-staged", "no-io", "echo"):
                other = records[prefix + mode]
                # Store per-point paired values before taking descriptive medians.
                row[f"ratio_to_{mode}"] = float(row["median_us"]) / float(other["median_us"])
        for mode in ("h-native", "h-staged", "d-staged", "no-io", "echo"):
            summarize("single_layer", task, f"lru_ratio_to_{mode}",
                      [r[f"ratio_to_{mode}"] for r in selected if r["mode"] == "l-cached"], "ratio")
        oracles = read_csv(out / "after32-oracle-summary.csv")
        for budget in (1, 2, 4):
            points = [r for r in oracles if float(r["b_over_k"]) == budget
                      and (task == "ALL" or wave_task[int(r["wave"])] == task)]
            for metric, unit in (("baseline_hit_ratio", "ratio"),
                                 ("ahead_reduction_vs_lru_pct", "percent"),
                                 ("additional_layer_window_reduction_pct", "percent")):
                summarize("two_layers", task, metric, [float(r[metric]) for r in points], unit, budget)
            for key in ("lru_us", "ahead_us", "query_perfect_us", "noio_us", "echo_style_us"):
                summarize("two_layers", task, key, [float(r[key]) for r in points], "us", budget)
            summarize("two_layers", task, "noio_reduction_vs_lru_pct",
                      [100 * (float(r["lru_us"]) - float(r["noio_us"])) / float(r["lru_us"]) for r in points],
                      "percent", budget)

    components = read_csv(out / "component-work-and-spans.csv")
    by_case = defaultdict(dict)
    for row in components:
        if row["mode"] == "h-native" and row["snapshot"] == "after32":
            assert row["actual_gpu_work_us"] != ""
            by_case[row["case"]][row["component_name"]] = float(row["actual_gpu_work_us"])
    for task in ["ALL", *sorted(workload)]:
        cases = [v for key, v in by_case.items() if task == "ALL" or records[key]["task"] == task]
        for component in sorted(set().union(*(v.keys() for v in cases))):
            summarize("native_gpu_work", task, component, [v.get(component, 0) for v in cases], "us")
        summarize("native_gpu_work", task, "indexer_fraction",
                  [sum(v[k] for k in ("indexer_query_prepare", "indexer_score", "indexer_topk")) / sum(v.values())
                   for v in cases], "ratio")

    capacity = read_csv(args.capacity / "capacity-by-group-layer.csv")
    for row in capacity:
        total, hit = int(row["selected_tokens"]), int(row["hit_tokens"])
        assert 0 <= hit <= total and int(row["logical_miss_bytes"]) == (total - hit) * 2048
        assert total == int(row["request_steps"]) * 2048
    cap_groups = defaultdict(dict)
    for row in capacity:
        key = tuple(row[k] for k in ("group", "layer", "allocation", "view", "capacity_tokens"))
        cap_groups[key][row["policy"]] = row
    for policies in cap_groups.values():
        assert len(policies) == 2
        a, b = policies["lru_hits_above_misses"], policies["belady_bypass"]
        assert a["selected_tokens"] == b["selected_tokens"]
        if a["view"] == "all":
            assert int(b["hit_tokens"]) >= int(a["hit_tokens"])
    pooled = defaultdict(list)
    for row in capacity:
        for task in ("ALL", row["task"]):
            key = (task, *[row[k] for k in ("concurrency", "allocation", "capacity_tokens", "view", "policy")])
            pooled[key].append(row)
    capacity_summary = []
    for key, group in sorted(pooled.items()):
        total = sum(int(r["selected_tokens"]) for r in group)
        hits = sum(int(r["hit_tokens"]) for r in group)
        layers = sorted({r["layer"] for r in group})
        layer_ratios = []
        for layer in layers:
            points = [r for r in group if r["layer"] == layer]
            denominator = sum(int(r["selected_tokens"]) for r in points)
            layer_ratios.append(sum(int(r["hit_tokens"]) for r in points) / denominator)
        row = dict(zip(("task", "concurrency", "allocation", "capacity_tokens", "view", "policy"), key))
        row.update(groups=len({r["group"] for r in group}), layers=len(layers),
                   B_over_k=group[0]["B_over_k"], B_over_Ck=group[0]["B_over_Ck"],
                   selected_tokens=total, hit_tokens=hits, pooled_hit_ratio=hits / total,
                   logical_miss_bytes=(total-hits)*2048,
                   layer_min_ratio=min(layer_ratios), layer_median_ratio=statistics.median(layer_ratios),
                   layer_max_ratio=max(layer_ratios), layer_std_ratio=statistics.pstdev(layer_ratios))
        capacity_summary.append(row)
    with (out / "capacity-summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(capacity_summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(capacity_summary)
    with (out / "report-summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    audit = dict(workloads=workload, cases=len(latency), summary_rows=len(summary),
                 capacity_rows=len(capacity), capacity_policy_pairs=len(cap_groups),
                 capacity_summary_rows=len(capacity_summary),
                 method="Medians of paired wave x target-layer snapshots; not independent requests or confidence intervals. Capacity views not pooled.",
                 files={str(p): sha(p) for p in [out / "single-layer-latency.csv", out / "two-layer-latency.csv",
                          out / "component-work-and-spans.csv", out / "report-summary.csv", out / "capacity-summary.csv",
                          args.capacity / "capacity-by-group-layer.csv", Path(__file__)]})
    (out / "report-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
