"""Export the frozen Indexer matrices and measured ablations for the research notes."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path("data/indexer-predict")
ORIGINAL = ROOT / "frozen-run-v1"
UPDATED = ROOT / "frozen-native-int4-v1"
WARP = ROOT / "frozen-native-warp-v1"
PERCENTILE = 0.95


def records(path):
    with path.open() as file:
        yield from (json.loads(line) for line in file)


def write_csv(output, name, rows):
    with (output / name).open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit(root):
    policy = json.loads((root / "policy.json").read_text())
    holdout = json.loads((root / "holdout.json").read_text())
    i = lambda r: (r["length_bucket"], r["batch"], r["layer"], r["step"], r["M"])
    expected = set(itertools.product((65536, 131072, 262144), (4, 16), (1, 12, 24, 47), (2, 4), (3072, 8192)))
    assert len(holdout["results"]) == len(expected) and {i(r) for r in holdout["results"]} == expected
    assert all(r["status"] == "valid" for r in holdout["results"])
    routes = {(r["length_bucket"], r["layer"], r["M"]): r["method"] for r in policy["routes"]}
    prior_root = UPDATED if root == WARP else ORIGINAL
    prior = {(r["length_bucket"], r["layer"], r["M"]): r["method"]
             for r in json.loads((prior_root / "policy.json").read_text())["routes"]}
    source = {p: list(records(Path(p))) for p in {r["source"] for r in holdout["results"]}}
    cells, requests = [], []
    anchor, cohort_ids = {}, {}
    for entry in holdout["results"]:
        length, batch, layer, step, m = i(entry)
        raw = source[entry["source"]][entry["line"] - 1]
        assert (raw["bucket"], raw["batch"], raw["layer"], raw["step"], raw["width"]) == i(entry)
        assert raw["split"] == "test" and len(set(raw["request_ids"])) == batch
        assert not set(raw["request_ids"]).intersection(policy["calibration_request_ids"])
        cohort = cohort_ids.setdefault((length, batch), raw["request_ids"])
        assert cohort == raw["request_ids"]
        assert entry["method"] == routes[length, layer, m]
        baseline = raw["fastest_equal_layout_bf16"]
        fastest = {}
        t = lambda name: raw["timings"][name]["median_us"]
        for group, prefix in (("plain", "bf16_b"), ("lossless", "lossless12:"), ("static", "static12_b")):
            valid = [name for name, r in raw["methods"].items()
                     if name.startswith(prefix) and r.get("baseline") and name in raw["timings"]]
            fastest[group] = min(valid, key=t)
        plain_sorted = min((name for name, r in raw["methods"].items()
                            if name.startswith("bf16_b") and name.endswith(":torch_sorted")
                            and r.get("baseline")), key=t)
        assert min(t(name) for name in fastest.values()) == t(baseline)
        current = entry["method"]
        prior_name = prior[length, layer, m]
        exact = raw["methods"][baseline]
        route = raw["methods"][current]
        assert exact["score_equal"] and exact["predicted_set_intersection"] == [m] * batch
        assert math.isclose(t(baseline) / t(current), entry["speedup_same_layout"])
        assert exact["true_top2048_intersection"] == raw["exact_candidate_intersection"]
        code = re.sub(r"_b[0-9]+.*", "", current)
        fields = dict(release=root.name, length_bucket=length, batch=batch, layer=layer, step=step, M=m)
        cell = dict(fields, method=current, baseline=baseline, route_us=t(current), exact_us=t(baseline),
                    prior_route=prior_name, prior_route_us=t(prior_name),
                    route_mad_us=raw["timings"][current]["mad_us"],
                    exact_mad_us=raw["timings"][baseline]["mad_us"],
                    speedup=entry["speedup_same_layout"], plain_method=fastest["plain"],
                    plain_us=t(fastest["plain"]), lossless_method=fastest["lossless"], lossless_us=t(fastest["lossless"]),
                    plain_sorted_method=plain_sorted, plain_sorted_us=t(plain_sorted),
                    fixed_exact_us=t("lossless12:torch_sorted"),
                    static_method=fastest["static"], static_us=t(fastest["static"]),
                    raw_history_bytes=128 * sum(raw["history_lengths"]),
                    route_append_us=t(f"append_{code}"),
                    exact_append_us=t("append_static12" if baseline.startswith("static12") else "append_lossless12"),
                    source_file=entry["source"], source_line=entry["line"])
        for prefix, method in (("route", route), ("exact", exact)):
            for field in ("logical_key_bytes", "index_key_bytes", "initialize_wall_us",
                          "workspace_bytes_before_selector", "logical_score_write_bytes",
                          "known_cute_workspace_bytes"):
                cell[f"{prefix}_{field}"] = method.get(field)
        cells.append(cell)
        data = zip(raw["request_ids"], raw["history_lengths"], exact["true_top2048_intersection"],
                   route["true_top2048_intersection"], route["predicted_set_intersection"],
                   entry["extra_miss_pp"], strict=True)
        for rid, history, ref_hits, hits, common, delta in data:
            assert (ref_hits - hits) * 100 / 2048 == delta
            requests.append(dict(fields, request=rid, history_tokens=history, exact_hits=ref_hits,
                                 route_hits=hits, topm_intersection=common, extra_miss_pp=delta))
        anchor[i(entry)] = (raw["request_ids"], raw["history_lengths"], exact["true_top2048_intersection"])
    heldout = set()
    for length in (65536, 131072, 262144):
        cohort = cohort_ids[length, 16]
        assert cohort[:4] == cohort_ids[length, 4] and not heldout.intersection(cohort)
        heldout.update(cohort)
    return cells, requests, anchor, holdout["summary"]


def ablations(native_warp):
    parents = (("native-warp-calibration-v1", WARP.name) if native_warp else
               ("certified-isolated-v1", "prefix-isolated-v1", "native-int4-isolated-v1",
                "native-int4-fused-query-calibration-v1", UPDATED.name))
    for suite in parents:
        path = ROOT / suite
        for file in sorted(path.glob("**/results.jsonl")):
            for line, raw in enumerate(records(file), 1):
                ablations = [(tag, s["control"], s["fused"])
                             for tag, s in raw["full_path_ablations"].items()
                             if tag.startswith(("native4_", "native8_", "prefix_histogram_"))]
                ablations.extend(("certified", raw["fastest_equal_layout_bf16"], name)
                                 for name in raw["timings"] if name.startswith("certified_int"))
                for tag, control, method in ablations:
                    orig, new = (raw["timings"][name]["median_us"] for name in (control, method))
                    yield dict(suite=suite, length_bucket=raw["bucket"], batch=raw["batch"], layer=raw["layer"],
                               step=raw["step"], M=raw["width"], split=raw["split"], experiment=tag,
                               control=control, method=method, control_us=orig, method_us=new,
                               speedup=orig/new, source_file=str(file), source_line=line)


def quality():
    for bucket in (65536, 131072, 262144):
        root = ORIGINAL / "quality"
        case, = records(root / f"requests-{bucket}.jsonl")
        file = root / f"run-{bucket}/task-quality.jsonl"
        rows = list(records(file))
        assert {r["mode"] for r in rows} == {"native", "bf16", "compact"} and len(rows) == 3
        for row in rows:
            expected = case["expected_answer"]
            score = sum(a.lower() in row["final_answer"].lower() for a in expected) / len(expected)
            assert score == row["final_score"] and row["rid"] == case["rid"]
            yield dict(length_bucket=bucket, **{k:row[k] for k in (
                "rid", "mode", "task", "prompt_tokens", "finish", "final_score", "output_tokens"
            )}, expected_answer="|".join(expected), prompt_sha256=case["input_ids_sha256"], source_file=str(file))


def run(output, native_warp):
    original = audit(ORIGINAL)
    runs = {ORIGINAL: original, UPDATED: audit(UPDATED)}
    if native_warp:
        runs[WARP] = audit(WARP)
    assert all(result[2] == original[2] for result in runs.values()), (
        "the implementation replacement must preserve the holdout and BF16 reference"
    )
    selected = {WARP: runs[WARP]} if native_warp else runs
    all_cells, all_requests = [], []
    ablation_rows, quality_rows = list(ablations(native_warp)), list(quality())
    stats = {}
    for path, (cells, requests, _, summary) in selected.items():
        values = sorted(r["extra_miss_pp"] for r in requests)
        assert summary["cells"] == len(cells) and summary["request_rows"] == len(requests)
        assert math.isclose(summary["geomean_speedup"], statistics.geometric_mean(c["speedup"] for c in cells))
        assert summary["p95_extra_miss_pp"] == values[math.ceil(PERCENTILE*len(values)) - 1]
        assert summary["mean_extra_miss_pp"] == statistics.mean(values)
        stats[path.name] = summary
        all_cells.extend(cells)
        all_requests.extend(requests)
    files = {root / name for root in runs for name in ("policy.json", "holdout.json")}
    for run in runs:
        policy = json.loads((run / "policy.json").read_text())
        files.update(Path(f) for f in policy["source_sha256"])
        impl = policy.get("implementation_replacement", {})
        files.update(Path(f) for f in impl.get("calibration_sha256", {}))
    files.update((ORIGINAL / "quality").glob("requests-*.jsonl"))
    for row in all_cells + ablation_rows + quality_rows:
        files.add(Path(row["source_file"]))
    files.update(file.parent / "metadata.json" for file in list(files) if file.name == "results.jsonl")
    source = {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
    exporter = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output, "cells.csv", all_cells)
    write_csv(output, "requests.csv", all_requests)
    write_csv(output, "ablations.csv", ablation_rows)
    write_csv(output, "ruler.csv", quality_rows)
    (output / "provenance.json").write_text(json.dumps(
        dict(summary=stats, export_script_sha256=exporter, source_sha256=source), indent=2,
    ) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native-warp", action="store_true")
    args = parser.parse_args()
    run(args.output, args.native_warp)
