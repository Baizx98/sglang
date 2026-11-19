"""Reduce raw segment-screening benchmark JSONL to a portable CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "raw_file", "source", "gpu_uuid", "batch", "length_bucket", "max_len",
    "layer", "step", "physical_layout", "path", "scope",
    "median_us", "mad_us", "min_us", "max_us", "trials",
    "set_recall", "rows_pass", "rows_total", "strictly_higher_missing",
    "selected_below_threshold", "max_threshold_regret", "repeat_set_equal",
    "history_summary_bytes_per_key", "best_native_exact_path", "best_native_exact_us",
)


def read_measurement(path):
    metadata = path.with_suffix(".metadata.json")
    gpu_uuid = json.loads(metadata.read_text()).get("gpu_uuid") if metadata.is_file() else None
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        inputs = entry["input"]
        evaluations = entry["evaluations"]
        timings = entry["timings"]
        exact_native = {
            name: time["median_us"]
            for name, time in timings.items()
            if name.startswith("sgl_score_") and name in evaluations
            and all(rep.get("valid", False) for rep in evaluations[name]["sgl"])
        }
        best = min(exact_native, key=exact_native.get) if exact_native else None
        for name, time in timings.items():
            if not name.startswith(("int8_", "bf16_sample4", "sgl_score")):
                continue
            evaluation = evaluations.get(name, {})
            reps = evaluation.get("sgl", ())
            correctness = reps[0] if reps else {}
            matches = correctness.get("set_overlap")
            selected = correctness.get("total_selected")
            yield {
                "raw_file": path.name,
                "source": inputs["source"],
                "gpu_uuid": gpu_uuid,
                **{key: inputs.get(key) for key in (
                    "batch", "length_bucket", "max_len", "layer", "step", "physical_layout"
                )},
                "path": name,
                "scope": "score_to_topk" if reps else "stage",
                **{key: time[key] for key in ("median_us", "mad_us", "min_us", "max_us")},
                "trials": len(time["samples_us"]),
                "set_recall": matches / selected if selected else None,
                "rows_pass": correctness.get("rows_pass"),
                "rows_total": correctness.get("rows_total"),
                "strictly_higher_missing": correctness.get("missing_strictly_above_threshold"),
                "selected_below_threshold": correctness.get("below_threshold"),
                "max_threshold_regret": correctness.get("max_threshold_regret"),
                "repeat_set_equal": evaluation.get("repeat_set_equal"),
                "history_summary_bytes_per_key": 80 if name.startswith("int8_") else 0,
                "best_native_exact_path": best,
                "best_native_exact_us": exact_native.get(best),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for path in args.raw:
            writer.writerows(read_measurement(path))


if __name__ == "__main__":
    main()
