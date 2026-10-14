"""Freeze Keye-tokenized workload manifests without looking at model outcomes.

Raw prompts/token IDs remain in ignored data/. Natural examples are never
truncated, padded, or repeated to meet a length target.
"""

import argparse
import hashlib
import json
import random
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from run_longbench_v2_compact_trace import format_prompt
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument(
        "--longbench-v2",
        type=Path,
        default=Path("/Tan/dataset/LongBench-v2/0000.parquet"),
    )
    parser.add_argument(
        "--longbench-v1", type=Path, default=Path("/Tan/dataset/LongBench/data.zip")
    )
    parser.add_argument("--longbench-prompts", type=Path, required=True)
    parser.add_argument("--model", default="/Tan/model/Keye-VL-2.0-30B-A3B")
    parser.add_argument("--samples-per-task-bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if args.samples_per_task_bucket <= 0:
        raise ValueError("sample count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "requests.jsonl"
    if output.exists():
        raise FileExistsError(
            "Do not overwrite a frozen manifest; use a new output directory"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    groups = defaultdict(list)
    sources = []
    excluded = Counter()

    def add(dataset, task, workload, sample_id, prompt, expected, revision, extra=None):
        if len(prompt) > 700000:
            excluded["char_guard_over_700000"] += 1
            return
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer.encode(formatted, add_special_tokens=False)
        length = len(ids)
        bucket = next(
            (cap for cap in [8192, 16384, 32768, 65536, 131072] if length <= cap), None
        )
        if bucket is None:
            excluded["over_128k"] += 1
            return
        key = (dataset, task, bucket)
        value = {
            "dataset": dataset,
            "task": task,
            "workload": workload,
            "sample_id": str(sample_id),
            "dataset_revision": revision,
            "length_bucket_upper": bucket,
            "prompt_tokens": length,
            "prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
            "input_ids_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "input_ids": ids,
            "expected_answer": expected,
            **(extra or {}),
        }
        groups[key].append(value)

    generation = args.ruler_root / "generation_manifest.json"
    if not generation.is_file():
        raise FileNotFoundError(
            "Complete official RULER generation before freezing requests"
        )
    source_generation = json.loads(generation.read_text())
    for record in source_generation["generated"]:
        source = Path(record["path"])
        if sha(source) != record["sha256"]:
            raise ValueError(f"RULER source hash mismatch: {source}")
        sources.append(record)
        for row in map(json.loads, source.read_text().splitlines()):
            task = record["task"]
            add(
                "RULER-v1",
                task,
                "reasoning" if task == "vt" else "retrieval",
                f"{record['length']}:{row['index']}",
                row["input"],
                row["outputs"],
                source_generation["official_commit"],
                {"generator_length": record["length"]},
            )
    sources.append({"path": str(generation.resolve()), "sha256": sha(generation)})

    domains = {
        "Single-Document QA": "qa",
        "Multi-Document QA": "qa",
        "Code Repository Understanding": "code",
        "Long Structured Data Understanding": "structured",
        "Long In-context Learning": "structured",
        "Long-dialogue History Understanding": "dialogue",
    }
    sources.append(
        {
            "path": str(args.longbench_v2.resolve()),
            "sha256": sha(args.longbench_v2),
            "dataset_revision": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
            "split": "train",
        }
    )
    for row in pd.read_parquet(args.longbench_v2).to_dict("records"):
        add(
            "LongBench-v2",
            row["domain"],
            domains[row["domain"]],
            row["_id"],
            format_prompt(row),
            row["answer"],
            "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
            {"sub_domain": row["sub_domain"], "difficulty": row["difficulty"]},
        )

    prompts = json.loads(args.longbench_prompts.read_text())
    sources.append(
        {
            "path": str(args.longbench_v1.resolve()),
            "sha256": sha(args.longbench_v1),
            "dataset_revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
            "split": "test",
        }
    )
    sources.append(
        {
            "path": str(args.longbench_prompts.resolve()),
            "sha256": sha(args.longbench_prompts),
            "source_url": "https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config/dataset2prompt.json",
        }
    )
    with zipfile.ZipFile(args.longbench_v1) as archive:
        for task in ["gov_report", "qmsum", "repobench-p", "lcc"]:
            name = next(
                n for n in archive.namelist() if n.endswith("/" + task + ".jsonl")
            )
            for index, row in enumerate(
                map(json.loads, archive.read(name).splitlines())
            ):
                add(
                    "LongBench-v1",
                    task,
                    "summary" if task in ["gov_report", "qmsum"] else "code",
                    row.get("_id", index),
                    prompts[task].format(**row),
                    row["answers"],
                    "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
                    {"language": row["language"]},
                )
    rng = random.Random(args.seed)
    selected = []
    cells = []
    for key, values in sorted(groups.items()):
        values.sort(key=lambda v: v["sample_id"])
        rng.shuffle(values)
        chosen = values[: args.samples_per_task_bucket]
        cells.append(
            {
                "dataset": key[0],
                "task": key[1],
                "bucket_upper_tokens": key[2],
                "eligible": len(values),
                "selected": len(chosen),
            }
        )
        selected.extend(chosen)
    with output.open("x") as f:
        for i, row in enumerate(selected):
            row["case_id"] = f"case{i:04d}"
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "selection": "stable ID sort then seeded shuffle within task and actual Keye-token bucket",
        "chat_template": True,
        "tokenizer_config_sha256": sha(Path(args.model) / "tokenizer_config.json"),
        "samples_per_task_bucket": args.samples_per_task_bucket,
        "requests_sha256": sha(output),
        "request_count": len(selected),
        "cells": cells,
        "excluded": dict(excluded),
        "sources": sources,
        "deferred_sources": ["InfiniteBench expanded tasks", "BABILong", "tau3 replay"],
        "natural_input_transforms": "none; complete examples only; no padding or truncation",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "requests": len(selected),
                "cells": len(cells),
                "workloads": dict(Counter(v["workload"] for v in selected)),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
