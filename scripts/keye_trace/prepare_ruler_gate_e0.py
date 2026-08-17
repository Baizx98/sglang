#!/usr/bin/env python3
"""Generate a pinned RULER pool and fail if the upstream wrapper hides errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def validate(path: Path, expected_rows: int) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"RULER generator did not create {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != expected_rows:
        raise RuntimeError(f"{path}: expected {expected_rows} rows, found {len(rows)}")
    indexes = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row.get("input"), str) or not row["input"]:
            raise RuntimeError(f"{path}:{ordinal + 1}: missing nonempty input")
        if not isinstance(row.get("outputs"), list) or not row["outputs"]:
            raise RuntimeError(f"{path}:{ordinal + 1}: missing nonempty outputs")
        indexes.append(row.get("index"))
    if len(set(map(str, indexes))) != len(indexes):
        raise RuntimeError(f"{path}: duplicate source indexes")
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--nltk-data", type=Path, default=Path("/Tan/dataset/nltk_data")
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    prepare = (args.ruler_repo / "scripts" / "data" / "prepare.py").resolve()
    if not prepare.is_file():
        raise FileNotFoundError(prepare)
    for resource in ["punkt", "punkt_tab"]:
        path = args.nltk_data / "tokenizers" / resource
        if not path.is_dir():
            raise FileNotFoundError(f"missing pinned NLTK {resource} at {path}")

    env = os.environ.copy()
    env["NLTK_DATA"] = str(args.nltk_data.resolve())
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    generated: list[dict[str, Any]] = []
    for length in args.lengths:
        save_dir = (args.output_root / str(length)).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        for task in args.tasks:
            command = [
                sys.executable,
                str(prepare),
                "--save_dir",
                str(save_dir),
                "--benchmark",
                "synthetic",
                "--task",
                task,
                "--tokenizer_path",
                str(args.model),
                "--tokenizer_type",
                "hf",
                "--max_seq_length",
                str(length),
                "--model_template_type",
                "base",
                "--num_samples",
                str(args.num_samples),
                "--random_seed",
                str(args.seed),
            ]
            completed = subprocess.run(
                command,
                cwd=prepare.parent,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"RULER preparation failed for {length}/{task}:\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            path = save_dir / task / "validation.jsonl"
            try:
                record = validate(path, args.num_samples)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"RULER output validation failed for {length}/{task}: {exc}\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                ) from exc
            record.update({"length": length, "task": task})
            generated.append(record)
            print(json.dumps(record), flush=True)

    manifest = {
        "schema_version": 1,
        "created_unix_s": time.time(),
        "official_repo": str(args.ruler_repo.resolve()),
        "official_commit": git_revision(args.ruler_repo),
        "python": sys.executable,
        "model": str(args.model.resolve()),
        "lengths": args.lengths,
        "tasks": args.tasks,
        "num_samples": args.num_samples,
        "random_seed": args.seed,
        "nltk_data": str(args.nltk_data.resolve()),
        "punkt_tab_zip_sha256": sha256_file(
            args.nltk_data / "tokenizers" / "punkt_tab.zip"
        ),
        "punkt_zip_sha256": sha256_file(
            args.nltk_data / "tokenizers" / "punkt.zip"
        ),
        "generated": generated,
    }
    manifest_path = args.output_root / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "files": len(generated)}))


if __name__ == "__main__":
    main()
