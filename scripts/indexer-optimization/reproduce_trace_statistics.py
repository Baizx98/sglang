"""Rebuild the fixed-segment oracle and observed-slot adjacency CSVs on CPU."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/horizonkv-motivation"
FIXTURES = DATA / "2026-10-21_bs16-v01"
TRACE = DATA / "2026-10-15_capacity-c4-v01/hbm/trace/rank_00"
REFERENCE = ROOT / "data/indexer-optimization/results"
STEPS = {0, 8, 24, 32}
LAYERS = {1, 12, 24, 36, 47}
SEGMENTS = (8, 16, 32, 64, 128, 256, 512)
ORACLE_FIELDS = (
    "request", "length_bucket", "layer", "step", "length", "segment_size",
    "occupied_segments", "retained_tokens", "oracle_skip_fraction",
)
ADJACENCY_FIELDS = (
    "request_id", "context", "observed_slots", "observed_adjacent_pairs",
    "consecutive_physical_pairs",
)


def jsonl(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def fixed_segment_rows(fixture_root):
    manifest = json.loads((fixture_root / "long-input-manifest.json").read_text())
    length_bucket = {
        f"replay__wave{wave:03d}__{case}": group["label"]
        for group in manifest["groups"]
        for wave, case in zip(group["waves"], group["case_ids"], strict=True)
    }
    output_lengths = {
        request["rid"]: len(request["response"]["output_ids"])
        for path in fixture_root.glob("input-wave*.json")
        for request in json.loads(path.read_text())
    }
    fixtures = fixture_root / "fixtures"
    seen = set()
    for entry in jsonl(fixtures / "manifest_rank00.jsonl"):
        if len(entry["request_ids"]) != 1 or entry["layer"] not in LAYERS:
            continue
        rid, = entry["request_ids"]
        step, = entry["steps"]
        layer = entry["layer"]
        if rid not in length_bucket or step not in STEPS or step >= output_lengths[rid] - 1:
            continue
        key = (rid, layer, step)
        assert key not in seen, key
        seen.add(key)
        fixture = torch.load(
            fixtures / entry["file"], map_location="cpu", mmap=True, weights_only=True
        )
        length = int(fixture["seq_lens"][0])
        logical = fixture["topk"].reshape(-1).numpy()
        logical = logical[logical >= 0]
        assert logical.size == min(length, 2048) and len(np.unique(logical)) == logical.size
        assert np.all(logical < length)
        for size in SEGMENTS:
            starts = np.unique(logical // size) * size
            retained = int(np.minimum(size, length - starts).sum())
            yield (
                rid, length_bucket[rid], layer, step, length, size, len(starts), retained,
                1.0 - retained / length,
            )


def slot_adjacency_rows(trace_prefix):
    events = [event for event in jsonl(trace_prefix.with_suffix(".jsonl"))]
    contexts = defaultdict(int)
    for event in events:
        if event["kind"] == "access":
            for rid, length in zip(event["request_ids"], event["context_tokens"], strict=True):
                contexts[rid] = max(contexts[rid], length)

    mapping = {rid: np.full(length, -1, dtype=np.int32) for rid, length in contexts.items()}
    binary = np.memmap(trace_prefix.with_suffix(".bin"), dtype="<i4", mode="r")
    expected_offset = 0
    for event in events:
        assert event["offset_bytes"] == expected_offset
        if event["kind"] == "reset":
            assert event["end_bytes"] == expected_offset
            continue
        assert event["kind"] == "access"
        batch, selected = event["shape"]
        assert (batch, selected) == (len(event["request_ids"]), 2048)
        total = batch * selected
        start = expected_offset // 4
        logical = binary[start : start + total].reshape(batch, selected)
        physical = binary[start + total : start + 2 * total].reshape(batch, selected)
        expected_offset += (2 * total + event["write_count"]) * 4
        assert expected_offset == event["end_bytes"]
        for row, (rid, context) in enumerate(zip(
            event["request_ids"], event["context_tokens"], strict=True
        )):
            known = logical[row] >= 0
            assert np.array_equal(known, physical[row] >= 0)
            log = logical[row, known]
            phys = physical[row, known]
            assert np.all(log < context)
            slots = mapping[rid]
            old = slots[log]
            assert np.all((old < 0) | (old == phys)), rid
            slots[log] = phys

    assert expected_offset == binary.nbytes
    for rid, slots in sorted(mapping.items()):
        seen = slots >= 0
        both = seen[:-1] & seen[1:]
        consecutive = both & (slots[1:] == slots[:-1] + 1)
        yield rid, contexts[rid], int(seen.sum()), int(both.sum()), int(consecutive.sum())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, default=FIXTURES)
    parser.add_argument("--trace-prefix", type=Path, default=TRACE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-against", type=Path, default=REFERENCE)
    args = parser.parse_args()
    if args.output.resolve() == args.verify_against.resolve():
        parser.error("output must be a separate directory from references")

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = (
        ("oracle-fixed-segments.csv", ORACLE_FIELDS, fixed_segment_rows(args.fixture_root), "\r\n"),
        ("observed-slot-adjacency.csv", ADJACENCY_FIELDS, slot_adjacency_rows(args.trace_prefix), "\n"),
    )
    for name, fields, rows, line_ending in outputs:
        output = args.output / name
        with output.open("x", newline="") as stream:
            writer = csv.writer(stream, lineterminator=line_ending)
            writer.writerow(fields)
            writer.writerows(rows)
        actual = digest(output)
        expected = digest(args.verify_against / name)
        if actual != expected:
            raise ValueError(f"{output}: generated {actual}, reference {expected}")
        print(f"{actual}  {output}")


if __name__ == "__main__":
    main()
