"""CPU replay and strict completeness checks for binary capacity access traces."""

import ctypes
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

CAPACITIES = [0, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 65536]


def compile_replay(directory):
    source = Path(__file__).with_name("horizonkv_capacity_lru.cpp")
    output = Path(directory) / "capacity_lru.so"
    subprocess.run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
    )
    library = ctypes.CDLL(str(output.resolve()))
    function = library.capacity_replay
    function.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int64] + [
        ctypes.c_void_p
    ] * 7
    function.restype = ctypes.c_int
    return function


def replay(function, capacity, slots, writes):
    offsets = np.array([0] + list(np.cumsum([v.size for v in slots])), dtype=np.int64)
    ordered = [np.unique(v[v >= 0]) for v in slots]
    sorted_offsets = np.array(
        [0] + list(np.cumsum([v.size for v in ordered])), dtype=np.int64
    )
    write_offsets = np.array(
        [0] + list(np.cumsum([v.size for v in writes])), dtype=np.int64
    )
    flat = np.concatenate([v.reshape(-1) for v in slots]).astype(np.int32)
    unique = np.concatenate(ordered).astype(np.int32)
    written = np.concatenate(writes).astype(np.int32)
    universe = max(int(flat.max(initial=0)), int(written.max(initial=0))) + 1
    result = np.empty(flat.size, dtype=np.uint8)
    values = [offsets, flat, sorted_offsets, unique, write_offsets, written, result]
    error = function(capacity, universe, len(slots), *(v.ctypes.data for v in values))
    if error:
        raise ValueError(f"LRU replay rejected input: {error}")
    return [
        result[offsets[i] : offsets[i + 1]].reshape(value.shape)
        for i, value in enumerate(slots)
    ]


def access_groups(directory, requests, allow_padding=False):
    """Yield verified raw events per rank/wave/layer, including audit-only tails."""
    lookup = {v["rid"]: v for v in requests}
    for rank in (0, 1):
        prefix = Path(directory) / "trace" / f"rank_{rank:02d}"
        records = [
            json.loads(line)
            for line in prefix.with_suffix(".jsonl").read_text().splitlines()
        ]
        binary = np.memmap(prefix.with_suffix(".bin"), dtype="<i4", mode="r")
        expected_offset, resets = 0, set()
        groups, seen = defaultdict(list), set()
        for record in records:
            assert record["rank"] == rank and record["offset_bytes"] == expected_offset
            if record["kind"] == "reset":
                assert (
                    record["resident_after_reset"] == 0 and record["wave"] not in resets
                )
                resets.add(record["wave"])
                continue
            assert record["kind"] == "access" and record["wave"] in resets
            shape = record["shape"]
            assert shape == [len(record["request_ids"]), 2048]
            count = int(np.prod(shape))
            start = expected_offset // 4
            logical = binary[start : start + count].reshape(shape)
            physical = binary[start + count : start + 2 * count].reshape(shape)
            writes = binary[
                start + 2 * count : start + 2 * count + record["write_count"]
            ]
            expected_offset += (2 * count + record["write_count"]) * 4
            assert expected_offset == record["end_bytes"] and len(writes) == shape[0]
            valid = []
            for i, (rid, step, context) in enumerate(
                zip(
                    record["request_ids"],
                    record["decode_steps"],
                    record["context_tokens"],
                )
            ):
                request = lookup[rid]
                assert context == request["prompt_tokens"] + step + 1
                valid_slots = logical[i] >= 0
                assert np.all(logical[i] >= -1)
                assert np.all(logical[i][valid_slots] < context)
                assert np.unique(logical[i][valid_slots]).size == min(2048, context)
                if not allow_padding:
                    assert valid_slots.all()
                assert np.array_equal(valid_slots, physical[i] >= 0)
                key = (rid, step, record["layer_id"])
                assert key not in seen
                seen.add(key)
                valid.append(step < len(request["response"]["output_ids"]) - 1)
            assert np.all(physical >= -1)
            assert np.unique(physical[physical >= 0]).size == np.count_nonzero(physical >= 0)
            record["valid_returned"] = valid
            groups[(record["wave"], record["layer_id"])].append(
                (record, logical, physical, writes)
            )
        assert expected_offset == binary.nbytes
        expected = {
            (rid, step, layer)
            for rid, request in lookup.items()
            for step in range(len(request["response"]["output_ids"]) - 1)
            for layer in range(48)
        }
        assert expected <= seen, (
            rank,
            "missing valid Decode/layer",
            len(expected - seen),
        )
        assert {layer for _, layer in groups} == set(range(48))
        for key, events in sorted(groups.items()):
            yield rank, key, events


def analyze(config_dir, output_dir, capacities=CAPACITIES):
    directory, output_dir = Path(config_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    function = compile_replay(output_dir)
    requests = [
        json.loads(line)
        for line in (directory / "requests.jsonl").read_text().splitlines()
    ]
    totals = defaultdict(lambda: [0, 0, 0])
    per_layer = []
    audited = 0
    for rank, (wave, layer), events in access_groups(directory, requests):
        slots, writes = [v[2] for v in events], [v[3] for v in events]
        wave_closed = False
        wave_started = False
        eligible = []
        for record, *_ in events:
            full = len(record["request_ids"]) == 4 and all(record["valid_returned"])
            if wave_started and not full:
                wave_closed = True
            wave_started = wave_started or full
            eligible.append(full and not wave_closed)
        for capacity in capacities:
            masks = replay(function, capacity, slots, writes)
            layer_totals = defaultdict(lambda: [0, 0, 0])
            for keep, event, mask in zip(eligible, events, masks):
                record = event[0]
                if record["online_hits"] is not None:
                    actual_capacity = int(directory.name.removeprefix("lru"))
                    if capacity == actual_capacity:
                        assert mask.sum(axis=1).tolist() == record["online_hits"], (
                            rank,
                            wave,
                            layer,
                            record["decode_steps"],
                        )
                        audited += len(record["request_ids"])
                if not keep:
                    continue
                for view in ("all_valid", "after32"):
                    if view == "after32" and min(record["decode_steps"]) < 32:
                        continue
                    value = [int(mask.sum()), mask.size, mask.shape[0]]
                    for target in (
                        totals[(rank, wave, capacity, view)],
                        layer_totals[view],
                    ):
                        for i in range(3):
                            target[i] += value[i]
                bucket = min(record["decode_steps"]) // 16
                target = totals[(rank, wave, capacity, f"step_bin_{bucket:02d}")]
                for i, value in enumerate([int(mask.sum()), mask.size, mask.shape[0]]):
                    target[i] += value
            for view, (hits, selected, request_steps) in layer_totals.items():
                per_layer.append(
                    {
                        "rank": rank,
                        "wave": wave,
                        "layer": layer,
                        "capacity_tokens": capacity,
                        "view": view,
                        "hit_tokens": hits,
                        "selected_tokens": selected,
                        "request_steps": request_steps,
                        "hit_ratio": hits / selected,
                    }
                )
        if layer == 47:
            print(
                json.dumps({"replayed": directory.name, "rank": rank, "wave": wave}),
                flush=True,
            )
    result = []
    for (rank, wave, capacity, view), (hits, selected, layer_request_steps) in sorted(
        totals.items()
    ):
        request_steps = layer_request_steps // 48
        result.append(
            {
                "config": directory.name,
                "rank": rank,
                "wave": wave,
                "capacity_tokens": capacity,
                "persistent_main_kv_mib": capacity * 48 / 1024,
                "view": view,
                "hit_tokens": hits,
                "selected_tokens": selected,
                "miss_tokens": selected - hits,
                "request_steps": request_steps,
                "miss_bytes": (selected - hits) * 1024,
                "hit_ratio": hits / selected,
            }
        )
    manifest = {
        "config": directory.name,
        "online_audited_layer_request_steps": audited,
        "requests": len(requests),
        "capacity_points": capacities,
        "full48_and_identity_audit": True,
        "timing_claims": False,
    }
    for rank in (0, 1):
        for suffix in (".bin", ".jsonl"):
            path = directory / "trace" / f"rank_{rank:02d}{suffix}"
            with path.open("rb") as stream:
                manifest[str(path.relative_to(directory)) + "_sha256"] = (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                )
    (output_dir / "audit.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result, per_layer, manifest
