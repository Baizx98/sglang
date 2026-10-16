"""Bounded decode-only CUPTI activities, independent of selection tracing.

External IDs attribute actual GPU kernels/copies to the innermost component.
CPU duration, GPU activity sums and elapsed latency stay distinct.
"""

import atexit
import ctypes
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

_library = None
_active = False
_wave = None
_step = 0
_captured = 0
_done = False
_ranges = []
_stack = []


def library():
    global _library
    if _library is None:
        _library = ctypes.CDLL(os.environ["KEYE_HORIZON_CUPTI_LIBRARY"])
        _library.hkv_clock.restype = ctypes.c_uint64
        _library.hkv_push.argtypes = [ctypes.c_uint64]
        _library.hkv_stop.argtypes = [ctypes.c_char_p]
    return _library


def checked(result):
    if result:
        raise RuntimeError(f"CUPTI collection failed with result {result}")


def active():
    return _active


@contextmanager
def component_scope(component, layer, fields):
    lib = library()
    identifier = len(_ranges) + 1
    record = {"id": identifier, "parent": _stack[-1] if _stack else None,
              "component": component, "layer": layer, **fields}
    _ranges.append(record)
    _stack.append(identifier)
    checked(lib.hkv_push(identifier))
    record["start_ns"] = lib.hkv_clock()
    try:
        yield
    finally:
        record["end_ns"] = lib.hkv_clock()
        checked(lib.hkv_pop())
        assert _stack.pop() == identifier


def finish():
    global _active, _done
    if not _active:
        return
    assert not _stack
    torch.cuda.synchronize()  # Once at the capture boundary, never per component.
    _active = False
    root = Path(os.environ["KEYE_HORIZON_KERNEL_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    prefix = root / f"{_wave}_rank{rank:02d}"
    raw = Path(str(prefix) + ".activities.jsonl")
    checked(library().hkv_stop(str(raw).encode()))
    activities = [json.loads(line) for line in raw.read_text().splitlines()]
    metadata = activities.pop()
    if metadata["dropped"] or not metadata["activity_count"]:
        raise RuntimeError(f"Incomplete CUPTI trace: {metadata}")
    by_id = {r["id"]: r for r in _ranges}
    for record in _ranges:
        record.update(gpu_activity_us=0., cpu_range_exclusive_us=(record["end_ns"]-record["start_ns"])/1000)
    for record in _ranges:
        if record["parent"]:
            by_id[record["parent"]]["cpu_range_exclusive_us"] -= (record["end_ns"]-record["start_ns"])/1000
    for event in activities:
        if event["end_ns"] <= event["start_ns"] or not event["start_ns"]:
            raise RuntimeError("Invalid CUPTI activity timestamp")
        if event["external_id"] in by_id:
            by_id[event["external_id"]]["gpu_activity_us"] += (event["end_ns"]-event["start_ns"])/1000
    value = {
        "rank": rank, "gpu_name": torch.cuda.get_device_name(), "wave": _wave,
        "decode_start": int(os.getenv("KEYE_HORIZON_KERNEL_START", "32")),
        "captured_steps": _captured, "ranges": _ranges, "collector": "CUPTI Activity API 13",
        "cuda_activity_sum_us": sum((e["end_ns"]-e["start_ns"])/1000 for e in activities),
        "collector_metadata": metadata,
        "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
        "contract": "exclusive cumulative GPU kernel/copy activities; not elapsed critical-path time; capture pass only",
    }
    Path(str(prefix) + ".summary.json").write_text(json.dumps(value) + "\n")
    _done = True


def begin(batch):
    global _active, _wave, _step, _captured, _done, _ranges
    if not os.getenv("KEYE_HORIZON_KERNEL_DIR"):
        return
    rids = list(getattr(batch, "rids", None) or [])
    if not batch.forward_mode.is_decode() or not rids or not all(r.startswith("kernel__") for r in rids):
        finish()
        return
    wave = "__".join(rids[0].split("__")[:3])
    if wave != _wave:
        finish()
        _wave, _step, _captured, _done, _ranges = wave, 0, 0, False, []
    if not _done and _step == int(os.getenv("KEYE_HORIZON_KERNEL_START", "32")):
        torch.cuda.synchronize()
        checked(library().hkv_start())
        _active = True


def end(batch):
    global _step, _captured
    if not os.getenv("KEYE_HORIZON_KERNEL_DIR") or not batch.forward_mode.is_decode():
        return
    if not (getattr(batch, "rids", None) or [""])[0].startswith("kernel__"):
        return
    _step += 1
    if active():
        _captured += 1
        if _captured >= int(os.getenv("KEYE_HORIZON_KERNEL_STEPS", "8")):
            finish()


atexit.register(finish)
