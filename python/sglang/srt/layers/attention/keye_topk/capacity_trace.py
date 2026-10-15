"""Opt-in access-only capacity tracing; deliberately makes no timing claims."""

import atexit
import json
import os
import queue
import threading
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

_writer = None
_wave = None
_steps = {}


def enabled(batch):
    rids = getattr(batch, "rids", None) or []
    return bool(
        os.getenv("KEYE_HORIZON_CAPACITY_DIR")
        and batch.forward_mode.is_decode()
        and rids
        and all(rid.startswith("capacity__") for rid in rids)
    )


class Writer:
    def __init__(self):
        self.root = Path(os.environ["KEYE_HORIZON_CAPACITY_DIR"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.items = queue.Queue(maxsize=512)
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, record, arrays=()):
        if self.error:
            raise RuntimeError("Capacity trace writer failed") from self.error
        self.items.put((record, arrays), timeout=120)

    def run(self):
        try:
            prefix = self.root / f"rank_{self.rank:02d}"
            with (
                prefix.with_suffix(".jsonl").open("w", buffering=1) as index,
                prefix.with_suffix(".bin").open("wb") as data,
            ):
                offset = 0
                for records, (record, arrays) in enumerate(
                    iter(self.items.get, None), 1
                ):
                    record.update(rank=self.rank, offset_bytes=offset)
                    for array in arrays:
                        value = array.astype("<i4", copy=False).tobytes()
                        data.write(value)
                        offset += len(value)
                    record["end_bytes"] = offset
                    index.write(json.dumps(record, separators=(",", ":")) + "\n")
                    if record.get("layer_id") == 47 or record["kind"] == "reset":
                        data.flush()
                        index.flush()
                        status = {
                            "records": records,
                            "data_bytes": offset,
                            "wave": record["wave"],
                            "steps": record.get("decode_steps"),
                            "writer_error": None,
                        }
                        tmp = prefix.with_suffix(".status.tmp")
                        tmp.write_text(json.dumps(status) + "\n")
                        tmp.replace(prefix.with_suffix(".status.json"))
        except BaseException as exc:  # noqa: BLE001 - propagate writer failure to inference
            self.error = exc
            (self.root / f"rank_{self.rank:02d}.error").write_text(repr(exc))

    def close(self):
        if self.thread.is_alive():
            self.items.put(None, timeout=120)
            self.thread.join(timeout=120)


def writer():
    global _writer
    if _writer is None:
        _writer = Writer()
    return _writer


def begin_layer(batch, layer_id, pool):
    global _wave
    if not enabled(batch) or layer_id != 0:
        return
    waves = {"__".join(rid.split("__")[:3]) for rid in batch.rids}
    if len(waves) != 1:
        raise RuntimeError("Capacity trace cannot mix waves")
    wave = waves.pop()
    if wave != _wave:
        store = getattr(pool, "horizon_store", None)
        if store is not None:
            for host in store.layers:
                host.resident.clear()
                host.free = deque(range(host.capacity))
            if any(host.resident for host in store.layers):
                raise RuntimeError("Main KV LRU reset failed")
        writer().submit(
            {
                "kind": "reset",
                "wave": wave,
                "resident_after_reset": 0,
                "storage_mode": os.getenv("KEYE_HORIZON_STORAGE", "hbm"),
                "capacity": int(os.getenv("KEYE_HORIZON_CACHE_TOKENS", "0")),
                "gpu_name": torch.cuda.get_device_name(),
            }
        )
        _wave = wave
    batch._capacity_steps = [_steps.get(rid, 0) for rid in batch.rids]
    for rid in batch.rids:
        _steps[rid] = _steps.get(rid, 0) + 1
    batch._capacity_online_hits = {}


def observe_hits(batch, layer_id, slots, hits):
    if enabled(batch):
        batch._capacity_online_hits[layer_id] = [
            sum(int(token) in hits for token in row if token >= 0)
            for row in slots.tolist()
        ]


def record(batch, layer_id, logical, physical, writes):
    if not enabled(batch):
        return
    arrays = [
        value.detach().cpu().numpy().astype(np.int32, copy=True)
        for value in (logical, physical, writes)
    ]
    writer().submit(
        {
            "kind": "access",
            "wave": _wave,
            "layer_id": layer_id,
            "request_ids": list(batch.rids),
            "decode_steps": batch._capacity_steps,
            "context_tokens": batch.seq_lens_cpu.tolist(),
            "shape": list(arrays[0].shape),
            "write_count": arrays[2].size,
            "online_hits": batch._capacity_online_hits.get(layer_id),
        },
        arrays,
    )


@atexit.register
def close():
    if _writer is not None:
        _writer.close()
