"""Opt-in Keye HBM/host-memory motivation probes; disabled by default.

Host storage uses exact token KV and a simple demand-only global-per-layer LRU.
This Python/CPU-managed prototype is not the optimized HiSparse implementation.
CUDA event spans include stream idle time caused by host work. They are not
cumulative kernel times, and nested spans must not be summed twice.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch
import torch.distributed as dist


def storage_mode():
    value = os.getenv("KEYE_HORIZON_STORAGE", "hbm")
    if value not in {"hbm", "dram", "lru"}:
        raise ValueError("KEYE_HORIZON_STORAGE must be hbm, dram or lru")
    if value != "hbm" and os.getenv("KEYE_SM80_DSA") != "1":
        raise ValueError("Horizon host storage requires KEYE_SM80_DSA=1")
    return value


_writer = None
_queue = None
_steps = {}


def _write_loop(path, items, device):
    torch.cuda.set_device(device)
    with path.open("a", buffering=1) as output:
        while True:
            item = items.get()
            if item is None:
                return
            record, events = item
            if events is not None:
                start, end, anchor = events
                # Waiting happens in the writer, never in an instrumented span.
                end.synchronize()
                record["cuda_stream_span_ms"] = start.elapsed_time(end)
                record["start_from_model_ms"] = anchor.elapsed_time(start)
                record["end_from_model_ms"] = anchor.elapsed_time(end)
            output.write(json.dumps(record, separators=(",", ":")) + "\n")


def emit(record, events=None):
    global _writer, _queue
    directory = os.getenv("KEYE_HORIZON_PROBE_DIR")
    if not directory:
        return
    if _queue is None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        (path / f"probe_metadata_rank_{rank:02d}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "gpu_name": torch.cuda.get_device_name(),
                    "device_index": torch.cuda.current_device(),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "storage_mode": storage_mode(),
                    "timing_contract": "CUDA stream spans include host idle; nested spans are not additive",
                    "selection_pass_has_timing_claims": False,
                },
                indent=2,
            )
            + "\n"
        )
        _queue = queue.Queue(maxsize=65536)
        _writer = threading.Thread(
            target=_write_loop,
            args=(
                path / f"probe_rank_{rank:02d}.jsonl",
                _queue,
                torch.cuda.current_device(),
            ),
            daemon=True,
        )
        _writer.start()
    _queue.put((record, events))


def _close_writer():
    if _queue is not None:
        _queue.put(None)
        _writer.join(timeout=30)


atexit.register(_close_writer)


def batch_metadata(batch, prefix=None):
    if not batch.forward_mode.is_decode():
        return None
    rids = list(getattr(batch, "rids", None) or [])
    prefix = (
        prefix
        if prefix is not None
        else os.getenv("KEYE_HORIZON_RID_PREFIX", "horizon__")
    )
    if not rids or not all(rid.startswith(prefix) for rid in rids):
        return None
    return {
        "request_ids": rids,
        "decode_steps": getattr(batch, "_horizon_steps", None),
        "context_tokens": batch.seq_lens_cpu.tolist(),
        "active_batch": len(rids),
    }


@contextmanager
def span(batch, layer_id, component, **fields):
    if os.getenv("KEYE_HORIZON_KERNEL_DIR"):
        from .horizonkv_kernel_profile import active, component_scope

        if active():
            if component == "model_total":
                fields.update(active_batch=len(batch.rids), request_ids=list(batch.rids),
                              context_tokens=batch.seq_lens_cpu.tolist())
            with component_scope(component, layer_id, fields):
                yield
            return
    metadata = batch_metadata(batch) if os.getenv("KEYE_HORIZON_PROBE_DIR") else None
    if metadata is None:
        yield
        return
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    if component == "model_total":
        batch._horizon_step_anchor = start
    anchor = getattr(batch, "_horizon_step_anchor", start)
    cpu_start = time.perf_counter()
    yield
    cpu_ms = (time.perf_counter() - cpu_start) * 1000
    end.record()
    if component == "model_total":
        fields.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(),
            cuda_reserved_bytes=torch.cuda.memory_reserved(),
            cuda_peak_allocated_bytes_since_startup=torch.cuda.max_memory_allocated(),
        )
    emit(
        {
            "kind": "span",
            "layer_id": layer_id,
            "component": component,
            "cpu_submission_ms": cpu_ms,
            **metadata,
            **fields,
        },
        (start, end, anchor),
    )


def profiled(component, batch_arg=3):
    def decorate(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            if os.getenv("KEYE_HORIZON_KERNEL_DIR"):
                from .horizonkv_kernel_profile import begin, end

                batch = kwargs.get("forward_batch")
                if batch is None:
                    batch = args[batch_arg - 1]
                if component == "model_total":
                    begin(batch)
                try:
                    with span(batch, getattr(self, "layer_id", -1), component):
                        return function(self, *args, **kwargs)
                finally:
                    if component == "model_total":
                        end(batch)
            if not os.getenv("KEYE_HORIZON_PROBE_DIR"):
                return function(self, *args, **kwargs)
            batch = kwargs.get("forward_batch")
            if batch is None:
                batch = args[batch_arg - 1]
            if component == "model_total" and batch.forward_mode.is_decode():
                rids = list(getattr(batch, "rids", None) or [])
                batch._horizon_steps = [_steps.get(rid, 0) for rid in rids]
                for rid in rids:
                    _steps[rid] = _steps.get(rid, 0) + 1
                batch._horizon_previous_selected = {}
                batch._horizon_previous_physical = {}
            with span(batch, getattr(self, "layer_id", -1), component):
                return function(self, *args, **kwargs)

        return wrapper

    return decorate


class HostLayer:
    def __init__(self, size, heads, dim, dtype, device, capacity):
        self.k = torch.zeros((size, heads, dim), dtype=dtype, pin_memory=True)
        self.v = torch.zeros_like(self.k, pin_memory=True)
        self.device = device
        self.capacity = capacity
        self.hot_k = torch.empty((capacity, heads, dim), dtype=dtype, device=device)
        self.hot_v = torch.empty_like(self.hot_k)
        self.resident = OrderedDict()  # physical pool slot -> hot-buffer slot
        self.free = deque(range(capacity))

    def write(self, slots, k, v):
        # CPU destination cannot be consumed until D2H finishes. Include this
        # synchronous writeback in the real decode path, not outside timing.
        self.k[slots] = k.reshape_as(self.k[slots]).cpu()
        self.v[slots] = v.reshape_as(self.v[slots]).cpu()
        for token in slots.tolist():
            hot = self.resident.pop(token, None)
            if hot is not None:
                self.free.append(hot)

    def stage(self, slots, batch, layer_id, logical_indices):
        selection_metadata = (
            batch_metadata(
                batch, os.getenv("KEYE_HORIZON_SELECTION_PREFIX", "selection__")
            )
            if os.getenv("KEYE_HORIZON_PROBE_DIR")
            else None
        )
        resident_before = set(self.resident) if selection_metadata is not None else None
        flat = slots.reshape(-1)
        valid = flat >= 0
        selected = flat[valid].tolist()
        with span(batch, layer_id, "kv_lookup_host"):
            # Snapshot all requests before any update; copy hits before evicting
            # so a working set larger than capacity still computes exact attention.
            hits = {
                token: self.resident[token]
                for token in selected
                if token in self.resident
            }
            if os.getenv("KEYE_HORIZON_CAPACITY_DIR"):
                from .capacity_trace import observe_hits

                observe_hits(batch, layer_id, slots, hits)
            miss_positions = [
                i
                for i, token in enumerate(flat.tolist())
                if token >= 0 and token not in hits
            ]
            hit_positions = [
                i
                for i, token in enumerate(flat.tolist())
                if token >= 0 and token in hits
            ]
            shape = (flat.numel(), *self.k.shape[1:])
            stage_k = torch.empty(shape, dtype=self.k.dtype, device=self.device)
            stage_v = torch.empty_like(stage_k)
            if hit_positions:
                output_rows = torch.tensor(hit_positions, device=self.device)
                hot_rows = torch.tensor(
                    [hits[int(flat[i])] for i in hit_positions], device=self.device
                )
                stage_k[output_rows] = self.hot_k[hot_rows]
                stage_v[output_rows] = self.hot_v[hot_rows]
            miss_slots = flat[miss_positions].long()
            cpu_k = torch.empty(
                (len(miss_positions), *self.k.shape[1:]),
                dtype=self.k.dtype,
                pin_memory=True,
            )
            cpu_v = torch.empty_like(cpu_k, pin_memory=True)
            if miss_positions:
                torch.index_select(self.k, 0, miss_slots, out=cpu_k)
                torch.index_select(self.v, 0, miss_slots, out=cpu_v)
            output_rows = torch.tensor(
                miss_positions, dtype=torch.long, device=self.device
            )
        row_bytes = self.k[0].numel() * self.k.element_size() * 2
        with span(batch, layer_id, "kv_h2d", bytes=len(miss_positions) * row_bytes):
            if miss_positions:
                stage_k[output_rows] = cpu_k.to(self.device, non_blocking=True)
                stage_v[output_rows] = cpu_v.to(self.device, non_blocking=True)
        with span(batch, layer_id, "kv_cache_update"):
            if self.capacity:
                # One batch is a simultaneous access. Break ties deterministically
                # by physical slot ID, never by future access or predicted scores.
                for token in sorted(set(selected)):
                    if token in self.resident:
                        self.resident.move_to_end(token)
                source_rows = {
                    int(token): i for i, token in enumerate(flat.tolist()) if token >= 0
                }
                updates = {}
                for token in sorted(set(selected) - set(hits)):
                    if not self.capacity:
                        break
                    if self.free:
                        hot = self.free.popleft()
                    else:
                        _, hot = self.resident.popitem(last=False)
                    # Capacity can be smaller than a batch's working set. Keep only
                    # the final write per destination to avoid duplicate-index races.
                    updates[hot] = source_rows[token]
                    self.resident[token] = hot
                if updates:
                    hot_rows = torch.tensor(
                        list(updates), dtype=torch.long, device=self.device
                    )
                    source = torch.tensor(
                        list(updates.values()), dtype=torch.long, device=self.device
                    )
                    self.hot_k[hot_rows] = stage_k[source]
                    self.hot_v[hot_rows] = stage_v[source]
        metadata = selection_metadata
        if metadata is not None:
            previous = getattr(batch, "_horizon_previous_selected", {})
            previous_physical = getattr(batch, "_horizon_previous_physical", {})
            steps = metadata["decode_steps"] or [None] * slots.shape[0]
            for i, rid in enumerate(metadata["request_ids"]):
                logical = logical_indices[i].tolist()
                s = {x for x in logical if x >= 0}
                d = {
                    x
                    for x, physical in zip(logical, slots[i].tolist())
                    if x >= 0 and physical not in hits
                }
                hint = previous.get(rid)
                record = {
                    "kind": "selection",
                    "layer_id": layer_id,
                    "request_id": rid,
                    "decode_step": steps[i],
                    "context_tokens": metadata["context_tokens"][i],
                    "active_batch": slots.shape[0],
                    "selected_count": len(s),
                    "miss_count": len(d),
                    "hit_count": len(s) - len(d),
                    "miss_bytes": len(d) * row_bytes,
                    "overlap_numerator": len(s & hint) if hint is not None else None,
                    "miss_coverage_numerator": len(d & hint)
                    if hint is not None
                    else None,
                    "hint_resident_count": sum(
                        physical in resident_before
                        for physical in previous_physical.get(rid, {}).values()
                    )
                    if hint is not None
                    else None,
                    "cache_capacity_tokens_per_layer": self.capacity,
                }
                if steps[i] is not None and steps[i] < int(
                    os.getenv("KEYE_HORIZON_SAVE_SET_STEPS", "2")
                ):
                    record.update(
                        selected=sorted(s),
                        miss=sorted(d),
                        hint=sorted(hint) if hint is not None else None,
                    )
                emit(record)
                previous[rid] = s
                previous_physical[rid] = {
                    x: physical
                    for x, physical in zip(logical, slots[i].tolist())
                    if x >= 0
                }
            batch._horizon_previous_selected = previous
            batch._horizon_previous_physical = previous_physical
        remapped = torch.arange(flat.numel(), dtype=torch.int32).reshape_as(slots)
        remapped[slots < 0] = -1
        return stage_k, stage_v, remapped.to(self.device)


class HostStore:
    def __init__(self, pool):
        mode = storage_mode()
        capacity = (
            int(os.getenv("KEYE_HORIZON_CACHE_TOKENS", "4096")) if mode == "lru" else 0
        )
        if capacity < 0 or (mode == "lru" and capacity == 0):
            raise ValueError("LRU cache token budget must be positive")
        ab = bool(os.getenv("KEYE_EVICTION_AB_DIR"))
        max_requests = int(os.getenv("KEYE_EVICTION_AB_MAX_REQUESTS", "1")) if ab else 1
        if ab and (capacity <= 0 or max_requests <= 0):
            raise ValueError("Eviction A/B requires positive per-request LRU and batch capacity")
        self.layers = [
            HostLayer(
                pool.size + pool.page_size,
                pool.head_num,
                pool.head_dim,
                pool.dtype,
                pool.device,
                capacity * max_requests,
            )
            for _ in range(pool.layer_num)
        ]
        self.start_layer = pool.start_layer
        self.device = pool.device
        if ab:
            from .eviction_study import LayerStudy

            stream = torch.cuda.Stream(device=pool.device)
            self.studies = [
                LayerStudy(layer, capacity, max_requests, stream) for layer in self.layers
            ]
        emit(
            {
                "kind": "allocation",
                "storage_mode": mode,
                "host_main_kv_bytes": sum(
                    layer.k.nbytes + layer.v.nbytes for layer in self.layers
                ),
                "persistent_hbm_main_kv_bytes": sum(
                    layer.hot_k.nbytes + layer.hot_v.nbytes for layer in self.layers
                ),
                "cache_capacity_tokens_per_layer": capacity,
                "logical_pool_tokens": pool.size,
                "layer_count": pool.layer_num,
                "gpu_name": torch.cuda.get_device_name(),
                "torch_version": torch.__version__,
            }
        )

    def write(self, layer, loc, k, v):
        self.layers[layer - self.start_layer].write(loc.long().cpu(), k, v)

    def prefill(self, layer, metadata, token_slots):
        host = self.layers[layer - self.start_layer]
        # Stage only this layer and actual request positions; other layers stay
        # exclusively in host memory, including throughout chunked prefill.
        logical = metadata.page_table_1.cpu()
        lengths = metadata.cache_seqlens_int32.cpu().tolist()
        selected = torch.cat(
            [logical[i, :length] for i, length in enumerate(lengths)]
        ).long()
        selected = torch.unique(selected)
        k = torch.index_select(host.k, 0, selected).to(self.device)
        v = torch.index_select(host.v, 0, selected).to(self.device)
        mapping = torch.full(
            (host.k.shape[0],), -1, dtype=torch.int32, device=self.device
        )
        mapping[selected.to(self.device)] = torch.arange(
            selected.numel(), dtype=torch.int32, device=self.device
        )
        slots = torch.where(
            token_slots >= 0, mapping[token_slots.clamp_min(0).long()], -1
        )
        return k, v, slots
