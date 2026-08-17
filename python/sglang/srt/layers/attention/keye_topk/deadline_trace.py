"""Low-overhead CUDA-event tracing for DSA prefetch deadlines.

The trace has two independent clocks:

* ``previous_step_same_layer`` pairs consecutive top-k-ready events from the
  same request and layer.
* ``cross_layer_candidate`` pairs lookahead-candidate-ready with the target
  layer's attention-consume point in the same decode step.

Events are queried before calling ``elapsed_time`` and JSONL writes happen on a
background thread.  The model hot path never synchronizes the device.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode


@dataclass(frozen=True)
class _Config:
    output_dir: Path
    layers: frozenset[int]
    rid_prefix: str
    interval_limit: int
    pending_limit: int


@dataclass
class _EventPoint:
    event: torch.cuda.Event
    decode_step: int


@dataclass
class _PendingInterval:
    kind: str
    request_id: str
    layer_id: int
    producer_decode_step: int
    consumer_decode_step: int
    start: torch.cuda.Event
    end: torch.cuda.Event
    context_tokens: Optional[int]
    batch_size: int


_UNSET = object()
_config: object | Optional[_Config] = _UNSET
_same_layer_previous: dict[tuple[str, int], _EventPoint] = {}
_cross_layer_ready: dict[tuple[str, int], _EventPoint] = {}
_decode_steps: dict[tuple[str, str, int], int] = {}
_interval_counts: dict[tuple[str, str, int], int] = {}
_pending: deque[_PendingInterval] = deque()
_writer_queue: Optional[queue.SimpleQueue[Optional[dict[str, Any]]]] = None
_writer_thread: Optional[threading.Thread] = None
_writer_rank: Optional[int] = None
_writer_shutdown = False


def _parse_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _get_config() -> Optional[_Config]:
    global _config
    if _config is not _UNSET:
        return _config  # type: ignore[return-value]

    output = os.getenv("KEYE_DEADLINE_TRACE_DIR", "").strip()
    if not output:
        _config = None
        return None

    raw_layers = os.getenv("KEYE_DEADLINE_TRACE_LAYERS", "all").strip()
    if raw_layers.lower() == "all":
        layers = frozenset(range(48))
    else:
        try:
            layers = frozenset(
                int(value.strip())
                for value in raw_layers.split(",")
                if value.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "KEYE_DEADLINE_TRACE_LAYERS must be 'all' or comma-separated integers"
            ) from exc
        if not layers or any(layer < 0 for layer in layers):
            raise ValueError(
                "KEYE_DEADLINE_TRACE_LAYERS must contain non-negative layer IDs"
            )

    _config = _Config(
        output_dir=Path(output),
        layers=layers,
        rid_prefix=os.getenv("KEYE_DEADLINE_TRACE_RID_PREFIX", "").strip(),
        interval_limit=_parse_positive_int("KEYE_DEADLINE_TRACE_INTERVALS", 32),
        pending_limit=_parse_positive_int("KEYE_DEADLINE_TRACE_PENDING_LIMIT", 8192),
    )
    return _config


def _global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))


def _writer_loop(
    output_path: Path,
    records: queue.SimpleQueue[Optional[dict[str, Any]]],
) -> None:
    with output_path.open("a", encoding="utf-8", buffering=1) as output:
        while True:
            record = records.get()
            if record is None:
                return
            output.write(json.dumps(record, separators=(",", ":")) + "\n")


def _ensure_writer(config: _Config) -> None:
    global _writer_queue, _writer_rank, _writer_thread
    if _writer_queue is not None:
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rank = _global_rank()
    device_index = torch.cuda.current_device()
    metadata = {
        "schema_version": 1,
        "created_unix_s": time.time(),
        "global_rank": rank,
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device_index),
        "interval_kinds": [
            "previous_step_same_layer",
            "cross_layer_candidate",
        ],
        "layers": sorted(config.layers),
        "rid_prefix": config.rid_prefix,
        "interval_limit_per_request_layer_kind": config.interval_limit,
        "timing_contract": {
            "previous_step_same_layer": (
                "top-k ready at (t-1,l) to top-k ready/attention consume at (t,l)"
            ),
            "cross_layer_candidate": (
                "lookahead candidate ready to target-layer attention consume"
            ),
            "device_synchronize_in_hot_path": False,
            "writer": "background JSONL thread",
        },
    }
    metadata_path = config.output_dir / f"deadline_metadata_rank_{rank:02d}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    records: queue.SimpleQueue[Optional[dict[str, Any]]] = queue.SimpleQueue()
    output_path = config.output_dir / f"deadline_events_rank_{rank:02d}.jsonl"
    writer = threading.Thread(
        target=_writer_loop,
        args=(output_path, records),
        name=f"keye-deadline-writer-rank-{rank}",
        daemon=True,
    )
    writer.start()
    _writer_queue = records
    _writer_thread = writer
    _writer_rank = rank


def _shutdown_writer() -> None:
    global _writer_shutdown
    if _writer_shutdown or _writer_queue is None:
        return
    _writer_shutdown = True
    _writer_queue.put(None)
    if _writer_thread is not None:
        _writer_thread.join(timeout=5)


atexit.register(_shutdown_writer)


def _request_rows(
    forward_batch: ForwardBatch,
    config: _Config,
) -> list[tuple[int, str, Optional[int]]]:
    request_ids = list(getattr(forward_batch, "rids", None) or [])
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    rows: list[tuple[int, str, Optional[int]]] = []
    for row_index, request_id in enumerate(request_ids):
        if config.rid_prefix and not request_id.startswith(config.rid_prefix):
            continue
        context_tokens = None
        if isinstance(seq_lens_cpu, torch.Tensor) and seq_lens_cpu.device.type == "cpu":
            if row_index < seq_lens_cpu.numel():
                context_tokens = int(seq_lens_cpu[row_index])
        rows.append((row_index, request_id, context_tokens))
    return rows


def _new_event() -> torch.cuda.Event:
    if get_is_capture_mode():
        raise RuntimeError(
            "KEYE deadline tracing requires --disable-cuda-graph; captured events "
            "do not represent per-step production and consumption points"
        )
    event = torch.cuda.Event(enable_timing=True)
    event.record(torch.cuda.current_stream())
    return event


def _append_pending(config: _Config, interval: _PendingInterval) -> None:
    if len(_pending) >= config.pending_limit:
        raise RuntimeError(
            "KEYE deadline trace pending-event ring is full; increase "
            "KEYE_DEADLINE_TRACE_PENDING_LIMIT or reduce traced layers/requests"
        )
    _pending.append(interval)


def _drain_ready(config: _Config) -> None:
    if not _pending:
        return
    _ensure_writer(config)
    assert _writer_queue is not None
    for _ in range(len(_pending)):
        interval = _pending.popleft()
        if not interval.end.query():
            _pending.append(interval)
            continue
        elapsed_ms = float(interval.start.elapsed_time(interval.end))
        _writer_queue.put(
            {
                "schema_version": 1,
                "timestamp_ns": time.time_ns(),
                "kind": interval.kind,
                "request_id": interval.request_id,
                "layer_id": interval.layer_id,
                "producer_decode_step": interval.producer_decode_step,
                "consumer_decode_step": interval.consumer_decode_step,
                "interval_ms": elapsed_ms,
                "context_tokens": interval.context_tokens,
                "batch_size": interval.batch_size,
                "global_rank": _writer_rank,
                "device_index": torch.cuda.current_device(),
            }
        )


def record_same_layer_topk_ready(
    forward_batch: ForwardBatch,
    layer_id: int,
) -> None:
    """Record the top-k-ready point and pair it with the previous decode step."""
    config = _get_config()
    if (
        config is None
        or layer_id not in config.layers
        or not forward_batch.forward_mode.is_decode()
    ):
        return
    rows = _request_rows(forward_batch, config)
    batch_size = int(getattr(forward_batch, "batch_size", len(rows)))
    eligible = [
        row
        for row in rows
        if _interval_counts.get(
            ("previous_step_same_layer", row[1], layer_id), 0
        )
        < config.interval_limit
    ]
    if not eligible:
        _drain_ready(config)
        return

    _ensure_writer(config)
    current = _new_event()
    for _, request_id, context_tokens in eligible:
        step_key = ("previous_step_same_layer", request_id, layer_id)
        decode_step = _decode_steps.get(step_key, 0)
        _decode_steps[step_key] = decode_step + 1
        state_key = (request_id, layer_id)
        previous = _same_layer_previous.get(state_key)
        if previous is not None:
            _append_pending(
                config,
                _PendingInterval(
                    kind="previous_step_same_layer",
                    request_id=request_id,
                    layer_id=layer_id,
                    producer_decode_step=previous.decode_step,
                    consumer_decode_step=decode_step,
                    start=previous.event,
                    end=current,
                    context_tokens=context_tokens,
                    batch_size=batch_size,
                ),
            )
            _interval_counts[step_key] = _interval_counts.get(step_key, 0) + 1
        if _interval_counts.get(step_key, 0) < config.interval_limit:
            _same_layer_previous[state_key] = _EventPoint(current, decode_step)
        else:
            _same_layer_previous.pop(state_key, None)
    _drain_ready(config)


def record_cross_layer_candidate_ready(
    forward_batch: ForwardBatch,
    layer_id: int,
) -> None:
    """Record when the target layer's lookahead candidate becomes available."""
    config = _get_config()
    if (
        config is None
        or layer_id not in config.layers
        or not forward_batch.forward_mode.is_decode()
    ):
        return
    rows = _request_rows(forward_batch, config)
    eligible = [
        row
        for row in rows
        if _interval_counts.get(("cross_layer_candidate", row[1], layer_id), 0)
        < config.interval_limit
    ]
    if not eligible:
        _drain_ready(config)
        return

    _ensure_writer(config)
    current = _new_event()
    for _, request_id, _ in eligible:
        step_key = ("cross_layer_candidate", request_id, layer_id)
        decode_step = _decode_steps.get(step_key, 0)
        _decode_steps[step_key] = decode_step + 1
        state_key = (request_id, layer_id)
        if state_key in _cross_layer_ready:
            raise RuntimeError(
                f"unconsumed cross-layer deadline event for {state_key}"
            )
        _cross_layer_ready[state_key] = _EventPoint(current, decode_step)
    _drain_ready(config)


def record_cross_layer_attention_consume(
    forward_batch: ForwardBatch,
    layer_id: int,
) -> None:
    """Pair candidate-ready events with the attention-consume point."""
    config = _get_config()
    if (
        config is None
        or layer_id not in config.layers
        or not forward_batch.forward_mode.is_decode()
    ):
        return
    rows = _request_rows(forward_batch, config)
    batch_size = int(getattr(forward_batch, "batch_size", len(rows)))
    ready_rows = [
        row for row in rows if (row[1], layer_id) in _cross_layer_ready
    ]
    if not ready_rows:
        _drain_ready(config)
        return

    _ensure_writer(config)
    consume = _new_event()
    for _, request_id, context_tokens in ready_rows:
        state_key = (request_id, layer_id)
        ready = _cross_layer_ready.pop(state_key)
        step_key = ("cross_layer_candidate", request_id, layer_id)
        _append_pending(
            config,
            _PendingInterval(
                kind="cross_layer_candidate",
                request_id=request_id,
                layer_id=layer_id,
                producer_decode_step=ready.decode_step,
                consumer_decode_step=ready.decode_step,
                start=ready.event,
                end=consume,
                context_tokens=context_tokens,
                batch_size=batch_size,
            ),
        )
        _interval_counts[step_key] = _interval_counts.get(step_key, 0) + 1
    _drain_ready(config)
