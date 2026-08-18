#!/usr/bin/env python3
"""Profile dense versus sparse Keye decode attention on SM8x GPUs.

This is a GPU kernel proxy for G2.5 Figure 1(b), not a GLM-5.2 FlashMLA
benchmark.  Both paths consume the same BF16 Keye GQA tensors.  Dense uses
SGLang's Triton decode kernel over the full context; sparse uses the validated
Keye exact-token DSA Triton kernel over top-k=2048 positions.

Each point runs in an isolated subprocess so an OOM or allocator fragmentation
does not contaminate later points.  CUDA events provide repeated kernel latency,
and a PyTorch Profiler Chrome trace is retained as an audit artifact.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


REPO = Path(__file__).resolve().parents[2]
SEED = 20260818
HQ = 32
HKV = 4
HEAD_DIM = 128
TOPK = 2048
DTYPE = torch.bfloat16
DEFAULT_CONTEXTS = (65536, 262144, 1048576)
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
CSV_FIELDS = (
    "model",
    "proxy_scope",
    "context_tokens",
    "context_label",
    "batch_size",
    "kernel",
    "status",
    "metric_name",
    "metric_value",
    "metric_unit",
    "latency_median_ms",
    "latency_p10_ms",
    "latency_p90_ms",
    "effective_tflops",
    "effective_bandwidth_gbps",
    "useful_flops",
    "logical_kv_read_bytes",
    "warmup_count",
    "repeat_count",
    "profiler_repeat_count",
    "profiler_device_time_ms_per_iter",
    "profiler_artifact",
    "gpu_name",
    "gpu_uuid",
    "compute_capability",
    "sm_count",
    "hbm_total_gib",
    "hbm_free_before_gib",
    "allocated_tensor_gib",
    "weight_dtype",
    "kv_dtype",
    "index_k_dtype",
    "tp_size",
    "dp_size",
    "software_commit",
    "torch_version",
    "cuda_version",
    "triton_version",
    "hostname",
    "seed",
    "error",
)


def load_keye_kernel():
    path = REPO / "python/sglang/srt/layers/attention/keye_dsa_sm80.py"
    spec = importlib.util.spec_from_file_location("keye_dsa_sm80_profile", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def context_label(context: int) -> str:
    if context == 1048576:
        return "1M"
    if context % 1024 == 0:
        return f"{context // 1024}K"
    return str(context)


def device_metadata(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    uuid = getattr(props, "uuid", "")
    return {
        "gpu_name": props.name,
        "gpu_uuid": str(uuid),
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": int(props.multi_processor_count),
        "hbm_total_gib": total_bytes / 2**30,
        "hbm_free_before_gib": free_bytes / 2**30,
    }


def dense_split_count(context: int, batch: int, sm_count: int, max_splits: int = 8) -> int:
    # Equivalent to the equal-length branch of TritonAttnBackend's dynamic
    # split heuristic for Keye's Hq/Hkv=8 grouped decode grid.
    extended_cores = int(sm_count * max(math.log2(context / 64.0), 1.0))
    token_grid = batch * math.ceil(HQ / min(16, HQ // HKV))
    target = min(math.ceil(extended_cores / token_grid), max_splits)
    target = max(target, 1)
    chunk = math.ceil(context / target)
    return max(1, math.ceil(context / chunk))


def event_latencies(fn: Callable[[], Any], warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]


def profiler_device_time(
    fn: Callable[[], Any], trace_path: Path, repeats: int
) -> float:
    # torch.profiler's Kineto trace reports zero CUDA time with this repository's
    # PyTorch 2.9.1+cu130 build.  The legacy autograd profiler still records the
    # CUDA-event-correlated device time, so keep it as an independent audit of
    # the repeated CUDA-event timing above.
    from torch.autograd.profiler import profile, record_function

    with profile(use_device="cuda") as prof:
        for _ in range(repeats):
            with record_function("attention_iteration"):
                fn()
        torch.cuda.synchronize()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    iterations = [
        event for event in prof.function_events
        if event.name == "attention_iteration"
    ]
    total_us = sum(float(event.device_time_total) for event in iterations)
    if total_us <= 0.0:
        raise RuntimeError(
            "CUDA profiler returned zero device time; profiler evidence is invalid"
        )
    return total_us / 1000.0 / repeats


def build_point(
    *, context: int, batch: int, kernel: str, device_id: int, warmups: int,
    repeats: int, profiler_repeats: int, trace_path: Path
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no GPU device is attached to this process")
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    metadata = device_metadata(device)
    if int(metadata["compute_capability"].split(".")[0]) < 8:
        raise RuntimeError("Keye sparse kernel requires NVIDIA Ampere or newer")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd,
    )

    keye = load_keye_kernel()
    workspace = keye.KeyeDSAWorkspace()
    total_tokens = batch * context
    q = torch.zeros((batch, HQ, HEAD_DIM), dtype=DTYPE, device=device)
    k_buffer = torch.empty((total_tokens, HKV, HEAD_DIM), dtype=DTYPE, device=device)
    v_buffer = torch.empty_like(k_buffer)
    # PyTorch 2.9.1+cu130's single fill kernel faults when one tensor reaches
    # 2^32 elements (for example 64K x B128 x Hkv4 x D128).  Chunked zeroing
    # keeps initialization outside the measured region and avoids that indexing
    # limit without changing the benchmark contents.
    init_chunk_tokens = 1 << 20
    for start in range(0, total_tokens, init_chunk_tokens):
        stop = min(start + init_chunk_tokens, total_tokens)
        k_buffer[start:stop].zero_()
        v_buffer[start:stop].zero_()
    output = torch.empty_like(q)

    if kernel == "dense":
        kv_indptr = torch.arange(
            batch + 1, dtype=torch.int32, device=device
        ) * context
        kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
        num_splits_value = dense_split_count(
            context, batch, int(metadata["sm_count"])
        )
        max_splits = 8
        num_kv_splits = torch.full(
            (batch,), num_splits_value, dtype=torch.int32, device=device
        )
        attn_logits = torch.empty(
            (batch, HQ, max_splits, HEAD_DIM), dtype=torch.float32, device=device
        )
        attn_lse = torch.empty(
            (batch, HQ, max_splits), dtype=torch.float32, device=device
        )

        def invoke():
            decode_attention_fwd(
                q,
                k_buffer,
                v_buffer,
                output,
                kv_indptr,
                kv_indices,
                attn_logits,
                attn_lse,
                num_kv_splits,
                max_splits,
                1.0 / math.sqrt(HEAD_DIM),
                1.0,
                1.0,
            )
            return output

        attended = context
        kernel_label = "dense GQA (Triton)"
        owned_extra_tensors = (
            kv_indptr, kv_indices, num_kv_splits, attn_logits, attn_lse
        )
    elif kernel == "sparse":
        sparse_q = q.unsqueeze(2)
        k_4d = k_buffer.view(batch, context, HKV, HEAD_DIM).permute(0, 2, 1, 3)
        v_4d = v_buffer.view(batch, context, HKV, HEAD_DIM).permute(0, 2, 1, 3)
        selected = min(TOPK, context)
        base = (
            torch.arange(selected, dtype=torch.int64, device=device) * context
            // selected
        )
        banks = []
        for bank_id in range(16):
            offset = (bank_id * max(context // (selected * 16), 1)) % context
            bank = ((base + offset) % context).to(torch.int32)
            banks.append(bank.view(1, 1, selected).expand(batch, 1, selected))
        topk_banks = torch.stack(banks, dim=0).contiguous()
        invocation = 0

        def invoke():
            nonlocal invocation
            topk_indices = topk_banks[invocation % topk_banks.shape[0]]
            invocation += 1
            return keye.keye_dsa_sparse_attention(
                sparse_q,
                k_4d,
                v_4d,
                topk_indices,
                sm_scale=1.0 / math.sqrt(HEAD_DIM),
                workspace=workspace,
            )

        attended = selected
        kernel_label = "sparse DSA top-2048 (Triton)"
        owned_extra_tensors = (topk_banks,)
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    allocated_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (q, k_buffer, v_buffer, output, *owned_extra_tensors)
    )
    latencies = event_latencies(invoke, warmups, repeats)
    median_ms = statistics.median(latencies)
    trace_device_ms = profiler_device_time(invoke, trace_path, profiler_repeats)
    useful_flops = 4 * batch * HQ * attended * HEAD_DIM
    logical_read_bytes = 2 * batch * HKV * attended * HEAD_DIM * DTYPE.itemsize
    effective_tflops = useful_flops / (median_ms / 1000.0) / 1e12
    effective_bandwidth = logical_read_bytes / (median_ms / 1000.0) / 1e9
    return {
        "model": "Keye-VL-2.0-30B-A3B",
        "proxy_scope": "GPU GQA kernel proxy for GLM-5.2 storage figure",
        "context_tokens": context,
        "context_label": context_label(context),
        "batch_size": batch,
        "kernel": kernel_label,
        "status": "measured",
        "metric_name": "Effective TFLOPS",
        "metric_value": effective_tflops,
        "metric_unit": "TFLOP/s",
        "latency_median_ms": median_ms,
        "latency_p10_ms": percentile(latencies, 0.10),
        "latency_p90_ms": percentile(latencies, 0.90),
        "effective_tflops": effective_tflops,
        "effective_bandwidth_gbps": effective_bandwidth,
        "useful_flops": useful_flops,
        "logical_kv_read_bytes": logical_read_bytes,
        "warmup_count": warmups,
        "repeat_count": repeats,
        "profiler_repeat_count": profiler_repeats,
        "profiler_device_time_ms_per_iter": trace_device_ms,
        "profiler_artifact": str(trace_path.resolve()),
        **metadata,
        "allocated_tensor_gib": allocated_bytes / 2**30,
        "weight_dtype": "not loaded",
        "kv_dtype": "BF16",
        "index_k_dtype": "not benchmarked",
        "tp_size": 1,
        "dp_size": 1,
        "software_commit": git_revision(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": __import__("triton").__version__,
        "hostname": platform.node(),
        "seed": SEED,
        "error": "",
    }


def failure_row(
    *, context: int, batch: int, kernel: str, status: str, error: str
) -> dict[str, Any]:
    return {
        "model": "Keye-VL-2.0-30B-A3B",
        "proxy_scope": "GPU GQA kernel proxy for GLM-5.2 storage figure",
        "context_tokens": context,
        "context_label": context_label(context),
        "batch_size": batch,
        "kernel": "dense GQA (Triton)" if kernel == "dense" else "sparse DSA top-2048 (Triton)",
        "status": status,
        "metric_name": "Effective TFLOPS",
        "metric_unit": "TFLOP/s",
        "weight_dtype": "not loaded",
        "kv_dtype": "BF16",
        "index_k_dtype": "not benchmarked",
        "tp_size": 1,
        "dp_size": 1,
        "software_commit": git_revision(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hostname": platform.node(),
        "seed": SEED,
        "error": error,
    }


def write_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")


def run_single(args: argparse.Namespace) -> int:
    assert args.context is not None and args.batch is not None and args.kernel is not None
    try:
        row = build_point(
            context=args.context,
            batch=args.batch,
            kernel=args.kernel,
            device_id=args.device,
            warmups=args.warmups,
            repeats=args.repeats,
            profiler_repeats=args.profiler_repeats,
            trace_path=args.trace_path,
        )
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        text = str(exc)
        is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in text.lower()
        row = failure_row(
            context=args.context,
            batch=args.batch,
            kernel=args.kernel,
            status="capacity_infeasible" if is_oom else "error",
            error=text,
        )
        write_json(args.output_json, row)
        return 0 if is_oom else 2
    write_json(args.output_json, row)
    return 0


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def run_sweep(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; attach an SM8x GPU before starting the sweep"
        )
    run_id = args.run_id or datetime.now(timezone.utc).astimezone().strftime(
        "%Y-%m-%d_%H%M_gpu-attention-saturation-v01"
    )
    output_dir = args.output_dir / run_id
    raw_dir = output_dir / "raw"
    trace_dir = output_dir / "profiler"
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for context in parse_int_list(args.contexts):
        for batch in parse_int_list(args.batches):
            for kernel in ("dense", "sparse"):
                stem = f"{context_label(context).lower()}_b{batch}_{kernel}"
                json_path = raw_dir / f"{stem}.json"
                trace_path = trace_dir / f"{stem}.pt.trace.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--single",
                    "--context", str(context),
                    "--batch", str(batch),
                    "--kernel", kernel,
                    "--device", str(args.device),
                    "--warmups", str(args.warmups),
                    "--repeats", str(args.repeats),
                    "--profiler-repeats", str(args.profiler_repeats),
                    "--trace-path", str(trace_path),
                    "--output-json", str(json_path),
                ]
                completed = subprocess.run(command, text=True, capture_output=True)
                if json_path.exists():
                    row = json.loads(json_path.read_text())
                else:
                    row = failure_row(
                        context=context,
                        batch=batch,
                        kernel=kernel,
                        status="error",
                        error=(completed.stderr or completed.stdout).strip(),
                    )
                    write_json(json_path, row)
                rows.append(row)
                print(
                    f"{context_label(context):>4} B={batch:<3} {kernel:<6} "
                    f"{row['status']} {row.get('metric_value', '')}"
                )
    csv_path = output_dir / "figure1b_gpu_attention_profile.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    manifest = {
        "run_id": run_id,
        "created": datetime.now(timezone.utc).astimezone().isoformat(),
        "command": sys.argv,
        "contexts": parse_int_list(args.contexts),
        "batches": parse_int_list(args.batches),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "profiler_repeats": args.profiler_repeats,
        "seed": SEED,
        "git_commit": git_revision(),
        "csv": str(csv_path.resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--context", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--kernel", choices=("dense", "sparse"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--profiler-repeats", type=int, default=5)
    parser.add_argument("--trace-path", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--contexts", default=",".join(str(v) for v in DEFAULT_CONTEXTS)
    )
    parser.add_argument(
        "--batches", default=",".join(str(v) for v in DEFAULT_BATCHES)
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "data/gpu-attention-profile"
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.single and (args.trace_path is None or args.output_json is None):
        parser.error("--single requires --trace-path and --output-json")
    return args


def main() -> int:
    args = parse_args()
    try:
        return run_single(args) if args.single else run_sweep(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
