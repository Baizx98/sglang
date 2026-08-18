#!/usr/bin/env python3
"""Measure GLM-5.2-shape dense MLA and sparse DSA GPU utilization.

The benchmark uses the BF16 absorbed-MLA layout used by GLM-5.2: 64 query
heads, a shared 512-dimensional latent KV vector, a 64-dimensional RoPE key,
and sparse top-k=2048.  Dense attention reads every historical token through
SGLang's Triton decode kernel.  Sparse attention uses the same implementation
with a compact top-k token index, making the full-history versus sparse-gather
comparison differ only in the historical tokens read.

Utilization is NVIDIA's sampled ``utilization.gpu`` metric collected by
``nvidia-smi`` while the kernel is launched continuously.  It is a measured
GPU compute duty-cycle proxy, not occupancy, theoretical FLOPs, or target-NPU
utilization.  Five independent measurement windows are retained per point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch


REPO = Path(__file__).resolve().parents[2]
SEED = 20260818
NUM_HEADS = 64
KV_LORA_RANK = 512
ROPE_DIM = 64
QK_DIM = KV_LORA_RANK + ROPE_DIM
TOPK = 2048
DTYPE = torch.bfloat16
DEFAULT_CONTEXTS = (65536, 262144, 524288, 1048576)
DEFAULT_BATCHES = (1, 8, 32, 64)
CSV_FIELDS = (
    "model", "context_tokens", "context_label", "batch_size", "kernel",
    "status", "metric_name", "metric_value", "metric_unit", "repeat_count",
    "repeat_values", "ci95_low", "ci95_high", "sample_interval_ms",
    "window_seconds", "warmup_count", "topk", "num_query_heads",
    "kv_lora_rank", "qk_rope_head_dim", "weight_dtype", "kv_dtype",
    "gpu_name", "gpu_index", "hbm_total_gib", "software_commit",
    "torch_version", "cuda_version", "nvidia_smi_version", "profiler",
    "seed", "error",
)


def context_label(context: int) -> str:
    return "1M" if context == 1048576 else f"{context // 1024}K"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def nvidia_smi_version() -> str:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    )
    return output.splitlines()[0].strip()


def ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean
    # Student-t critical value for n=5.  The CLI fixes five independent windows.
    critical = 2.776 if len(values) == 5 else 1.96
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return max(0.0, mean - half), min(100.0, mean + half)


def dense_split_count(context: int, batch: int, sm_count: int) -> int:
    extended_cores = int(sm_count * max(math.log2(context / 64.0), 1.0))
    token_grid = batch * math.ceil(NUM_HEADS / 16)
    target = min(math.ceil(extended_cores / token_grid), 8)
    chunk = math.ceil(context / max(target, 1))
    return max(1, math.ceil(context / chunk))


def collect_utilization(
    invoke: Callable[[], object], *, gpu_index: int, window_seconds: float,
    sample_interval_ms: int,
) -> tuple[float, list[int], int]:
    command = [
        "nvidia-smi", f"--id={gpu_index}",
        "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
        f"--loop-ms={sample_interval_ms}",
    ]
    monitor = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        bufsize=1,
    )
    samples: list[int] = []

    def reader() -> None:
        assert monitor.stdout is not None
        for line in monitor.stdout:
            value = line.strip()
            if value.isdigit():
                samples.append(int(value))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    deadline = time.monotonic() + window_seconds
    launches = 0
    while time.monotonic() < deadline:
        invoke()
        launches += 1
    torch.cuda.synchronize()
    monitor.terminate()
    try:
        monitor.wait(timeout=3)
    except subprocess.TimeoutExpired:
        monitor.kill()
        monitor.wait(timeout=3)
    thread.join(timeout=1)
    # The first sample can precede the first launch; discard it consistently.
    usable = samples[1:] if len(samples) > 1 else samples
    if not usable:
        stderr = monitor.stderr.read().strip() if monitor.stderr else ""
        raise RuntimeError(f"nvidia-smi returned no utilization samples: {stderr}")
    return statistics.fmean(usable), usable, launches


def build_kernels(context: int, batch: int, device: torch.device):
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd,
    )

    total_tokens = context * batch
    q = torch.zeros((batch, NUM_HEADS, QK_DIM), dtype=DTYPE, device=device)
    kv = torch.empty((total_tokens, 1, QK_DIM), dtype=DTYPE, device=device)
    for start in range(0, total_tokens, 1 << 20):
        kv[start : min(start + (1 << 20), total_tokens)].zero_()
    v = kv[..., :KV_LORA_RANK]
    output = torch.empty((batch, NUM_HEADS, KV_LORA_RANK), dtype=DTYPE, device=device)

    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * context
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
    props = torch.cuda.get_device_properties(device)
    splits = dense_split_count(context, batch, props.multi_processor_count)
    num_kv_splits = torch.full((batch,), splits, dtype=torch.int32, device=device)
    attn_logits = torch.empty(
        (batch, NUM_HEADS, 8, KV_LORA_RANK), dtype=torch.float32, device=device
    )
    attn_lse = torch.empty(
        (batch, NUM_HEADS, 8), dtype=torch.float32, device=device
    )

    def dense() -> torch.Tensor:
        decode_attention_fwd(
            q, kv, v, output, kv_indptr, kv_indices, attn_logits, attn_lse,
            num_kv_splits, 8, 1.0 / math.sqrt(QK_DIM), 1.0, 1.0,
        )
        return output

    selected = min(TOPK, context)
    base = torch.arange(selected, dtype=torch.int64, device=device) * context // selected
    sparse_indices = torch.cat(
        [base.to(torch.int32) + request * context for request in range(batch)]
    )
    sparse_indptr = torch.arange(
        batch + 1, dtype=torch.int32, device=device
    ) * selected
    sparse_splits_value = dense_split_count(
        selected, batch, props.multi_processor_count
    )
    sparse_num_splits = torch.full(
        (batch,), sparse_splits_value, dtype=torch.int32, device=device
    )
    sparse_logits = torch.empty_like(attn_logits)
    sparse_lse = torch.empty_like(attn_lse)
    sparse_invocation = 0

    def sparse() -> torch.Tensor:
        nonlocal sparse_invocation
        sparse_invocation += 1
        decode_attention_fwd(
            q, kv, v, output, sparse_indptr, sparse_indices,
            sparse_logits, sparse_lse, sparse_num_splits, 8,
            1.0 / math.sqrt(QK_DIM), 1.0, 1.0,
        )
        return output

    # Keep all owned tensors alive for the lifetime of the closures.
    _owned = (q, kv, v, output, kv_indptr, kv_indices, num_kv_splits,
              attn_logits, attn_lse, sparse_indices, sparse_indptr,
              sparse_num_splits, sparse_logits, sparse_lse)
    return {"Dense": dense, "GLM-5.2 DSA": sparse}, _owned


def profile_point(args: argparse.Namespace, context: int, batch: int, kernel: str) -> dict:
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    del free_bytes
    common = {
        "model": "zai-org/GLM-5.2", "context_tokens": context,
        "context_label": context_label(context), "batch_size": batch,
        "kernel": kernel, "metric_name": "NVML GPU compute utilization",
        "metric_unit": "%", "repeat_count": args.independent_repeats,
        "sample_interval_ms": args.sample_interval_ms,
        "window_seconds": args.window_seconds, "warmup_count": args.warmups,
        "topk": TOPK, "num_query_heads": NUM_HEADS,
        "kv_lora_rank": KV_LORA_RANK, "qk_rope_head_dim": ROPE_DIM,
        "weight_dtype": "not loaded", "kv_dtype": "BF16",
        "gpu_name": props.name, "gpu_index": args.device,
        "hbm_total_gib": total_bytes / 2**30, "software_commit": git_revision(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "nvidia_smi_version": nvidia_smi_version(),
        "profiler": "nvidia-smi/NVML utilization.gpu",
        "seed": SEED, "error": "",
    }
    try:
        kernels, owned = build_kernels(context, batch, device)
        invoke = kernels[kernel]
        for _ in range(args.warmups):
            invoke()
        torch.cuda.synchronize()
        values: list[float] = []
        raw_samples: list[list[int]] = []
        launches: list[int] = []
        for _ in range(args.independent_repeats):
            value, samples, count = collect_utilization(
                invoke, gpu_index=args.nvidia_smi_gpu,
                window_seconds=args.window_seconds,
                sample_interval_ms=args.sample_interval_ms,
            )
            values.append(value)
            raw_samples.append(samples)
            launches.append(count)
            time.sleep(args.repeat_gap_seconds)
        low, high = ci95(values)
        common.update({
            "status": "measured", "metric_value": statistics.fmean(values),
            "repeat_values": json.dumps(values), "ci95_low": low,
            "ci95_high": high,
            "error": json.dumps({"samples": raw_samples, "launches": launches}),
        })
        del owned
        return common
    except torch.OutOfMemoryError as exc:
        common.update({"status": "OOM", "metric_value": "", "repeat_values": "",
                       "ci95_low": "", "ci95_high": "", "error": str(exc)})
        return common
    except Exception as exc:
        common.update({"status": "N/A", "metric_value": "", "repeat_values": "",
                       "ci95_low": "", "ci95_high": "", "error": repr(exc)})
        return common


def parse_list(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nvidia-smi-gpu", type=int, default=0)
    parser.add_argument("--contexts", default=",".join(map(str, DEFAULT_CONTEXTS)))
    parser.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)))
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--independent-repeats", type=int, default=5)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--repeat-gap-seconds", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "data/gpu-attention-profile")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.independent_repeats != 5:
        parser.error("Figure 1 protocol requires exactly five independent repeats")

    run_id = args.run_id or datetime.now(timezone.utc).astimezone().strftime(
        "%Y-%m-%d_%H%M_glm52-dsa-utilization-v01"
    )
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for batch in parse_list(args.batches):
        for context in parse_list(args.contexts):
            for kernel in ("Dense", "GLM-5.2 DSA"):
                row = profile_point(args, context, batch, kernel)
                rows.append(row)
                print(context_label(context), f"B={batch}", kernel,
                      row["status"], row["metric_value"], flush=True)
                torch.cuda.empty_cache()
    csv_path = output_dir / "figure1b_glm52_gpu_utilization.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "manifest.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).astimezone().isoformat(),
        "command": sys.argv, "git_commit": git_revision(),
        "csv": str(csv_path.resolve()), "metric_limit": (
            "NVML utilization.gpu is a sampled device compute duty cycle; "
            "it is not per-SM active cycles and not target-NPU utilization."
        ),
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
