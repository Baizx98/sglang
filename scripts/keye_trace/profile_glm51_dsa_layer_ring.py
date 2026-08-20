#!/usr/bin/env python3
"""Profile GLM-5.1-shaped active DSA attention with a 78-layer KV ring.

This benchmark measures one TP=8 rank of BF16 MLA attention over an already
selected top-2048 working set.  Each of GLM-5.1's 78 attention layers owns a
different KV buffer, so a layer's reuse distance exceeds the L40S L2 cache even
at batch one.  The timed CUDA graph contains one full 78-layer decode step.

The result is a GPU kernel proxy.  It excludes index scoring, top-k selection,
gather, communication, and the rest of the transformer layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch


REPO = Path(__file__).resolve().parents[2]
SEED = 20260820
LAYERS = 78
TP_SIZE = 8
LOCAL_QUERY_HEADS = 8
TOPK = 2048
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
VALUE_LATENT_DIM = KV_LORA_RANK
PAGE_SIZE = 64
DTYPE = torch.bfloat16
DEFAULT_BATCHES = (1, 2, 4, 8, 9, 12, 16, 20, 24, 28, 32, 48, 64)

FIELDS = (
    "run_id", "repeat_id", "batch_size", "layers", "tp_size",
    "local_query_heads", "topk", "page_size", "qk_latent_dim",
    "value_latent_dim", "active_kv_bytes_per_layer",
    "layer_reuse_distance_bytes", "l2_cache_bytes", "executed_flops",
    "decode_step_gpu_time_s", "per_layer_gpu_time_s", "achieved_tflops",
    "warmup_steps", "measured_steps", "status", "kernel_backend",
    "dispatch_mode", "dispatch_target", "dispatch_group_sizes",
    "dispatch_kernels_per_decode_step", "plan_details",
    "plan_num_blks_x", "plan_num_blks_y", "plan_kv_len_limit",
    "plan_total_work_items", "plan_active_clusters",
    "plan_max_work_per_cluster", "plan_work_histogram", "gpu_name",
    "gpu_index", "hbm_total_gib", "hbm_free_before_gib",
    "allocated_tensor_gib", "power_limit_w", "default_power_limit_w",
    "application_sm_clock_mhz", "application_memory_clock_mhz",
    "sm_clock_mhz_before", "sm_clock_mhz_after",
    "memory_clock_mhz_before", "memory_clock_mhz_after",
    "temperature_c_before", "temperature_c_after", "power_w_before",
    "power_w_after", "software_commit", "torch_version", "cuda_version",
    "flashinfer_version", "hostname", "seed", "error",
)


def parse_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def telemetry(device_id: int) -> dict[str, float]:
    fields = (
        "clocks.sm,clocks.mem,temperature.gpu,power.draw"
    )
    output = subprocess.check_output(
        [
            "nvidia-smi", f"--id={device_id}", f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    sm, memory, temperature, power = (
        float(value.strip()) for value in output.split(",")
    )
    return {
        "sm_clock_mhz": sm,
        "memory_clock_mhz": memory,
        "temperature_c": temperature,
        "power_w": power,
    }


def kv_len_limit(batch: int, num_clusters: int) -> int:
    average = max(math.ceil(batch * TOPK / num_clusters), 1)
    if average <= 8:
        return 32
    if average <= 16:
        return 64
    if average <= 32:
        return 128
    if average <= 64:
        return 192
    return math.ceil(average / 256) * 256


def plan_metadata(wrapper, batch: int) -> dict[str, object]:
    plan = [int(value) for value in wrapper._plan_info]
    num_clusters = plan[1]
    work_indptr_offset = plan[15] // torch.int32.itemsize
    work_indptr = wrapper._int_workspace_buffer.view(torch.int32)[
        work_indptr_offset : work_indptr_offset + num_clusters + 1
    ].cpu().tolist()
    loads = [
        work_indptr[index + 1] - work_indptr[index]
        for index in range(num_clusters)
    ]
    histogram = Counter(loads)
    return {
        "plan_num_blks_x": plan[0],
        "plan_num_blks_y": plan[1],
        "plan_kv_len_limit": kv_len_limit(batch, num_clusters),
        "plan_total_work_items": sum(loads),
        "plan_active_clusters": sum(value > 0 for value in loads),
        "plan_max_work_per_cluster": max(loads),
        "plan_work_histogram": json.dumps(
            {str(key): histogram[key] for key in sorted(histogram)},
            separators=(",", ":"),
        ),
    }


def balanced_groups(batch: int, target: int) -> list[int]:
    if target <= 0 or batch <= target:
        return [batch]
    num_groups = math.ceil(batch / target)
    base, remainder = divmod(batch, num_groups)
    return [base + (index < remainder) for index in range(num_groups)]


def build_layer_ring(batch: int, device: torch.device, dispatch_target: int):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    if TOPK % PAGE_SIZE:
        raise ValueError("top-k must be divisible by page size")
    pages_per_request = TOPK // PAGE_SIZE
    group_sizes = balanced_groups(batch, dispatch_target)
    wrappers = {}
    plan_tensors = []
    for group_size in sorted(set(group_sizes)):
        workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
        qo_indptr = torch.arange(
            group_size + 1, dtype=torch.int32, device=device
        )
        kv_indptr = qo_indptr * pages_per_request
        kv_indices = torch.arange(
            group_size * pages_per_request, dtype=torch.int32, device=device
        )
        kv_lens = torch.full(
            (group_size,), TOPK, dtype=torch.int32, device=device
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            LOCAL_QUERY_HEADS,
            KV_LORA_RANK,
            QK_ROPE_HEAD_DIM,
            PAGE_SIZE,
            False,
            1.0 / math.sqrt(QK_LATENT_DIM),
            DTYPE,
            DTYPE,
        )
        wrappers[group_size] = (workspace, wrapper)
        plan_tensors.extend((workspace, qo_indptr, kv_indptr, kv_indices, kv_lens))
    q_nope = torch.randn(
        (batch, LOCAL_QUERY_HEADS, KV_LORA_RANK), dtype=DTYPE, device=device
    )
    q_rope = torch.randn(
        (batch, LOCAL_QUERY_HEADS, QK_ROPE_HEAD_DIM), dtype=DTYPE, device=device
    )
    compressed_kv = torch.empty(
        (LAYERS, batch, pages_per_request, PAGE_SIZE, KV_LORA_RANK),
        dtype=DTYPE,
        device=device,
    )
    k_rope = torch.empty(
        (LAYERS, batch, pages_per_request, PAGE_SIZE, QK_ROPE_HEAD_DIM),
        dtype=DTYPE,
        device=device,
    )
    # Initialize per layer to avoid very-large-tensor indexing limits while
    # ensuring every page is physically backed before timing begins.
    for layer in range(LAYERS):
        compressed_kv[layer].zero_()
        k_rope[layer].zero_()
    output = torch.empty(
        (batch, LOCAL_QUERY_HEADS, KV_LORA_RANK), dtype=DTYPE, device=device
    )

    group_views = []
    offset = 0
    for group_size in group_sizes:
        stop = offset + group_size
        _, wrapper = wrappers[group_size]
        layer_ckv = [
            compressed_kv[layer, offset:stop].reshape(
                group_size * pages_per_request, PAGE_SIZE, KV_LORA_RANK
            )
            for layer in range(LAYERS)
        ]
        layer_kpe = [
            k_rope[layer, offset:stop].reshape(
                group_size * pages_per_request, PAGE_SIZE, QK_ROPE_HEAD_DIM
            )
            for layer in range(LAYERS)
        ]
        group_views.append(
            (
                wrapper,
                q_nope[offset:stop],
                q_rope[offset:stop],
                output[offset:stop],
                layer_ckv,
                layer_kpe,
            )
        )
        offset = stop

    def invoke_step():
        for layer in range(LAYERS):
            for wrapper, group_q_nope, group_q_rope, group_output, layer_ckv, layer_kpe in group_views:
                wrapper.run(
                    group_q_nope,
                    group_q_rope,
                    layer_ckv[layer],
                    layer_kpe[layer],
                    out=group_output,
                )
        return output

    active_bytes_per_layer = batch * TOPK * QK_LATENT_DIM * DTYPE.itemsize
    flops_per_layer = (
        2
        * batch
        * LOCAL_QUERY_HEADS
        * TOPK
        * (QK_LATENT_DIM + VALUE_LATENT_DIM)
    )
    owned = (
        *plan_tensors, q_nope, q_rope, compressed_kv, k_rope, output,
    )
    allocated_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in owned
        if isinstance(tensor, torch.Tensor)
    )
    return (
        invoke_step,
        flops_per_layer * LAYERS,
        active_bytes_per_layer,
        allocated_bytes,
        wrappers,
        group_sizes,
        owned,
    )


def capture(fn: Callable[[], torch.Tensor], warmups: int):
    for _ in range(warmups):
        output = fn()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise RuntimeError("non-finite output before CUDA graph capture")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    return graph, static_output


def time_graph(graph: torch.cuda.CUDAGraph, steps: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0 / steps


def device_metadata(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    device_id = device.index if device.index is not None else torch.cuda.current_device()
    power_output = subprocess.check_output(
        [
            "nvidia-smi", f"--id={device_id}",
            "--query-gpu=power.limit,power.default_limit,"
            "clocks.applications.graphics,clocks.applications.memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    power_limit, default_power_limit, app_sm, app_memory = (
        float(value.strip()) for value in power_output.split(",")
    )
    return {
        "gpu_name": properties.name,
        "hbm_total_gib": total_bytes / 2**30,
        "hbm_free_before_gib": free_bytes / 2**30,
        "l2_cache_bytes": int(properties.L2_cache_size),
        "power_limit_w": power_limit,
        "default_power_limit_w": default_power_limit,
        "application_sm_clock_mhz": app_sm,
        "application_memory_clock_mhz": app_memory,
        "software_commit": git_revision(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "flashinfer_version": __import__("flashinfer").__version__,
        "hostname": platform.node(),
    }


def worker(args: argparse.Namespace) -> int:
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    metadata = device_metadata(device)
    common = {
        "run_id": args.run_id,
        "repeat_id": args.repeat_id,
        "batch_size": args.batch,
        "layers": LAYERS,
        "tp_size": TP_SIZE,
        "local_query_heads": LOCAL_QUERY_HEADS,
        "topk": TOPK,
        "page_size": PAGE_SIZE,
        "qk_latent_dim": QK_LATENT_DIM,
        "value_latent_dim": VALUE_LATENT_DIM,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "gpu_index": args.device,
        "kernel_backend": "FlashInfer FA2 MLA; 78-layer active-KV ring",
        "seed": SEED,
        "error": "",
        **metadata,
    }
    try:
        (
            invoke, flops, active_bytes, allocated_bytes, wrappers,
            group_sizes, owned,
        ) = build_layer_ring(args.batch, device, args.dispatch_target)
        torch.cuda.synchronize()
        plans = {
            group_size: plan_metadata(wrapper, group_size)
            for group_size, (_, wrapper) in wrappers.items()
        }
        plan_details = {
            str(group_size): plans[group_size] for group_size in sorted(plans)
        }
        plan = {
            "plan_num_blks_x": max(
                int(value["plan_num_blks_x"]) for value in plans.values()
            ),
            "plan_num_blks_y": max(
                int(value["plan_num_blks_y"]) for value in plans.values()
            ),
            "plan_kv_len_limit": json.dumps(
                {
                    str(size): plans[size]["plan_kv_len_limit"]
                    for size in sorted(plans)
                },
                separators=(",", ":"),
            ),
            "plan_total_work_items": sum(
                int(plans[size]["plan_total_work_items"])
                for size in group_sizes
            ),
            "plan_active_clusters": sum(
                int(plans[size]["plan_active_clusters"])
                for size in group_sizes
            ),
            "plan_max_work_per_cluster": max(
                int(value["plan_max_work_per_cluster"])
                for value in plans.values()
            ),
            "plan_work_histogram": json.dumps(
                {
                    str(size): plans[size]["plan_work_histogram"]
                    for size in sorted(plans)
                },
                separators=(",", ":"),
            ),
        }
        before = telemetry(args.device)
        graph, static_output = capture(invoke, args.warmup_steps)
        elapsed = time_graph(graph, args.measured_steps)
        after = telemetry(args.device)
        row = {
            **common,
            **plan,
            "active_kv_bytes_per_layer": active_bytes,
            "layer_reuse_distance_bytes": active_bytes * LAYERS,
            "allocated_tensor_gib": allocated_bytes / 2**30,
            "dispatch_mode": (
                "balanced microbatch" if len(group_sizes) > 1 else "single kernel"
            ),
            "dispatch_target": args.dispatch_target,
            "dispatch_group_sizes": json.dumps(group_sizes, separators=(",", ":")),
            "dispatch_kernels_per_decode_step": LAYERS * len(group_sizes),
            "plan_details": json.dumps(
                plan_details, sort_keys=True, separators=(",", ":")
            ),
            "executed_flops": flops,
            "decode_step_gpu_time_s": elapsed,
            "per_layer_gpu_time_s": elapsed / LAYERS,
            "achieved_tflops": flops / elapsed / 1e12,
            "status": "measured",
            "sm_clock_mhz_before": before["sm_clock_mhz"],
            "sm_clock_mhz_after": after["sm_clock_mhz"],
            "memory_clock_mhz_before": before["memory_clock_mhz"],
            "memory_clock_mhz_after": after["memory_clock_mhz"],
            "temperature_c_before": before["temperature_c"],
            "temperature_c_after": after["temperature_c"],
            "power_w_before": before["power_w"],
            "power_w_after": after["power_w"],
        }
        del static_output, graph, owned
    except torch.OutOfMemoryError as error:
        row = {**common, "status": "OOM", "error": str(error)}
    except Exception as error:
        row = {**common, "status": "N/A", "error": repr(error)}
    print(json.dumps(row, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--repeat-id", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)))
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--dispatch-target", type=int, default=16)
    parser.add_argument("--independent-repeats", type=int, default=5)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--output-root", type=Path, default=REPO / "data/motivation"
    )
    args = parser.parse_args()
    if args.worker:
        if args.batch is None or args.run_id is None:
            parser.error("worker requires --batch and --run-id")
        return worker(args)
    if args.warmup_steps < 20 or args.measured_steps < 100:
        parser.error("protocol requires at least 20 warmups and 100 measured steps")
    if not args.diagnostic and args.independent_repeats < 5:
        parser.error("formal protocol requires at least five independent repeats")
    if args.independent_repeats < 1:
        parser.error("independent repeats must be positive")

    run_id = args.run_id or datetime.now(timezone.utc).astimezone().strftime(
        "%Y-%m-%d_%H%M_glm51-dsa-layer-ring-v01"
    )
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    batches = parse_ints(args.batches)
    jobs = [
        (batch, repeat)
        for repeat in range(args.independent_repeats)
        for batch in batches
    ]
    random.Random(SEED).shuffle(jobs)
    rows = []
    for batch, repeat in jobs:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--batch",
            str(batch),
            "--repeat-id",
            str(repeat),
            "--device",
            str(args.device),
            "--run-id",
            run_id,
            "--warmup-steps",
            str(args.warmup_steps),
            "--measured-steps",
            str(args.measured_steps),
            "--dispatch-target",
            str(args.dispatch_target),
        ]
        result = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        try:
            row = json.loads(result.stdout.strip().splitlines()[-1])
        except Exception:
            row = {
                "run_id": run_id,
                "repeat_id": repeat,
                "batch_size": batch,
                "status": "N/A",
                "error": (result.stdout + result.stderr)[-8000:],
            }
        rows.append(row)
        print(
            f"B={batch} repeat={repeat} status={row.get('status')} "
            f"TFLOPS={row.get('achieved_tflops', '')}",
            flush=True,
        )

    rows.sort(key=lambda row: (int(row["batch_size"]), int(row["repeat_id"])))
    csv_path = output_dir / "saturation_layer_ring_raw.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "created": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "diagnostic" if args.diagnostic else "formal",
        "command": sys.argv,
        "repository_commit": git_revision(),
        "seed": SEED,
        "jobs_randomized": True,
        "dispatch": {
            "mode": "balanced microbatch",
            "target": args.dispatch_target,
            "definition": (
                "resident batch is split into the minimum number of nearly "
                "equal microbatches, each no larger than the target"
            ),
        },
        "scope": (
            "one TP=8 GLM-5.1-shaped rank; 78-layer BF16 active top-2048 "
            "FlashInfer FA2 MLA attention ring"
        ),
        "excluded": [
            "index scoring and top-k selection",
            "active-KV gather or prefetch",
            "full-history KV residency",
            "communication and non-attention layers",
        ],
        "flop_formula_per_layer": (
            "2*B*8*2048*((512+64)+512)"
        ),
        "timing": (
            "CUDA-event elapsed time for one CUDA-graph replay containing "
            "78 distinct-layer attention kernels"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if any(row.get("status") != "measured" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
