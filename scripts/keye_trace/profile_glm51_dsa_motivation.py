#!/usr/bin/env python3
"""Profile the GLM-5.1-shaped DSA motivation experiment on one GPU.

The benchmark represents one rank of an eight-way tensor-parallel GLM-5.1
decode replica.  It deliberately separates two concerns:

* ``saturation`` times only the active top-k MLA attention working set.  A
  FlashInfer MLA paged-decode kernel executes QK, softmax, and PV for eight
  local query heads, 2,048 selected tokens, a 576-wide compressed QK
  representation, and a 512-wide latent value representation.
* ``capacity`` allocates the exact logical bytes of all 78 layers' BF16 MLA KV
  and FP8 index-K state.  A configurable per-rank KV budget determines fit;
  points above the budget are reported as such and are never called OOM.

This is a GLM-5.1-shaped CUDA microbenchmark, not the official Hopper
FlashMLA/DSA production kernel and not an end-to-end serving benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch


REPO = Path(__file__).resolve().parents[2]
SEED = 20260819
LAYERS = 78
TP_SIZE = 8
GLOBAL_QUERY_HEADS = 64
LOCAL_QUERY_HEADS = GLOBAL_QUERY_HEADS // TP_SIZE
TOPK = 2048
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
VALUE_LATENT_DIM = KV_LORA_RANK
MLA_BYTES_PER_TOKEN_LAYER = QK_LATENT_DIM * 2
INDEX_BYTES_PER_TOKEN_LAYER = 128 + 4
TOTAL_BYTES_PER_TOKEN_LAYER = (
    MLA_BYTES_PER_TOKEN_LAYER + INDEX_BYTES_PER_TOKEN_LAYER
)
DEFAULT_BATCHES = (1, 2, 4, 8, 9, 16, 32, 48, 64, 96, 128)
DEFAULT_CONTEXTS = (32768, 65536, 131072, 200000)
DEFAULT_BUDGETS_GIB = (25.0, 30.0, 35.0, 40.0)

SATURATION_FIELDS = (
    "run_id", "repeat_id", "batch_size", "local_query_heads", "topk",
    "qk_latent_dim", "value_latent_dim", "executed_flops",
    "kernel_time_s", "achieved_tflops", "warmup_steps", "measured_steps",
    "status", "kernel_backend", "gpu_name", "gpu_index", "hbm_total_gib",
    "software_commit", "torch_version", "cuda_version", "flashinfer_version",
    "hostname", "seed", "error",
)
CAPACITY_FIELDS = (
    "run_id", "context_tokens", "batch_size", "budget_gib",
    "logical_mla_bytes", "logical_index_bytes", "logical_total_bytes",
    "logical_total_gib", "budget_status", "physical_probe",
    "allocation_status", "torch_allocated_delta_bytes",
    "allocator_overhead_bytes",
    "cuda_free_before_bytes", "cuda_free_after_bytes", "gpu_name",
    "gpu_index", "hbm_total_gib", "software_commit", "error",
)
MODEL_FIELDS = (
    "context_tokens", "budget_gib", "kv_gib_per_request",
    "max_batch_under_budget", "bytes_per_token_layer", "layers",
    "mla_bytes_per_token_layer", "index_bytes_per_token_layer",
)


def parse_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def device_metadata(device: torch.device) -> dict:
    props = torch.cuda.get_device_properties(device)
    _, total = torch.cuda.mem_get_info(device)
    return {
        "gpu_name": props.name,
        "hbm_total_gib": total / 2**30,
        "software_commit": git_revision(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "flashinfer_version": __import__("flashinfer").__version__,
        "hostname": platform.node(),
    }


def build_active_mla_attention(batch: int, device: torch.device):
    """Build QK-softmax-PV over only the selected 2K-token working set."""
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    # Each selected token becomes a one-token page.  The active DSA working set
    # is therefore exact and contiguous, while full-history residency remains
    # outside this performance-only experiment.
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="auto")
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)
    kv_indptr = qo_indptr * TOPK
    kv_indices = torch.arange(batch * TOPK, dtype=torch.int32, device=device)
    kv_lens = torch.full((batch,), TOPK, dtype=torch.int32, device=device)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        LOCAL_QUERY_HEADS,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        1,
        False,
        1.0 / math.sqrt(QK_LATENT_DIM),
        torch.bfloat16,
        torch.bfloat16,
    )
    q_nope = torch.randn(
        (batch, LOCAL_QUERY_HEADS, KV_LORA_RANK),
        dtype=torch.bfloat16,
        device=device,
    )
    q_rope = torch.randn(
        (batch, LOCAL_QUERY_HEADS, QK_ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    compressed_kv = torch.randn(
        (batch * TOPK, 1, KV_LORA_RANK),
        dtype=torch.bfloat16,
        device=device,
    )
    k_rope = torch.randn(
        (batch * TOPK, 1, QK_ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )

    def invoke():
        return wrapper.run(q_nope, q_rope, compressed_kv, k_rope)

    flops = (
        2
        * batch
        * LOCAL_QUERY_HEADS
        * TOPK
        * (QK_LATENT_DIM + VALUE_LATENT_DIM)
    )
    owned = (
        workspace, wrapper, qo_indptr, kv_indptr, kv_indices, kv_lens,
        q_nope, q_rope, compressed_kv, k_rope,
    )
    return invoke, flops, owned


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


def saturation_worker(args: argparse.Namespace) -> int:
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    common = {
        "run_id": args.run_id,
        "repeat_id": args.repeat_id,
        "batch_size": args.batch,
        "local_query_heads": LOCAL_QUERY_HEADS,
        "topk": TOPK,
        "qk_latent_dim": QK_LATENT_DIM,
        "value_latent_dim": VALUE_LATENT_DIM,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "gpu_index": args.device,
        "seed": SEED,
        "kernel_backend": "FlashInfer MLA paged decode (active top-k only)",
        "error": "",
        **device_metadata(device),
    }
    try:
        invoke, flops, owned = build_active_mla_attention(args.batch, device)
        graph, static_output = capture(invoke, args.warmup_steps)
        elapsed = time_graph(graph, args.measured_steps)
        row = {
            **common,
            "executed_flops": flops,
            "kernel_time_s": elapsed,
            "achieved_tflops": flops / elapsed / 1e12,
            "status": "measured",
        }
        del static_output, owned
    except torch.OutOfMemoryError as exc:
        row = {**common, "status": "OOM", "error": str(exc)}
    except Exception as exc:
        row = {**common, "status": "N/A", "error": repr(exc)}
    print(json.dumps(row))
    return 0


def ncu_worker(args: argparse.Namespace) -> int:
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    invoke, _, owned = build_active_mla_attention(args.batch, device)
    graph, static_output = capture(invoke, args.warmup_steps)
    torch.cuda.nvtx.range_push("glm51_dsa_active_attention")
    graph.replay()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    del static_output, owned
    return 0


def capacity_worker(args: argparse.Namespace) -> int:
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    metadata = device_metadata(device)
    tokens = args.context * args.batch
    mla_bytes = tokens * LAYERS * MLA_BYTES_PER_TOKEN_LAYER
    index_bytes = tokens * LAYERS * INDEX_BYTES_PER_TOKEN_LAYER
    total_bytes = mla_bytes + index_bytes
    budget_bytes = round(args.budget_gib * 2**30)
    within_budget = total_bytes <= budget_bytes
    row = {
        "run_id": args.run_id,
        "context_tokens": args.context,
        "batch_size": args.batch,
        "budget_gib": args.budget_gib,
        "logical_mla_bytes": mla_bytes,
        "logical_index_bytes": index_bytes,
        "logical_total_bytes": total_bytes,
        "logical_total_gib": total_bytes / 2**30,
        "budget_status": "within_budget" if within_budget else "over_budget",
        "physical_probe": within_budget,
        "allocation_status": "not_attempted_over_budget",
        "torch_allocated_delta_bytes": "",
        "allocator_overhead_bytes": "",
        "cuda_free_before_bytes": "",
        "cuda_free_after_bytes": "",
        "gpu_index": args.device,
        "error": "",
        **metadata,
    }
    if within_budget:
        try:
            torch.cuda.empty_cache()
            free_before, _ = torch.cuda.mem_get_info(device)
            allocated_before = torch.cuda.memory_allocated(device)
            # One layer retains semantic shapes.  The remaining 77 layers are
            # byte-accurate dummy allocations because capacity, not compute,
            # is under test.
            current_mla = torch.empty(
                (tokens, QK_LATENT_DIM), dtype=torch.bfloat16, device=device
            )
            current_index = torch.empty(
                (tokens, INDEX_BYTES_PER_TOKEN_LAYER),
                dtype=torch.uint8,
                device=device,
            )
            remaining = torch.empty(
                tokens * (LAYERS - 1) * TOTAL_BYTES_PER_TOKEN_LAYER,
                dtype=torch.uint8,
                device=device,
            )
            torch.cuda.synchronize()
            allocated_after = torch.cuda.memory_allocated(device)
            free_after, _ = torch.cuda.mem_get_info(device)
            delta = allocated_after - allocated_before
            if delta < total_bytes:
                raise RuntimeError(
                    f"allocation delta {delta} is smaller than logical {total_bytes}"
                )
            row.update({
                "allocation_status": "measured_fit",
                "torch_allocated_delta_bytes": delta,
                "allocator_overhead_bytes": delta - total_bytes,
                "cuda_free_before_bytes": free_before,
                "cuda_free_after_bytes": free_after,
            })
            del current_mla, current_index, remaining
        except torch.OutOfMemoryError as exc:
            row.update({"allocation_status": "measured_oom", "error": str(exc)})
        except Exception as exc:
            row.update({"allocation_status": "N/A", "error": repr(exc)})
    print(json.dumps(row))
    return 0


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run_json_worker(command: list[str]) -> dict:
    result = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "status": "N/A",
            "allocation_status": "N/A",
            "error": (result.stdout + result.stderr)[-8000:],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saturation-worker", action="store_true")
    parser.add_argument("--capacity-worker", action="store_true")
    parser.add_argument("--ncu-worker", action="store_true")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--context", type=int)
    parser.add_argument("--budget-gib", type=float, default=30.0)
    parser.add_argument("--repeat-id", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--measured-steps", type=int, default=500)
    parser.add_argument("--independent-repeats", type=int, default=5)
    parser.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)))
    parser.add_argument("--contexts", default=",".join(map(str, DEFAULT_CONTEXTS)))
    parser.add_argument(
        "--capacity-budgets-gib",
        default=",".join(map(str, DEFAULT_BUDGETS_GIB)),
    )
    parser.add_argument("--physical-probe-budget-gib", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=REPO / "data/motivation")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    if args.saturation_worker:
        if args.batch is None:
            parser.error("saturation worker requires --batch")
        return saturation_worker(args)
    if args.capacity_worker:
        if args.batch is None or args.context is None:
            parser.error("capacity worker requires --batch and --context")
        return capacity_worker(args)
    if args.ncu_worker:
        if args.batch is None:
            parser.error("NCU worker requires --batch")
        return ncu_worker(args)
    if args.warmup_steps < 100 or args.measured_steps < 500:
        parser.error("protocol requires >=100 warmups and >=500 measured replays")
    if args.independent_repeats < 5:
        parser.error("protocol requires >=5 independent repeats")

    run_id = args.run_id or datetime.now(timezone.utc).astimezone().strftime(
        "%Y-%m-%d_%H%M_glm51-dsa-motivation-v01"
    )
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    script = str(Path(__file__).resolve())

    saturation_rows: list[dict] = []
    for batch in parse_ints(args.batches):
        for repeat_id in range(args.independent_repeats):
            command = [
                sys.executable, script, "--saturation-worker",
                "--batch", str(batch), "--repeat-id", str(repeat_id),
                "--device", str(args.device), "--run-id", run_id,
                "--warmup-steps", str(args.warmup_steps),
                "--measured-steps", str(args.measured_steps),
            ]
            row = run_json_worker(command)
            saturation_rows.append(row)
            print(
                "saturation", batch, repeat_id, row.get("status"),
                row.get("achieved_tflops", ""), flush=True,
            )
    write_csv(output_dir / "saturation_raw.csv", SATURATION_FIELDS, saturation_rows)

    contexts = parse_ints(args.contexts)
    capacity_model_rows: list[dict] = []
    for budget in parse_floats(args.capacity_budgets_gib):
        budget_bytes = round(budget * 2**30)
        for context in contexts:
            per_request = context * LAYERS * TOTAL_BYTES_PER_TOKEN_LAYER
            capacity_model_rows.append({
                "context_tokens": context,
                "budget_gib": budget,
                "kv_gib_per_request": per_request / 2**30,
                "max_batch_under_budget": budget_bytes // per_request,
                "bytes_per_token_layer": TOTAL_BYTES_PER_TOKEN_LAYER,
                "layers": LAYERS,
                "mla_bytes_per_token_layer": MLA_BYTES_PER_TOKEN_LAYER,
                "index_bytes_per_token_layer": INDEX_BYTES_PER_TOKEN_LAYER,
            })
    write_csv(output_dir / "capacity_model.csv", MODEL_FIELDS, capacity_model_rows)

    # Probe exactly the matrix requested in data/motivation/figure.md.  Points
    # outside the nominal budget remain explicit over-budget records.
    probe_max_batch = {32768: 10, 65536: 6, 131072: 3, 200000: 2}
    capacity_rows: list[dict] = []
    for context in contexts:
        for batch in range(1, probe_max_batch[context] + 1):
            command = [
                sys.executable, script, "--capacity-worker",
                "--context", str(context), "--batch", str(batch),
                "--budget-gib", str(args.physical_probe_budget_gib),
                "--device", str(args.device), "--run-id", run_id,
            ]
            row = run_json_worker(command)
            capacity_rows.append(row)
            print(
                "capacity", context, batch, row.get("budget_status"),
                row.get("allocation_status"), flush=True,
            )
    write_csv(output_dir / "capacity_probe_raw.csv", CAPACITY_FIELDS, capacity_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "command": sys.argv,
        "repository_commit_before_new_script_commit": git_revision(),
        "scope": "GLM-5.1-shaped one-TP-rank CUDA microbenchmark",
        "not_claimed": [
            "official GLM-5.1 FlashMLA performance",
            "H200 absolute performance",
            "end-to-end decode throughput",
        ],
        "performance_kernel_scope": (
            "CUDA-graph replay of FlashInfer BF16 MLA paged decode over the "
            "contiguous active top-2048 working set; planning excluded"
        ),
        "flop_formula": "2*B*H_local*K*(D_qk+D_v)",
        "capacity_formula": "B*T*L*(1152 BF16 MLA bytes + 132 FP8 index bytes)",
        "model_revision": "zai-org/GLM-5.1@5b48738d05b580d2ed4d7277ce40febc8d7b9420",
        "tp_size": TP_SIZE,
        "local_query_heads": LOCAL_QUERY_HEADS,
        "topk": TOPK,
        "layers": LAYERS,
        "qk_latent_dim": QK_LATENT_DIM,
        "value_latent_dim": VALUE_LATENT_DIM,
        "index_layout_bytes_per_token_layer": INDEX_BYTES_PER_TOKEN_LAYER,
        "nominal_kv_budget_gib": args.physical_probe_budget_gib,
        "budget_sensitivity_gib": parse_floats(args.capacity_budgets_gib),
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "independent_repeats": args.independent_repeats,
        "environment": {
            **device_metadata(torch.device(f"cuda:{args.device}")),
            "gpu_index": args.device,
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
