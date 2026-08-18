#!/usr/bin/env python3
"""Measure Keye decode achieved TFLOPS for G2.5 Figure 1(b).

Dense measures full-history Keye-shape GQA decode attention.  DSA measures the
actual SM80 prepared-query index-score kernel, optimized top-k=2048 selection,
and Keye sparse-attention kernels in one CUDA-event window.  Indexer projection,
the rest of the transformer layer, communication, and scheduler gaps are out of
scope and are stated explicitly in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

os.environ.setdefault("KEYE_SM80_DSA", "1")

import torch


REPO = Path(__file__).resolve().parents[2]
SEED = 20260818
HQ = 32
HKV = 4
HEAD_DIM = 128
INDEX_HEADS = 16
INDEX_DIM = 64
TOPK = 2048
DTYPE = torch.bfloat16
CONTEXTS = (131072, 1048576)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
FIELDS = (
    "context_tokens", "batch_size", "attention_type", "run_id",
    "executed_flops", "kernel_time_s", "achieved_tflops", "peak_tflops",
    "status", "warmup_steps", "measured_steps", "flop_scope",
    "kernel_scope", "topk", "gpu_name", "gpu_index", "hbm_total_gb",
    "sm_clock_mhz_before", "sm_clock_mhz_after", "temperature_c_before",
    "temperature_c_after",
    "software_commit", "torch_version", "cuda_version", "triton_version",
    "hostname", "seed", "error",
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def parse_list(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def dense_split_count(context: int, batch: int, sm_count: int) -> int:
    extended_cores = int(sm_count * max(math.log2(context / 64.0), 1.0))
    token_grid = batch * math.ceil(HQ / min(16, HQ // HKV))
    target = max(1, min(math.ceil(extended_cores / token_grid), 8))
    chunk = math.ceil(context / target)
    return max(1, math.ceil(context / chunk))


def initialize_by_chunks(tensor: torch.Tensor, *, random: bool = False) -> None:
    chunk = 1 << 20
    for start in range(0, tensor.shape[0], chunk):
        view = tensor[start : min(start + chunk, tensor.shape[0])]
        view.normal_(mean=0.0, std=0.02) if random else view.zero_()


def time_operation(fn: Callable[[], object], warmups: int, steps: int) -> float:
    graph, static_output = capture_operation(fn, warmups)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        graph.replay()
    end.record()
    end.synchronize()
    # Keep graph-owned allocations alive until timing has completed.
    del static_output
    return start.elapsed_time(end) / 1000.0 / steps


def capture_operation(fn: Callable[[], object], warmups: int):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    return graph, static_output


def build_dense(context: int, batch: int, device: torch.device):
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd,
    )

    total_tokens = context * batch
    q = torch.randn((batch, HQ, HEAD_DIM), dtype=DTYPE, device=device) * 0.02
    k = torch.empty((total_tokens, HKV, HEAD_DIM), dtype=DTYPE, device=device)
    v = torch.empty_like(k)
    initialize_by_chunks(k)
    initialize_by_chunks(v)
    output = torch.empty_like(q)
    indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * context
    indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
    props = torch.cuda.get_device_properties(device)
    num_splits = dense_split_count(context, batch, props.multi_processor_count)
    splits = torch.full((batch,), num_splits, dtype=torch.int32, device=device)
    logits = torch.empty((batch, HQ, 8, HEAD_DIM), dtype=torch.float32, device=device)
    lse = torch.empty((batch, HQ, 8), dtype=torch.float32, device=device)

    def invoke():
        decode_attention_fwd(
            q, k, v, output, indptr, indices, logits, lse, splits, 8,
            1.0 / math.sqrt(HEAD_DIM), 1.0, 1.0,
        )
        return output

    useful_flops = 4 * batch * HQ * context * HEAD_DIM
    owned = (q, k, v, output, indptr, indices, splits, logits, lse)
    return invoke, useful_flops, owned


def build_dsa(context: int, batch: int, device: torch.device):
    from sgl_kernel import fast_topk_v2
    from sglang.srt.layers.attention.keye_dsa_sm80 import (
        KeyeDSAWorkspace,
        keye_dsa_sparse_attention,
    )
    from sglang.srt.layers.attention.keye_topk.ampere_indexer import (
        keye_indexer_score_paged,
    )

    total_tokens = context * batch
    query = torch.randn((batch, INDEX_HEADS, INDEX_DIM), dtype=DTYPE, device=device) * 0.02
    index_k = torch.empty((total_tokens, INDEX_DIM), dtype=DTYPE, device=device)
    initialize_by_chunks(index_k, random=True)
    page_table = (
        torch.arange(context, dtype=torch.int32, device=device).unsqueeze(0)
        + torch.arange(batch, dtype=torch.int32, device=device).unsqueeze(1) * context
    )
    weights = torch.randn((batch, INDEX_HEADS), dtype=torch.float32, device=device)
    seqlens = torch.full((batch,), context, dtype=torch.int32, device=device)

    q = torch.randn((batch, HQ, 1, HEAD_DIM), dtype=DTYPE, device=device) * 0.02
    k_flat = torch.empty((total_tokens, HKV, HEAD_DIM), dtype=DTYPE, device=device)
    v_flat = torch.empty_like(k_flat)
    initialize_by_chunks(k_flat)
    initialize_by_chunks(v_flat)
    k = k_flat.view(batch, context, HKV, HEAD_DIM).permute(0, 2, 1, 3)
    v = v_flat.view(batch, context, HKV, HEAD_DIM).permute(0, 2, 1, 3)
    workspace = KeyeDSAWorkspace()

    def invoke():
        scores = keye_indexer_score_paged(
            query, index_k, page_table, weights, seqlens,
            softmax_scale=1.0 / math.sqrt(INDEX_DIM),
        )
        topk_indices = fast_topk_v2(scores, seqlens, TOPK).unsqueeze(1)
        return keye_dsa_sparse_attention(
            q, k, v, topk_indices, sm_scale=1.0 / math.sqrt(HEAD_DIM),
            workspace=workspace,
        )

    # Verified useful-FLOP formula for the kernels in the timed region.
    # Top-k comparisons and address arithmetic are not floating-point ops, but
    # their device time remains in the denominator.
    indexer_flops = batch * context * (2 * INDEX_HEADS * INDEX_DIM + 2 * INDEX_HEADS - 1)
    attention_flops = 4 * batch * HQ * TOPK * HEAD_DIM
    useful_flops = indexer_flops + attention_flops
    owned = (query, index_k, page_table, weights, seqlens, q, k_flat, v_flat, k, v)
    return invoke, useful_flops, owned


def metadata(device: torch.device) -> dict:
    props = torch.cuda.get_device_properties(device)
    _, total = torch.cuda.mem_get_info(device)
    return {
        "gpu_name": props.name,
        "hbm_total_gb": total / 1e9,
        "software_commit": git_revision(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": __import__("triton").__version__,
        "hostname": platform.node(),
    }


def telemetry(device_id: int) -> tuple[int, int]:
    output = subprocess.check_output([
        "nvidia-smi", f"--id={device_id}",
        "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader,nounits",
    ], text=True).strip()
    clock, temperature = (int(value.strip()) for value in output.split(","))
    return clock, temperature


def worker(args: argparse.Namespace) -> int:
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    # Independent processes use identical data; repeat variance therefore
    # measures runtime noise rather than fast-top-k input-distribution changes.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    common = {
        "context_tokens": args.context,
        "batch_size": args.batch,
        "attention_type": args.attention_type,
        "run_id": f"{args.run_id}-r{args.repeat_index}",
        "peak_tflops": args.peak_tflops,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "topk": TOPK,
        "gpu_index": args.device,
        "seed": SEED,
        "error": "",
        **metadata(device),
    }
    try:
        if args.attention_type == "Dense attention":
            invoke, flops, owned = build_dense(args.context, args.batch, device)
            scope = "CUDA-graph replay: full-history GQA decode attention"
            flop_scope = "QK+PV useful FLOPs"
        else:
            invoke, flops, owned = build_dsa(args.context, args.batch, device)
            scope = (
                "CUDA-graph replay: prepared-query index score + "
                "fast_topk_v2 + sparse attention"
            )
            flop_scope = "index-score + sparse QK+PV useful FLOPs; top-k comparisons excluded"
        # Correctness/sanity gate before timing.
        output = invoke()
        torch.cuda.synchronize()
        if not torch.isfinite(output).all().item():
            raise RuntimeError("non-finite attention output")
        clock_before, temperature_before = telemetry(args.device)
        elapsed = time_operation(
            invoke, warmups=args.warmup_steps, steps=args.measured_steps
        )
        clock_after, temperature_after = telemetry(args.device)
        row = {
            **common, "executed_flops": flops, "kernel_time_s": elapsed,
            "achieved_tflops": flops / elapsed / 1e12, "status": "measured",
            "flop_scope": flop_scope, "kernel_scope": scope,
            "sm_clock_mhz_before": clock_before,
            "sm_clock_mhz_after": clock_after,
            "temperature_c_before": temperature_before,
            "temperature_c_after": temperature_after,
        }
        del owned
    except torch.OutOfMemoryError as exc:
        row = {**common, "status": "OOM", "error": str(exc)}
    except Exception as exc:
        row = {**common, "status": "N/A", "error": repr(exc)}
    print(json.dumps(row))
    return 0


def profiler_worker(args: argparse.Namespace) -> int:
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if args.attention_type == "Dense attention":
        invoke, _, owned = build_dense(args.context, args.batch, device)
    else:
        invoke, _, owned = build_dsa(args.context, args.batch, device)
    graph, static_output = capture_operation(invoke, args.warmup_steps)
    torch.cuda.nvtx.range_push("figure1b_decode")
    graph.replay()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    del static_output, owned
    return 0


def measure_peak(device_id: int, warmups: int = 20, steps: int = 100) -> float:
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    n = 8192
    a = torch.randn((n, n), dtype=DTYPE, device=device)
    b = torch.randn((n, n), dtype=DTYPE, device=device)
    out = torch.empty_like(a)

    def invoke():
        return torch.mm(a, b, out=out)

    elapsed = time_operation(invoke, warmups=warmups, steps=steps)
    return 2 * n**3 / elapsed / 1e12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--profiler-worker", action="store_true")
    parser.add_argument("--context", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--attention-type", choices=("Dense attention", "DSA"))
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--peak-tflops", type=float, default=0.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    parser.add_argument("--batches", default=",".join(map(str, BATCHES)))
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--independent-repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "data/gpu-attention-profile")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.worker or args.profiler_worker:
        if args.context is None or args.batch is None or args.attention_type is None:
            parser.error("worker requires context, batch, and attention type")
        return profiler_worker(args) if args.profiler_worker else worker(args)
    if args.warmup_steps < 20 or args.measured_steps < 100:
        parser.error("protocol requires at least 20 warmups and 100 measured steps")
    if args.independent_repeats < 5:
        parser.error("protocol requires at least five independent repeats")

    run_id = args.run_id or datetime.now(timezone.utc).astimezone().strftime(
        "%Y-%m-%d_%H%M_keye-decode-achieved-tflops-v01"
    )
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    peak = measure_peak(args.device)
    rows = []
    for context in parse_list(args.contexts):
        for batch in parse_list(args.batches):
            for attention_type in ("Dense attention", "DSA"):
                for repeat in range(args.independent_repeats):
                    command = [
                        sys.executable, str(Path(__file__).resolve()), "--worker",
                        "--context", str(context), "--batch", str(batch),
                        "--attention-type", attention_type,
                        "--repeat-index", str(repeat), "--peak-tflops", str(peak),
                        "--device", str(args.device), "--run-id", run_id,
                        "--warmup-steps", str(args.warmup_steps),
                        "--measured-steps", str(args.measured_steps),
                    ]
                    result = subprocess.run(
                        command, cwd=REPO, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, check=False,
                    )
                    try:
                        row = json.loads(result.stdout.strip().splitlines()[-1])
                    except Exception:
                        row = {
                            "context_tokens": context, "batch_size": batch,
                            "attention_type": attention_type,
                            "run_id": f"{run_id}-r{repeat}", "status": "N/A",
                            "error": (result.stdout + result.stderr)[-4000:],
                        }
                    rows.append(row)
                    print(context, batch, attention_type, repeat,
                          row.get("status"), row.get("achieved_tflops", ""), flush=True)
    csv_path = output_dir / "figure1b_keye_decode_achieved_tflops.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "manifest.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).astimezone().isoformat(),
        "command": sys.argv,
        "repository_commit": git_revision(),
        "gpu_bf16_available_peak_tflops": peak,
        "peak_protocol": "8192x8192 BF16 torch.mm; 20 warmups; 100 CUDA-event steps",
        "scope": {
            "dense": "full-history Keye GQA decode attention",
            "dsa": "prepared-query index score + fast_topk_v2 + Keye sparse attention",
            "excluded": [
                "indexer q/k projection, norm, RoPE, and Hadamard",
                "transformer projections and FFN/MoE", "communication",
                "scheduler and CPU launch gaps",
            ],
        },
        "flop_formula": {
            "dense": "4*B*32*T*128",
            "dsa": "B*T*(2*16*64+2*16-1) + 4*B*32*2048*128",
            "note": "top-k comparisons/address arithmetic are timed but are not FLOPs",
        },
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
