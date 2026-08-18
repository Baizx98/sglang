#!/usr/bin/env python3
"""Profile GLM-5.2 dense MLA and DSA with Nsight Compute counters.

The controller launches one fresh Nsight Compute process per independent
repeat.  The worker marks exactly one attention operation with an NVTX range;
warm-up, allocation, and JIT compilation kernels are therefore excluded.
Metrics from the operation's stage-1 and reduction kernels are aggregated with
kernel duration as the weight.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from profile_glm52_dsa_utilization import (
    DEFAULT_BATCHES,
    DEFAULT_CONTEXTS,
    REPO,
    SEED,
    TOPK,
    build_kernels,
    ci95,
    context_label,
    git_revision,
    nvidia_smi_version,
)


NCU = Path("/home/bzx/local/cuda/bin/ncu")
PRIMARY = "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"
METRICS = (
    PRIMARY,
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "gpu__time_duration.sum",
)
CSV_FIELDS = (
    "model", "context_tokens", "context_label", "batch_size", "kernel",
    "status", "metric_name", "metric_value", "metric_unit", "repeat_count",
    "repeat_values", "ci95_low", "ci95_high", "sm_throughput_pct",
    "dram_throughput_pct", "achieved_occupancy_pct", "operation_duration_us",
    "aggregation", "warmup_count", "topk", "gpu_name", "gpu_index",
    "hbm_total_gib", "software_commit", "torch_version", "cuda_version",
    "driver_version", "profiler", "profiler_version", "seed", "error",
)


def parse_list(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def worker(context: int, batch: int, kernel: str, device: int, warmups: int) -> int:
    cuda_device = torch.device(f"cuda:{device}")
    torch.cuda.set_device(cuda_device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    kernels, owned = build_kernels(context, batch, cuda_device)
    invoke = kernels[kernel]
    for _ in range(warmups):
        invoke()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("figure1b_attention")
    invoke()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    del owned
    return 0


def ncu_version() -> str:
    output = subprocess.check_output([str(NCU), "--version"], text=True)
    return output.strip().splitlines()[-1]


def parse_ncu_csv(output: str) -> dict[str, float]:
    lines = output.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"')), None)
    if start is None:
        raise RuntimeError(f"Nsight Compute CSV header missing: {output[-1000:]}")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    # The second CSV row contains units rather than a kernel measurement.
    rows = [row for row in rows if row.get("Kernel Name")]
    if not rows:
        raise RuntimeError("Nsight Compute did not capture an attention kernel")
    durations = [float(row["gpu__time_duration.sum"]) for row in rows]
    total_duration = sum(durations)
    if total_duration <= 0:
        raise RuntimeError("Nsight Compute reported zero operation duration")
    result = {"operation_duration_us": total_duration / 1000.0}
    for metric in METRICS[:-1]:
        result[metric] = sum(float(row[metric]) * duration for row, duration in zip(rows, durations)) / total_duration
    result["kernel_count"] = float(len(rows))
    return result


def run_repeat(args: argparse.Namespace, context: int, batch: int, kernel: str) -> dict[str, float]:
    command = [
        "sudo", "-E", str(NCU), "--csv", "--page", "raw",
        "--disable-extra-suffixes", "--target-processes", "all", "--nvtx",
        "--nvtx-include", "figure1b_attention/", "--metrics", ",".join(METRICS),
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--context", str(context), "--batch", str(batch),
        "--kernel", kernel, "--device", str(args.device),
        "--warmups", str(args.warmups),
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"ncu exited {completed.returncode}: {completed.stdout[-2000:]}")
    return parse_ncu_csv(completed.stdout)


def profile_point(args: argparse.Namespace, context: int, batch: int, kernel: str) -> dict:
    device = torch.device(f"cuda:{args.device}")
    props = torch.cuda.get_device_properties(device)
    _, total_bytes = torch.cuda.mem_get_info(device)
    common = {
        "model": "zai-org/GLM-5.2", "context_tokens": context,
        "context_label": context_label(context), "batch_size": batch,
        "kernel": kernel, "metric_name": "NCU SM active cycles",
        "metric_unit": "%", "repeat_count": args.independent_repeats,
        "aggregation": "duration-weighted across attention kernels",
        "warmup_count": args.warmups, "topk": TOPK, "gpu_name": props.name,
        "gpu_index": args.device, "hbm_total_gib": total_bytes / 2**30,
        "software_commit": git_revision(), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "driver_version": nvidia_smi_version(),
        "profiler": "NVIDIA Nsight Compute", "profiler_version": ncu_version(),
        "seed": SEED, "error": "",
    }
    try:
        repeats = [run_repeat(args, context, batch, kernel)
                   for _ in range(args.independent_repeats)]
        values = [repeat[PRIMARY] for repeat in repeats]
        low, high = ci95(values)
        common.update({
            "status": "measured", "metric_value": statistics.fmean(values),
            "repeat_values": json.dumps(values), "ci95_low": low,
            "ci95_high": high,
            "sm_throughput_pct": statistics.fmean(
                x["sm__throughput.avg.pct_of_peak_sustained_elapsed"] for x in repeats),
            "dram_throughput_pct": statistics.fmean(
                x["dram__throughput.avg.pct_of_peak_sustained_elapsed"] for x in repeats),
            "achieved_occupancy_pct": statistics.fmean(
                x["sm__warps_active.avg.pct_of_peak_sustained_active"] for x in repeats),
            "operation_duration_us": statistics.fmean(
                x["operation_duration_us"] for x in repeats),
        })
    except Exception as exc:
        common.update({"status": "N/A", "metric_value": "", "repeat_values": "",
                       "ci95_low": "", "ci95_high": "", "error": repr(exc)})
    return common


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--context", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--kernel", choices=("Dense", "GLM-5.2 DSA"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--contexts", default=",".join(map(str, DEFAULT_CONTEXTS)))
    parser.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)))
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--independent-repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "data/gpu-attention-profile")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.worker:
        if args.context is None or args.batch is None or args.kernel is None:
            parser.error("worker requires --context, --batch, and --kernel")
        return worker(args.context, args.batch, args.kernel, args.device, args.warmups)
    if args.independent_repeats != 5:
        parser.error("Figure 1 protocol requires exactly five independent repeats")
    run_id = args.run_id or datetime.now(timezone.utc).astimezone().strftime(
        "%Y-%m-%d_%H%M_glm52-dsa-ncu-v01")
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
    csv_path = output_dir / "figure1b_glm52_gpu_ncu.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "manifest.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).astimezone().isoformat(),
        "command": sys.argv, "git_commit": git_revision(),
        "primary_metric": PRIMARY, "metrics": METRICS,
        "metric_semantics": (
            "Primary value is measured SM active cycles as a percentage of "
            "peak sustained elapsed cycles, duration-weighted over the two "
            "kernels in one attention operation."
        ),
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
