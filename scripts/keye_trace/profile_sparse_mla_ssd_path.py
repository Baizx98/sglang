#!/usr/bin/env python3
"""Measure SSD-miss exposure in a 78-layer GLM-5.1-shaped sparse-MLA path.

For every layer, the benchmark waits for all 4-KiB O_DIRECT io_uring reads in
that layer's miss batch, then executes one BF16 top-2048 MLA kernel on an L40S.
The reported path metric is exactly the sum of measured SSD completion wait and
CUDA-event kernel time across 78 layers.  It excludes indexer, top-k selection,
H2D copies, communication, MoE, and all other transformer work.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
HELPER_SOURCE = Path(__file__).with_name("io_uring_direct_reader.c")
SEED = 20260821
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
DEFAULT_HIT_RATES = (0.0, 90.0, 95.0, 97.0, 99.0, 99.5, 100.0)
RAW_FIELDS = (
    "run_id", "repeat_id", "path_sample_id", "dram_hit_rate_pct",
    "realized_dram_hit_rate_pct", "batch_size", "layers", "tp_size",
    "local_query_heads", "topk", "qk_latent_dim", "value_latent_dim",
    "active_kv_bytes_per_layer", "logical_4k_reads_per_layer",
    "ssd_miss_reads_per_layer", "block_size_bytes", "queue_depth",
    "ssd_wait_s", "max_layer_ssd_wait_s", "mla_kernel_time_s",
    "path_latency_s", "status", "gpu_name", "gpu_uuid", "gpu_index",
    "hbm_total_gib", "ssd_file", "ssd_file_size_bytes", "ssd_device",
    "ssd_model", "filesystem", "mount_options", "io_engine",
    "direct_io", "software_commit", "torch_version", "cuda_version",
    "flashinfer_version", "kernel_release", "hostname", "seed", "error",
)
SUMMARY_FIELDS = (
    "dram_hit_rate_pct", "realized_dram_hit_rate_pct", "samples",
    "independent_repeats", "mean_path_latency_s", "p99_path_latency_s",
    "mean_ssd_wait_s", "p99_ssd_wait_s", "mean_mla_kernel_time_s",
    "p99_mla_kernel_time_s", "normalized_mean", "normalized_p99",
    "normalization", "status",
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def parse_hit_rates(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",") if value.strip()]
    if any(value < 0.0 or value > 100.0 for value in values):
        raise ValueError("DRAM hit rates must be in [0, 100]")
    if len(values) != len(set(values)):
        raise ValueError("DRAM hit rates must be unique")
    return values


def hit_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def compile_helper(output: Path) -> str:
    command = [
        "gcc", "-O3", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
        str(HELPER_SOURCE), "-o", str(output),
    ]
    subprocess.run(command, check=True, cwd=REPO)
    return subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0]


def prepare_backing_file(path: Path, size_gib: int) -> None:
    target_size = size_gib * 2**30
    if path.exists():
        stat = path.stat()
        if stat.st_size != target_size:
            raise RuntimeError(
                f"existing SSD file has size {stat.st_size}, expected {target_size}: {path}"
            )
        if stat.st_blocks * 512 < target_size:
            raise RuntimeError(f"SSD file is sparse or incompletely allocated: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if size_gib <= 0:
        raise ValueError("SSD file size must be positive")
    subprocess.run(
        [
            "dd", "if=/dev/zero", f"of={path}", "bs=4M",
            f"count={size_gib * 256}", "oflag=direct", "conv=fsync",
            "status=progress",
        ],
        check=True,
    )
    stat = path.stat()
    if stat.st_size != target_size or stat.st_blocks * 512 < target_size:
        raise RuntimeError(f"failed to create fully allocated SSD file: {path}")


def storage_metadata(path: Path) -> dict[str, str]:
    findmnt = subprocess.check_output(
        ["findmnt", "-T", str(path), "-n", "-o", "SOURCE,FSTYPE,OPTIONS"],
        text=True,
    ).strip().split(maxsplit=2)
    source = findmnt[0]
    filesystem = findmnt[1]
    options = findmnt[2] if len(findmnt) > 2 else ""
    parent = subprocess.check_output(
        ["lsblk", "-ndo", "PKNAME", source], text=True
    ).strip()
    block_device = f"/dev/{parent}" if parent else source
    model = subprocess.check_output(
        ["lsblk", "-ndo", "MODEL", block_device], text=True
    ).strip()
    return {
        "ssd_device": block_device,
        "ssd_model": model,
        "filesystem": filesystem,
        "mount_options": options,
    }


class DirectReader:
    def __init__(self, library: Path, path: Path, queue_depth: int, block_size: int):
        self.lib = ctypes.CDLL(str(library), use_errno=True)
        self.lib.direct_reader_create.argtypes = [
            ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint
        ]
        self.lib.direct_reader_create.restype = ctypes.c_void_p
        self.lib.direct_reader_destroy.argtypes = [ctypes.c_void_p]
        self.lib.direct_reader_file_size.argtypes = [ctypes.c_void_p]
        self.lib.direct_reader_file_size.restype = ctypes.c_uint64
        self.lib.direct_reader_last_error.argtypes = [ctypes.c_void_p]
        self.lib.direct_reader_last_error.restype = ctypes.c_char_p
        self.lib.direct_reader_read_random.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint,
        ]
        self.lib.direct_reader_read_random.restype = ctypes.c_longlong
        self.handle = self.lib.direct_reader_create(
            os.fsencode(path), queue_depth, block_size
        )
        if not self.handle:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(path))
        self.file_size = int(self.lib.direct_reader_file_size(self.handle))
        self.block_size = block_size

    def close(self) -> None:
        if self.handle:
            self.lib.direct_reader_destroy(self.handle)
            self.handle = None

    def read_offsets(self, offsets: list[int]) -> float:
        if not offsets:
            return 0.0
        values = (ctypes.c_uint64 * len(offsets))(*offsets)
        elapsed_ns = self.lib.direct_reader_read_random(
            self.handle, values, len(offsets)
        )
        if elapsed_ns < 0:
            message = self.lib.direct_reader_last_error(self.handle).decode()
            raise OSError(-elapsed_ns, message)
        return elapsed_ns / 1e9

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def build_sparse_mla_layers(batch: int, device: torch.device):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    pages_per_request = TOPK // PAGE_SIZE
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)
    kv_indptr = qo_indptr * pages_per_request
    kv_indices = torch.arange(
        batch * pages_per_request, dtype=torch.int32, device=device
    )
    kv_lens = torch.full((batch,), TOPK, dtype=torch.int32, device=device)
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
    for layer in range(LAYERS):
        compressed_kv[layer].zero_()
        k_rope[layer].zero_()
    layer_ckv = [
        compressed_kv[layer].reshape(
            batch * pages_per_request, PAGE_SIZE, KV_LORA_RANK
        )
        for layer in range(LAYERS)
    ]
    layer_kpe = [
        k_rope[layer].reshape(
            batch * pages_per_request, PAGE_SIZE, QK_ROPE_HEAD_DIM
        )
        for layer in range(LAYERS)
    ]
    output = torch.empty(
        (batch, LOCAL_QUERY_HEADS, KV_LORA_RANK), dtype=DTYPE, device=device
    )

    def invoke_layer(layer: int):
        wrapper.run(
            q_nope,
            q_rope,
            layer_ckv[layer],
            layer_kpe[layer],
            out=output,
        )
        return output

    owned = (
        workspace, qo_indptr, kv_indptr, kv_indices, kv_lens, q_nope, q_rope,
        compressed_kv, k_rope, layer_ckv, layer_kpe, output, wrapper,
    )
    return invoke_layer, output, owned


def gpu_metadata(device: torch.device, requested_index: int) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    _, total = torch.cuda.mem_get_info(device)
    query = subprocess.check_output(
        [
            "nvidia-smi", f"--id={requested_index}",
            "--query-gpu=uuid", "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return {
        "gpu_name": properties.name,
        "gpu_uuid": query,
        "gpu_index": requested_index,
        "hbm_total_gib": total / 2**30,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "flashinfer_version": __import__("flashinfer").__version__,
    }


def time_layer(
    invoke_layer: Callable[[int], torch.Tensor], layer: int,
    start: torch.cuda.Event, end: torch.cuda.Event,
) -> float:
    start.record()
    invoke_layer(layer)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def worker(args: argparse.Namespace) -> int:
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED + args.repeat_id)
    torch.cuda.manual_seed_all(SEED + args.repeat_id)
    hit_rate = args.hit_rate
    storage = storage_metadata(args.ssd_file)
    common = {
        "run_id": args.run_id,
        "repeat_id": args.repeat_id,
        "dram_hit_rate_pct": hit_rate,
        "batch_size": args.batch,
        "layers": LAYERS,
        "tp_size": TP_SIZE,
        "local_query_heads": LOCAL_QUERY_HEADS,
        "topk": TOPK,
        "qk_latent_dim": QK_LATENT_DIM,
        "value_latent_dim": VALUE_LATENT_DIM,
        "block_size_bytes": args.block_size,
        "queue_depth": args.queue_depth,
        "ssd_file": str(args.ssd_file),
        "ssd_file_size_bytes": args.ssd_file.stat().st_size,
        "io_engine": "raw Linux io_uring ABI; IORING_OP_READV",
        "direct_io": "O_DIRECT",
        "software_commit": git_revision(),
        "kernel_release": platform.release(),
        "hostname": platform.node(),
        "seed": SEED,
        "error": "",
        **storage,
    }
    rows: list[dict[str, object]] = []
    try:
        metadata = gpu_metadata(device, args.device)
        if args.expected_gpu_name.lower() not in str(metadata["gpu_name"]).lower():
            raise RuntimeError(
                f"expected GPU containing {args.expected_gpu_name!r}, got {metadata['gpu_name']!r}"
            )
        common.update(metadata)
        invoke_layer, output, owned = build_sparse_mla_layers(args.batch, device)
        torch.cuda.synchronize()
        for _ in range(args.warmup_paths):
            for layer in range(LAYERS):
                invoke_layer(layer)
        torch.cuda.synchronize()
        if not torch.isfinite(output).all().item():
            raise RuntimeError("non-finite sparse-MLA output after warmup")

        active_bytes = args.batch * TOPK * QK_LATENT_DIM * DTYPE.itemsize
        logical_reads = math.ceil(active_bytes / args.block_size)
        miss_reads = round(logical_reads * (1.0 - hit_rate / 100.0))
        realized_hit_rate = 100.0 * (1.0 - miss_reads / logical_reads)
        common.update(
            {
                "active_kv_bytes_per_layer": active_bytes,
                "logical_4k_reads_per_layer": logical_reads,
                "ssd_miss_reads_per_layer": miss_reads,
                "realized_dram_hit_rate_pct": realized_hit_rate,
            }
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        rng = random.Random(
            SEED + args.repeat_id * 100003 + int(round(hit_rate * 1000))
        )
        with DirectReader(
            args.helper_lib, args.ssd_file, args.queue_depth, args.block_size
        ) as reader:
            file_blocks = reader.file_size // args.block_size
            for _ in range(args.io_warmup_layers):
                offsets = [
                    rng.randrange(file_blocks) * args.block_size
                    for _ in range(miss_reads)
                ]
                reader.read_offsets(offsets)

            for sample in range(args.path_samples):
                ssd_waits: list[float] = []
                kernel_times: list[float] = []
                for layer in range(LAYERS):
                    offsets = [
                        rng.randrange(file_blocks) * args.block_size
                        for _ in range(miss_reads)
                    ]
                    ssd_waits.append(reader.read_offsets(offsets))
                    kernel_times.append(time_layer(invoke_layer, layer, start, end))
                ssd_wait = sum(ssd_waits)
                kernel_time = sum(kernel_times)
                rows.append(
                    {
                        **common,
                        "path_sample_id": sample,
                        "ssd_wait_s": ssd_wait,
                        "max_layer_ssd_wait_s": max(ssd_waits, default=0.0),
                        "mla_kernel_time_s": kernel_time,
                        "path_latency_s": ssd_wait + kernel_time,
                        "status": "measured",
                    }
                )
                print(
                    f"hit={hit_rate:g}% repeat={args.repeat_id} "
                    f"sample={sample + 1}/{args.path_samples} "
                    f"path_ms={(ssd_wait + kernel_time) * 1e3:.3f}",
                    flush=True,
                )
        del owned
    except Exception as error:
        rows = [
            {
                **common,
                "path_sample_id": -1,
                "status": "N/A",
                "error": repr(error),
            }
        ]

    with args.worker_output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=RAW_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    measured = [row for row in rows if row.get("status") == "measured"]
    summary = {
        "hit_rate": hit_rate,
        "repeat_id": args.repeat_id,
        "status": "measured" if len(measured) == args.path_samples else "N/A",
        "samples": len(measured),
        "output": str(args.worker_output),
        "error": rows[-1].get("error", ""),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "measured" else 1


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_combined_raw(paths: list[Path], output: Path) -> list[dict[str, str]]:
    rows = [row for path in paths for row in read_rows(path)]
    rows.sort(
        key=lambda row: (
            float(row["dram_hit_rate_pct"]),
            int(row["repeat_id"]),
            int(row["path_sample_id"]),
        )
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary(rows: list[dict[str, str]], output: Path) -> None:
    measured = [row for row in rows if row["status"] == "measured"]
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in measured:
        grouped.setdefault(float(row["dram_hit_rate_pct"]), []).append(row)
    if 100.0 not in grouped:
        raise RuntimeError("100% DRAM-hit baseline is required for normalization")

    def stats(group: list[dict[str, str]], field: str) -> tuple[float, float]:
        values = [float(row[field]) for row in group]
        return statistics.fmean(values), percentile(values, 0.99)

    baseline_mean, baseline_p99 = stats(grouped[100.0], "path_latency_s")
    output_rows = []
    for hit_rate in sorted(grouped):
        group = grouped[hit_rate]
        mean_path, p99_path = stats(group, "path_latency_s")
        mean_ssd, p99_ssd = stats(group, "ssd_wait_s")
        mean_kernel, p99_kernel = stats(group, "mla_kernel_time_s")
        output_rows.append(
            {
                "dram_hit_rate_pct": f"{hit_rate:g}",
                "realized_dram_hit_rate_pct": group[0]["realized_dram_hit_rate_pct"],
                "samples": len(group),
                "independent_repeats": len({row["repeat_id"] for row in group}),
                "mean_path_latency_s": mean_path,
                "p99_path_latency_s": p99_path,
                "mean_ssd_wait_s": mean_ssd,
                "p99_ssd_wait_s": p99_ssd,
                "mean_mla_kernel_time_s": mean_kernel,
                "p99_mla_kernel_time_s": p99_kernel,
                "normalized_mean": mean_path / baseline_mean,
                "normalized_p99": p99_path / baseline_p99,
                "normalization": (
                    "mean/100%-hit mean; p99/100%-hit p99"
                ),
                "status": "measured",
            }
        )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=REPO / "data/motivation")
    parser.add_argument(
        "--ssd-file", type=Path,
        default=Path("/Tan/keye_figure2/ssd_random_read_backing_8g.bin"),
    )
    parser.add_argument("--ssd-file-size-gib", type=int, default=8)
    parser.add_argument("--prepare-ssd-file", action="store_true")
    parser.add_argument("--helper-lib", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expected-gpu-name", default="L40S")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hit-rates", default=",".join(map(str, DEFAULT_HIT_RATES)))
    parser.add_argument("--hit-rate", type=float)
    parser.add_argument("--repeat-id", type=int, default=0)
    parser.add_argument("--queue-depth", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--warmup-paths", type=int, default=20)
    parser.add_argument("--io-warmup-layers", type=int, default=10)
    parser.add_argument("--path-samples", type=int, default=100)
    parser.add_argument("--independent-repeats", type=int, default=5)
    args = parser.parse_args()

    if args.worker:
        if args.run_id is None or args.hit_rate is None:
            parser.error("worker requires --run-id and --hit-rate")
        if args.helper_lib is None or args.worker_output is None:
            parser.error("worker requires --helper-lib and --worker-output")
        return worker(args)

    hit_rates = parse_hit_rates(args.hit_rates)
    if 100.0 not in hit_rates:
        parser.error("hit-rate sweep must include 100 for compute-only normalization")
    if not args.diagnostic:
        if args.independent_repeats < 5 or args.path_samples < 100:
            parser.error("formal protocol requires >=5 repeats and >=100 path samples")
        if args.warmup_paths < 20:
            parser.error("formal protocol requires >=20 compute warmup paths")
    if args.independent_repeats < 1 or args.path_samples < 1:
        parser.error("repeat and sample counts must be positive")
    if args.batch != 8:
        parser.error("Figure 2 protocol fixes batch size at 8")
    if args.queue_depth != 128 or args.block_size != 4096:
        parser.error("Figure 2 protocol fixes QD=128 and block size=4096")

    if args.prepare_ssd_file:
        prepare_backing_file(args.ssd_file, args.ssd_file_size_gib)
    if not args.ssd_file.is_file():
        parser.error(f"SSD backing file does not exist: {args.ssd_file}")
    prepare_backing_file(args.ssd_file, args.ssd_file_size_gib)

    run_id = args.run_id or datetime.now(timezone.utc).astimezone().strftime(
        "%Y-%m-%d_%H%M_figure2-sparse-mla-ssd-path-v01"
    )
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw_jobs"
    raw_dir.mkdir()
    helper_lib = output_dir / "io_uring_direct_reader.so"
    compiler = compile_helper(helper_lib)

    jobs = [
        (hit_rate, repeat)
        for repeat in range(args.independent_repeats)
        for hit_rate in hit_rates
    ]
    random.Random(SEED).shuffle(jobs)
    job_outputs: list[Path] = []
    failures = []
    for hit_rate, repeat in jobs:
        worker_output = raw_dir / (
            f"hit-{hit_label(hit_rate)}_repeat-{repeat:02d}.csv"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--run-id", run_id,
            "--ssd-file", str(args.ssd_file),
            "--helper-lib", str(helper_lib),
            "--worker-output", str(worker_output),
            "--device", str(args.device),
            "--expected-gpu-name", args.expected_gpu_name,
            "--batch", str(args.batch),
            "--hit-rate", str(hit_rate),
            "--repeat-id", str(repeat),
            "--queue-depth", str(args.queue_depth),
            "--block-size", str(args.block_size),
            "--warmup-paths", str(args.warmup_paths),
            "--io-warmup-layers", str(args.io_warmup_layers),
            "--path-samples", str(args.path_samples),
        ]
        result = subprocess.run(command, cwd=REPO, text=True)
        if worker_output.exists():
            job_outputs.append(worker_output)
        if result.returncode != 0:
            failures.append({"hit_rate": hit_rate, "repeat": repeat})
        print(
            f"completed hit={hit_rate:g}% repeat={repeat} rc={result.returncode}",
            flush=True,
        )

    rows = write_combined_raw(job_outputs, output_dir / "path_latency_raw.csv")
    if not failures:
        write_summary(rows, output_dir / "path_latency_summary.csv")
    manifest = {
        "created": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "diagnostic" if args.diagnostic else "formal",
        "command": sys.argv,
        "repository_commit": git_revision(),
        "compiler": compiler,
        "helper_source": str(HELPER_SOURCE.relative_to(REPO)),
        "seed": SEED,
        "jobs_randomized": True,
        "failures": failures,
        "scope": (
            "one TP=8 GLM-5.1-shaped rank; 78 sequential BF16 active "
            "top-2048 FlashInfer FA2 MLA layers; B=8"
        ),
        "path_definition": (
            "sum over 78 layers of final io_uring completion wait plus "
            "CUDA-event MLA kernel elapsed time"
        ),
        "io": {
            "engine": "raw Linux io_uring ABI; IORING_OP_READV",
            "direct_io": "O_DIRECT",
            "block_size_bytes": args.block_size,
            "queue_depth": args.queue_depth,
            "ssd_file": str(args.ssd_file),
            "ssd_file_size_bytes": args.ssd_file.stat().st_size,
            **storage_metadata(args.ssd_file),
        },
        "hit_rates_pct": hit_rates,
        "normalization": "mean/100%-hit mean; p99/100%-hit p99",
        "excluded": [
            "index scoring and top-k selection",
            "SSD-to-GPU or host-to-device copy",
            "communication, MoE, and non-attention layers",
            "request scheduler and end-to-end serving overhead",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
