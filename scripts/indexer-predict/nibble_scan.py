"""SM86 dense INT4 Key scan using two native nibble MMA instructions."""

from __future__ import annotations

import ctypes
import hashlib
import os
from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile

import torch


@lru_cache
def _load_scan():
    source = Path(__file__).with_suffix(".cu")
    root = source.resolve().parents[2]
    cuda_home = Path(os.environ.get("CUDA_HOME", root / "data/indexer-optimization/cuda130/nvidia/cu13"))
    nvcc = cuda_home / "bin/nvcc"
    fingerprint = hashlib.sha256(source.read_bytes() + subprocess.check_output([nvcc, "--version"]))
    directory = root / "data/indexer-predict/build" / f"nibble-sm86-{fingerprint.hexdigest()[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "scan.so"
    if not artifact.exists():
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            work = Path(tmp) / artifact.name
            subprocess.run([
                nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler=-fPIC", "--fmad=false",
                "-I", cuda_home / "include/cccl", "-L", cuda_home / "lib",
                "-arch=sm_86", source, "-o", work,
            ], check=True)
            work.replace(artifact)
    native = ctypes.CDLL(str(artifact)).nibble_scan
    native.argtypes = ([ctypes.c_void_p] * 9 + [ctypes.c_int] * 3
                       + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p])
    native.restype = ctypes.c_int
    return native


def prepare_scan(state, lengths, scale, score_dtype):
    if state.bits != 4 or state.groups != 1 or state.fp4 or state.pair_pq:
        raise ValueError("native nibble MMA requires one signed linear INT4 Key group")
    if score_dtype not in (torch.float32, torch.float16):
        raise ValueError("native nibble MMA outputs FP16 or FP32 scores")
    if torch.cuda.get_device_capability(lengths.device) != (8, 6):
        raise ValueError("native nibble MMA is compiled for SM86")
    native = _load_scan()
    width = ((state.plan.width + 127) // 128) * 128
    args = (state.query_scales, state.codes, state.scales)
    packed = torch.empty((state.batch, 16, 8, 2), device=lengths.device, dtype=torch.int32)
    scores = torch.empty((state.batch, width), device=lengths.device, dtype=score_dtype)

    def scan(gates):
        if gates.dtype != torch.bfloat16 or gates.stride(1) != 1:
            raise ValueError("native nibble MMA requires adjacent BF16 gates per row")
        stream = torch.cuda.current_stream(lengths.device).cuda_stream
        buffers = (state.query_codes, packed, *args, gates, state.plan.bases, lengths, scores)
        error = native(*(t.data_ptr() for t in buffers), state.batch, width, gates.stride(0),
                       scale, score_dtype == torch.float16, stream)
        if error:
            raise RuntimeError(f"native nibble scan launch returned CUDA error {error}")
        return scores

    return scan, packed.nbytes + scores.nbytes
