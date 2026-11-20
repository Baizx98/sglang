"""Gather-and-score variants and exact logical-ID selectors for Keye refinement."""

from __future__ import annotations

import ctypes
import fcntl
import functools
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[2]
K = 2048
ROUND1 = ROOT / "scripts/indexer-optimization"
if str(ROUND1) not in sys.path:
    sys.path.insert(0, str(ROUND1))

from optimize import _load_packed_key


@triton.jit
def _score_candidates(
    q, cache, page, gates, lengths, candidates, scores, ids,
    ms, exps, headers, exceptions,
    qb: tl.constexpr, qh: tl.constexpr, qd: tl.constexpr,
    ks: tl.constexpr, kd: tl.constexpr,
    pb: tl.constexpr, pn: tl.constexpr,
    wb: tl.constexpr, wh: tl.constexpr,
    m: tl.constexpr, width: tl.constexpr, scale: tl.constexpr,
    tile: tl.constexpr, sort_local: tl.constexpr, packed_key: tl.constexpr, wide: tl.constexpr,
):
    row = tl.program_id(1)
    position = tl.program_id(0) * tile + tl.arange(0, tile)
    present = position < m
    length = tl.load(lengths + row)
    token = tl.load(candidates + row * m + position, present, other=-1)
    valid = present & (token >= 0) & (token < length) & (token < width)
    physical = tl.load(page + row * pb + token * pn, valid, other=-1)
    valid &= physical >= 0

    if sort_local:
        # A local order changes transactions without materializing a gathered K.
        packed = (physical.to(tl.uint64) << 32) | token.to(tl.uint32).to(tl.uint64)
        packed = tl.where(valid, packed, 0xffffffffffffffff)
        ordered = tl.sort(packed)
        physical = (ordered >> 32).to(tl.int32)
        token = ordered.to(tl.int32)
        valid = present & (physical >= 0)
        tl.store(ids + row * m + position, token, present)

    dim = tl.arange(0, 64)
    head = tl.arange(0, 16)
    if wide:
        physical = physical.to(tl.int64)
    if packed_key:
        key = _load_packed_key(ms, exps, headers, exceptions, physical, valid, dim, wide)
    else:
        key = tl.load(cache + physical[:, None] * ks + dim[None, :] * kd, valid[:, None], other=0.0)
    query = tl.load(q + row * qb + head[:, None] * qh + dim[None, :] * qd)
    weight = tl.load(gates + row * wb + head * wh).to(tl.float32)
    per_head = tl.dot(query, tl.trans(key), allow_tf32=False)
    per_head = per_head.to(tl.bfloat16).to(tl.float32)
    values = tl.sum(tl.maximum(per_head * scale, 0.0) * weight[:, None], 0)
    tl.store(scores + row * m + position, tl.where(valid, values, -float("inf")), present)


@triton.jit
def _map_selected(indices, candidates, output, m: tl.constexpr):
    row = tl.program_id(1)
    column = tl.program_id(0) * 256 + tl.arange(0, 256)
    offset = tl.load(indices + row * 2048 + column)
    logical = tl.load(candidates + row * m + offset, (offset >= 0) & (offset < m), other=-1)
    tl.store(output + row * 2048 + column, logical)


def prepare_scorer(
    q, key, table, weights, lengths, candidates, scale, tile, warps,
    *, sort_local=False, encoded=None,
):
    batch, m = candidates.shape
    scores = torch.empty((batch, m), dtype=torch.float32, device=q.device)
    logical = torch.empty_like(candidates) if sort_local else candidates
    packed_buffers = encoded if encoded is not None else (key, key, key, key)
    wide_address = key.size(0) * key.stride(0) >= 2**31

    def score(query=q, gates=weights):
        _score_candidates[(triton.cdiv(m, tile), batch)](
            query, key, table, gates, lengths, candidates, scores, logical,
            *packed_buffers, *query.stride(), *key.stride(), *table.stride(), *gates.stride(),
            m, table.size(1), float(scale), tile, sort_local, encoded is not None, wide_address,
            num_warps=warps,
        )
        return scores, logical

    return score


@functools.lru_cache
def _load_extension(architecture):
    source = Path(__file__).with_name("refine_select.cu")
    cuda_home = Path(os.environ.get("CUDA_HOME", ROOT / "data/indexer-optimization/cuda130/nvidia/cu13"))
    nvcc = cuda_home / "bin/nvcc"
    fingerprint = hashlib.sha256(source.read_bytes() + subprocess.check_output([nvcc, "--version"]))
    build = ROOT / "data/indexer-round2/build" / f"refine-sm{architecture}-{fingerprint.hexdigest()[:12]}"
    build.mkdir(parents=True, exist_ok=True)
    artifact = build / "refine.so"
    with (build / "compile.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not artifact.exists():
            work = artifact.with_suffix(".partial.so")
            subprocess.run(
                [
                    nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler=-fPIC",
                    "-I", cuda_home / "include/cccl",
                    "-gencode", f"arch=compute_{architecture},code=sm_{architecture}",
                    source, "-o", work,
                ],
                check=True,
            )
            work.replace(artifact)
    native = ctypes.CDLL(str(artifact)).refine_select
    native.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    native.restype = ctypes.c_int
    return native


def prepare_logical_radix(lengths, width, *, threads=256, wide_digits=False):
    batch = lengths.numel()
    major, minor = torch.cuda.get_device_capability(lengths.device)
    native = _load_extension(f"{major}{minor}")
    out = torch.empty((batch, K), dtype=torch.int32, device=lengths.device)

    def select(scores, logical):
        result = native(
            scores.data_ptr(), logical.data_ptr(), lengths.data_ptr(), out.data_ptr(),
            batch, width, threads, wide_digits, torch.cuda.current_stream(lengths.device).cuda_stream,
        )
        if result:
            raise RuntimeError(f"refinement selector failed to launch with CUDA status {result}")
        return out

    return select


def prepare_mapping_selector(select_offsets, logical):
    batch, width = logical.shape
    output = torch.empty((batch, K), dtype=torch.int32, device=logical.device)

    def select(scores, candidate_ids):
        offsets = select_offsets(scores)
        _map_selected[(K // 256, batch)](offsets, candidate_ids, output, width, num_warps=4)
        return output

    return select
