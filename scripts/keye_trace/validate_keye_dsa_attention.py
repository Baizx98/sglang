#!/usr/bin/env python3
"""Validate the Keye SM8x exact-token DSA attention against eager PyTorch."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def load_kernel_module():
    """Load the standalone kernel without triggering SGLang model registration."""
    path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/keye_dsa_sm80.py"
    )
    spec = importlib.util.spec_from_file_location("keye_dsa_sm80_validation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_KERNEL = load_kernel_module()
KeyeDSAWorkspace = _KERNEL.KeyeDSAWorkspace
keye_dsa_paged_attention = _KERNEL.keye_dsa_paged_attention
keye_dsa_sparse_attention = _KERNEL.keye_dsa_sparse_attention

SEED = 20260804
HQ = 32
HKV = 4
HEAD_DIM = 128
TOPK = 2048


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def eager_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Mirror the kernel's low-precision QK/scale/P materialization."""
    batch, hq, q_len, _ = q.shape
    hkv = k.shape[1]
    group_size = hq // hkv
    out = torch.zeros_like(q)
    for batch_id in range(batch):
        for query_id in range(q_len):
            selected = indices[batch_id, query_id]
            selected = selected[
                (selected >= 0) & (selected < k.shape[2])
            ].to(torch.long)
            if selected.numel() == 0:
                continue
            for q_head in range(hq):
                kv_head = q_head // group_size
                key = k[batch_id, kv_head, selected]
                value = v[batch_id, kv_head, selected]
                logits = torch.matmul(q[batch_id, q_head, query_id], key.T)
                logits = logits.to(q.dtype)
                logits = (logits * sm_scale).to(q.dtype)
                probability = torch.softmax(logits.float(), dim=-1).to(q.dtype)
                out[batch_id, q_head, query_id] = torch.matmul(
                    probability.unsqueeze(0), value
                ).squeeze(0)
    return out


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float()
    expected_f = expected.float()
    difference = (actual_f - expected_f).abs()
    flat_actual = actual_f.flatten()
    flat_expected = expected_f.flatten()
    cosine = torch.nn.functional.cosine_similarity(
        flat_actual.unsqueeze(0), flat_expected.unsqueeze(0)
    ).item()
    return {
        "all_finite": bool(torch.isfinite(actual_f).all().item()),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
        "cosine_similarity": float(cosine),
    }


def thresholds(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.bfloat16:
        return {"max_abs": 4e-3, "mean_abs": 5e-4, "cosine_similarity": 0.9999}
    return {"max_abs": 2e-3, "mean_abs": 2.5e-4, "cosine_similarity": 0.99995}


def assess(result: dict[str, Any], dtype: torch.dtype) -> bool:
    limit = thresholds(dtype)
    return bool(
        result["all_finite"]
        and result["max_abs"] <= limit["max_abs"]
        and result["mean_abs"] <= limit["mean_abs"]
        and result["cosine_similarity"] >= limit["cosine_similarity"]
    )


def unique_indices(
    batch: int,
    q_len: int,
    kv_len: int,
    valid_counts: list[int],
    device: torch.device,
) -> torch.Tensor:
    output = torch.full(
        (batch, q_len, TOPK), -1, device=device, dtype=torch.int32
    )
    for batch_id in range(batch):
        for query_id in range(q_len):
            valid = min(valid_counts[batch_id], TOPK, kv_len)
            output[batch_id, query_id, :valid] = torch.randperm(
                kv_len, device=device, dtype=torch.int64
            )[:valid].to(torch.int32)
    return output


def run_direct_case(
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    q_len: int,
    kv_len: int,
    valid_counts: list[int],
    workspace: KeyeDSAWorkspace,
) -> dict[str, Any]:
    q = torch.randn(
        batch, HQ, q_len, HEAD_DIM, device=device, dtype=dtype
    )
    k = torch.randn(
        batch, HKV, kv_len, HEAD_DIM, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    indices = unique_indices(batch, q_len, kv_len, valid_counts, device)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    actual = keye_dsa_sparse_attention(
        q, k, v, indices, sm_scale=scale, workspace=workspace
    )
    torch.cuda.synchronize(device)
    kernel_ms = (time.perf_counter() - started) * 1000
    expected = eager_reference(q, k, v, indices, scale)
    result = metrics(actual, expected)
    repeated = keye_dsa_sparse_attention(
        q, k, v, indices, sm_scale=scale, workspace=workspace
    )
    result.update(
        {
            "name": name,
            "path": "direct",
            "dtype": str(dtype).removeprefix("torch."),
            "shape": {
                "batch": batch,
                "hq": HQ,
                "hkv": HKV,
                "q_len": q_len,
                "kv_len": kv_len,
                "head_dim": HEAD_DIM,
                "topk": TOPK,
                "valid_counts": valid_counts,
            },
            "kernel_first_call_ms": kernel_ms,
            "repeat_max_abs": float((actual.float() - repeated.float()).abs().max().item()),
        }
    )
    result["passed"] = assess(result, dtype) and result["repeat_max_abs"] == 0.0
    return result


def run_paged_case(
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    is_decode: bool,
    num_tokens: int,
    pool_tokens: int,
    valid_count: int,
    workspace: KeyeDSAWorkspace,
) -> dict[str, Any]:
    q_packed = torch.randn(
        num_tokens, HQ, HEAD_DIM, device=device, dtype=dtype
    )
    k_cache = torch.randn(
        pool_tokens, HKV, HEAD_DIM, device=device, dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    token_slots = torch.full(
        (num_tokens, TOPK), -1, device=device, dtype=torch.int32
    )
    for token in range(num_tokens):
        token_slots[token, :valid_count] = torch.randperm(
            pool_tokens, device=device
        )[:valid_count].to(torch.int32)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    actual = keye_dsa_paged_attention(
        q_packed,
        k_cache,
        v_cache,
        token_slots,
        num_q_heads=HQ,
        head_dim=HEAD_DIM,
        sm_scale=scale,
        is_decode=is_decode,
        workspace=workspace,
    )
    k_base = k_cache.permute(1, 0, 2).unsqueeze(0)
    v_base = v_cache.permute(1, 0, 2).unsqueeze(0)
    if is_decode:
        q_4d = q_packed.unsqueeze(2)
        k_4d = k_base.expand(num_tokens, -1, -1, -1)
        v_4d = v_base.expand(num_tokens, -1, -1, -1)
        indices = token_slots.view(num_tokens, 1, -1)
        expected = eager_reference(q_4d, k_4d, v_4d, indices, scale)
        expected = expected.squeeze(2).reshape(num_tokens, HQ * HEAD_DIM)
    else:
        q_4d = q_packed.permute(1, 0, 2).unsqueeze(0)
        expected = eager_reference(
            q_4d, k_base, v_base, token_slots.unsqueeze(0), scale
        )
        expected = expected.squeeze(0).permute(1, 0, 2).reshape(
            num_tokens, HQ * HEAD_DIM
        )
    result = metrics(actual, expected)
    result.update(
        {
            "name": name,
            "path": "paged_decode" if is_decode else "paged_prefill",
            "dtype": str(dtype).removeprefix("torch."),
            "shape": {
                "num_tokens": num_tokens,
                "pool_tokens": pool_tokens,
                "hq": HQ,
                "hkv": HKV,
                "head_dim": HEAD_DIM,
                "topk": TOPK,
                "valid_count": valid_count,
            },
        }
    )
    result["passed"] = assess(result, dtype)
    return result


def run_device(device_id: int) -> dict[str, Any]:
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED + device_id)
    torch.cuda.manual_seed_all(SEED + device_id)
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 8:
        raise RuntimeError(f"GPU {device_id} is not SM8x: {capability}")
    workspace = KeyeDSAWorkspace()
    cases = [
        run_direct_case(
            name="decode_real_shape_split_bf16",
            device=device,
            dtype=torch.bfloat16,
            batch=2,
            q_len=1,
            kv_len=4096,
            valid_counts=[TOPK, TOPK],
            workspace=workspace,
        ),
        run_direct_case(
            name="decode_ragged_padding_split_bf16",
            device=device,
            dtype=torch.bfloat16,
            batch=2,
            q_len=1,
            kv_len=3072,
            valid_counts=[257, TOPK],
            workspace=workspace,
        ),
        run_direct_case(
            name="prefill_real_shape_nonsplit_bf16",
            device=device,
            dtype=torch.bfloat16,
            batch=1,
            q_len=8,
            kv_len=4096,
            valid_counts=[TOPK],
            workspace=workspace,
        ),
        run_direct_case(
            name="decode_real_shape_split_fp16",
            device=device,
            dtype=torch.float16,
            batch=1,
            q_len=1,
            kv_len=4096,
            valid_counts=[TOPK],
            workspace=workspace,
        ),
        run_paged_case(
            name="physical_slot_adapter_decode_bf16",
            device=device,
            dtype=torch.bfloat16,
            is_decode=True,
            num_tokens=3,
            pool_tokens=5000,
            valid_count=TOPK,
            workspace=workspace,
        ),
        run_paged_case(
            name="physical_slot_adapter_prefill_bf16",
            device=device,
            dtype=torch.bfloat16,
            is_decode=False,
            num_tokens=8,
            pool_tokens=5000,
            valid_count=TOPK,
            workspace=workspace,
        ),
    ]
    return {
        "device_id": device_id,
        "name": properties.name,
        "pci_bus_id": properties.pci_bus_id,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_bytes": properties.total_memory,
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0, 1],
        help="PyTorch CUDA ordinals; verify recorded GPU names, not nvidia-smi row numbers",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    devices = [run_device(device_id) for device_id in args.devices]
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "repository": str(Path.cwd()),
        "commit": git_revision(Path.cwd()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "model_shape": {
            "hq": HQ,
            "hkv": HKV,
            "head_dim": HEAD_DIM,
            "topk": TOPK,
        },
        "thresholds": {
            "bfloat16": thresholds(torch.bfloat16),
            "float16": thresholds(torch.float16),
        },
        "devices": devices,
        "passed": all(device["passed"] for device in devices),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
