#!/usr/bin/env python3
"""Generate Figure 1(a) GLM-5.2 per-accelerator capacity data.

The default is an explicitly representative 16-accelerator decode replica.
Model tensors are evenly resident across TP/EP ranks and KV-family state is
sequence-sharded across the same 16 accelerators.  The script never presents
the modeled rows as a live 16-device memory snapshot.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


CONTEXTS = (131072, 1048576)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
LAYERS = 78
INDEX_LAYERS = 21
INDEX_DIM = 128
KV_LATENT_DIM = 512
KV_ROPE_DIM = 64
TOPK = 2048
BF16_BYTES = 2

# Official checkpoint tensor payload previously audited from safetensors
# metadata at revision b4734de4facf877f85769a911abafc5283eab3d9.
BF16_PARAMETER_COUNT = 753_329_921_024
FP32_PARAMETER_COUNT = 19_456
MODEL_PARAMETER_BYTES = BF16_PARAMETER_COUNT * 2 + FP32_PARAMETER_COUNT * 4

FIELDS = (
    "context_tokens", "batch_size", "model_weight_gb", "full_kv_gb",
    "index_kv_gb", "topk_kv_gb", "total_gb", "hbm_threshold_gb",
    "dram_threshold_gb", "ssd_threshold_gb", "sharding_scope",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerators", type=int, default=16)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--ep", type=int, default=16)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--kv-sequence-shards", type=int, default=16)
    parser.add_argument("--hbm-gb", type=float, default=141.0)
    parser.add_argument("--node-dram-gb", type=float, default=4000.0)
    parser.add_argument("--node-ssd-gb", type=float, default=61440.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot-status", type=Path, required=True)
    args = parser.parse_args()

    if args.tp != args.accelerators or args.ep != args.accelerators:
        parser.error("default model assumes dense/shared TP and expert EP span all accelerators")
    if args.kv_sequence_shards != args.accelerators:
        parser.error("all states must use the same per-accelerator scope")
    if args.dp != 1:
        parser.error("DP replicas do not divide capacity within one decode replica")

    model_gb = MODEL_PARAMETER_BYTES / args.accelerators / 1e9
    hbm_gb = args.hbm_gb
    dram_gb = args.node_dram_gb / args.accelerators
    ssd_gb = args.node_ssd_gb / args.accelerators
    scope = (
        f"representative_per_accelerator:TP{args.tp},EP{args.ep},DP{args.dp},"
        f"KV-sequence-shards={args.kv_sequence_shards},devices={args.accelerators}"
    )
    rows = []
    for context in CONTEXTS:
        for batch in BATCHES:
            full_bytes = (
                batch * context * LAYERS * (KV_LATENT_DIM + KV_ROPE_DIM) * BF16_BYTES
            )
            index_bytes = batch * context * INDEX_LAYERS * INDEX_DIM * BF16_BYTES
            topk_bytes = (
                batch * LAYERS * TOPK * (KV_LATENT_DIM + KV_ROPE_DIM) * BF16_BYTES
            )
            full_gb = full_bytes / args.kv_sequence_shards / 1e9
            index_gb = index_bytes / args.kv_sequence_shards / 1e9
            topk_gb = topk_bytes / args.kv_sequence_shards / 1e9
            rows.append({
                "context_tokens": context,
                "batch_size": batch,
                "model_weight_gb": model_gb,
                "full_kv_gb": full_gb,
                "index_kv_gb": index_gb,
                "topk_kv_gb": topk_gb,
                "total_gb": model_gb + full_gb + index_gb + topk_gb,
                "hbm_threshold_gb": hbm_gb,
                "dram_threshold_gb": dram_gb,
                "ssd_threshold_gb": ssd_gb,
                "sharding_scope": scope,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    created = datetime.now(timezone.utc).astimezone().isoformat()
    args.manifest.write_text(json.dumps({
        "created": created,
        "model": "zai-org/GLM-5.2",
        "model_revision": "b4734de4facf877f85769a911abafc5283eab3d9",
        "precision": "BF16 parameter/KV/index capacity model",
        "parameter_payload_bytes": MODEL_PARAMETER_BYTES,
        "formula": {
            "full_kv": "B*T*78*(512+64)*2 / 16",
            "index_kv": "B*T*21*128*2 / 16",
            "topk_kv": "B*78*2048*(512+64)*2 / 16",
        },
        "parallelism": {
            "tp": args.tp, "ep": args.ep, "dp": args.dp,
            "accelerators_per_replica": args.accelerators,
            "kv_sequence_shards": args.kv_sequence_shards,
            "status": "representative_not_experimentally_deployed",
        },
        "thresholds": {
            "hbm_gb_per_accelerator": hbm_gb,
            "dram_gb_per_accelerator": dram_gb,
            "ssd_gb_per_accelerator": ssd_gb,
            "representative_source": (
                "Two NVIDIA DGX H200 systems: 16 accelerators, 2*2 TB system "
                "memory, and 2*(8*3.84 TB) data NVMe."
            ),
            "source_url": (
                "https://docs.nvidia.com/dgx/dgxh100-user-guide/"
                "introduction-to-dgxh100.html"
            ),
        },
    }, indent=2) + "\n")
    args.snapshot_status.write_text(json.dumps({
        "created": created,
        "status": "unavailable",
        "reason": (
            "No 16-accelerator GLM-5.2 BF16 deployment is attached. Per-card "
            "values are formula-derived checkpoint tensor payloads, not CUDA/NPU snapshots."
        ),
        "mean_card_bytes": None,
        "max_card_bytes": None,
        "max_deviation_pct": None,
        "required_followup": (
            "Replace modeled rows with post-load resident parameter bytes and "
            "16 per-card memory snapshots when the Table 2 platform is frozen."
        ),
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
