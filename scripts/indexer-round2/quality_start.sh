#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${INDEXER_QUALITY_GPUS:-GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822,GPU-79886749-8f57-63a5-8c8e-8a68a085565a}"
export KEYE_SM80_DSA=1 KEYE_SM80_EXACT_TOPK=1 KEYE_SM80_PREFILL_SCORE_CHUNK=64
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
if [[ ${INDEXER_QUALITY_FI074:-0} == 1 ]]; then
    experiment="$PWD/data/indexer-optimization"
    dsl="$experiment/fi-dsl"
    export PYTHONPATH="$experiment/fi074:$dsl/nvidia_cutlass_dsl/dsl_packages:$dsl${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_HOME="$experiment/cuda130/nvidia/cu13"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CPATH="$CUDA_HOME/include/cccl:$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/include${CPATH:+:$CPATH}"
    export FLASHINFER_CUDA_ARCH_LIST="8.6 8.9"
fi

exec .venv/bin/python scripts/indexer-round2/quality_server.py \
    --model-path /Tan/model/Keye-VL-2.0-30B-A3B --trust-remote-code \
    --host 127.0.0.1 --port "${INDEXER_QUALITY_PORT:-31249}" --tp-size 1 --pp-size 2 \
    --attention-backend keye_sa --context-length 131328 --max-total-tokens 140000 \
    --max-running-requests 1 --chunked-prefill-size 1024 --mem-fraction-static 0.90 \
    --disable-cuda-graph --disable-custom-all-reduce --disable-radix-cache \
    --disable-overlap-schedule --dtype bfloat16 --random-seed 20261119 --skip-server-warmup
