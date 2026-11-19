#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/../.." && pwd)
previous="$repo/data/indexer-optimization"
export CUDA_VISIBLE_DEVICES="${INDEXER_GPU:-GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822}"
export CUDA_HOME="${INDEXER_CUDA_HOME:-$previous/cuda130/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include/cccl:$repo/.venv/lib/python3.12/site-packages/nvidia/cu13/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export FLASHINFER_CUDA_ARCH_LIST="${INDEXER_CUDA_ARCH:-8.6}"
export INDEXER_FI074="${INDEXER_FI074:-$previous/fi074}"
export INDEXER_FI_DSL="${INDEXER_FI_DSL:-$previous/fi-dsl}"

target="${INDEXER_FULL_TARGET:-$script_dir/full_index.py}"
exec "$repo/.venv/bin/python" -c 'import os, pathlib, runpy, sys
dsl = pathlib.Path(os.environ["INDEXER_FI_DSL"])
sys.path[:0] = [os.environ["INDEXER_FI074"], str(dsl / "nvidia_cutlass_dsl/dsl_packages"), str(dsl)]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parent))
runpy.run_path(sys.argv[0], run_name="__main__")' "$target" "$@"
