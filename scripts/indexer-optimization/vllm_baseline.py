"""Run the upstream vLLM FP32 persistent Top-K kernel on Keye scores.

The CUDA build embeds the upstream launch_persistent_topk template unchanged and
includes its CUDA headers directly. A small raw-tensor adapter replaces the
stable PyTorch ABI. The supported experiment range is k=2048 and one visible
CUDA GPU. Calls requiring vLLM's separate generic decode fallback report an error.
"""

from __future__ import annotations

import ctypes
import fcntl
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess

import torch


_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "data/indexer-optimization"
_K = 2048

_ADAPTER = r"""
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include "persistent_topk.cuh"
#include "sampled_topk.cuh"

namespace torch {
namespace headeronly {
enum class ScalarType { Byte };
}
namespace stable {
struct Tensor {
  void* data;
  int64_t rows;
  int64_t width;
  int64_t row_stride;
  int get_device_index() const { return 0; }
  int64_t size(int dim) const { return dim == 0 ? rows : width; }
  int64_t stride(int dim) const { return dim == 0 ? row_stride : 1; }
  int dim() const { return 1; }
  bool is_cuda() const { return true; }
  headeronly::ScalarType scalar_type() const { return headeronly::ScalarType::Byte; }
  const void* const_data_ptr() const { return data; }
  template<class T> const T* const_data_ptr() const { return static_cast<const T*>(data); }
  template<class T> T* mutable_data_ptr() const { return static_cast<T*>(data); }
};
namespace accelerator {
struct DeviceGuard { explicit DeviceGuard(int) {} };
}
}
}

template <typename... Message>
void check(bool condition, const Message&... message) {
  if (!condition) {
    std::ostringstream text;
    (text << ... << message);
    throw std::runtime_error(text.str());
  }
}
#define STD_TORCH_CHECK(...) check(__VA_ARGS__)

thread_local cudaStream_t stream_from_torch;
cudaStream_t get_current_cuda_stream() { return stream_from_torch; }

const cudaDeviceProp* get_device_prop() {
  static cudaDeviceProp properties;
  int device;
  cudaError_t result = cudaGetDevice(&device);
  check(result == cudaSuccess, cudaGetErrorString(result));
  result = cudaGetDeviceProperties(&properties, device);
  check(result == cudaSuccess, cudaGetErrorString(result));
  return &properties;
}

void top_k_per_row_decode(const torch::stable::Tensor&, int64_t,
                          const torch::stable::Tensor&, torch::stable::Tensor&,
                          int64_t, int64_t, int64_t, int64_t) {
  throw std::runtime_error("upstream dispatch requires its separate generic decode fallback");
}
"""

_ENTRY = r"""
extern "C" const char* keye_vllm_topk(
    void* scores, void* lengths, void* output, void* workspace,
    int64_t rows, int64_t width, int64_t stride, int64_t workspace_bytes,
    int64_t max_seq_len, void* stream) {
  try {
    stream_from_torch = static_cast<cudaStream_t>(stream);
    using torch::stable::Tensor;
    Tensor out{output, rows, 2048, 2048};
    Tensor scratch{workspace, workspace_bytes, 1, 1};
    launch_persistent_topk<2048, float>(
        Tensor{scores, rows, width, stride}, Tensor{lengths, rows, 1, 1},
        out, scratch, max_seq_len);
    return nullptr;
  } catch (const std::exception& error) {
    static thread_local std::string message;
    message = error.what();
    return message.c_str();
  }
}
"""


@lru_cache
def _extension(architecture: str):
    checkout = Path(
        os.environ.get("KEYE_VLLM_CHECKOUT", _DATA / "references/vllm-current")
    )
    cuda_home = Path(
        os.environ.get("CUDA_HOME", _DATA / "cuda130/nvidia/cu13")
    )
    cuda_include = cuda_home / "include/cccl"
    nvcc = cuda_home / "bin/nvcc"
    csrc = checkout / "csrc"
    upstream_source = (csrc / "topk.cu").read_text()
    start = upstream_source.index("namespace {")
    end = upstream_source.index("template <int TopK>\nvoid launch_persistent_topk_dtype")
    source = _ADAPTER + upstream_source[start:end] + "}\n" + _ENTRY
    digest = hashlib.sha256(source.encode())
    for dependency in (
        "persistent_topk.cuh",
        "persistent_topk_overflow.cuh",
        "sampled_topk.cuh",
        "topk_histogram_4096.cuh",
        "topk_histogram_4096_overflow.cuh",
        "core/cuda_compat.h",
    ):
        digest.update((csrc / dependency).read_bytes())
    digest.update(subprocess.check_output([nvcc, "--version"]))
    build = _DATA / "build" / f"vllm-persistent-sm{architecture}-{digest.hexdigest()[:12]}"
    build.mkdir(parents=True, exist_ok=True)
    library = build / "persistent_topk.so"
    with (build / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file():
            translation_unit = build / "persistent_topk.cu"
            translation_unit.write_text(source)
            temporary = library.with_suffix(".tmp.so")
            subprocess.run(
                [
                    nvcc,
                    "-O3",
                    "-std=c++17",
                    "--shared",
                    "-Xcompiler=-fPIC",
                    "-I", csrc,
                    "-I", cuda_include,
                    "-gencode", f"arch=compute_{architecture},code=sm_{architecture}",
                    translation_unit,
                    "-o", temporary,
                ],
                check=True,
            )
            temporary.replace(library)
    launch = ctypes.CDLL(str(library)).keye_vllm_topk
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 5 + [ctypes.c_void_p]
    launch.restype = ctypes.c_char_p
    return launch


def prepare_selector(lengths: torch.Tensor, max_seq_len: int):
    lengths = lengths.contiguous()
    if lengths.dtype != torch.int32 or lengths.ndim != 1 or not lengths.is_cuda:
        raise ValueError("expected CUDA int32 lengths [batch]")
    if torch.cuda.device_count() != 1:
        raise ValueError("use CUDA_VISIBLE_DEVICES to select one benchmark GPU")
    major, minor = torch.cuda.get_device_capability(lengths.device)
    launch = _extension(f"{major}{minor}")
    scratch = torch.empty(1024 * 1024, dtype=torch.uint8, device=lengths.device)
    output = torch.empty((lengths.numel(), _K), dtype=torch.int32, device=lengths.device)

    def select(scores: torch.Tensor):
        if (
            scores.dtype != torch.float32
            or scores.device != lengths.device
            or scores.ndim != 2
            or scores.stride(1) != 1
            or scores.size(0) != lengths.numel()
            or not 0 < max_seq_len <= scores.size(1)
        ):
            raise ValueError("scores must be CUDA FP32 and match the configured lengths")
        stream = torch.cuda.current_stream(scores.device)
        error = launch(
            scores.data_ptr(), lengths.data_ptr(), output.data_ptr(),
            scratch.data_ptr(), scores.size(0), scores.size(1), scores.stride(0),
            scratch.numel(), max_seq_len, stream.cuda_stream,
        )
        if error:
            raise RuntimeError(error.decode())
        return output

    return select


def prepare(q, key_cache, page_table, weights, lengths, scale):
    source = (
        _ROOT / "python/sglang/srt/layers/attention/keye_topk/ampere_indexer.py"
    )
    spec = importlib.util.spec_from_file_location("keye_vllm_baseline_amperescore", source)
    score_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_module)
    select = prepare_selector(lengths, page_table.size(1))

    def call():
        scores = score_module.keye_indexer_score_paged(
            q, key_cache, page_table, weights, lengths, scale
        )
        return select(scores)

    return {"sglang_score_vllm_persistent": call}
