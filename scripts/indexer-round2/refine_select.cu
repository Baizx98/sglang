// Select exact short-list Top-2048 and emit logical token IDs in one CTA per row.
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cstdint>

namespace {

constexpr int kTop = 2048;
constexpr int kBins = 2048;

__device__ __forceinline__ uint32_t ordered_bits(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
}

__device__ __forceinline__ void increment_histogram(int* histogram, bool active,
                                                    int bucket) {
  const unsigned ballot = __ballot_sync(0xffffffffu, active);
  if (active) {
    const unsigned peers = __match_any_sync(ballot, bucket);
    if ((threadIdx.x & 31) == __ffs(peers) - 1) {
      atomicAdd(histogram + bucket, __popc(peers));
    }
  }
}

template <int bins, int threads>
__device__ void choose_bucket(int rank, const int* histogram,
                              typename cub::BlockScan<int, threads>::TempStorage& scan,
                              int& chosen, int& remaining) {
  constexpr int items = (bins + threads - 1) / threads;
  int counts[items], prefix[items];
#pragma unroll
  for (int item = 0; item < items; ++item) {
    const int bucket = bins - 1 - threadIdx.x * items - item;
    counts[item] = bucket >= 0 ? histogram[bucket] : 0;
  }
  cub::BlockScan<int, threads>(scan).InclusiveSum(counts, prefix);
#pragma unroll
  for (int item = 0; item < items; ++item) {
    if (prefix[item] >= rank && prefix[item] - counts[item] < rank) {
      chosen = bins - 1 - threadIdx.x * items - item;
      remaining = rank - prefix[item] + counts[item];
    }
  }
  __syncthreads();
}

template <int bits, int shift, int width, int threads>
__device__ void refine_prefix(const uint32_t (&keys)[width / threads], int live,
                              uint32_t& prefix, uint32_t& mask, int& rank, int* histogram,
                              typename cub::BlockScan<int, threads>::TempStorage& scan,
                              int& chosen, int& remaining) {
  constexpr int bins = 1 << bits;
  const int lane = threadIdx.x;
  for (int i = lane; i < bins; i += threads) histogram[i] = 0;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < width / threads; ++i) {
    if (lane + i * threads < live && (keys[i] & mask) == prefix) {
      atomicAdd(histogram + ((keys[i] >> shift) & (bins - 1)), 1);
    }
  }
  __syncthreads();
  choose_bucket<bins, threads>(rank, histogram, scan, chosen, remaining);
  prefix |= static_cast<uint32_t>(chosen) << shift;
  mask |= static_cast<uint32_t>(bins - 1) << shift;
  rank = remaining;
}

template <int width, int threads, bool wide_digits>
__global__ void refine_topk(const float* scores, const int32_t* logical,
                            const int32_t* lengths, int32_t* output) {
  constexpr int items = width / threads;
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const int live = lengths[row];
  const float* values = scores + row * width;
  const int32_t* ids = logical + row * width;
  int32_t* dst = output + row * kTop;
  if (live <= kTop) {
    for (int pos = lane; pos < kTop; pos += threads) {
      dst[pos] = pos < live ? ids[pos] : -1;
    }
    return;
  }

  __shared__ int histogram[kBins];
  __shared__ int chosen, remaining;
  __shared__ typename cub::BlockScan<int, threads>::TempStorage scan;
  uint32_t keys[items];

  for (int i = lane; i < kBins; i += threads) histogram[i] = 0;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < items; ++i) {
    const int pos = lane + i * threads;
    const bool active = pos < live;
    keys[i] = active ? ordered_bits(values[pos]) : 0;
    increment_histogram(histogram, active, keys[i] >> 21);
  }
  __syncthreads();
  choose_bucket<kBins, threads>(kTop, histogram, scan, chosen, remaining);
  uint32_t prefix = static_cast<uint32_t>(chosen) << 21;
  uint32_t mask = 0xffe00000;
  int need = remaining;
  if constexpr (wide_digits) {
    refine_prefix<11, 10, width, threads>(
        keys, live, prefix, mask, need, histogram, scan, chosen, remaining);
    refine_prefix<10, 0, width, threads>(
        keys, live, prefix, mask, need, histogram, scan, chosen, remaining);
  } else {
#pragma unroll
    for (int shift = 14; shift >= 0; shift -= 7) {
      for (int i = lane; i < 128; i += threads) histogram[i] = 0;
      __syncthreads();
#pragma unroll
      for (int i = 0; i < items; ++i) {
        const bool active = lane + i * threads < live && (keys[i] & mask) == prefix;
        increment_histogram(histogram, active, (keys[i] >> shift) & 127);
      }
      __syncthreads();
      choose_bucket<128, threads>(need, histogram, scan, chosen, remaining);
      prefix |= static_cast<uint32_t>(chosen) << shift;
      mask |= 127u << shift;
      need = remaining;
    }
  }

  unsigned strict = 0, equal = 0;
#pragma unroll
  for (int i = 0; i < items; ++i) {
    if (lane + i * threads < live) {
      if (keys[i] > prefix) strict |= 1u << i;
      if (keys[i] == prefix) equal |= 1u << i;
    }
  }
  int tied_before;
  cub::BlockScan<int, threads>(scan).ExclusiveSum(__popc(equal), tied_before);
  __syncthreads();
  while (equal && tied_before < need) {
    const unsigned first = equal & -equal;
    strict |= first;
    equal &= ~first;
    ++tied_before;
  }
  int write_pos;
  cub::BlockScan<int, threads>(scan).ExclusiveSum(__popc(strict), write_pos);
  while (strict) {
    const int item = __ffs(strict) - 1;
    dst[write_pos++] = ids[lane + item * threads];
    strict &= strict - 1;
  }
}

template <int threads, bool wide_digits>
void launch(const float* scores, const int32_t* logical, const int32_t* lengths,
            int32_t* output, int rows, int width, cudaStream_t stream) {
  if (width == 3072) {
    refine_topk<3072, threads, wide_digits><<<rows, threads, 0, stream>>>(scores, logical, lengths, output);
  } else {
    refine_topk<8192, threads, wide_digits><<<rows, threads, 0, stream>>>(scores, logical, lengths, output);
  }
}

}  // namespace

extern "C" int refine_select(const float* scores, const int32_t* logical,
                             const int32_t* lengths, int32_t* output,
                             int rows, int width, int threads, int wide_digits, void* stream) {
  if (width != 3072 && width != 8192) return cudaErrorInvalidValue;
  const auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (threads == 256 && !wide_digits) {
    launch<256, false>(scores, logical, lengths, output, rows, width, cuda_stream);
  } else if (threads == 512 && !wide_digits) {
    launch<512, false>(scores, logical, lengths, output, rows, width, cuda_stream);
  } else if (threads == 512 && wide_digits) {
    launch<512, true>(scores, logical, lengths, output, rows, width, cuda_stream);
  } else if (threads == 1024 && wide_digits) {
    launch<1024, true>(scores, logical, lengths, output, rows, width, cuda_stream);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
