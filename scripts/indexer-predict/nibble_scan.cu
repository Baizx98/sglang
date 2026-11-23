#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int kHeads = 16;
constexpr int kDimensions = 64;
constexpr unsigned kWarpMask = 0xffffffffu;

struct QueryFragment {
    uint32_t high;
    uint32_t low;
};

__global__ void pack_query(const int8_t* q, QueryFragment* packed) {
    const int word = blockIdx.x * blockDim.x + threadIdx.x;
    const auto* bytes = reinterpret_cast<const uint2*>(q);
    const uint2 x = bytes[word];
    QueryFragment result{};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int j = 8 * i;
        const int dst = 4 * i;
        result.low |= ((x.x >> j) & 15u) << dst;
        result.low |= ((x.y >> j) & 15u) << (dst + 16);
        result.high |= ((x.x >> (j + 4)) & 15u) << dst;
        result.high |= ((x.y >> (j + 4)) & 15u) << (dst + 16);
    }
    packed[word] = result;
}

__device__ void dot(const QueryFragment (&a)[4], uint32_t b0, uint32_t b1, int (&out)[4]) {
    // Each signed q is 16 * signed(high nibble) + unsigned(low nibble).
    asm volatile(
        "{\n"
        " .reg .s32 h0,h1,h2,h3,l0,l1,l2,l3,z;\n"
        " mov.s32 z, 0;\n"
        " mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "  {h0,h1,h2,h3}, {%4,%5,%6,%7}, {%12,%13}, {z,z,z,z};\n"
        " mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
        "  {l0,l1,l2,l3}, {%8,%9,%10,%11}, {%12,%13}, {z,z,z,z};\n"
        " mad.lo.s32 %0, h0, 16, l0;\n"
        " mad.lo.s32 %1, h1, 16, l1;\n"
        " mad.lo.s32 %2, h2, 16, l2;\n"
        " mad.lo.s32 %3, h3, 16, l3;\n"
        "}\n"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(a[0].high), "r"(a[1].high), "r"(a[2].high), "r"(a[3].high),
          "r"(a[0].low), "r"(a[1].low), "r"(a[2].low), "r"(a[3].low), "r"(b0), "r"(b1));
}

__device__ void dot(const uint32_t (&a)[8], const uint32_t (&b)[4], int (&out)[4]) {
    asm volatile(
        "{\n"
        " .reg .s32 x0,x1,x2,x3,z;\n"
        " mov.s32 z, 0;\n"
        " mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "  {x0,x1,x2,x3}, {%4,%5,%6,%7}, {%12,%13}, {z,z,z,z};\n"
        " mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "  {%0,%1,%2,%3}, {%8,%9,%10,%11}, {%14,%15}, {x0,x1,x2,x3};\n"
        "}\n"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]),
          "r"(a[6]), "r"(a[7]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
}

__device__ float term(int dot, float qs, float ks, float gate, float scale) {
    const float bf16 = __bfloat162float(__float2bfloat16_rn(float(dot) * qs * ks));
    return fmaxf(bf16 * scale, 0.0f) * gate;
}

template<typename Score, int Bits>
__global__ void scan(const void* q, const float* query_scales, const uint32_t* keys,
                     const __nv_bfloat16* key_scales, const __nv_bfloat16* gates,
                     const int* bases, const int* lengths, Score* output,
                     int width, int gate_stride, float scale) {
    using Word = std::conditional_t<Bits == 4, QueryFragment, uint32_t>;
    constexpr int kQueryWords = kDimensions / sizeof(Word);
    constexpr int kKeyWords = kDimensions * Bits / 32;
    const int lane = threadIdx.x % 32;
    const int group = lane / 4;
    const int offset = lane % 4;
    const int row = blockIdx.y;
    const int first = blockIdx.x * 128 + (threadIdx.x / 32) * 8;
    const int length = lengths[row];
    const int base = bases[row];
    const auto* head0 = static_cast<const Word*>(q) + (row * kHeads + group) * kQueryWords + offset;
    const auto* head8 = head0 + 8 * kQueryWords;
    Word fragments[Bits];
    #pragma unroll
    for (int i = 0; i < Bits / 2; ++i) {
        fragments[2 * i] = head0[4 * i];
        fragments[2 * i + 1] = head8[4 * i];
    }
    const float q0 = query_scales[row * kHeads + group];
    const float q8 = query_scales[row * kHeads + group + 8];
    const float g0 = __bfloat162float(gates[row * gate_stride + group]);
    const float g8 = __bfloat162float(gates[row * gate_stride + group + 8]);

    #pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
        const int token = first + chunk * 64;
        if (token >= length) {
            if (lane < 8) {
                output[row * width + token + lane] = Score(-CUDART_INF_F);
            }
            continue;
        }
        const int key_token = token + group;
        const bool live = key_token < length;
        uint32_t b[Bits / 2] = {};
        if (live) {
            const auto* key = keys + (int64_t(base) + key_token) * kKeyWords + offset;
            #pragma unroll
            for (int i = 0; i < Bits / 2; ++i) {
                b[i] = key[4 * i];
            }
        }
        int products[4];
        if constexpr (Bits == 4) {
            constexpr unsigned kBias = 0x88888888u;
            dot(fragments, live ? b[0] ^ kBias : 0, live ? b[1] ^ kBias : 0, products);
        } else {
            dot(fragments, b, products);
        }
        const float local_scale = live ? __bfloat162float(key_scales[base + key_token]) : 0;
        #pragma unroll
        for (int col = 0; col < 2; ++col) {
            const int position = token + offset * 2 + col;
            const float ks = __shfl_sync(kWarpMask, local_scale, (offset * 2 + col) * 4);
            float value = term(products[col], q0, ks, g0, scale)
                        + term(products[2 + col], q8, ks, g8, scale);
            #pragma unroll
            for (int xor_mask = 16; xor_mask >= 4; xor_mask /= 2) {
                value += __shfl_xor_sync(kWarpMask, value, xor_mask);
            }
            if (group == 0) {
                output[row * width + position] = Score(position < length ? value : -CUDART_INF_F);
            }
        }
    }
}

}  // namespace

template<int Bits>
void launch(const void* fragments, const float* query_scales, const uint32_t* keys,
            const __nv_bfloat16* key_scales, const __nv_bfloat16* gates, const int* bases,
            const int* lengths, void* output, int batch, int width, int gate_stride,
            float scale, int half_output, cudaStream_t stream) {
    const dim3 grid(width / 128, batch);
    if (half_output) {
        scan<__half, Bits><<<grid, 256, 0, stream>>>(fragments, query_scales, keys, key_scales, gates,
                                                    bases, lengths, static_cast<__half*>(output),
                                                    width, gate_stride, scale);
    } else {
        scan<float, Bits><<<grid, 256, 0, stream>>>(fragments, query_scales, keys, key_scales, gates,
                                                   bases, lengths, static_cast<float*>(output),
                                                   width, gate_stride, scale);
    }
}

extern "C" int warp_scan(const int8_t* q, void* packed, const float* query_scales,
                         const uint32_t* keys, const __nv_bfloat16* key_scales,
                         const __nv_bfloat16* gates, const int* bases, const int* lengths,
                         void* output, int batch, int width, int gate_stride, float scale,
                         int half_output, int key_bits, cudaStream_t stream) {
    if (key_bits == 4) {
        auto* fragments = static_cast<QueryFragment*>(packed);
        if (q != nullptr) {
            pack_query<<<batch, 128, 0, stream>>>(q, fragments);
        }
        launch<4>(fragments, query_scales, keys, key_scales, gates, bases, lengths,
                  output, batch, width, gate_stride, scale, half_output, stream);
    } else {
        launch<8>(q, query_scales, keys, key_scales, gates, bases, lengths,
                  output, batch, width, gate_stride, scale, half_output, stream);
    }
    return cudaGetLastError();
}
