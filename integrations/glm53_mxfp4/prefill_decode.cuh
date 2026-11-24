// Independently authored MxWave reconstruction experiment for the licensed
// ZipServ triple-bitmap layout. Included after the hash-bound external L_API.cu.
// No GEMM, FP4 value, packed layout, or floating-point arithmetic is changed.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint32_t mxwave_decode_pair(
    const uint8_t* common, const __nv_bfloat16* rare,
    uint64_t b0, uint64_t b1, uint64_t b2,
    int common_start, int rare_start, int position, unsigned exponent) {
    const uint64_t selected = b0 | b1 | b2;
    const int preceding = __popcll(selected & ((uint64_t(1) << position) - 1));
    const bool first_common = (selected >> position) & 1;
    uint32_t result = 0;
    #pragma unroll
    for (int item = 0; item < 2; ++item) {
        const int bit = position + item;
        const int count = preceding + (item && first_common);
        uint16_t value;
        if ((selected >> bit) & 1) {
            const unsigned code = ((b0 >> bit) & 1)
                | (((b1 >> bit) & 1) << 1) | (((b2 >> bit) & 1) << 2);
            const unsigned packed = common[common_start + count];
            value = uint16_t((packed & 127) | ((packed & 128) << 8)
                             | (((exponent + code) & 255u) << 7));
        } else {
            value = reinterpret_cast<const uint16_t*>(rare)[rare_start + bit - count];
        }
        result |= uint32_t(value) << (16 * item);
    }
    return result;
}

template<bool Coalesced>
__global__ void mxwave_prefill_decode_kernel(
    const uint8_t* common, const __nv_bfloat16* rare,
    const uint64_t* b0, const uint64_t* b1, const uint64_t* b2,
    const int* medium, const int* global, int max_common, int max_rare,
    uint8_t exponent, __nv_bfloat16* output, int rows, int width) {
    using Config = TilingConfigBF16TripleBitmap<4, 1, 4>;
    extern __shared__ __align__(128) unsigned char storage[];
    auto* shared_b0 = reinterpret_cast<uint64_t*>(storage);
    auto* shared_b1 = shared_b0 + 64;
    auto* shared_b2 = shared_b1 + 64;
    auto* shared_common = reinterpret_cast<uint8_t*>(shared_b2 + 64);
    const int padding = (128 - max_common % 128) % 128;
    auto* shared_rare = reinterpret_cast<__nv_bfloat16*>(shared_common + max_common + padding);
    const int tile = blockIdx.y * (width / 64) + blockIdx.x;
    const int start_common = global[2 * tile];
    const int start_rare = global[2 * tile + 1];
    CopyTripleBitmapToShared<Config::TILE_BITMAP_M_V3, Config>(
        shared_b0, shared_b1, shared_b2, b0 + tile * 64, b1 + tile * 64, b2 + tile * 64);
    CopyCompressedDataToShared<Config>(
        shared_common, shared_rare, common + start_common, rare + start_rare,
        global[2 * tile + 2] - start_common, global[2 * tile + 3] - start_rare);
    cp_async_group_commit();
    cp_async_wait_group<0>();
    __syncthreads();

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int origin_row = blockIdx.y * 64 + warp * 16;
    const int origin_column = blockIdx.x * 64;
    const int warp_common = medium[2 * (tile * 4 + warp)];
    const int warp_rare = medium[2 * (tile * 4 + warp) + 1];
    shared_b0 += warp * 16;
    shared_b1 += warp * 16;
    shared_b2 += warp * 16;
    if constexpr (Coalesced) {
        // Lanes 0..15 own the 16 small tiles for one median tile. Compute
        // exclusive common-value offsets once, then each warp writes full rows.
        const int own = lane < 16
            ? __popcll(shared_b0[lane] | shared_b1[lane] | shared_b2[lane]) : 0;
        int inclusive = own;
        #pragma unroll
        for (int distance = 1; distance < 16; distance *= 2) {
            const int previous = __shfl_up_sync(0xffffffff, inclusive, distance);
            if (lane >= distance) inclusive += previous;
        }
        const int prefix = inclusive - own;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int small = (lane / 4) * 2 + half;
            const int before = __shfl_sync(0xffffffff, prefix, small);
            const uint64_t v0 = shared_b0[small];
            const uint64_t v1 = shared_b1[small];
            const uint64_t v2 = shared_b2[small];
            #pragma unroll
            for (int row = 0; row < 8; ++row) {
                const int position = row * 8 + (lane % 4) * 2;
                const uint32_t value = mxwave_decode_pair(
                    shared_common, shared_rare, v0, v1, v2,
                    warp_common + before, warp_rare + small * 64 - before,
                    position, exponent);
                auto* destination = output + (origin_row + half * 8 + row) * width
                    + origin_column + lane * 2;
                *reinterpret_cast<uint32_t*>(destination) = value;
            }
        }
    } else {
        // Control the removal of shared output staging separately from the
        // row-coalesced mapping; this preserves the original pair-to-lane map.
        int high = warp_common;
        int low = warp_rare;
        #pragma unroll
        for (int small = 0; small < 16; ++small) {
            const uint64_t v0 = shared_b0[small];
            const uint64_t v1 = shared_b1[small];
            const uint64_t v2 = shared_b2[small];
            const uint32_t value = mxwave_decode_pair(
                shared_common, shared_rare, v0, v1, v2, high, low, lane * 2, exponent);
            const int row = origin_row + (small % 2) * 8 + lane / 4;
            const int column = origin_column + (small / 2) * 8 + (lane % 4) * 2;
            *reinterpret_cast<uint32_t*>(output + row * width + column) = value;
            const int count = __popcll(v0 | v1 | v2);
            high += count;
            low += 64 - count;
        }
    }
}

extern "C" int mxwave_prefill_decode(
    int mode, void* stream, const void* common, const void* rare,
    const void* b0, const void* b1, const void* b2, const void* medium, const void* global,
    int max_common, int max_rare, int exponent, void* output, int rows, int width) {
    if (mode < 0 || mode > 2 || rows <= 0 || width <= 0 || rows % 64 || width % 64
        || max_common < 0 || max_common > 4096 || max_rare < 0 || max_rare > 4096
        || exponent < 0 || exponent > 255 || !output) return int(cudaErrorInvalidValue);
    if (mode == 0 && exponent > 248) return int(cudaErrorInvalidValue);
    if (mode == 0) return int(BF16TripleBitmap_Decompress_API(
        static_cast<cudaStream_t>(stream), static_cast<const uint8_t*>(common),
        static_cast<const __nv_bfloat16*>(rare), static_cast<const uint64_t*>(b0),
        static_cast<const uint64_t*>(b1), static_cast<const uint64_t*>(b2),
        static_cast<const int*>(medium), static_cast<const int*>(global), max_common, max_rare,
        uint8_t(exponent), static_cast<__nv_bfloat16*>(output), rows, width));
    const int shared = 64 * 8 * 3 + max_common + max_rare * 2 + 256;
    const dim3 grid(width / 64, rows / 64);
    if (mode == 1) mxwave_prefill_decode_kernel<false><<<grid, 128, shared,
        static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(common),
        static_cast<const __nv_bfloat16*>(rare), static_cast<const uint64_t*>(b0),
        static_cast<const uint64_t*>(b1), static_cast<const uint64_t*>(b2),
        static_cast<const int*>(medium), static_cast<const int*>(global), max_common, max_rare,
        uint8_t(exponent), static_cast<__nv_bfloat16*>(output), rows, width);
    else mxwave_prefill_decode_kernel<true><<<grid, 128, shared,
        static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(common),
        static_cast<const __nv_bfloat16*>(rare), static_cast<const uint64_t*>(b0),
        static_cast<const uint64_t*>(b1), static_cast<const uint64_t*>(b2),
        static_cast<const int*>(medium), static_cast<const int*>(global), max_common, max_rare,
        uint8_t(exponent), static_cast<__nv_bfloat16*>(output), rows, width);
    return int(cudaGetLastError());
}

extern "C" int mxwave_prefill_resources(int mode, int common, int rare, int* result) {
    using Config = TilingConfigBF16TripleBitmap<4, 1, 4>;
    if (!result || mode < 0 || mode > 2) return int(cudaErrorInvalidValue);
    const int shared = 64 * 8 * 3 + common + rare * 2 + 256
        + (mode == 0 ? 64 * (64 + PADDING_SHARED_MEM_FOR_DECOMP) * 2 : 0);
    const void* kernel = mode == 0
        ? reinterpret_cast<const void*>(BF16TripleBitmap_Decompress_Kernel<Config>)
        : mode == 1 ? reinterpret_cast<const void*>(mxwave_prefill_decode_kernel<false>)
                    : reinterpret_cast<const void*>(mxwave_prefill_decode_kernel<true>);
    cudaFuncAttributes attributes{};
    auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) return int(status);
    int blocks = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 128, shared);
    if (status != cudaSuccess) return int(status);
    result[0] = shared;
    result[1] = attributes.numRegs;
    result[2] = int(attributes.sharedSizeBytes);
    result[3] = blocks;
    result[4] = 128;
    return 0;
}
