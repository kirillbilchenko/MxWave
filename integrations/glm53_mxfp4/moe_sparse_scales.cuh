// SPDX-License-Identifier: Apache-2.0
// Copyright MxWave contributors.
#pragma once
#include <cstdint>

namespace MARLIN_NAMESPACE_NAME {
struct MxWaveMoeScales {
  const uint32_t* records;
  const int4* exceptions;
  uint64_t palette;
};
static_assert(sizeof(MxWaveMoeScales) == 24);

__device__ __forceinline__ int4 mxwave_sparse_exception(
    const MxWaveMoeScales* descriptor, uint32_t record) {
  return descriptor->exceptions[record & 0x3fffffffu];
}

__device__ __forceinline__ bool mxwave_sparse_escape(uint32_t q) {
  return (q >> 30) == 3;
}

__device__ __forceinline__ int2 mxwave_sparse_raw_half(uint32_t palette, uint32_t q) {
  uint32_t selectors = (q | (q << 8)) & 0x00ff00ffu;
  selectors = (selectors | (selectors << 4)) & 0x0f0f0f0fu;
  selectors = (selectors | (selectors << 2)) & 0x33333333u;
  const uint32_t lo = __byte_perm(palette, 0, selectors);
  const uint32_t hi = __byte_perm(palette, 0, selectors >> 16);
  return make_int2(__byte_perm(lo, hi, 0x5410), __byte_perm(lo, hi, 0x7632));
}

__device__ __forceinline__ int4 mxwave_sparse_raw(
    const MxWaveMoeScales* descriptor, uint32_t record) {
  if (mxwave_sparse_escape(record)) return mxwave_sparse_exception(descriptor, record);
  const uint32_t palette = descriptor->palette;
  const int2 lo = mxwave_sparse_raw_half(palette, record & 0xffffu);
  const int2 hi = mxwave_sparse_raw_half(palette, record >> 16);
  return make_int4(lo.x, lo.y, hi.x, hi.y);
}

__device__ __forceinline__ bool mxwave_sparse_uses_nibbles(uint64_t palette) {
  return (palette & (uint64_t(1) << 40)) != 0;
}

__device__ __forceinline__ uint32_t mxwave_sparse_nibble_base(uint64_t palette) {
  return (palette >> 32) & 255u;
}

__device__ __forceinline__ uint32_t mxwave_sparse_pack_nibbles(int2 bytes, uint32_t base) {
  // Every source byte, including exceptions, was range-checked on CPU. Each
  // byte is >= base, so subtraction cannot borrow across byte boundaries.
  const uint32_t lo = uint32_t(bytes.x) - base * 0x01010101u;
  const uint32_t hi = uint32_t(bytes.y) - base * 0x01010101u;
  // Preserve the existing aligned-nibble loader order [0,1,4,5,2,3,6,7].
  return (lo & 0x0000000fu) | ((lo >> 4) & 0x000000f0u) |
         ((hi << 8) & 0x00000f00u) | ((hi << 4) & 0x0000f000u) |
         (lo & 0x000f0000u) | ((lo >> 4) & 0x00f00000u) |
         ((hi << 8) & 0x0f000000u) | ((hi << 4) & 0xf0000000u);
}

__device__ __forceinline__ uint32_t mxwave_sparse_nibble_half(
    uint32_t palette, uint32_t selectors, uint32_t base) {
  if (palette & (1u << 24)) {
    uint32_t spread = (selectors | (selectors << 8)) & 0x00ff00ffu;
    spread = (spread | (spread << 4)) & 0x0f0f0f0fu;
    spread = (spread | (spread << 2)) & 0x33333333u;
    return spread + ((palette & 255u) - base) * 0x11111111u;
  }
  return mxwave_sparse_pack_nibbles(mxwave_sparse_raw_half(palette, selectors), base);
}

__device__ __forceinline__ int2 mxwave_sparse_nibbles(
    const MxWaveMoeScales* descriptor, uint32_t record) {
  const uint32_t base = mxwave_sparse_nibble_base(descriptor->palette);
  if (mxwave_sparse_escape(record)) {
    const int4 raw = mxwave_sparse_exception(descriptor, record);
    return make_int2(mxwave_sparse_pack_nibbles(make_int2(raw.x, raw.y), base),
                     mxwave_sparse_pack_nibbles(make_int2(raw.z, raw.w), base));
  }
  const uint32_t palette = descriptor->palette;
  return make_int2(mxwave_sparse_nibble_half(palette, record & 0xffffu, base),
                   mxwave_sparse_nibble_half(palette, record >> 16, base));
}

__device__ __forceinline__ int4 mxwave_sparse_nibble_bf16(uint32_t q, uint32_t base_pair) {
  constexpr uint32_t mask = 0x07800780u;
  return make_int4(((q << 7) & mask) | base_pair,
                   ((q << 3) & mask) | base_pair,
                   ((q >> 1) & mask) | base_pair,
                   ((q >> 5) & mask) | base_pair);
}

__device__ __forceinline__ int4 mxwave_sparse_bytes_bf16(int2 bytes) {
  int4 result;
  dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(
      bytes.x, reinterpret_cast<nv_bfloat162*>(&result));
  dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(
      bytes.y, reinterpret_cast<nv_bfloat162*>(&result) + 2);
  return result;
}

__device__ __forceinline__ void mxwave_sparse_copy(
    int4* destination, const uint32_t* source) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
               :: "r"(address), "l"(source));
}

template <bool affine_only = false>
__device__ __forceinline__ int4 mxwave_sparse_fragment(
    const MxWaveMoeScales* descriptor, uint32_t record, int half) {
  if (mxwave_sparse_escape(record)) {
    const int4 raw = mxwave_sparse_exception(descriptor, record);
    const int2 bytes = half ? make_int2(raw.z, raw.w) : make_int2(raw.x, raw.y);
    int4 result;
    dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(
        bytes.x, reinterpret_cast<nv_bfloat162*>(&result));
    dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(
        bytes.y, reinterpret_cast<nv_bfloat162*>(&result) + 2);
    return result;
  }
  const uint32_t q = (record >> (16 * half)) & 0xffffu;
  const uint32_t palette = descriptor->palette;
  if (affine_only || (palette & (1u << 24))) {
    const uint32_t spread = (q & 0xffu) | ((q & 0xff00u) << 8);
    const uint32_t base = (palette & 255u) * 0x00800080u;
    constexpr uint32_t mask = 0x01800180u;
    if constexpr (!affine_only) {
      if (palette & (1u << 25)) {
        return make_int4(((spread << 7) & mask) | base, ((spread << 5) & mask) | base,
                         ((spread << 3) & mask) | base, ((spread << 1) & mask) | base);
      }
    }
    return make_int4(((spread << 7) & mask) + base, ((spread << 5) & mask) + base,
                     ((spread << 3) & mask) + base, ((spread << 1) & mask) + base);
  }
  uint32_t selectors = (q | (q << 8)) & 0x00ff00ffu;
  selectors = (selectors | (selectors << 4)) & 0x0f0f0f0fu;
  selectors = (selectors | (selectors << 2)) & 0x33333333u;
  const uint32_t lo = __byte_perm(palette, 0, selectors);
  const uint32_t hi = __byte_perm(palette, 0, selectors >> 16);
  const uint32_t first = __byte_perm(lo, hi, 0x5410);
  const uint32_t second = __byte_perm(lo, hi, 0x7632);
  int4 result;
  dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(
      first, reinterpret_cast<nv_bfloat162*>(&result));
  dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(
      second, reinterpret_cast<nv_bfloat162*>(&result) + 2);
  return result;
}

__global__ void mxwave_decode_moe_scales(
    const MxWaveMoeScales* descriptor, int4* output, int vectors) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  output[index] = mxwave_sparse_raw(descriptor, descriptor->records[index]);
}

__global__ void mxwave_decode_moe_bf16_scales(
    const MxWaveMoeScales* descriptor, int4* output, int vectors) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  const uint32_t q = descriptor->records[index];
  #if MXWAVE_MOE_SPARSE_NIBBLE_STAGE
  if (mxwave_sparse_uses_nibbles(descriptor->palette)) {
    const int2 nibbles = mxwave_sparse_nibbles(descriptor, q);
    const uint32_t base_pair = mxwave_sparse_nibble_base(descriptor->palette) * 0x00800080u;
    output[2 * index] = mxwave_sparse_nibble_bf16(nibbles.x, base_pair);
    output[2 * index + 1] = mxwave_sparse_nibble_bf16(nibbles.y, base_pair);
  } else {
    const int4 raw = mxwave_sparse_raw(descriptor, q);
    output[2 * index] = mxwave_sparse_bytes_bf16(make_int2(raw.x, raw.y));
    output[2 * index + 1] = mxwave_sparse_bytes_bf16(make_int2(raw.z, raw.w));
  }
  #else
  output[2 * index] = mxwave_sparse_fragment(descriptor, q, 0);
  output[2 * index + 1] = mxwave_sparse_fragment(descriptor, q, 1);
  #endif
}
}  // namespace MARLIN_NAMESPACE_NAME
