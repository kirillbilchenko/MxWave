// SPDX-License-Identifier: Apache-2.0
// Copyright MxWave contributors.
static thread_local std::string nv_error;
#define NV_EXPORT extern "C" __attribute__((visibility("default")))
NV_EXPORT const char* nv_last_error() { return nv_error.c_str(); }
NV_EXPORT int nv_plan(int m, int n, int k, int block, int topk, int experts, int* p) {
  int sms = 0, shared = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  auto c = MARLIN_NAMESPACE_NAME::determine_exec_config(
      vllm::kBFloat16, vllm::kFE2M1f, vllm::kBFloat16, vllm::kFE4M3fn,
      m, n, k, experts, topk, (block + 15) / 16, block == 8, 4, 16,
      false, false, false, 4, shared, sms);
  p[0] = c.tb_cfg.thread_k; p[1] = c.tb_cfg.thread_n; p[2] = c.blocks_per_sm;
  return p[0] > 0 && p[1] > 0 ? 0 : 1;
}
NV_EXPORT int nv_run(const void* x, const void* w, void* s, void* g, void* y,
    void* tmp, void* locks, void* ids, void* experts, void* padded, void* weights,
    int m, int n, int k, int e, int topk, int block, int multiply,
    int tk, int tn, int bpsm, uintptr_t stream) {
  try {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    MARLIN_NAMESPACE_NAME::marlin_mm(x, w, y, tmp, nullptr, nullptr, s, g, nullptr,
        ids, experts, padded, weights, block, e, topk, multiply != 0, m, n, k,
        locks, vllm::kBFloat16, vllm::kFE2M1f, vllm::kBFloat16, vllm::kFE4M3fn,
        false, false, 16, 0, reinterpret_cast<cudaStream_t>(stream), tk, tn,
        sms, bpsm, false, true, false);
    auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return 0;
  } catch (const std::exception& err) { nv_error = err.what(); return 1; }
}
NV_EXPORT int nv_decode(const void* d, void* out, int vpe, int vectors, uintptr_t stream) {
  MARLIN_NAMESPACE_NAME::nv_decode_all<<<(vectors + 127) / 128, 128, 0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const MARLIN_NAMESPACE_NAME::NvScales*>(d),
      reinterpret_cast<int4*>(out), vpe, vectors);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}
