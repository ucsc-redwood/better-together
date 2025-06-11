#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include "helpers.cuh"

namespace bm = benchmark;

class CudaEventTimer {
 public:
  explicit CudaEventTimer(bm::State &state,
                          const bool flush_l2_cache = false,
                          const cudaStream_t stream = nullptr)
      : stream_(stream), p_state_(&state) {
    // flush all of L2$
    if (flush_l2_cache) {
      int current_device = 0;
      CheckCuda(cudaGetDevice(&current_device));

      int l2_cache_bytes = 0;
      CheckCuda(cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize, current_device));

      if (l2_cache_bytes > 0) {
        const int memset_value = 0;
        int *l2_cache_buffer = nullptr;
        CheckCuda(cudaMalloc(reinterpret_cast<void **>(&l2_cache_buffer), l2_cache_bytes));
        CheckCuda(cudaMemsetAsync(l2_cache_buffer, memset_value, l2_cache_bytes, stream_));
        CheckCuda(cudaFree(l2_cache_buffer));
      }
    }

    CheckCuda(cudaEventCreate(&start_));
    CheckCuda(cudaEventCreate(&stop_));
    CheckCuda(cudaEventRecord(start_, stream_));
  }

  CudaEventTimer() = delete;

  /**
   * @brief Destroy the `CudaEventTimer` and ending the manual time range.
   *
   */
  ~CudaEventTimer() {
    CheckCuda(cudaEventRecord(stop_, stream_));
    CheckCuda(cudaEventSynchronize(stop_));
    float milliseconds = 0.0f;
    CheckCuda(cudaEventElapsedTime(&milliseconds, start_, stop_));
    p_state_->SetIterationTime(milliseconds / 1000.0f);
    CheckCuda(cudaEventDestroy(start_));
    CheckCuda(cudaEventDestroy(stop_));
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  cudaStream_t stream_;
  bm::State *p_state_;
};
