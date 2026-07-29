#pragma once

#include <map>

#include "apps/cifar-dense/appdata.hpp"
#include "platform/engine/cuda/manager.cuh"

namespace cifar_dense::cuda {

// Device-resident mirror of one AppData's weights/biases. The model parameters
// are READ-ONLY and never part of the CPU<->GPU pipeline handoff, so there is no
// reason for GPU kernels to stream them from uncached zero-copy pinned memory
// (measured ~5.7 GB/s on Orin vs ~60 GB/s DRAM). Activations stay zero-copy --
// they ARE the handoff surface.
struct DeviceWeights {
  float* c_w[5];  // conv1..conv5 weights
  float* c_b[5];  // conv1..conv5 biases
  float* f_w[3];  // fc1..fc3 weights
  float* f_b[3];  // fc1..fc3 biases
};

class CudaDispatcher {
 public:
  CudaDispatcher() = default;

  CudaDispatcher(const CudaDispatcher&) = delete;
  CudaDispatcher& operator=(const CudaDispatcher&) = delete;
  CudaDispatcher(CudaDispatcher&&) = delete;
  CudaDispatcher& operator=(CudaDispatcher&&) = delete;

  ~CudaDispatcher();

  ::cuda::CudaPinnedResource& get_mr() { return mgr_.get_mr(); }

  // Lazily upload appdata's weights to device memory (once per AppData; pool
  // slots are stable for the pipeline's lifetime) and return the mirror.
  const DeviceWeights& dev_weights(const cifar_dense::AppData& appdata);

  void run_stage_1_async(cifar_dense::AppData& appdata);
  void run_stage_2_async(cifar_dense::AppData& appdata);
  void run_stage_3_async(cifar_dense::AppData& appdata);
  void run_stage_4_async(cifar_dense::AppData& appdata);
  void run_stage_5_async(cifar_dense::AppData& appdata);
  void run_stage_6_async(cifar_dense::AppData& appdata);
  void run_stage_7_async(cifar_dense::AppData& appdata);
  void run_stage_8_async(cifar_dense::AppData& appdata);
  void run_stage_9_async(cifar_dense::AppData& appdata);
  void run_stage_10_async(cifar_dense::AppData& appdata);
  void run_stage_11_async(cifar_dense::AppData& appdata);

  using StageFn = void (CudaDispatcher::*)(cifar_dense::AppData&);

  static constexpr std::array<StageFn, 11> stage_functions = {
      &CudaDispatcher::run_stage_1_async,
      &CudaDispatcher::run_stage_2_async,
      &CudaDispatcher::run_stage_3_async,
      &CudaDispatcher::run_stage_4_async,
      &CudaDispatcher::run_stage_5_async,
      &CudaDispatcher::run_stage_6_async,
      &CudaDispatcher::run_stage_7_async,
      &CudaDispatcher::run_stage_8_async,
      &CudaDispatcher::run_stage_9_async,
      &CudaDispatcher::run_stage_10_async,
      &CudaDispatcher::run_stage_11_async,
  };

  void dispatch_stage(AppData& appdata, const int stage) {
    assert(stage >= 1 && stage <= 11);

    (this->*stage_functions[stage - 1])(appdata);

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }

  void dispatch_multi_stage(AppData& appdata, const int start_stage, const int end_stage) {
    assert(start_stage >= 1 && end_stage <= 11);

    for (int stage = start_stage; stage <= end_stage; stage++) {
      (this->*stage_functions[stage - 1])(appdata);
    }

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }

 private:
  ::cuda::CudaManager<::cuda::CudaPinnedResource> mgr_;
  std::map<const cifar_dense::AppData*, DeviceWeights> devw_;
};

}  // namespace cifar_dense::cuda
