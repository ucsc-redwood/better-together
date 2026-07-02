#pragma once

#include "apps/cifar-dense/appdata.hpp"
#include "platform/engine/cuda/manager.cuh"

namespace cifar_dense::cuda {

class CudaDispatcher {
 public:
  CudaDispatcher() = default;

  CudaDispatcher(const CudaDispatcher&) = delete;
  CudaDispatcher& operator=(const CudaDispatcher&) = delete;
  CudaDispatcher(CudaDispatcher&&) = delete;
  CudaDispatcher& operator=(CudaDispatcher&&) = delete;

  ::cuda::CudaPinnedResource& get_mr() { return mgr_.get_mr(); }

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
};

}  // namespace cifar_dense::cuda
