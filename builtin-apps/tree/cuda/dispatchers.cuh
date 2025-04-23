#pragma once

#include "../../common/cuda/manager.cuh"
#include "../safe_tree_appdata.hpp"

namespace tree::cuda {

class CudaDispatcher {
 public:
  CudaDispatcher() = default;

  CudaDispatcher(const CudaDispatcher &) = delete;
  CudaDispatcher &operator=(const CudaDispatcher &) = delete;
  CudaDispatcher(CudaDispatcher &&) = delete;
  CudaDispatcher &operator=(CudaDispatcher &&) = delete;

  ::cuda::CudaManagedResource &get_mr() { return mgr_.get_mr(); }

  void run_stage_1_async(tree::SafeAppData &appdata);
  void run_stage_2_async(tree::SafeAppData &appdata);
  void run_stage_3_async(tree::SafeAppData &appdata);
  void run_stage_4_async(tree::SafeAppData &appdata);
  void run_stage_5_async(tree::SafeAppData &appdata);
  void run_stage_6_async(tree::SafeAppData &appdata);
  void run_stage_7_async(tree::SafeAppData &appdata);

  using StageFn = void (CudaDispatcher::*)(tree::SafeAppData &);

  static constexpr std::array<StageFn, 7> stage_functions = {
      &CudaDispatcher::run_stage_1_async,
      &CudaDispatcher::run_stage_2_async,
      &CudaDispatcher::run_stage_3_async,
      &CudaDispatcher::run_stage_4_async,
      &CudaDispatcher::run_stage_5_async,
      &CudaDispatcher::run_stage_6_async,
      &CudaDispatcher::run_stage_7_async,
  };

  void dispatch_stage(SafeAppData &appdata, const int stage) {
    assert(stage >= 1 && stage <= 7);

    (this->*stage_functions[stage - 1])(appdata);

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }

  void dispatch_multi_stage(SafeAppData &appdata, const int start_stage, const int end_stage) {
    assert(start_stage >= 1 && end_stage <= 7);

    for (int stage = start_stage; stage <= end_stage; stage++) {
      (this->*stage_functions[stage - 1])(appdata);
    }

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }

 private:
  ::cuda::CudaManager<::cuda::CudaManagedResource> mgr_;
};

}  // namespace tree::cuda
