#pragma once

#include "apps/tree/safe_tree_appdata.hpp"
#include "platform/engine/cuda/manager.cuh"

namespace tree::cuda {

class CudaDispatcher {
 public:
  CudaDispatcher() = default;

  CudaDispatcher(const CudaDispatcher&) = delete;
  CudaDispatcher& operator=(const CudaDispatcher&) = delete;
  CudaDispatcher(CudaDispatcher&&) = delete;
  CudaDispatcher& operator=(CudaDispatcher&&) = delete;

  ::cuda::CudaPinnedResource& get_mr() { return mgr_.get_mr(); }

  void run_stage_1_async(tree::SafeAppData& appdata);
  void run_stage_2_async(tree::SafeAppData& appdata);
  void run_stage_3_async(tree::SafeAppData& appdata);
  void run_stage_4_async(tree::SafeAppData& appdata);
  void run_stage_5_async(tree::SafeAppData& appdata);
  void run_stage_6_async(tree::SafeAppData& appdata);
  void run_stage_7_async(tree::SafeAppData& appdata);

  using StageFn = void (CudaDispatcher::*)(tree::SafeAppData&);

  static constexpr std::array<StageFn, 7> stage_functions = {
      &CudaDispatcher::run_stage_1_async,
      &CudaDispatcher::run_stage_2_async,
      &CudaDispatcher::run_stage_3_async,
      &CudaDispatcher::run_stage_4_async,
      &CudaDispatcher::run_stage_5_async,
      &CudaDispatcher::run_stage_6_async,
      &CudaDispatcher::run_stage_7_async,
  };

  void dispatch_stage(tree::SafeAppData& appdata, const int stage) {
    assert(stage >= 1 && stage <= 7);

    (this->*stage_functions[stage - 1])(appdata);

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }

  void dispatch_multi_stage(tree::SafeAppData& appdata,
                            const int start_stage,
                            const int end_stage) {
    assert(start_stage >= 1 && end_stage <= 7);

    for (int stage = start_stage; stage <= end_stage; stage++) {
      (this->*stage_functions[stage - 1])(appdata);
    }

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }

  // --------------------------------------------------------------------------
  // AppData overloads: genuinely chains stage-to-stage (single buffer per
  // field, no golden/_out split) -- for real-workload profiling, not
  // correctness testing. See apps/tree/omp/dispatchers.hpp for the OMP
  // rationale; identical reasoning applies here. SafeAppData above remains
  // the differential/oracle path, unchanged. The chunk-end
  // cudaGetLastError()/cudaDeviceSynchronize() below is preserved exactly --
  // it's the existing GPU->CPU visibility point a hybrid schedule relies on.
  // --------------------------------------------------------------------------

  void run_stage_1_async(tree::AppData& appdata);
  void run_stage_2_async(tree::AppData& appdata);
  void run_stage_3_async(tree::AppData& appdata);
  void run_stage_4_async(tree::AppData& appdata);
  void run_stage_5_async(tree::AppData& appdata);
  void run_stage_6_async(tree::AppData& appdata);
  void run_stage_7_async(tree::AppData& appdata);

  using StageFnAppData = void (CudaDispatcher::*)(tree::AppData&);

  static constexpr std::array<StageFnAppData, 7> stage_functions_appdata = {
      &CudaDispatcher::run_stage_1_async,
      &CudaDispatcher::run_stage_2_async,
      &CudaDispatcher::run_stage_3_async,
      &CudaDispatcher::run_stage_4_async,
      &CudaDispatcher::run_stage_5_async,
      &CudaDispatcher::run_stage_6_async,
      &CudaDispatcher::run_stage_7_async,
  };

  void dispatch_stage(tree::AppData& appdata, const int stage) {
    assert(stage >= 1 && stage <= 7);

    (this->*stage_functions_appdata[stage - 1])(appdata);

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }

  void dispatch_multi_stage(tree::AppData& appdata, const int start_stage, const int end_stage) {
    assert(start_stage >= 1 && end_stage <= 7);

    for (int stage = start_stage; stage <= end_stage; stage++) {
      (this->*stage_functions_appdata[stage - 1])(appdata);
    }

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }

 private:
  ::cuda::CudaManager<::cuda::CudaPinnedResource> mgr_;
};

}  // namespace tree::cuda
