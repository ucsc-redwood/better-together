#pragma once

#include "../../common/cuda/manager.cuh"
#include "../appdata.hpp"

namespace cifar_dense::cuda {

class CudaDispatcher {
 public:
  CudaDispatcher() = default;

  ::cuda::CudaManagedResource &get_mr() { return mgr_.get_mr(); }

  void run_stage_1_async(cifar_dense::AppData &appdata);
  void run_stage_2_async(cifar_dense::AppData &appdata);
  void run_stage_3_async(cifar_dense::AppData &appdata);
  void run_stage_4_async(cifar_dense::AppData &appdata);
  void run_stage_5_async(cifar_dense::AppData &appdata);
  void run_stage_6_async(cifar_dense::AppData &appdata);
  void run_stage_7_async(cifar_dense::AppData &appdata);
  void run_stage_8_async(cifar_dense::AppData &appdata);
  void run_stage_9_async(cifar_dense::AppData &appdata);

  using StageFn = void (CudaDispatcher::*)(cifar_dense::AppData &);

  static constexpr std::array<StageFn, 9> stage_functions = {
      &CudaDispatcher::run_stage_1_async,
      &CudaDispatcher::run_stage_2_async,
      &CudaDispatcher::run_stage_3_async,
      &CudaDispatcher::run_stage_4_async,
      &CudaDispatcher::run_stage_5_async,
      &CudaDispatcher::run_stage_6_async,

      &CudaDispatcher::run_stage_7_async,
      &CudaDispatcher::run_stage_8_async,
      &CudaDispatcher::run_stage_9_async,
  };

 private:
  ::cuda::CudaManager<::cuda::CudaManagedResource> mgr_;
};

}  // namespace cifar_dense::cuda
