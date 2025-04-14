#pragma once

#include <cuda_runtime_api.h>

#include "../sparse_appdata.hpp"
#include "builtin-apps/common/cuda/manager.cuh"

namespace cifar_sparse::cuda {

// How many images to process per iteration together
constexpr auto kNumBatches = 16;

void process_stage_1(AppData &appdata);
void process_stage_2(AppData &appdata);
void process_stage_3(AppData &appdata);
void process_stage_4(AppData &appdata);
void process_stage_5(AppData &appdata);
void process_stage_6(AppData &appdata);
void process_stage_7(AppData &appdata);
void process_stage_8(AppData &appdata);
void process_stage_9(AppData &appdata);

template <int Stage>
  requires(Stage >= 1 && Stage <= 9)
void run_stage(AppData &appdata) {
  if constexpr (Stage == 1) {
    process_stage_1(appdata);
  } else if constexpr (Stage == 2) {
    process_stage_2(appdata);
  } else if constexpr (Stage == 3) {
    process_stage_3(appdata);
  } else if constexpr (Stage == 4) {
    process_stage_4(appdata);
  } else if constexpr (Stage == 5) {
    process_stage_5(appdata);
  } else if constexpr (Stage == 6) {
    process_stage_6(appdata);
  } else if constexpr (Stage == 7) {
    process_stage_7(appdata);
  } else if constexpr (Stage == 8) {
    process_stage_8(appdata);
  } else if constexpr (Stage == 9) {
    process_stage_9(appdata);
  }
}

namespace v2 {

template <typename MemResourceT>
  requires std::is_same_v<MemResourceT, ::cuda::CudaManagedResource> ||
           std::is_same_v<MemResourceT, ::cuda::CudaPinnedResource>
class CudaDispatcher final : public ::cuda::CudaManager<MemResourceT> {
 public:
  CudaDispatcher() = default;

  void run_stage_1_async(cifar_sparse::v2::AppData &appdata);
  void run_stage_2_async(cifar_sparse::v2::AppData &appdata);
  void run_stage_3_async(cifar_sparse::v2::AppData &appdata);
  void run_stage_4_async(cifar_sparse::v2::AppData &appdata);
  void run_stage_5_async(cifar_sparse::v2::AppData &appdata);
  void run_stage_6_async(cifar_sparse::v2::AppData &appdata);
  void run_stage_7_async(cifar_sparse::v2::AppData &appdata);
  void run_stage_8_async(cifar_sparse::v2::AppData &appdata);
  void run_stage_9_async(cifar_sparse::v2::AppData &appdata);

  using StageFn = void (CudaDispatcher::*)(cifar_sparse::v2::AppData &);

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
  // create a manager
  ::cuda::CudaManager<MemResourceT> mgr_;
};

}  // namespace v2

}  // namespace cifar_sparse::cuda
