#pragma once

#include "../appdata.hpp"
#include "builtin-apps/common/cuda/helpers.cuh"
#include "builtin-apps/common/cuda/manager.cuh"
#include "builtin-apps/debug_logger.hpp"
#include "model_data.cuh"

namespace cuda {

// clang-format off
void run_stage_1_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData& d_model_data, cuda::CudaManager& mgr);  // Conv 1
void run_stage_2_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData& d_model_data, cuda::CudaManager& mgr);  // MaxPool 1
void run_stage_3_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData& d_model_data, cuda::CudaManager& mgr);  // Conv 2
void run_stage_4_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData& d_model_data, cuda::CudaManager& mgr);  // MaxPool 2
void run_stage_5_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData& d_model_data, cuda::CudaManager& mgr);  // Conv 3
void run_stage_6_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData& d_model_data, cuda::CudaManager& mgr);  // Conv 4
void run_stage_7_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData& d_model_data, cuda::CudaManager& mgr);  // Conv 5
void run_stage_8_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData& d_model_data, cuda::CudaManager& mgr);  // MaxPool 3
void run_stage_9_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData& d_model_data, cuda::CudaManager& mgr);  // Linear

using DispatchFnBatch = void (*)(cifar_dense::AppDataBatch&, const cuda::DeviceModelData&, cuda::CudaManager&);

const DispatchFnBatch dispatch_fns_batch[] = {
    run_stage_1_async,
    run_stage_2_async,
    run_stage_3_async,
    run_stage_4_async,
    run_stage_5_async,
    run_stage_6_async,
    run_stage_7_async,
    run_stage_8_async,
    run_stage_9_async,
};
// clang-format on

inline void dispatch_multi_stage(cifar_dense::AppDataBatch& appdata,
                                 const cuda::DeviceModelData& d_model_data,
                                 const int start_stage,
                                 const int end_stage,
                                 cuda::CudaManager& mgr) {
  assert(start_stage >= 1 && end_stage <= 9);

  // Sync to GPU

  for (int stage = start_stage; stage <= end_stage; stage++) {
    dispatch_fns_batch[stage - 1](appdata, d_model_data, mgr);
  }

  CheckCuda(cudaStreamSynchronize(mgr.get_stream()));
}

}  // namespace cuda