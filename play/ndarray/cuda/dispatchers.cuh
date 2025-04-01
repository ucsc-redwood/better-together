#pragma once

#include "../appdata.hpp"
#include "builtin-apps/common/cuda/helpers.cuh"
#include "builtin-apps/common/cuda/manager.cuh"

namespace cuda {

void run_stage_1(cifar_dense::AppDataBatch& appdata, cuda::CudaManager& mgr);  // Conv 1
void run_stage_2(cifar_dense::AppDataBatch& appdata, cuda::CudaManager& mgr);  // MaxPool 1
void run_stage_3(cifar_dense::AppDataBatch& appdata, cuda::CudaManager& mgr);  // Conv 2
void run_stage_4(cifar_dense::AppDataBatch& appdata, cuda::CudaManager& mgr);  // MaxPool 2
void run_stage_5(cifar_dense::AppDataBatch& appdata, cuda::CudaManager& mgr);  // Conv 3
void run_stage_6(cifar_dense::AppDataBatch& appdata, cuda::CudaManager& mgr);  // Conv 4
void run_stage_7(cifar_dense::AppDataBatch& appdata, cuda::CudaManager& mgr);  // Conv 5
void run_stage_8(cifar_dense::AppDataBatch& appdata, cuda::CudaManager& mgr);  // MaxPool 3
void run_stage_9(cifar_dense::AppDataBatch& appdata, cuda::CudaManager& mgr);  // Linear

using DispatchFnBatch = void (*)(cifar_dense::AppDataBatch&, cuda::CudaManager&);

const DispatchFnBatch dispatch_fns_batch[] = {
    run_stage_1,
    run_stage_2,
    run_stage_3,
    run_stage_4,
    run_stage_5,
    run_stage_6,
    run_stage_7,
    run_stage_8,
    run_stage_9,
};

inline void dispatch_multi_stage_unrestricted(const int num_threads,
                                              cifar_dense::AppDataBatch& appdata,
                                              const int start_stage,
                                              const int end_stage,
                                              cuda::CudaManager& mgr) {
  assert(start_stage >= 1 && end_stage <= 9);

#pragma omp parallel num_threads(num_threads)
  {
    for (int stage = start_stage; stage <= end_stage; stage++) {
      dispatch_fns_batch[stage - 1](appdata, mgr);
    }
  }
}

inline void dispatch_multi_stage(cifar_dense::AppDataBatch& appdata,
                                 const int start_stage,
                                 const int end_stage,
                                 cuda::CudaManager& mgr) {
  assert(start_stage >= 1 && end_stage <= 9);

  // Sync to GPU

  for (int stage = start_stage; stage <= end_stage; stage++) {
    dispatch_fns_batch[stage - 1](appdata, mgr);
  }

  CheckCuda(cudaDeviceSynchronize());
  // Sync to Host
}

}  // namespace cuda