#pragma once

#include <cassert>

#include "../appdata.hpp"
#include "builtin-apps/affinity.hpp"

namespace cifar_sparse::omp {

namespace v2 {

void run_stage_1(cifar_sparse::v2::AppData& appdata);  // Conv 1
void run_stage_2(cifar_sparse::v2::AppData& appdata);  // MaxPool 1
void run_stage_3(cifar_sparse::v2::AppData& appdata);  // Conv 2
void run_stage_4(cifar_sparse::v2::AppData& appdata);  // MaxPool 2
void run_stage_5(cifar_sparse::v2::AppData& appdata);  // Conv 3
void run_stage_6(cifar_sparse::v2::AppData& appdata);  // Conv 4
void run_stage_7(cifar_sparse::v2::AppData& appdata);  // Conv 5
void run_stage_8(cifar_sparse::v2::AppData& appdata);  // MaxPool 3
void run_stage_9(cifar_sparse::v2::AppData& appdata);  // Linear

using DispatchFnBatch = void (*)(cifar_sparse::v2::AppData&);

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

inline void dispatch_multi_stage(const std::vector<int>& cores_to_use,
                                 const int num_threads,
                                 cifar_sparse::v2::AppData& appdata,
                                 const int start_stage,
                                 const int end_stage) {
  assert(start_stage >= 1 && end_stage <= 9);

#pragma omp parallel num_threads(num_threads)
  {
    bind_thread_to_cores(cores_to_use);

    for (int stage = start_stage; stage <= end_stage; stage++) {
      dispatch_fns_batch[stage - 1](appdata);
    }
  }
}

}  // namespace v2

}  // namespace cifar_sparse::omp
