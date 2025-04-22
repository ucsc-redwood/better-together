#pragma once

#include <cassert>

#include "../../affinity.hpp"
#include "../appdata.hpp"

namespace cifar_dense::omp {

void run_stage_1(AppData& appdata);  // Conv 1
void run_stage_2(AppData& appdata);  // MaxPool 1
void run_stage_3(AppData& appdata);  // Conv 2
void run_stage_4(AppData& appdata);  // MaxPool 2
void run_stage_5(AppData& appdata);  // Conv 3
void run_stage_6(AppData& appdata);  // Conv 4
void run_stage_7(AppData& appdata);  // Conv 5
void run_stage_8(AppData& appdata);  // MaxPool 3
void run_stage_9(AppData& appdata);  // Linear

using DispatchFnBatch = void (*)(AppData&);

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
                                 AppData& appdata,
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

}  // namespace cifar_dense::omp
