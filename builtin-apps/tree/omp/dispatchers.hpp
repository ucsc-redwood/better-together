#pragma once

#include "../safe_tree_appdata.hpp"
#include "builtin-apps/affinity.hpp"

namespace tree::omp {

void run_stage_1(SafeAppData &appdata);
void run_stage_2(SafeAppData &appdata);
void run_stage_3(SafeAppData &appdata);
void run_stage_4(SafeAppData &appdata);
void run_stage_5(SafeAppData &appdata);
void run_stage_6(SafeAppData &appdata);
void run_stage_7(SafeAppData &appdata);

using DispatchFnBatch = void (*)(SafeAppData &);

const DispatchFnBatch dispatch_fns_batch[] = {
    run_stage_1,
    run_stage_2,
    run_stage_3,
    run_stage_4,
    run_stage_5,
    run_stage_6,
    run_stage_7,
};

static inline void dispatch_stage(SafeAppData &appdata, const int stage) {
  assert(stage >= 1 && stage <= 7);

#pragma omp parallel
  { dispatch_fns_batch[stage - 1](appdata); }
}

static inline void dispatch_multi_stage(SafeAppData &appdata,
                                        const int start_stage,
                                        const int end_stage) {
  assert(start_stage >= 1 && end_stage <= 7);

#pragma omp parallel
  {
    for (int stage = start_stage; stage <= end_stage; stage++) {
      dispatch_fns_batch[stage - 1](appdata);
    }
  }
}

static inline void dispatch_stage(const std::vector<int> &cores_to_use,
                                  const int num_threads,
                                  SafeAppData &appdata,
                                  const int stage) {
  assert(stage >= 1 && stage <= 7);

#pragma omp parallel num_threads(num_threads)
  {
    bind_thread_to_cores(cores_to_use);

    dispatch_fns_batch[stage - 1](appdata);
  }
}

static inline void dispatch_multi_stage(const std::vector<int> &cores_to_use,
                                        const int num_threads,
                                        SafeAppData &appdata,
                                        const int start_stage,
                                        const int end_stage) {
  assert(start_stage >= 1 && end_stage <= 7);

#pragma omp parallel num_threads(num_threads)
  {
    bind_thread_to_cores(cores_to_use);

    for (int stage = start_stage; stage <= end_stage; stage++) {
      dispatch_fns_batch[stage - 1](appdata);
    }
  }
}

}  // namespace tree::omp
