#pragma once

#include "apps/tree/safe_tree_appdata.hpp"
#include "platform/registry/affinity.hpp"

namespace tree::omp {

void run_stage_1(SafeAppData& appdata);
void run_stage_2(SafeAppData& appdata);
void run_stage_3(SafeAppData& appdata);
void run_stage_4(SafeAppData& appdata);
void run_stage_5(SafeAppData& appdata);
void run_stage_6(SafeAppData& appdata);
void run_stage_7(SafeAppData& appdata);

using DispatchFnBatch = void (*)(SafeAppData&);

const DispatchFnBatch dispatch_fns_batch[] = {
    run_stage_1,
    run_stage_2,
    run_stage_3,
    run_stage_4,
    run_stage_5,
    run_stage_6,
    run_stage_7,
};

static inline void dispatch_stage(SafeAppData& appdata, const int stage) {
  assert(stage >= 1 && stage <= 7);

#pragma omp parallel
  {
    dispatch_fns_batch[stage - 1](appdata);
  }
}

static inline void dispatch_multi_stage(SafeAppData& appdata,
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

static inline void dispatch_stage(const std::vector<int>& cores_to_use,
                                  const int num_threads,
                                  SafeAppData& appdata,
                                  const int stage) {
  assert(stage >= 1 && stage <= 7);

#pragma omp parallel num_threads(num_threads)
  {
    bind_thread_to_cores(cores_to_use);

    dispatch_fns_batch[stage - 1](appdata);
  }
}

static inline void dispatch_multi_stage(const std::vector<int>& cores_to_use,
                                        const int num_threads,
                                        SafeAppData& appdata,
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

// --------------------------------------------------------------------------
// AppData overload: genuinely chains stage-to-stage (single buffer per field,
// no golden/_out split) -- extracted from what HostTreeManager::initialize()
// (tree_appdata.cpp) already did to build the golden every SafeAppData test
// compares against. For real-workload profiling, not correctness testing --
// SafeAppData above remains the differential/oracle path, unchanged.
//
// Each run_stage_N(AppData&) is self-contained: it owns its own #pragma omp
// parallel for where the original golden-building code had one, and is plain
// single-threaded where the golden always was (stages 2/3/6). So, unlike the
// SafeAppData overloads above, this wrapper is a plain sequential loop, NOT
// wrapped in its own #pragma omp parallel {} -- stages 2/3/6 have no
// worksharing construct and would race if forced into a shared parallel team.
// --------------------------------------------------------------------------

void run_stage_1(tree::AppData& appdata);
void run_stage_2(tree::AppData& appdata);
void run_stage_3(tree::AppData& appdata);
void run_stage_4(tree::AppData& appdata);
void run_stage_5(tree::AppData& appdata);
void run_stage_6(tree::AppData& appdata);
void run_stage_7(tree::AppData& appdata);

using DispatchFnBatchAppData = void (*)(tree::AppData&);

const DispatchFnBatchAppData dispatch_fns_batch_appdata[] = {
    run_stage_1,
    run_stage_2,
    run_stage_3,
    run_stage_4,
    run_stage_5,
    run_stage_6,
    run_stage_7,
};

static inline void dispatch_stage(tree::AppData& appdata, const int stage) {
  assert(stage >= 1 && stage <= 7);
  dispatch_fns_batch_appdata[stage - 1](appdata);
}

static inline void dispatch_multi_stage(tree::AppData& appdata,
                                        const int start_stage,
                                        const int end_stage) {
  assert(start_stage >= 1 && end_stage <= 7);
  for (int stage = start_stage; stage <= end_stage; stage++) {
    dispatch_fns_batch_appdata[stage - 1](appdata);
  }
}

// Core-pinning intentionally unsupported for the AppData path -- each
// run_stage_N(AppData&) manages its own parallelism internally (its own
// #pragma omp parallel for where needed), so there's no single outer team to
// pin. Cores/num_threads are accepted (and ignored) only so call sites shared
// with the SafeAppData path don't need a separate signature.
static inline void dispatch_multi_stage(const std::vector<int>& /*cores_to_use*/,
                                        const int /*num_threads*/,
                                        tree::AppData& appdata,
                                        const int start_stage,
                                        const int end_stage) {
  dispatch_multi_stage(appdata, start_stage, end_stage);
}

}  // namespace tree::omp
