#pragma once

#include "../appdata.hpp"
#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"

namespace omp {

// ----------------------------------------------------------------------------
// Batched version
// ----------------------------------------------------------------------------

// clang-format off
void run_stage_1(cifar_dense::AppDataBatch& appdata);  // Conv 1
void run_stage_2(cifar_dense::AppDataBatch& appdata);  // MaxPool 1
void run_stage_3(cifar_dense::AppDataBatch& appdata);  // Conv 2
void run_stage_4(cifar_dense::AppDataBatch& appdata);  // MaxPool 2
void run_stage_5(cifar_dense::AppDataBatch& appdata);  // Conv 3
void run_stage_6(cifar_dense::AppDataBatch& appdata);  // Conv 4
void run_stage_7(cifar_dense::AppDataBatch& appdata);  // Conv 5
void run_stage_8(cifar_dense::AppDataBatch& appdata);  // MaxPool 3
void run_stage_9(cifar_dense::AppDataBatch& appdata);  // Linear
// clang-format on

using DispatchFnBatch = void (*)(cifar_dense::AppDataBatch&);

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
                                              const int end_stage) {
  assert(start_stage >= 1 && end_stage <= 9);

#pragma omp parallel num_threads(num_threads)
  {
    for (int stage = start_stage; stage <= end_stage; stage++) {
      dispatch_fns_batch[stage - 1](appdata);
    }
  }
}

inline void dispatch_multi_stage(const std::vector<int>& cores_to_use,
                                 const int num_threads,
                                 cifar_dense::AppDataBatch& appdata,
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

// template <ProcessorType PT>
// void dispatch_multi_stage(const int num_threads,
//                           cifar_dense::AppDataBatch& appdata,
//                           const int start_stage,
//                           const int end_stage) {
//   if constexpr (PT == ProcessorType::kLittleCore) {
//     dispatch_multi_stage(g_lit_cores, num_threads, appdata, start_stage, end_stage);
//   } else if constexpr (PT == ProcessorType::kMediumCore) {
//     dispatch_multi_stage(g_med_cores, num_threads, appdata, start_stage, end_stage);
//   } else if constexpr (PT == ProcessorType::kBigCore) {
//     dispatch_multi_stage(g_big_cores, num_threads, appdata, start_stage, end_stage);
//   }
// }

inline void dispatch_multi_stage(const int start_stage,
                                 const int end_stage,
                                 const ProcessorType pt,
                                 const int num_threads,
                                 cifar_dense::AppDataBatch& appdata) {
  if (pt == ProcessorType::kLittleCore) {
    dispatch_multi_stage(g_lit_cores, num_threads, appdata, start_stage, end_stage);
  } else if (pt == ProcessorType::kMediumCore) {
    dispatch_multi_stage(g_med_cores, num_threads, appdata, start_stage, end_stage);
  } else if (pt == ProcessorType::kBigCore) {
    dispatch_multi_stage(g_big_cores, num_threads, appdata, start_stage, end_stage);
  }
}

}  // namespace omp
