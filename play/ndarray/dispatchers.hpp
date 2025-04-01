#pragma once

#include "appdata.hpp"

namespace omp {

void run_stage_1(AppData& appdata);  // Conv 1
void run_stage_2(AppData& appdata);  // MaxPool 1
void run_stage_3(AppData& appdata);  // Conv 2
void run_stage_4(AppData& appdata);  // MaxPool 2
void run_stage_5(AppData& appdata);  // Conv 3
void run_stage_6(AppData& appdata);  // Conv 4
void run_stage_7(AppData& appdata);  // Conv 5
void run_stage_8(AppData& appdata);  // MaxPool 3
void run_stage_9(AppData& appdata);  // Linear

using DispatchFn = void (*)(AppData&);

const DispatchFn dispatch_fns[] = {
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

inline void dispatch_single_stage(AppData& appdata, const int stage, const int num_threads) {
  assert(stage >= 1 && stage <= 9);

#pragma omp parallel num_threads(num_threads)
  {
    // TODO: affinity
    dispatch_fns[stage - 1](appdata);
  }
}

inline void dispatch_multi_stage(AppData& appdata,
                                 const int start_stage,
                                 const int end_stage,
                                 const int num_threads) {
  assert(start_stage >= 1 && end_stage <= 9);

#pragma omp parallel num_threads(num_threads)
  {
    // TODO: affinity

    for (int stage = start_stage; stage <= end_stage; stage++) {
      dispatch_fns[stage - 1](appdata);
    }
  }
}

}  // namespace omp
