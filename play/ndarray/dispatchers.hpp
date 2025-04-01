#pragma once

#include "appdata.hpp"

namespace omp {

void dispatch_stage_1(AppData& appdata);  // Conv 1
void dispatch_stage_2(AppData& appdata);  // MaxPool 1
void dispatch_stage_3(AppData& appdata);  // Conv 2
void dispatch_stage_4(AppData& appdata);  // MaxPool 2
void dispatch_stage_5(AppData& appdata);  // Conv 3
void dispatch_stage_6(AppData& appdata);  // Conv 4
void dispatch_stage_7(AppData& appdata);  // Conv 5
void dispatch_stage_8(AppData& appdata);  // MaxPool 3
void dispatch_stage_9(AppData& appdata);  // Linear

using DispatchFn = void (*)(AppData&);

const DispatchFn dispatch_fns[] = {
    dispatch_stage_1,
    dispatch_stage_2,
    dispatch_stage_3,
    dispatch_stage_4,
    dispatch_stage_5,
    dispatch_stage_6,
    dispatch_stage_7,
    dispatch_stage_8,
    dispatch_stage_9,
};

inline void dispatch_multi_stage(AppData& appdata, const int start_stage, const int end_stage) {
  assert(start_stage >= 1 && end_stage <= 9);

#pragma omp parallel
  {
    // TODO: affinity

    for (int stage = start_stage; stage <= end_stage; stage++) {
      dispatch_fns[stage - 1](appdata);
    }
  }
}

}  // namespace omp
