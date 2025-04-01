#pragma once

#include <cuda_runtime_api.h>

#include "../sparse_appdata.hpp"

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

}  // namespace cifar_sparse::cuda
