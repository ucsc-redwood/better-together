#pragma once

#include "../dense_appdata.hpp"

namespace cifar_dense::omp {

// Stage processing functions
void process_stage_1(cifar_dense::AppData &app_data);
void process_stage_2(cifar_dense::AppData &app_data);
void process_stage_3(cifar_dense::AppData &app_data);
void process_stage_4(cifar_dense::AppData &app_data);
void process_stage_5(cifar_dense::AppData &app_data);
void process_stage_6(cifar_dense::AppData &app_data);
void process_stage_7(cifar_dense::AppData &app_data);
void process_stage_8(cifar_dense::AppData &app_data);
void process_stage_9(cifar_dense::AppData &app_data);

template <int Stage>
  requires(Stage >= 1 && Stage <= 9)
void run_stage(cifar_dense::AppData &app_data) {
  if constexpr (Stage == 1) {
    process_stage_1(app_data);
  } else if constexpr (Stage == 2) {
    process_stage_2(app_data);
  } else if constexpr (Stage == 3) {
    process_stage_3(app_data);
  } else if constexpr (Stage == 4) {
    process_stage_4(app_data);
  } else if constexpr (Stage == 5) {
    process_stage_5(app_data);
  } else if constexpr (Stage == 6) {
    process_stage_6(app_data);
  } else if constexpr (Stage == 7) {
    process_stage_7(app_data);
  } else if constexpr (Stage == 8) {
    process_stage_8(app_data);
  } else if constexpr (Stage == 9) {
    process_stage_9(app_data);
  }
}

}  // namespace cifar_dense::omp
