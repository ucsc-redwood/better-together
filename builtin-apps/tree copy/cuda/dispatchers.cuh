#pragma once

#include "../safe_tree_appdata.hpp"
// #include "../tree_appdata.hpp"

namespace tree::cuda {

// void process_stage_1(AppData &app_data);
// void process_stage_2(AppData &app_data);
// void process_stage_3(AppData &app_data);
// void process_stage_4(AppData &app_data);
// void process_stage_5(AppData &app_data);
// void process_stage_6(AppData &app_data);
// void process_stage_7(AppData &app_data);

// template <int Stage>
//   requires(Stage >= 1 && Stage <= 7)
// void run_stage(AppData &appdata) {
//   if constexpr (Stage == 1) {
//     process_stage_1(appdata);
//   } else if constexpr (Stage == 2) {
//     process_stage_2(appdata);
//   } else if constexpr (Stage == 3) {
//     process_stage_3(appdata);
//   } else if constexpr (Stage == 4) {
//     process_stage_4(appdata);
//   } else if constexpr (Stage == 5) {
//     process_stage_5(appdata);
//   } else if constexpr (Stage == 6) {
//     process_stage_6(appdata);
//   } else if constexpr (Stage == 7) {
//     process_stage_7(appdata);
//   }
// }

void process_stage_1(SafeAppData &app_data);
void process_stage_2(SafeAppData &app_data);
void process_stage_3(SafeAppData &app_data);
void process_stage_4(SafeAppData &app_data);
void process_stage_5(SafeAppData &app_data);
void process_stage_6(SafeAppData &app_data);
void process_stage_7(SafeAppData &app_data);

template <int Stage>
  requires(Stage >= 1 && Stage <= 7)
void run_stage(SafeAppData &appdata) {
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
  }
}

}  // namespace tree::cuda
