#pragma once

#include "../safe_tree_appdata.hpp"
// #include "../tree_appdata.hpp"

namespace tree::omp {

// void process_stage_1(AppData &appdata);
// void process_stage_2(AppData &appdata);
// void process_stage_3(AppData &appdata);
// void process_stage_4(AppData &appdata);
// void process_stage_5(AppData &appdata);
// void process_stage_6(AppData &appdata);
// void process_stage_7(AppData &appdata);

void process_stage_1(SafeAppData &appdata);
void process_stage_2(SafeAppData &appdata);
void process_stage_3(SafeAppData &appdata);
void process_stage_4(SafeAppData &appdata);
void process_stage_5(SafeAppData &appdata);
void process_stage_6(SafeAppData &appdata);
void process_stage_7(SafeAppData &appdata);

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

}  // namespace tree::omp
