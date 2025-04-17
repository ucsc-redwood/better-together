#include "dispatcher.hpp"

#include "../../debug_logger.hpp"

namespace octree::omp {

void run_stage_1(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 1, &appdata);

}

void run_stage_2(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 2, &appdata);
}

void run_stage_3(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 3, &appdata);
}

void run_stage_4(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 4, &appdata);
}

void run_stage_5(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 5, &appdata);
}

void run_stage_6(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 6, &appdata);
}

void run_stage_7(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 7, &appdata);
}

}
