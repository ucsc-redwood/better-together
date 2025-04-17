#include "dispatchers.hpp"

#include "../../debug_logger.hpp"
#include "../../tree/omp/func_sort.hpp"
#include "kernels.hpp"

namespace octree::omp {

// ----------------------------------------------------------------------------
// Stage 1 (compute morton codes)
// ----------------------------------------------------------------------------

void run_stage_1(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 1, &appdata);

  compute_morton_codes(appdata.u_positions.data(), appdata.n_input, appdata.u_morton_codes.data());

#pragma omp barrier
}

// ----------------------------------------------------------------------------
// Stage 2 (sort morton codes)
// ----------------------------------------------------------------------------

void run_stage_2(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 2, &appdata);

  const auto num_threads = omp_get_num_threads();
  const int tid = omp_get_thread_num();
  ::tree::omp::parallel_sort(appdata.u_morton_codes, appdata.u_morton_codes, tid, num_threads);
}

// ----------------------------------------------------------------------------
// Stage 3 (unique morton codes)
// ----------------------------------------------------------------------------

void run_stage_3(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 3, &appdata);

#pragma omp single
  {
    auto it = std::unique(appdata.u_morton_codes.begin(), appdata.u_morton_codes.end());
    appdata.u_morton_codes.erase(it, appdata.u_morton_codes.end());
  }

#pragma omp barrier
}

void run_stage_4(AppData &appdata) { LOG_KERNEL(LogKernelType::kOMP, 4, &appdata); }

void run_stage_5(AppData &appdata) { LOG_KERNEL(LogKernelType::kOMP, 5, &appdata); }

void run_stage_6(AppData &appdata) { LOG_KERNEL(LogKernelType::kOMP, 6, &appdata); }

void run_stage_7(AppData &appdata) { LOG_KERNEL(LogKernelType::kOMP, 7, &appdata); }

}  // namespace octree::omp
