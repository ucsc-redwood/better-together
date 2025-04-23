
#include <cub/cub.cuh>
#include <cub/util_math.cuh>

#include "../../debug_logger.hpp"
#include "dispatchers.cuh"

// #include "01_morton.cuh"
// #include "04_radix_tree.cuh"
// #include "05_edge_count.cuh"
// #include "07_octree.cuh"
#include "../../common/cuda/helpers.cuh"


namespace tree::cuda {

void CudaDispatcher::run_stage_1_async(tree::SafeAppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 1, &appdata);
}

void CudaDispatcher::run_stage_2_async(tree::SafeAppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 2, &appdata);
}

void CudaDispatcher::run_stage_3_async(tree::SafeAppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 3, &appdata);
}

void CudaDispatcher::run_stage_4_async(tree::SafeAppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 4, &appdata);
}

void CudaDispatcher::run_stage_5_async(tree::SafeAppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 5, &appdata);
}

void CudaDispatcher::run_stage_6_async(tree::SafeAppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 6, &appdata);
}

void CudaDispatcher::run_stage_7_async(tree::SafeAppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 7, &appdata);
}

}  // namespace tree::cuda
