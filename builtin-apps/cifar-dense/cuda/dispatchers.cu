#include "../../debug_logger.hpp"
#include "dispatchers.cuh"

namespace cifar_dense::cuda {

void CudaDispatcher::run_stage_1_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 1, &appdata);
}

void CudaDispatcher::run_stage_2_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 2, &appdata);
}

void CudaDispatcher::run_stage_3_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 3, &appdata);
}

void CudaDispatcher::run_stage_4_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 4, &appdata);
}

void CudaDispatcher::run_stage_5_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 5, &appdata);
}

void CudaDispatcher::run_stage_6_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 6, &appdata);
}

void CudaDispatcher::run_stage_7_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 7, &appdata);
}

void CudaDispatcher::run_stage_8_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 8, &appdata);
}

void CudaDispatcher::run_stage_9_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 9, &appdata);
}

}  // namespace cifar_dense::cuda
