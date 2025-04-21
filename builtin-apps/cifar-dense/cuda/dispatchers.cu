#include "../../debug_logger.hpp"
#include "all_kernels.cuh"
#include "dispatchers.cuh"

namespace cifar_dense::cuda {

void CudaDispatcher::run_stage_1_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 1, &appdata);

  const int N = appdata.u_input.d0();
  const int inC = appdata.u_input.d1();
  const int inH = appdata.u_input.d2();
  const int inW = appdata.u_input.d3();
  const int outC = appdata.u_conv1_w.d0();
  const int outH = appdata.u_conv1_out.d2();
  const int outW = appdata.u_conv1_out.d3();

  conv2d_batch_cuda(appdata.u_input.data(),
                    appdata.u_conv1_w.data(),
                    appdata.u_conv1_b.data(),
                    appdata.u_conv1_out.data(),
                    N,
                    inC,
                    inH,
                    inW,
                    outC,
                    kKernelSize,
                    kKernelSize,
                    outH,
                    outW,
                    kStride,
                    kPadding,
                    kRelu);

  CheckCuda(cudaGetLastError());
  CheckCuda(cudaDeviceSynchronize());
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
