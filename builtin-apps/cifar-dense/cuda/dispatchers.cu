#include "../../debug_logger.hpp"
#include "all_kernels.cuh"
#include "dispatchers.cuh"

namespace cifar_dense::cuda {

constexpr bool kSync = true;

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

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_2_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 2, &appdata);

  const int N = appdata.u_conv1_out.d0();     // 128
  const int C = appdata.u_conv1_out.d1();     // 16
  const int inH = appdata.u_conv1_out.d2();   // 32
  const int inW = appdata.u_conv1_out.d3();   // 32
  const int outH = appdata.u_pool1_out.d2();  // 16
  const int outW = appdata.u_pool1_out.d3();  // 16

  maxpool2d_batch_cuda(appdata.u_conv1_out.data(),
                       appdata.u_pool1_out.data(),
                       N,
                       C,
                       inH,
                       inW,
                       outH,
                       outW,
                       kPoolSize,
                       kPoolStride);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_3_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 3, &appdata);

  const int N = appdata.u_pool1_out.d0();     // 128
  const int inC = appdata.u_pool1_out.d1();   // 16
  const int inH = appdata.u_pool1_out.d2();   // 16
  const int inW = appdata.u_pool1_out.d3();   // 16
  const int outC = appdata.u_conv2_w.d0();    // 32
  const int outH = appdata.u_conv2_out.d2();  // 16
  const int outW = appdata.u_conv2_out.d3();  // 16

  conv2d_batch_cuda(appdata.u_pool1_out.data(),
                    appdata.u_conv2_w.data(),
                    appdata.u_conv2_b.data(),
                    appdata.u_conv2_out.data(),
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

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_4_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 4, &appdata);

  const int N = appdata.u_conv2_out.d0();     // 128
  const int C = appdata.u_conv2_out.d1();     // 32
  const int inH = appdata.u_conv2_out.d2();   // 16
  const int inW = appdata.u_conv2_out.d3();   // 16
  const int outH = appdata.u_pool2_out.d2();  // 8
  const int outW = appdata.u_pool2_out.d3();  // 8

  maxpool2d_batch_cuda(appdata.u_conv2_out.data(),
                       appdata.u_pool2_out.data(),
                       N,
                       C,
                       inH,
                       inW,
                       outH,
                       outW,
                       kPoolSize,
                       kPoolStride);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_5_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 5, &appdata);

  const int N = appdata.u_pool2_out.d0();     // 128
  const int inC = appdata.u_pool2_out.d1();   // 32
  const int inH = appdata.u_pool2_out.d2();   // 8
  const int inW = appdata.u_pool2_out.d3();   // 8
  const int outC = appdata.u_conv3_w.d0();    // 64
  const int outH = appdata.u_conv3_out.d2();  // 8
  const int outW = appdata.u_conv3_out.d3();  // 8

  conv2d_batch_cuda(appdata.u_pool2_out.data(),
                    appdata.u_conv3_w.data(),
                    appdata.u_conv3_b.data(),
                    appdata.u_conv3_out.data(),
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

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_6_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 6, &appdata);

  const int N = appdata.u_conv3_out.d0();     // 128
  const int inC = appdata.u_conv3_out.d1();   // 64
  const int inH = appdata.u_conv3_out.d2();   // 8
  const int inW = appdata.u_conv3_out.d3();   // 8
  const int outC = appdata.u_conv4_w.d0();    // 64
  const int outH = appdata.u_conv4_out.d2();  // 8
  const int outW = appdata.u_conv4_out.d3();  // 8

  conv2d_batch_cuda(appdata.u_conv3_out.data(),
                    appdata.u_conv4_w.data(),
                    appdata.u_conv4_b.data(),
                    appdata.u_conv4_out.data(),
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

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_7_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 7, &appdata);

  const int N = appdata.u_conv4_out.d0();     // 128
  const int inC = appdata.u_conv4_out.d1();   // 64
  const int inH = appdata.u_conv4_out.d2();   // 8
  const int inW = appdata.u_conv4_out.d3();   // 8
  const int outC = appdata.u_conv5_w.d0();    // 64
  const int outH = appdata.u_conv5_out.d2();  // 8
  const int outW = appdata.u_conv5_out.d3();  // 8

  conv2d_batch_cuda(appdata.u_conv4_out.data(),
                    appdata.u_conv5_w.data(),
                    appdata.u_conv5_b.data(),
                    appdata.u_conv5_out.data(),
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

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_8_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 8, &appdata);

  const int N = appdata.u_conv5_out.d0();     // 128
  const int C = appdata.u_conv5_out.d1();     // 64
  const int inH = appdata.u_conv5_out.d2();   // 8
  const int inW = appdata.u_conv5_out.d3();   // 8
  const int outH = appdata.u_pool3_out.d2();  // 4
  const int outW = appdata.u_pool3_out.d3();  // 4

  maxpool2d_batch_cuda(appdata.u_conv5_out.data(),
                       appdata.u_pool3_out.data(),
                       N,
                       C,
                       inH,
                       inW,
                       outH,
                       outW,
                       kPoolSize,
                       kPoolStride);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_9_async(cifar_dense::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 9, &appdata);

  const int N = appdata.u_pool3_out.d0();    // 128
  const int C = appdata.u_pool3_out.d1();    // 64
  const int H = appdata.u_pool3_out.d2();    // 4
  const int W = appdata.u_pool3_out.d3();    // 4
  const int inF = C * H * W;                 // 1024
  const int outF = appdata.u_linear_w.d0();  // 10

  linear_batch_cuda(appdata.u_pool3_out.data(),
                    appdata.u_linear_w.data(),
                    appdata.u_linear_b.data(),
                    appdata.u_linear_out.data(),
                    N,
                    inF,
                    outF);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

}  // namespace cifar_dense::cuda
