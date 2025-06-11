#include "../../debug_logger.hpp"
#include "all_kernels.cuh"
#include "dispatchers.cuh"

namespace cifar_sparse::cuda {

constexpr bool kSync = false;

// Stage 1: Sparse conv1
void CudaDispatcher::run_stage_1_async(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 1, &appdata);
  const int N = appdata.u_input.d0();
  const int inC = appdata.u_input.d1();
  const int inH = appdata.u_input.d2();
  const int inW = appdata.u_input.d3();
  const int outC = appdata.conv1_sparse.rows;

  conv2d_csr_batch_cuda(appdata.u_input.data(),
                        N,
                        inC,
                        inH,
                        inW,
                        appdata.conv1_sparse.values_data(),
                        appdata.conv1_sparse.row_ptr_data(),
                        appdata.conv1_sparse.col_idx_data(),
                        outC,
                        appdata.u_conv1_b.data(),
                        appdata.u_conv1_b.d0(),
                        kKernelSize,
                        kStride,
                        kPadding,
                        kRelu,
                        appdata.u_conv1_out.data());

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

// Stage 2: Pool1
void CudaDispatcher::run_stage_2_async(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 2, &appdata);
  const int N = appdata.u_conv1_out.d0();
  const int C = appdata.u_conv1_out.d1();
  const int inH = appdata.u_conv1_out.d2();
  const int inW = appdata.u_conv1_out.d3();
  const int outH = appdata.u_pool1_out.d2();
  const int outW = appdata.u_pool1_out.d3();

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

// Stage 3: Sparse conv2
void CudaDispatcher::run_stage_3_async(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 3, &appdata);
  const int N = appdata.u_pool1_out.d0();
  const int inC = appdata.u_pool1_out.d1();
  const int inH = appdata.u_pool1_out.d2();
  const int inW = appdata.u_pool1_out.d3();
  const int outC = appdata.conv2_sparse.rows;

  conv2d_csr_batch_cuda(appdata.u_pool1_out.data(),
                        N,
                        inC,
                        inH,
                        inW,
                        appdata.conv2_sparse.values_data(),
                        appdata.conv2_sparse.row_ptr_data(),
                        appdata.conv2_sparse.col_idx_data(),
                        outC,
                        appdata.u_conv2_b.data(),
                        appdata.u_conv2_b.d0(),
                        kKernelSize,
                        kStride,
                        kPadding,
                        kRelu,
                        appdata.u_conv2_out.data());

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

// Stage 4: Pool2
void CudaDispatcher::run_stage_4_async(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 4, &appdata);
  const int N = appdata.u_conv2_out.d0();
  const int C = appdata.u_conv2_out.d1();
  const int inH = appdata.u_conv2_out.d2();
  const int inW = appdata.u_conv2_out.d3();
  const int outH = appdata.u_pool2_out.d2();
  const int outW = appdata.u_pool2_out.d3();

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

// Stage 5: Sparse conv3
void CudaDispatcher::run_stage_5_async(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 5, &appdata);
  const int N = appdata.u_pool2_out.d0();
  const int inC = appdata.u_pool2_out.d1();
  const int inH = appdata.u_pool2_out.d2();
  const int inW = appdata.u_pool2_out.d3();
  const int outC = appdata.conv3_sparse.rows;

  conv2d_csr_batch_cuda(appdata.u_pool2_out.data(),
                        N,
                        inC,
                        inH,
                        inW,
                        appdata.conv3_sparse.values_data(),
                        appdata.conv3_sparse.row_ptr_data(),
                        appdata.conv3_sparse.col_idx_data(),
                        outC,
                        appdata.u_conv3_b.data(),
                        appdata.u_conv3_b.d0(),
                        kKernelSize,
                        kStride,
                        kPadding,
                        kRelu,
                        appdata.u_conv3_out.data());

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

// Stage 6: Sparse conv4
void CudaDispatcher::run_stage_6_async(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 6, &appdata);
  const int N = appdata.u_conv3_out.d0();
  const int inC = appdata.u_conv3_out.d1();
  const int inH = appdata.u_conv3_out.d2();
  const int inW = appdata.u_conv3_out.d3();
  const int outC = appdata.conv4_sparse.rows;

  conv2d_csr_batch_cuda(appdata.u_conv3_out.data(),
                        N,
                        inC,
                        inH,
                        inW,
                        appdata.conv4_sparse.values_data(),
                        appdata.conv4_sparse.row_ptr_data(),
                        appdata.conv4_sparse.col_idx_data(),
                        outC,
                        appdata.u_conv4_b.data(),
                        appdata.u_conv4_b.d0(),
                        kKernelSize,
                        kStride,
                        kPadding,
                        kRelu,
                        appdata.u_conv4_out.data());

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

// Stage 7: Sparse conv5
void CudaDispatcher::run_stage_7_async(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 7, &appdata);
  const int N = appdata.u_conv4_out.d0();
  const int inC = appdata.u_conv4_out.d1();
  const int inH = appdata.u_conv4_out.d2();
  const int inW = appdata.u_conv4_out.d3();
  const int outC = appdata.conv5_sparse.rows;

  conv2d_csr_batch_cuda(appdata.u_conv4_out.data(),
                        N,
                        inC,
                        inH,
                        inW,
                        appdata.conv5_sparse.values_data(),
                        appdata.conv5_sparse.row_ptr_data(),
                        appdata.conv5_sparse.col_idx_data(),
                        outC,
                        appdata.u_conv5_b.data(),
                        appdata.u_conv5_b.d0(),
                        kKernelSize,
                        kStride,
                        kPadding,
                        kRelu,
                        appdata.u_conv5_out.data());

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

// Stage 8: Pool3
void CudaDispatcher::run_stage_8_async(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 8, &appdata);
  const int N = appdata.u_conv5_out.d0();
  const int C = appdata.u_conv5_out.d1();
  const int inH = appdata.u_conv5_out.d2();
  const int inW = appdata.u_conv5_out.d3();
  const int outH = appdata.u_pool3_out.d2();
  const int outW = appdata.u_pool3_out.d3();

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

// Stage 9: Sparse linear
void CudaDispatcher::run_stage_9_async(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 9, &appdata);
  const int N = appdata.u_pool3_out.d0();
  const int C = appdata.u_pool3_out.d1();
  const int H = appdata.u_pool3_out.d2();
  const int W = appdata.u_pool3_out.d3();
  const int inF = C * H * W;
  const int outF = appdata.linear_sparse.rows;

  linear_csr_batch_cuda(appdata.u_pool3_out.data(),
                        N,
                        inF,
                        appdata.linear_sparse.values_data(),
                        appdata.linear_sparse.row_ptr_data(),
                        appdata.linear_sparse.col_idx_data(),
                        appdata.u_linear_b.data(),
                        outF,
                        appdata.u_linear_out.data());

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

}  // namespace cifar_sparse::cuda
