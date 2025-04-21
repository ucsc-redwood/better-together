#include "../../common/cuda/helpers.cuh"
#include "../../debug_logger.hpp"
#include "all_kernels.cuh"
#include "dispatchers.cuh"

namespace cifar_sparse::cuda {

// -----------------------------------------------------------------------------
// Stage 1 (first conv2d)
// -----------------------------------------------------------------------------

constexpr bool kAutoSync = false;
constexpr int kGpuBatchSize = 16;

// Input Image dimensions
constexpr int kInputChannels = 3;
constexpr int kInputHeight = 32;
constexpr int kInputWidth = 32;

// Convolution parameters
constexpr int kKernelSize = 3;
constexpr int kStride = 1;
constexpr int kPadding = 1;

// Pooling parameters
constexpr int kPoolSize = 2;
constexpr int kPoolStride = 2;

constexpr bool kRelu = true;

void process_stage_1(AppData &appdata) {
  const auto total_iterations = appdata.conv1_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 512);
  LOG_KERNEL(LogKernelType::kCUDA, 1, &appdata);

  for (auto i = 0; i < kGpuBatchSize; i++) {
    conv2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_image_data.data(),
                                                kInputChannels,
                                                kInputHeight,
                                                kInputWidth,
                                                appdata.conv1_weights.values,
                                                appdata.conv1_weights.row_ptr,
                                                appdata.conv1_weights.col_idx,
                                                appdata.conv1_weights.rows,
                                                appdata.conv1_weights.cols,
                                                appdata.conv1_weights.nnz,
                                                appdata.u_conv1_bias.data(),
                                                64,
                                                kKernelSize,
                                                kStride,
                                                kPadding,
                                                kRelu,
                                                appdata.u_conv1_output.data());

    if constexpr (kAutoSync) {
      CheckCuda(cudaDeviceSynchronize());
    }
  }
}

// -----------------------------------------------------------------------------
// Stage 2 (first maxpool)
// -----------------------------------------------------------------------------

void process_stage_2(AppData &appdata) {
  constexpr auto output_height = (kInputHeight - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (kInputWidth - kPoolSize) / kPoolStride + 1;
  auto total_iterations = kInputChannels * output_height * output_width;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 512);
  LOG_KERNEL(LogKernelType::kCUDA, 2, &appdata);

  for (auto i = 0; i < kGpuBatchSize; i++) {
    maxpool2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_conv1_output.data(),
                                                   kInputChannels,
                                                   kInputHeight,
                                                   kInputWidth,
                                                   kPoolSize,
                                                   kPoolStride,
                                                   appdata.u_pool1_output.data());

    if constexpr (kAutoSync) {
      CheckCuda(cudaDeviceSynchronize());
    }
  }
}

void process_stage_3(AppData &appdata) {
  const auto total_iterations = appdata.conv2_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 512);

  for (auto i = 0; i < kGpuBatchSize; i++) {
    conv2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_pool1_output.data(),
                                                64,
                                                16,
                                                16,
                                                appdata.conv2_weights.values,
                                                appdata.conv2_weights.row_ptr,
                                                appdata.conv2_weights.col_idx,
                                                appdata.conv2_weights.rows,
                                                appdata.conv2_weights.cols,
                                                appdata.conv2_weights.nnz,
                                                appdata.u_conv2_bias.data(),
                                                192,
                                                kKernelSize,
                                                kStride,
                                                kPadding,
                                                kRelu,
                                                appdata.u_conv2_output.data());

    if constexpr (kAutoSync) {
      CheckCuda(cudaDeviceSynchronize());
    }
  }
}

void process_stage_4(AppData &appdata) {
  constexpr auto input_channels = 192;
  constexpr auto input_height = 16;
  constexpr auto input_width = 16;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations = input_channels * output_height * output_width;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 512);
  LOG_KERNEL(LogKernelType::kCUDA, 3, &appdata);

  for (auto i = 0; i < kGpuBatchSize; i++) {
    maxpool2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_conv2_output.data(),
                                                   input_channels,
                                                   input_height,
                                                   input_width,
                                                   kPoolSize,
                                                   kPoolStride,
                                                   appdata.u_pool2_output.data());

    if constexpr (kAutoSync) {
      CheckCuda(cudaDeviceSynchronize());
    }
  }
}

void process_stage_5(AppData &appdata) {
  const auto total_iterations = appdata.conv3_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 512);
  LOG_KERNEL(LogKernelType::kCUDA, 4, &appdata);

  for (auto i = 0; i < kGpuBatchSize; i++) {
    conv2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_pool2_output.data(),
                                                192,
                                                8,
                                                8,
                                                appdata.conv3_weights.values,
                                                appdata.conv3_weights.row_ptr,
                                                appdata.conv3_weights.col_idx,
                                                appdata.conv3_weights.rows,
                                                appdata.conv3_weights.cols,
                                                appdata.conv3_weights.nnz,
                                                appdata.u_conv3_bias.data(),
                                                384,
                                                kKernelSize,
                                                kStride,
                                                kPadding,
                                                kRelu,
                                                appdata.u_conv3_output.data());

    if constexpr (kAutoSync) {
      CheckCuda(cudaDeviceSynchronize());
    }
  }
}

void process_stage_6(AppData &appdata) {
  const auto total_iterations = appdata.conv4_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 512);
  LOG_KERNEL(LogKernelType::kCUDA, 5, &appdata);

  for (auto i = 0; i < kGpuBatchSize; i++) {
    conv2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_conv3_output.data(),
                                                384,
                                                8,
                                                8,
                                                appdata.conv4_weights.values,
                                                appdata.conv4_weights.row_ptr,
                                                appdata.conv4_weights.col_idx,
                                                appdata.conv4_weights.rows,
                                                appdata.conv4_weights.cols,
                                                appdata.conv4_weights.nnz,
                                                appdata.u_conv4_bias.data(),
                                                512,
                                                kKernelSize,
                                                kStride,
                                                kPadding,
                                                kRelu,
                                                appdata.u_conv4_output.data());

    if constexpr (kAutoSync) {
      CheckCuda(cudaDeviceSynchronize());
    }
  }
}

void process_stage_7(AppData &appdata) {
  const auto total_iterations = appdata.conv5_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 512);
  LOG_KERNEL(LogKernelType::kCUDA, 6, &appdata);

  for (auto i = 0; i < kGpuBatchSize; i++) {
    conv2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_conv4_output.data(),
                                                512,
                                                8,
                                                8,
                                                appdata.conv5_weights.values,
                                                appdata.conv5_weights.row_ptr,
                                                appdata.conv5_weights.col_idx,
                                                appdata.conv5_weights.rows,
                                                appdata.conv5_weights.cols,
                                                appdata.conv5_weights.nnz,
                                                appdata.u_conv5_bias.data(),
                                                512,
                                                kKernelSize,
                                                kStride,
                                                kPadding,
                                                kRelu,
                                                appdata.u_conv5_output.data());

    if constexpr (kAutoSync) {
      CheckCuda(cudaDeviceSynchronize());
    }
  }
}

void process_stage_8(AppData &appdata) {
  constexpr auto input_channels = 512;
  constexpr auto input_height = 8;
  constexpr auto input_width = 8;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations = input_channels * output_height * output_width;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 512);
  LOG_KERNEL(LogKernelType::kCUDA, 7, &appdata);

  for (auto i = 0; i < kGpuBatchSize; i++) {
    maxpool2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_conv5_output.data(),
                                                   input_channels,
                                                   input_height,
                                                   input_width,
                                                   kPoolSize,
                                                   kPoolStride,
                                                   appdata.u_pool3_output.data());

    if constexpr (kAutoSync) {
      CheckCuda(cudaDeviceSynchronize());
    }
  }
}

void process_stage_9(AppData &appdata) {
  const auto total_iterations = appdata.linear_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 512);
  LOG_KERNEL(LogKernelType::kCUDA, 8, &appdata);

  for (auto i = 0; i < kGpuBatchSize; i++) {
    linear<<<grid_dim, block_dim, shared_mem>>>(appdata.u_pool3_output.data(),
                                                appdata.linear_weights.values,
                                                appdata.linear_weights.row_ptr,
                                                appdata.linear_weights.col_idx,
                                                appdata.linear_weights.rows,
                                                appdata.linear_weights.cols,
                                                appdata.linear_weights.nnz,
                                                appdata.u_linear_bias.data(),
                                                appdata.u_linear_output.data());

    if constexpr (kAutoSync) {
      CheckCuda(cudaDeviceSynchronize());
    }
  }
}

}  // namespace cifar_sparse::cuda