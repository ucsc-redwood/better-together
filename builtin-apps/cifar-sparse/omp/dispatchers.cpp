
#include "dispatchers.hpp"

#include <omp.h>
#include <spdlog/spdlog.h>

#include "../../debug_logger.hpp"
#include "all_kernels.hpp"

namespace cifar_sparse::omp {

// ----------------------------------------------------------------------------
// Pipeline Processing Stages
// ----------------------------------------------------------------------------

void run_stage_1(cifar_sparse::AppData &app_data) {
  constexpr auto start = 0;
  const auto end = app_data.conv1_weights.rows;

  LOG_KERNEL(LogKernelType::kOMP, 1, &app_data);

  for (auto batch = 0; batch < kNumBatches; ++batch) {
    conv2d_omp(app_data.u_image_data.data(),
               kInputChannels,
               kInputHeight,
               kInputWidth,
               app_data.conv1_weights,
               app_data.u_conv1_bias.data(),
               64,
               kKernelSize,
               kStride,
               kPadding,
               kRelu,
               app_data.u_conv1_output.data(),
               start,
               end);
  }
}

void run_stage_2(cifar_sparse::AppData &app_data) {
  constexpr auto start = 0;

  constexpr auto input_channels = 64;
  constexpr auto input_height = 32;
  constexpr auto input_width = 32;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations = input_channels * output_height * output_width;

  constexpr auto end = total_iterations;

  LOG_KERNEL(LogKernelType::kOMP, 2, &app_data);

  for (auto batch = 0; batch < kNumBatches; ++batch) {
    maxpool2d_omp(app_data.u_conv1_output.data(),
                  input_channels,
                  input_height,
                  input_width,
                  kPoolSize,
                  kPoolStride,
                  app_data.u_pool1_output.data(),
                  start,
                  end);
  }
}

void run_stage_3(cifar_sparse::AppData &app_data) {
  const auto start = 0;
  const auto end = app_data.conv2_weights.rows;

  LOG_KERNEL(LogKernelType::kOMP, 3, &app_data);

  for (auto batch = 0; batch < kNumBatches; ++batch) {
    conv2d_omp(app_data.u_pool1_output.data(),
               64,
               16,
               16,
               app_data.conv2_weights,
               app_data.u_conv2_bias.data(),
               192,
               kKernelSize,
               kStride,
               kPadding,
               kRelu,
               app_data.u_conv2_output.data(),
               start,
               end);
  }
}

void run_stage_4(cifar_sparse::AppData &app_data) {
  constexpr auto input_channels = 192;
  constexpr auto input_height = 16;
  constexpr auto input_width = 16;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations = input_channels * output_height * output_width;

  const auto start = 0;
  const auto end = total_iterations;

  LOG_KERNEL(LogKernelType::kOMP, 4, &app_data);

  for (auto batch = 0; batch < kNumBatches; ++batch) {
    maxpool2d_omp(app_data.u_conv2_output.data(),
                  input_channels,
                  input_height,
                  input_width,
                  kPoolSize,
                  kPoolStride,
                  app_data.u_pool2_output.data(),
                  start,
                  end);
  }
}

void run_stage_5(cifar_sparse::AppData &app_data) {
  const auto start = 0;
  const auto end = app_data.conv3_weights.rows;

  LOG_KERNEL(LogKernelType::kOMP, 5, &app_data);

  for (auto batch = 0; batch < kNumBatches; ++batch) {
    conv2d_omp(app_data.u_pool2_output.data(),
               192,
               8,
               8,
               app_data.conv3_weights,
               app_data.u_conv3_bias.data(),
               384,
               kKernelSize,
               kStride,
               kPadding,
               kRelu,
               app_data.u_conv3_output.data(),
               start,
               end);
  }
}

void run_stage_6(cifar_sparse::AppData &app_data) {
  const auto start = 0;
  const auto end = app_data.conv4_weights.rows;

  LOG_KERNEL(LogKernelType::kOMP, 6, &app_data);

  for (auto batch = 0; batch < kNumBatches; ++batch) {
    conv2d_omp(app_data.u_conv3_output.data(),
               384,
               8,
               8,
               app_data.conv4_weights,
               app_data.u_conv4_bias.data(),
               256,
               kKernelSize,
               kStride,
               kPadding,
               kRelu,
               app_data.u_conv4_output.data(),
               start,
               end);
  }
}

void run_stage_7(cifar_sparse::AppData &app_data) {
  const auto start = 0;
  const auto end = app_data.conv5_weights.rows;

  LOG_KERNEL(LogKernelType::kOMP, 7, &app_data);

  for (auto batch = 0; batch < kNumBatches; ++batch) {
    conv2d_omp(app_data.u_conv4_output.data(),
               256,
               8,
               8,
               app_data.conv5_weights,
               app_data.u_conv5_bias.data(),
               256,
               kKernelSize,
               kStride,
               kPadding,
               kRelu,
               app_data.u_conv5_output.data(),
               start,
               end);
  }
}

void run_stage_8(cifar_sparse::AppData &app_data) {
  constexpr auto input_channels = 256;
  constexpr auto input_height = 8;
  constexpr auto input_width = 8;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations = input_channels * output_height * output_width;

  const auto start = 0;
  const auto end = total_iterations;

  LOG_KERNEL(LogKernelType::kOMP, 8, &app_data);

  for (auto batch = 0; batch < kNumBatches; ++batch) {
    maxpool2d_omp(app_data.u_conv5_output.data(),
                  input_channels,
                  input_height,
                  input_width,
                  kPoolSize,
                  kPoolStride,
                  app_data.u_pool3_output.data(),
                  start,
                  end);
  }
}

void run_stage_9(cifar_sparse::AppData &app_data) {
  const auto start = 0;
  const auto end = app_data.linear_weights.rows;

  LOG_KERNEL(LogKernelType::kOMP, 9, &app_data);

  for (auto batch = 0; batch < kNumBatches; ++batch) {
    linear_omp(app_data.u_pool3_output.data(),
               app_data.linear_weights,
               app_data.u_linear_bias.data(),
               app_data.u_linear_output.data(),
               start,
               end);
  }
}

}  // namespace cifar_sparse::omp