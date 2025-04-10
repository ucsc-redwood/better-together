
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

// ----------------------------------------------------------------------------
// v2
// ----------------------------------------------------------------------------

namespace v2 {

// constexpr int out_channels = 16;   // conv1 output channels
// const int kernel_size  = 3;    // conv1 kernel size 3×3
// const int stride       = 1;    // Typically stride 1
// const int padding      = 1;    // For same padding, usually 1 when kernel size is 3
// const bool relu        = true; // Apply ReLU activation after convolution

void run_stage_1(cifar_sparse::v2::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 1, &appdata);

  // inline void conv2d_omp_batched(const float *input_data,
  //                                const int batch_size,
  //                                const int in_channels,
  //                                const int in_height,
  //                                const int in_width,
  //                                // Sparse weights for this convolution layer:
  //                                const float *weight_vals,
  //                                const int *weight_row_ptr,
  //                                const int *weight_col_idx,
  //                                const int out_channels,  // equals number of rows in CSR matrix
  //                                const float *bias_data,  // may be nullptr if no bias is used
  //                                const int bias_size,     // usually equals out_channels
  //                                const int kernel_size,
  //                                const int stride,
  //                                const int padding,
  //                                const bool relu,
  //                                float *output_data)  // preallocated output array

  const int batch_size = appdata.u_input.d0();   // Expected 128
  const int in_channels = appdata.u_input.d1();  // Expected 3 (RGB)
  const int in_height = appdata.u_input.d2();    // Expected 32
  const int in_width = appdata.u_input.d3();     // Expected 32

  const int out_channels = appdata.conv1_sparse.rows;  // Expected 16

  // Ndarray4D u_input;      // (128, 3, 32, 32)
  // Ndarray4D u_conv1_out;  // (128, 16, 32, 32)

  v2::conv2d_omp_batched(appdata.u_input.data(),
                         batch_size,   // 128
                         in_channels,  // 3
                         in_height,    // 32
                         in_width,     // 32
                         appdata.conv1_sparse.values_data(),
                         appdata.conv1_sparse.row_ptr_data(),
                         appdata.conv1_sparse.col_idx_data(),
                         out_channels,  // 16
                         appdata.u_conv1_b.data(),
                         appdata.u_conv1_b.size(),
                         kKernelSize,
                         kStride,
                         kPadding,
                         kRelu,
                         appdata.u_conv1_out.data());
}

}  // namespace v2

}  // namespace cifar_sparse::omp