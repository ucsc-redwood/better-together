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

void run_stage_1(cifar_sparse::v2::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 1, &appdata);

  const int batch_size = appdata.u_input.d0();   // Expected 128
  const int in_channels = appdata.u_input.d1();  // Expected 3 (RGB)
  const int in_height = appdata.u_input.d2();    // Expected 32
  const int in_width = appdata.u_input.d3();     // Expected 32

  const int out_channels = appdata.conv1_sparse.rows;  // Expected 16

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

void run_stage_2(cifar_sparse::v2::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 2, &appdata);

  // Extract dimensions from the convolution output NDArray4D.
  const int batch_size = appdata.u_conv1_out.d0();  // Expected: 128
  const int channels = appdata.u_conv1_out.d1();    // Expected: 16
  const int in_height = appdata.u_conv1_out.d2();   // Expected: 32
  const int in_width = appdata.u_conv1_out.d3();    // Expected: 32

  // Call the clean batched max pool kernel.
  maxpool2d_omp_batched_clean(appdata.u_conv1_out.data(),  // input_data pointer
                              batch_size,                  // number of images
                              channels,                    // number of channels per image
                              in_height,                   // height of the input feature map
                              in_width,                    // width of the input feature map
                              kPoolSize,
                              kPoolStride,
                              appdata.u_pool1_out.data());
}

void run_stage_3(cifar_sparse::v2::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 3, &appdata);

  // Extract dimensions from the pool1 output NDArray4D
  const int batch_size = appdata.u_pool1_out.d0();   // Expected: 128
  const int in_channels = appdata.u_pool1_out.d1();  // Expected: 16
  const int in_height = appdata.u_pool1_out.d2();    // Expected: 16
  const int in_width = appdata.u_pool1_out.d3();     // Expected: 16

  const int out_channels = appdata.conv2_sparse.rows;  // Expected: 32

  v2::conv2d_omp_batched(appdata.u_pool1_out.data(),
                         batch_size,   // 128
                         in_channels,  // 16
                         in_height,    // 16
                         in_width,     // 16
                         appdata.conv2_sparse.values_data(),
                         appdata.conv2_sparse.row_ptr_data(),
                         appdata.conv2_sparse.col_idx_data(),
                         out_channels,  // 32
                         appdata.u_conv2_b.data(),
                         appdata.u_conv2_b.size(),
                         kKernelSize,
                         kStride,
                         kPadding,
                         kRelu,
                         appdata.u_conv2_out.data());
}

void run_stage_4(cifar_sparse::v2::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 4, &appdata);

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv2_out.d0();  // Expected: 128
  const int channels = appdata.u_conv2_out.d1();    // Expected: 32
  const int in_height = appdata.u_conv2_out.d2();   // Expected: 16
  const int in_width = appdata.u_conv2_out.d3();    // Expected: 16

  // Call the clean batched max pool kernel
  maxpool2d_omp_batched_clean(appdata.u_conv2_out.data(),  // input_data pointer
                              batch_size,                  // number of images
                              channels,                    // number of channels per image
                              in_height,                   // height of the input feature map
                              in_width,                    // width of the input feature map
                              kPoolSize,
                              kPoolStride,
                              appdata.u_pool2_out.data());
}

void run_stage_5(cifar_sparse::v2::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 5, &appdata);

  // Extract dimensions from the pool2 output NDArray4D
  const int batch_size = appdata.u_pool2_out.d0();   // Expected: 128
  const int in_channels = appdata.u_pool2_out.d1();  // Expected: 32
  const int in_height = appdata.u_pool2_out.d2();    // Expected: 8
  const int in_width = appdata.u_pool2_out.d3();     // Expected: 8

  const int out_channels = appdata.conv3_sparse.rows;  // Expected: 64

  v2::conv2d_omp_batched(appdata.u_pool2_out.data(),
                         batch_size,   // 128
                         in_channels,  // 32
                         in_height,    // 8
                         in_width,     // 8
                         appdata.conv3_sparse.values_data(),
                         appdata.conv3_sparse.row_ptr_data(),
                         appdata.conv3_sparse.col_idx_data(),
                         out_channels,  // 64
                         appdata.u_conv3_b.data(),
                         appdata.u_conv3_b.size(),
                         kKernelSize,
                         kStride,
                         kPadding,
                         kRelu,
                         appdata.u_conv3_out.data());
}

void run_stage_6(cifar_sparse::v2::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 6, &appdata);

  // Extract dimensions from the conv3 output NDArray4D
  const int batch_size = appdata.u_conv3_out.d0();   // Expected: 128
  const int in_channels = appdata.u_conv3_out.d1();  // Expected: 64
  const int in_height = appdata.u_conv3_out.d2();    // Expected: 8
  const int in_width = appdata.u_conv3_out.d3();     // Expected: 8

  const int out_channels = appdata.conv4_sparse.rows;  // Expected: 64

  v2::conv2d_omp_batched(appdata.u_conv3_out.data(),
                         batch_size,   // 128
                         in_channels,  // 64
                         in_height,    // 8
                         in_width,     // 8
                         appdata.conv4_sparse.values_data(),
                         appdata.conv4_sparse.row_ptr_data(),
                         appdata.conv4_sparse.col_idx_data(),
                         out_channels,  // 64
                         appdata.u_conv4_b.data(),
                         appdata.u_conv4_b.size(),
                         kKernelSize,
                         kStride,
                         kPadding,
                         kRelu,
                         appdata.u_conv4_out.data());
}

void run_stage_7(cifar_sparse::v2::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 7, &appdata);

  // Extract dimensions from the conv4 output NDArray4D
  const int batch_size = appdata.u_conv4_out.d0();   // Expected: 128
  const int in_channels = appdata.u_conv4_out.d1();  // Expected: 64
  const int in_height = appdata.u_conv4_out.d2();    // Expected: 8
  const int in_width = appdata.u_conv4_out.d3();     // Expected: 8

  const int out_channels = appdata.conv5_sparse.rows;  // Expected: 64

  v2::conv2d_omp_batched(appdata.u_conv4_out.data(),
                         batch_size,   // 128
                         in_channels,  // 64
                         in_height,    // 8
                         in_width,     // 8
                         appdata.conv5_sparse.values_data(),
                         appdata.conv5_sparse.row_ptr_data(),
                         appdata.conv5_sparse.col_idx_data(),
                         out_channels,  // 64
                         appdata.u_conv5_b.data(),
                         appdata.u_conv5_b.size(),
                         kKernelSize,
                         kStride,
                         kPadding,
                         kRelu,
                         appdata.u_conv5_out.data());
}

void run_stage_8(cifar_sparse::v2::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 8, &appdata);

  // Extract dimensions from the conv5 output NDArray4D
  const int batch_size = appdata.u_conv5_out.d0();  // Expected: 128
  const int channels = appdata.u_conv5_out.d1();    // Expected: 64
  const int in_height = appdata.u_conv5_out.d2();   // Expected: 8
  const int in_width = appdata.u_conv5_out.d3();    // Expected: 8

  // Call the clean batched max pool kernel
  maxpool2d_omp_batched_clean(appdata.u_conv5_out.data(),  // input_data pointer
                              batch_size,                  // number of images
                              channels,                    // number of channels per image
                              in_height,                   // height of the input feature map
                              in_width,                    // width of the input feature map
                              kPoolSize,
                              kPoolStride,
                              appdata.u_pool3_out.data());
}

void run_stage_9(cifar_sparse::v2::AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 9, &appdata);

  int out_neurons = 10;  // the output dimension of the linear layer

  // inline void linear_omp_batched(
  //     const float *input_data,
  //     const int batch_size,
  //     const int input_features,  // needed for indexing in each sample's input
  //     const float *weight_vals,
  //     const int *weight_row_ptr,
  //     const int *weight_col_idx,
  //     const float *bias_data,
  //     float *output_data,
  //     const int out_neurons) {

  linear_omp_batched(appdata.u_pool3_out.data(),
                     128,
                     1024,
                     appdata.linear_sparse.values_data(),
                     appdata.linear_sparse.row_ptr_data(),
                     appdata.linear_sparse.col_idx_data(),
                     appdata.u_linear_b.data(),
                     appdata.u_linear_out.data(),
                     out_neurons);
}

}  // namespace v2

}  // namespace cifar_sparse::omp