#include "dispatchers.hpp"

#include <omp.h>
#include <spdlog/spdlog.h>

#include "../../debug_logger.hpp"
#include "all_kernels.hpp"

namespace cifar_sparse::omp {

// // Convolution parameters
// constexpr int kKernelSize = 3;
// constexpr int kStride = 1;
// constexpr int kPadding = 1;

// // Pooling parameters
// constexpr int kPoolSize = 2;
// constexpr int kPoolStride = 2;

// constexpr bool kRelu = true;

void run_stage_1(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 1, &appdata);

  const int batch_size = appdata.u_input.d0();   // Expected 128
  const int in_channels = appdata.u_input.d1();  // Expected 3 (RGB)
  const int in_height = appdata.u_input.d2();    // Expected 32
  const int in_width = appdata.u_input.d3();     // Expected 32

  const int out_channels = appdata.conv1_sparse.rows;  // Expected 16

  conv2d_omp_batched(appdata.u_input.data(),
                     batch_size,   // 128
                     in_channels,  // 3
                     in_height,    // 32
                     in_width,     // 32
                     appdata.conv1_sparse.values_data(),
                     appdata.conv1_sparse.row_ptr_data(),
                     appdata.conv1_sparse.col_idx_data(),
                     out_channels,  // 16
                     appdata.u_conv1_b.data(),
                     appdata.u_conv1_b.d0(),
                     kKernelSize,
                     kStride,
                     kPadding,
                     kRelu,
                     appdata.u_conv1_out.data());
}

void run_stage_2(AppData &appdata) {
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

void run_stage_3(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 3, &appdata);

  // Extract dimensions from the pool1 output NDArray4D
  const int batch_size = appdata.u_pool1_out.d0();   // Expected: 128
  const int in_channels = appdata.u_pool1_out.d1();  // Expected: 16
  const int in_height = appdata.u_pool1_out.d2();    // Expected: 16
  const int in_width = appdata.u_pool1_out.d3();     // Expected: 16

  const int out_channels = appdata.conv2_sparse.rows;  // Expected: 32

  conv2d_omp_batched(appdata.u_pool1_out.data(),
                     batch_size,   // 128
                     in_channels,  // 16
                     in_height,    // 16
                     in_width,     // 16
                     appdata.conv2_sparse.values_data(),
                     appdata.conv2_sparse.row_ptr_data(),
                     appdata.conv2_sparse.col_idx_data(),
                     out_channels,  // 32
                     appdata.u_conv2_b.data(),
                     appdata.u_conv2_b.d0(),
                     kKernelSize,
                     kStride,
                     kPadding,
                     kRelu,
                     appdata.u_conv2_out.data());
}

void run_stage_4(AppData &appdata) {
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

void run_stage_5(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 5, &appdata);

  // Extract dimensions from the pool2 output NDArray4D
  const int batch_size = appdata.u_pool2_out.d0();   // Expected: 128
  const int in_channels = appdata.u_pool2_out.d1();  // Expected: 32
  const int in_height = appdata.u_pool2_out.d2();    // Expected: 8
  const int in_width = appdata.u_pool2_out.d3();     // Expected: 8

  const int out_channels = appdata.conv3_sparse.rows;  // Expected: 64

  conv2d_omp_batched(appdata.u_pool2_out.data(),
                     batch_size,   // 128
                     in_channels,  // 32
                     in_height,    // 8
                     in_width,     // 8
                     appdata.conv3_sparse.values_data(),
                     appdata.conv3_sparse.row_ptr_data(),
                     appdata.conv3_sparse.col_idx_data(),
                     out_channels,  // 64
                     appdata.u_conv3_b.data(),
                     appdata.u_conv3_b.d0(),
                     kKernelSize,
                     kStride,
                     kPadding,
                     kRelu,
                     appdata.u_conv3_out.data());
}

void run_stage_6(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 6, &appdata);

  // Extract dimensions from the conv3 output NDArray4D
  const int batch_size = appdata.u_conv3_out.d0();   // Expected: 128
  const int in_channels = appdata.u_conv3_out.d1();  // Expected: 64
  const int in_height = appdata.u_conv3_out.d2();    // Expected: 8
  const int in_width = appdata.u_conv3_out.d3();     // Expected: 8

  const int out_channels = appdata.conv4_sparse.rows;  // Expected: 64

  conv2d_omp_batched(appdata.u_conv3_out.data(),
                     batch_size,   // 128
                     in_channels,  // 64
                     in_height,    // 8
                     in_width,     // 8
                     appdata.conv4_sparse.values_data(),
                     appdata.conv4_sparse.row_ptr_data(),
                     appdata.conv4_sparse.col_idx_data(),
                     out_channels,  // 64
                     appdata.u_conv4_b.data(),
                     appdata.u_conv4_b.d0(),
                     kKernelSize,
                     kStride,
                     kPadding,
                     kRelu,
                     appdata.u_conv4_out.data());
}

void run_stage_7(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 7, &appdata);

  // Extract dimensions from the conv4 output NDArray4D
  const int batch_size = appdata.u_conv4_out.d0();   // Expected: 128
  const int in_channels = appdata.u_conv4_out.d1();  // Expected: 64
  const int in_height = appdata.u_conv4_out.d2();    // Expected: 8
  const int in_width = appdata.u_conv4_out.d3();     // Expected: 8

  const int out_channels = appdata.conv5_sparse.rows;  // Expected: 64

  conv2d_omp_batched(appdata.u_conv4_out.data(),
                     batch_size,   // 128
                     in_channels,  // 64
                     in_height,    // 8
                     in_width,     // 8
                     appdata.conv5_sparse.values_data(),
                     appdata.conv5_sparse.row_ptr_data(),
                     appdata.conv5_sparse.col_idx_data(),
                     out_channels,  // 64
                     appdata.u_conv5_b.data(),
                     appdata.u_conv5_b.d0(),
                     kKernelSize,
                     kStride,
                     kPadding,
                     kRelu,
                     appdata.u_conv5_out.data());
}

void run_stage_8(AppData &appdata) {
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

void run_stage_9(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 9, &appdata);

  // The pooled output is (128, 64, 4, 4) which becomes a flattened input
  // of (128, 1024) for the linear layer
  const int batch_size = appdata.u_pool3_out.d0();   // Expected: 128
  const int channels = appdata.u_pool3_out.d1();     // Expected: 64
  const int pool_height = appdata.u_pool3_out.d2();  // Expected: 4
  const int pool_width = appdata.u_pool3_out.d3();   // Expected: 4

  // Total features per image = channels * height * width
  const int input_features = channels * pool_height * pool_width;  // 64 * 4 * 4 = 1024

  // Output neurons = number of classes
  const int out_neurons = appdata.linear_sparse.rows;  // Expected: 10

  // Use the batched sparse linear layer kernel
  linear_omp_batched(appdata.u_pool3_out.data(),  // Input data (flattened 4D->2D)
                     batch_size,                  // 128
                     input_features,              // 1024
                     appdata.linear_sparse.values_data(),
                     appdata.linear_sparse.row_ptr_data(),
                     appdata.linear_sparse.col_idx_data(),
                     appdata.u_linear_b.data(),
                     appdata.u_linear_out.data(),
                     out_neurons);  // 10
}

}  // namespace cifar_sparse::omp