#include "dispatchers.hpp"

#include <omp.h>
#include <spdlog/spdlog.h>

#include "../../debug_logger.hpp"
#include "all_kernels.hpp"

namespace cifar_dense::omp {

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

  const int out_channels = appdata.u_conv1_w.d0();  // Expected 16
  const int out_height = appdata.u_conv1_out.d2();  // Expected 32`
  const int out_width = appdata.u_conv1_out.d3();   // Expected 32

  conv2d_batch_u(appdata.u_input.data(),
                 appdata.u_conv1_w.data(),
                 appdata.u_conv1_b.data(),
                 appdata.u_conv1_out.data(),
                 batch_size,    // 128
                 in_channels,   // 3
                 in_height,     // 32
                 in_width,      // 32
                 out_channels,  // 16
                 kKernelSize,   // 3
                 kKernelSize,   // 3
                 out_height,    // 32
                 out_width,     // 32
                 kStride,       // 1
                 kPadding,      // 1
                 kRelu);
}

void run_stage_2(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 2, &appdata);

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv1_out.d0();  // Expected: 128
  const int channels = appdata.u_conv1_out.d1();    // Expected: 16
  const int in_height = appdata.u_conv1_out.d2();   // Expected: 32
  const int in_width = appdata.u_conv1_out.d3();    // Expected: 32

  const int out_height = appdata.u_pool1_out.d2();  // Expected: 16
  const int out_width = appdata.u_pool1_out.d3();   // Expected: 16

  // Call the batched max pool kernel
  maxpool2d_batch_u(appdata.u_conv1_out.data(),  // input_data pointer
                    appdata.u_pool1_out.data(),  // output_data pointer
                    batch_size,                  // number of images
                    channels,                    // number of channels per image
                    in_height,                   // height of the input feature map
                    in_width,                    // width of the input feature map
                    out_height,                  // height of the output feature map
                    out_width,                   // width of the output feature map
                    kPoolSize,                   // pool size (2)
                    kPoolStride);                // pool stride (2)
}

void run_stage_3(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 3, &appdata);

  // Extract dimensions from the pool1 output NDArray4D
  const int batch_size = appdata.u_pool1_out.d0();   // Expected: 128
  const int in_channels = appdata.u_pool1_out.d1();  // Expected: 16
  const int in_height = appdata.u_pool1_out.d2();    // Expected: 16
  const int in_width = appdata.u_pool1_out.d3();     // Expected: 16

  const int out_channels = appdata.u_conv2_w.d0();  // Expected: 32
  const int out_height = appdata.u_conv2_out.d2();  // Expected: 16
  const int out_width = appdata.u_conv2_out.d3();   // Expected: 16

  conv2d_batch_u(appdata.u_pool1_out.data(),
                 appdata.u_conv2_w.data(),
                 appdata.u_conv2_b.data(),
                 appdata.u_conv2_out.data(),
                 batch_size,    // 128
                 in_channels,   // 16
                 in_height,     // 16
                 in_width,      // 16
                 out_channels,  // 32
                 kKernelSize,   // 3
                 kKernelSize,   // 3
                 out_height,    // 16
                 out_width,     // 16
                 kStride,       // 1
                 kPadding,      // 1
                 kRelu);
}

void run_stage_4(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 4, &appdata);

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv2_out.d0();  // Expected: 128
  const int channels = appdata.u_conv2_out.d1();    // Expected: 32
  const int in_height = appdata.u_conv2_out.d2();   // Expected: 16
  const int in_width = appdata.u_conv2_out.d3();    // Expected: 16

  const int out_height = appdata.u_pool2_out.d2();  // Expected: 8
  const int out_width = appdata.u_pool2_out.d3();   // Expected: 8

  // Call the batched max pool kernel
  maxpool2d_batch_u(appdata.u_conv2_out.data(),  // input_data pointer
                    appdata.u_pool2_out.data(),  // output_data pointer
                    batch_size,                  // number of images
                    channels,                    // number of channels per image
                    in_height,                   // height of the input feature map
                    in_width,                    // width of the input feature map
                    out_height,                  // height of the output feature map
                    out_width,                   // width of the output feature map
                    kPoolSize,                   // pool size (2)
                    kPoolStride);                // pool stride (2)
}

void run_stage_5(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 5, &appdata);

  // Extract dimensions from the pool2 output NDArray4D
  const int batch_size = appdata.u_pool2_out.d0();   // Expected: 128
  const int in_channels = appdata.u_pool2_out.d1();  // Expected: 32
  const int in_height = appdata.u_pool2_out.d2();    // Expected: 8
  const int in_width = appdata.u_pool2_out.d3();     // Expected: 8

  const int out_channels = appdata.u_conv3_w.d0();  // Expected: 64
  const int out_height = appdata.u_conv3_out.d2();  // Expected: 8
  const int out_width = appdata.u_conv3_out.d3();   // Expected: 8

  conv2d_batch_u(appdata.u_pool2_out.data(),
                 appdata.u_conv3_w.data(),
                 appdata.u_conv3_b.data(),
                 appdata.u_conv3_out.data(),
                 batch_size,    // 128
                 in_channels,   // 32
                 in_height,     // 8
                 in_width,      // 8
                 out_channels,  // 64
                 kKernelSize,   // 3
                 kKernelSize,   // 3
                 out_height,    // 8
                 out_width,     // 8
                 kStride,       // 1
                 kPadding,      // 1
                 kRelu);
}

void run_stage_6(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 6, &appdata);

  // Extract dimensions from the conv3 output NDArray4D
  const int batch_size = appdata.u_conv3_out.d0();   // Expected: 128
  const int in_channels = appdata.u_conv3_out.d1();  // Expected: 64
  const int in_height = appdata.u_conv3_out.d2();    // Expected: 8
  const int in_width = appdata.u_conv3_out.d3();     // Expected: 8

  const int out_channels = appdata.u_conv4_w.d0();  // Expected: 64
  const int out_height = appdata.u_conv4_out.d2();  // Expected: 8
  const int out_width = appdata.u_conv4_out.d3();   // Expected: 8

  conv2d_batch_u(appdata.u_conv3_out.data(),
                 appdata.u_conv4_w.data(),
                 appdata.u_conv4_b.data(),
                 appdata.u_conv4_out.data(),
                 batch_size,    // 128
                 in_channels,   // 64
                 in_height,     // 8
                 in_width,      // 8
                 out_channels,  // 64
                 kKernelSize,   // 3
                 kKernelSize,   // 3
                 out_height,    // 8
                 out_width,     // 8
                 kStride,       // 1
                 kPadding,      // 1
                 kRelu);
}

void run_stage_7(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 7, &appdata);

  // Extract dimensions from the conv4 output NDArray4D
  const int batch_size = appdata.u_conv4_out.d0();   // Expected: 128
  const int in_channels = appdata.u_conv4_out.d1();  // Expected: 64
  const int in_height = appdata.u_conv4_out.d2();    // Expected: 8
  const int in_width = appdata.u_conv4_out.d3();     // Expected: 8

  const int out_channels = appdata.u_conv5_w.d0();  // Expected: 64
  const int out_height = appdata.u_conv5_out.d2();  // Expected: 8
  const int out_width = appdata.u_conv5_out.d3();   // Expected: 8

  conv2d_batch_u(appdata.u_conv4_out.data(),
                 appdata.u_conv5_w.data(),
                 appdata.u_conv5_b.data(),
                 appdata.u_conv5_out.data(),
                 batch_size,    // 128
                 in_channels,   // 64
                 in_height,     // 8
                 in_width,      // 8
                 out_channels,  // 64
                 kKernelSize,   // 3
                 kKernelSize,   // 3
                 out_height,    // 8
                 out_width,     // 8
                 kStride,       // 1
                 kPadding,      // 1
                 kRelu);
}

void run_stage_8(AppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 8, &appdata);

  // Extract dimensions from the conv5 output NDArray4D
  const int batch_size = appdata.u_conv5_out.d0();  // Expected: 128
  const int channels = appdata.u_conv5_out.d1();    // Expected: 64
  const int in_height = appdata.u_conv5_out.d2();   // Expected: 8
  const int in_width = appdata.u_conv5_out.d3();    // Expected: 8

  const int out_height = appdata.u_pool3_out.d2();  // Expected: 4
  const int out_width = appdata.u_pool3_out.d3();   // Expected: 4

  // Call the batched max pool kernel
  maxpool2d_batch_u(appdata.u_conv5_out.data(),  // input_data pointer
                    appdata.u_pool3_out.data(),  // output_data pointer
                    batch_size,                  // number of images
                    channels,                    // number of channels per image
                    in_height,                   // height of the input feature map
                    in_width,                    // width of the input feature map
                    out_height,                  // height of the output feature map
                    out_width,                   // width of the output feature map
                    kPoolSize,                   // pool size (2)
                    kPoolStride);                // pool stride (2)
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
  const int out_features = appdata.u_linear_w.d0();                // Expected: 10

  // Since the linear_batch_u function expects a flattened input, we need to ensure
  // that the data is correctly laid out. The pool3_out is already in the right layout,
  // but we need to interpret it as a 2D array.

  // Use the batched dense linear layer kernel
  linear_batch_u(appdata.u_pool3_out.data(),  // Input data (flattened 4D->2D)
                 appdata.u_linear_w.data(),
                 appdata.u_linear_b.data(),
                 appdata.u_linear_out.data(),
                 batch_size,      // 128
                 input_features,  // 1024
                 out_features);   // 10
}

}  // namespace cifar_dense::omp