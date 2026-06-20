#include "dispatchers.hpp"

#include <cstdint>

#include "platform/util/debug_logger.hpp"

namespace cifar_dense::vulkan {

// // Convolution parameters
// constexpr int kKernelSize = 3;
// constexpr int kStride = 1;
// constexpr int kPadding = 1;

// // Pooling parameters
// constexpr int kPoolSize = 2;
// constexpr int kPoolStride = 2;

// constexpr bool kRelu = true;

// Push constants for the conv2d shader
struct Conv2dPushConstants_v2 {
  int32_t N;           // Batch size
  int32_t C;           // Number of input channels
  int32_t H;           // Input height
  int32_t W;           // Input width
  int32_t K;           // Number of output channels
  int32_t R;           // Kernel height
  int32_t S;           // Kernel width
  int32_t stride;      // Convolution stride
  int32_t padding;     // Convolution padding
  int32_t apply_relu;  // 1 to apply ReLU, 0 otherwise
};

// Push constants for the maxpool shader
struct MaxpoolPushConstants_v2 {
  int32_t N;       // Batch size
  int32_t C;       // Number of channels
  int32_t H;       // Input height
  int32_t W;       // Input width
  int32_t pool_h;  // Pooling kernel height
  int32_t pool_w;  // Pooling kernel width
  int32_t stride;  // Pooling stride
};

// Push constants for the linear shader
struct LinearPushConstants_v2 {
  int32_t N;             // Batch size
  int32_t in_features;   // Number of input features
  int32_t out_features;  // Number of output features
};

// ----------------------------------------------------------------------------
// Constructor (v2)
// ----------------------------------------------------------------------------

VulkanDispatcher::VulkanDispatcher() : engine(), seq(engine.make_seq()) {
  spdlog::debug("VulkanDispatcher::VulkanDispatcher(), Initializing VulkanDispatcher");

  // conv2d is reused by stages 1,3,5,6,7 and maxpool by stages 2,4,8. When a chunk
  // records several of those into one command buffer, each stage must bind its OWN
  // descriptor set (a single shared set would be overwritten and every dispatch would
  // see the last binding) -- so allocate one set per stage that uses the algo.
  auto conv2d_algo = engine.make_algo("new_cifar_dense_conv2d")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(5)     // stages 1,3,5,6,7
                         ->num_buffers(4)  // Input, Weight, Bias, Output
                         ->push_constant<Conv2dPushConstants_v2>()
                         ->build();

  cached_algorithms.try_emplace("conv2d", std::move(conv2d_algo));

  // Create algorithm for maxpool
  auto maxpool_algo = engine.make_algo("new_cifar_dense_maxpool")
                          ->work_group_size(256, 1, 1)
                          ->num_sets(3)     // stages 2,4,8
                          ->num_buffers(2)  // Input, Output
                          ->push_constant<MaxpoolPushConstants_v2>()
                          ->build();

  cached_algorithms.try_emplace("maxpool", std::move(maxpool_algo));

  // Create algorithm for linear
  auto linear_algo = engine.make_algo("new_cifar_dense_linear")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)     // stage 9
                         ->num_buffers(4)  // Input, Weight, Bias, Output
                         ->push_constant<LinearPushConstants_v2>()
                         ->build();

  cached_algorithms.try_emplace("linear", std::move(linear_algo));
}

// ----------------------------------------------------------------------------
// run_stage_k: per-stage single-submit wrappers (bm_main per-stage device timing).
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_1(AppData& appdata) { dispatch_multi_stage(appdata, 1, 1); }
void VulkanDispatcher::run_stage_2(AppData& appdata) { dispatch_multi_stage(appdata, 2, 2); }
void VulkanDispatcher::run_stage_3(AppData& appdata) { dispatch_multi_stage(appdata, 3, 3); }
void VulkanDispatcher::run_stage_4(AppData& appdata) { dispatch_multi_stage(appdata, 4, 4); }
void VulkanDispatcher::run_stage_5(AppData& appdata) { dispatch_multi_stage(appdata, 5, 5); }
void VulkanDispatcher::run_stage_6(AppData& appdata) { dispatch_multi_stage(appdata, 6, 6); }
void VulkanDispatcher::run_stage_7(AppData& appdata) { dispatch_multi_stage(appdata, 7, 7); }
void VulkanDispatcher::run_stage_8(AppData& appdata) { dispatch_multi_stage(appdata, 8, 8); }
void VulkanDispatcher::run_stage_9(AppData& appdata) { dispatch_multi_stage(appdata, 9, 9); }

// ----------------------------------------------------------------------------
// Stage 1 (v2) - Conv1  (conv2d descriptor set 0)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_1(AppData& appdata, vk::CommandBuffer cmd) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 1, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_input.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv1_w.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv1_b.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv1_out.pmr_vec()),
                              });

  const int batch_size = appdata.u_input.d0();   // Expected 128
  const int in_channels = appdata.u_input.d1();  // Expected 3 (RGB)
  const int in_height = appdata.u_input.d2();    // Expected 32
  const int in_width = appdata.u_input.d3();     // Expected 32

  const int out_channels = appdata.u_conv1_w.d0();                                // Expected 16
  const int out_height = (in_height + 2 * kPadding - kKernelSize) / kStride + 1;  // 32
  const int out_width = (in_width + 2 * kPadding - kKernelSize) / kStride + 1;    // 32

  const int total_output = batch_size * out_channels * out_height * out_width;

  algo->update_push_constant(Conv2dPushConstants_v2{
      .N = batch_size,
      .C = in_channels,
      .H = in_height,
      .W = in_width,
      .K = out_channels,
      .R = kKernelSize,
      .S = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .apply_relu = kRelu ? 1 : 0,
  });

  algo->record_bind_core(cmd, 0);
  algo->record_bind_push(cmd);
  algo->record_dispatch(cmd, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 2 (v2) - MaxPool1  (maxpool descriptor set 0)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_2(AppData& appdata, vk::CommandBuffer cmd) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 2, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv1_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_pool1_out.pmr_vec()),
                              });

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv1_out.d0();  // Expected: 128
  const int channels = appdata.u_conv1_out.d1();    // Expected: 16
  const int in_height = appdata.u_conv1_out.d2();   // Expected: 32
  const int in_width = appdata.u_conv1_out.d3();    // Expected: 32

  const int out_height = (in_height - kPoolSize) / kPoolStride + 1;  // Expected: 16
  const int out_width = (in_width - kPoolSize) / kPoolStride + 1;    // Expected: 16

  const int total_output = batch_size * channels * out_height * out_width;

  algo->update_push_constant(MaxpoolPushConstants_v2{
      .N = batch_size,
      .C = channels,
      .H = in_height,
      .W = in_width,
      .pool_h = kPoolSize,
      .pool_w = kPoolSize,
      .stride = kPoolStride,
  });

  algo->record_bind_core(cmd, 0);
  algo->record_bind_push(cmd);
  algo->record_dispatch(cmd, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 3 (v2) - Conv2  (conv2d descriptor set 1)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_3(AppData& appdata, vk::CommandBuffer cmd) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 3, &appdata);

  algo->update_descriptor_set(1,
                              {
                                  engine.get_buffer_info(appdata.u_pool1_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv2_w.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv2_b.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv2_out.pmr_vec()),
                              });

  const int batch_size = appdata.u_pool1_out.d0();   // Expected: 128
  const int in_channels = appdata.u_pool1_out.d1();  // Expected: 16
  const int in_height = appdata.u_pool1_out.d2();    // Expected: 16
  const int in_width = appdata.u_pool1_out.d3();     // Expected: 16

  const int out_channels = appdata.u_conv2_w.d0();                                // Expected: 32
  const int out_height = (in_height + 2 * kPadding - kKernelSize) / kStride + 1;  // 16
  const int out_width = (in_width + 2 * kPadding - kKernelSize) / kStride + 1;    // 16

  const int total_output = batch_size * out_channels * out_height * out_width;

  algo->update_push_constant(Conv2dPushConstants_v2{
      .N = batch_size,
      .C = in_channels,
      .H = in_height,
      .W = in_width,
      .K = out_channels,
      .R = kKernelSize,
      .S = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .apply_relu = kRelu ? 1 : 0,
  });

  algo->record_bind_core(cmd, 1);
  algo->record_bind_push(cmd);
  algo->record_dispatch(cmd, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 4 (v2) - MaxPool2  (maxpool descriptor set 1)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_4(AppData& appdata, vk::CommandBuffer cmd) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 4, &appdata);

  algo->update_descriptor_set(1,
                              {
                                  engine.get_buffer_info(appdata.u_conv2_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_pool2_out.pmr_vec()),
                              });

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv2_out.d0();  // Expected: 128
  const int channels = appdata.u_conv2_out.d1();    // Expected: 32
  const int in_height = appdata.u_conv2_out.d2();   // Expected: 16
  const int in_width = appdata.u_conv2_out.d3();    // Expected: 16

  const int out_height = (in_height - kPoolSize) / kPoolStride + 1;  // Expected: 8
  const int out_width = (in_width - kPoolSize) / kPoolStride + 1;    // Expected: 8

  const int total_output = batch_size * channels * out_height * out_width;

  algo->update_push_constant(MaxpoolPushConstants_v2{
      .N = batch_size,
      .C = channels,
      .H = in_height,
      .W = in_width,
      .pool_h = kPoolSize,
      .pool_w = kPoolSize,
      .stride = kPoolStride,
  });

  algo->record_bind_core(cmd, 1);
  algo->record_bind_push(cmd);
  algo->record_dispatch(cmd, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 5 (v2) - Conv3  (conv2d descriptor set 2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_5(AppData& appdata, vk::CommandBuffer cmd) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 5, &appdata);

  algo->update_descriptor_set(2,
                              {
                                  engine.get_buffer_info(appdata.u_pool2_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv3_w.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv3_b.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv3_out.pmr_vec()),
                              });

  const int batch_size = appdata.u_pool2_out.d0();   // Expected: 128
  const int in_channels = appdata.u_pool2_out.d1();  // Expected: 32
  const int in_height = appdata.u_pool2_out.d2();    // Expected: 8
  const int in_width = appdata.u_pool2_out.d3();     // Expected: 8

  const int out_channels = appdata.u_conv3_w.d0();                                // Expected: 64
  const int out_height = (in_height + 2 * kPadding - kKernelSize) / kStride + 1;  // 8
  const int out_width = (in_width + 2 * kPadding - kKernelSize) / kStride + 1;    // 8

  const int total_output = batch_size * out_channels * out_height * out_width;

  algo->update_push_constant(Conv2dPushConstants_v2{
      .N = batch_size,
      .C = in_channels,
      .H = in_height,
      .W = in_width,
      .K = out_channels,
      .R = kKernelSize,
      .S = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .apply_relu = kRelu ? 1 : 0,
  });

  algo->record_bind_core(cmd, 2);
  algo->record_bind_push(cmd);
  algo->record_dispatch(cmd, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 6 (v2) - Conv4  (conv2d descriptor set 3)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_6(AppData& appdata, vk::CommandBuffer cmd) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 6, &appdata);

  algo->update_descriptor_set(3,
                              {
                                  engine.get_buffer_info(appdata.u_conv3_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv4_w.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv4_b.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv4_out.pmr_vec()),
                              });

  const int batch_size = appdata.u_conv3_out.d0();   // Expected: 128
  const int in_channels = appdata.u_conv3_out.d1();  // Expected: 64
  const int in_height = appdata.u_conv3_out.d2();    // Expected: 8
  const int in_width = appdata.u_conv3_out.d3();     // Expected: 8

  const int out_channels = appdata.u_conv4_w.d0();                                // Expected: 64
  const int out_height = (in_height + 2 * kPadding - kKernelSize) / kStride + 1;  // 8
  const int out_width = (in_width + 2 * kPadding - kKernelSize) / kStride + 1;    // 8

  const int total_output = batch_size * out_channels * out_height * out_width;

  algo->update_push_constant(Conv2dPushConstants_v2{
      .N = batch_size,
      .C = in_channels,
      .H = in_height,
      .W = in_width,
      .K = out_channels,
      .R = kKernelSize,
      .S = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .apply_relu = kRelu ? 1 : 0,
  });

  algo->record_bind_core(cmd, 3);
  algo->record_bind_push(cmd);
  algo->record_dispatch(cmd, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 7 (v2) - Conv5  (conv2d descriptor set 4)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_7(AppData& appdata, vk::CommandBuffer cmd) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 7, &appdata);

  algo->update_descriptor_set(4,
                              {
                                  engine.get_buffer_info(appdata.u_conv4_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv5_w.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv5_b.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv5_out.pmr_vec()),
                              });

  const int batch_size = appdata.u_conv4_out.d0();   // Expected: 128
  const int in_channels = appdata.u_conv4_out.d1();  // Expected: 64
  const int in_height = appdata.u_conv4_out.d2();    // Expected: 8
  const int in_width = appdata.u_conv4_out.d3();     // Expected: 8

  const int out_channels = appdata.u_conv5_w.d0();                                // Expected: 64
  const int out_height = (in_height + 2 * kPadding - kKernelSize) / kStride + 1;  // 8
  const int out_width = (in_width + 2 * kPadding - kKernelSize) / kStride + 1;    // 8

  const int total_output = batch_size * out_channels * out_height * out_width;

  algo->update_push_constant(Conv2dPushConstants_v2{
      .N = batch_size,
      .C = in_channels,
      .H = in_height,
      .W = in_width,
      .K = out_channels,
      .R = kKernelSize,
      .S = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .apply_relu = kRelu ? 1 : 0,
  });

  algo->record_bind_core(cmd, 4);
  algo->record_bind_push(cmd);
  algo->record_dispatch(cmd, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 8 (v2) - MaxPool3  (maxpool descriptor set 2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_8(AppData& appdata, vk::CommandBuffer cmd) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 8, &appdata);

  algo->update_descriptor_set(2,
                              {
                                  engine.get_buffer_info(appdata.u_conv5_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_pool3_out.pmr_vec()),
                              });

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv5_out.d0();  // Expected: 128
  const int channels = appdata.u_conv5_out.d1();    // Expected: 64
  const int in_height = appdata.u_conv5_out.d2();   // Expected: 8
  const int in_width = appdata.u_conv5_out.d3();    // Expected: 8

  const int out_height = (in_height - kPoolSize) / kPoolStride + 1;  // Expected: 4
  const int out_width = (in_width - kPoolSize) / kPoolStride + 1;    // Expected: 4

  const int total_output = batch_size * channels * out_height * out_width;

  algo->update_push_constant(MaxpoolPushConstants_v2{
      .N = batch_size,
      .C = channels,
      .H = in_height,
      .W = in_width,
      .pool_h = kPoolSize,
      .pool_w = kPoolSize,
      .stride = kPoolStride,
  });

  algo->record_bind_core(cmd, 2);
  algo->record_bind_push(cmd);
  algo->record_dispatch(cmd, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 9 (v2) - Linear  (linear descriptor set 0)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_9(AppData& appdata, vk::CommandBuffer cmd) {
  auto algo = cached_algorithms.at("linear").get();

  LOG_KERNEL(LogKernelType::kVK, 9, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_pool3_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_linear_w.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_linear_b.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_linear_out.pmr_vec()),
                              });

  // Calculate flattened input size (total number of features per sample)
  const int batch_size = appdata.u_pool3_out.d0();       // Expected: 128
  const int channels = appdata.u_pool3_out.d1();         // Expected: 64
  const int height = appdata.u_pool3_out.d2();           // Expected: 4
  const int width = appdata.u_pool3_out.d3();            // Expected: 4
  const int input_features = channels * height * width;  // 64 * 4 * 4 = 1024

  // Number of output features from the linear layer
  const int out_features = appdata.u_linear_w.d0();  // Expected: 10

  const int total_output = batch_size * out_features;

  algo->update_push_constant(LinearPushConstants_v2{
      .N = batch_size,
      .in_features = input_features,
      .out_features = out_features,
  });

  algo->record_bind_core(cmd, 0);
  algo->record_bind_push(cmd);
  algo->record_dispatch(cmd, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

}  // namespace cifar_dense::vulkan
