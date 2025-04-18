#include "dispatchers.hpp"

#include <cstdint>

#include "../../debug_logger.hpp"

namespace cifar_dense::vulkan {

// Convolution parameters
constexpr int kKernelSize = 3;
constexpr int kStride = 1;
constexpr int kPadding = 1;

// Pooling parameters
constexpr int kPoolSize = 2;
constexpr int kPoolStride = 2;

constexpr bool kRelu = true;

// ----------------------------------------------------------------------------
// V2
// ----------------------------------------------------------------------------

namespace v2 {

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
  int32_t N;        // Batch size
  int32_t C;        // Number of channels
  int32_t H;        // Input height
  int32_t W;        // Input width
  int32_t pool_h;   // Pooling kernel height
  int32_t pool_w;   // Pooling kernel width
  int32_t stride;   // Pooling stride
  int32_t padding;  // Pooling padding
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

  // Create algorithm for conv2d
  auto conv2d_algo = engine.make_algo("new_cifar_dense_conv2d")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(4)  // Input, Weight, Bias, Output
                         ->push_constant<Conv2dPushConstants_v2>()
                         ->build();

  cached_algorithms.try_emplace("conv2d", std::move(conv2d_algo));

  // Create algorithm for maxpool
  auto maxpool_algo = engine.make_algo("new_cifar_dense_maxpool")
                          ->work_group_size(256, 1, 1)
                          ->num_sets(1)
                          ->num_buffers(2)  // Input, Output
                          ->push_constant<MaxpoolPushConstants_v2>()
                          ->build();

  cached_algorithms.try_emplace("maxpool", std::move(maxpool_algo));

  // Create algorithm for linear
  auto linear_algo = engine.make_algo("new_cifar_dense_linear")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(4)  // Input, Weight, Bias, Output
                         ->push_constant<LinearPushConstants_v2>()
                         ->build();

  cached_algorithms.try_emplace("linear", std::move(linear_algo));
}

// ----------------------------------------------------------------------------
// Stage 1 (v2) - Conv1
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_1(cifar_dense::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 1, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_input.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv1_w.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv1_b.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv1_out.pmr_vec()),
                              });

  const int batch_size = appdata.u_input.d0();   // Expected 512
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

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 2 (v2) - MaxPool1
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_2(cifar_dense::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 2, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv1_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_pool1_out.pmr_vec()),
                              });

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv1_out.d0();  // Expected: 512
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
      .padding = 0,  // No padding in MaxPool
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 3 (v2) - Conv2
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_3(cifar_dense::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 3, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_pool1_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv2_w.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv2_b.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv2_out.pmr_vec()),
                              });

  const int batch_size = appdata.u_pool1_out.d0();   // Expected: 512
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

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 4 (v2) - MaxPool2
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_4(cifar_dense::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 4, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv2_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_pool2_out.pmr_vec()),
                              });

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv2_out.d0();  // Expected: 512
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
      .padding = 0,  // No padding in MaxPool
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 5 (v2) - Conv3
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_5(cifar_dense::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 5, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_pool2_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv3_w.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv3_b.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv3_out.pmr_vec()),
                              });

  const int batch_size = appdata.u_pool2_out.d0();   // Expected: 512
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

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 6 (v2) - Conv4
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_6(cifar_dense::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 6, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv3_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv4_w.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv4_b.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv4_out.pmr_vec()),
                              });

  const int batch_size = appdata.u_conv3_out.d0();   // Expected: 512
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

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 7 (v2) - Conv5
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_7(cifar_dense::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 7, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv4_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv5_w.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv5_b.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv5_out.pmr_vec()),
                              });

  const int batch_size = appdata.u_conv4_out.d0();   // Expected: 512
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

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 8 (v2) - MaxPool3
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_8(cifar_dense::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 8, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv5_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_pool3_out.pmr_vec()),
                              });

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv5_out.d0();  // Expected: 512
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
      .padding = 0,  // No padding in MaxPool
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 9 (v2) - Linear
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_9(cifar_dense::v2::AppData& appdata) {
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
  const int batch_size = appdata.u_pool3_out.d0();       // Expected: 512
  const int channels = appdata.u_pool3_out.d1();         // Expected: 64
  const int height = appdata.u_pool3_out.d2();           // Expected: 4
  const int width = appdata.u_pool3_out.d3();            // Expected: 4
  const int input_features = channels * height * width;  // 64 * 4 * 4 = 1024

  // Number of output features from the linear layer
  const int out_features = appdata.u_linear_w.d1();  // Expected: 10

  const int total_output = batch_size * out_features;

  algo->update_push_constant(LinearPushConstants_v2{
      .N = batch_size,
      .in_features = input_features,
      .out_features = out_features,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

}  // namespace v2

}  // namespace cifar_dense::vulkan
