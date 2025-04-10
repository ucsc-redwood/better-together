#include "dispatchers.hpp"

#include <cstdint>

#include "../../debug_logger.hpp"

namespace cifar_sparse::vulkan {

// ----------------------------------------------------------------------------
// Uniform parameters
// ----------------------------------------------------------------------------

// layout(push_constant) uniform Params {
//   uint input_height;
//   uint input_width;
//   uint weight_output_channels;
//   uint weight_input_channels;
//   uint weight_height;
//   uint weight_width;
//   uint bias_number_of_elements;
//   uint kernel_size;
//   uint stride;
//   uint padding;
//   uint output_height;
//   uint output_width;
//   bool relu;
// }
// params;

struct Conv2dPushConstants {
  uint32_t input_height;
  uint32_t input_width;
  uint32_t weight_output_channels;
  uint32_t weight_input_channels;
  uint32_t weight_height;
  uint32_t weight_width;
  uint32_t kernel_size;
  uint32_t stride;
  uint32_t padding;
  bool relu;
};

// layout(push_constant) uniform Params {
//   uint input_channels;
//   uint input_height;
//   uint input_width;
//   uint pool_size;
//   uint stride;
//   uint output_height;
//   uint output_width;
// }
// params;

struct MaxpoolPushConstants {
  uint32_t input_channels;
  uint32_t input_height;
  uint32_t input_width;
  uint32_t pool_size;
  uint32_t stride;
};

// layout(push_constant) uniform Params {
//   int weight_matrix_rows;
//   int weight_matrix_cols;
// }
// params;

struct LinearPushConstants {
  uint32_t weight_matrix_rows;
  uint32_t weight_matrix_cols;
};

// ----------------------------------------------------------------------------
// Model parameters (from the paper)
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------

VulkanDispatcher::VulkanDispatcher() : engine(), seq(engine.make_seq()) {
  spdlog::debug("VulkanDispatcher::VulkanDispatcher(), Initializing VulkanDispatcher");

  // conv2d

  auto conv2d_algo = engine.make_algo("cifar_sparse_conv2d")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(6)
                         ->push_constant<Conv2dPushConstants>()
                         ->build();

  cached_algorithms.try_emplace("conv2d", std::move(conv2d_algo));

  // maxpool2d

  auto maxpool2d_algo = engine.make_algo("cifar_sparse_maxpool")
                            ->work_group_size(256, 1, 1)
                            ->num_sets(1)
                            ->num_buffers(2)
                            ->push_constant<MaxpoolPushConstants>()
                            ->build();

  cached_algorithms.try_emplace("maxpool2d", std::move(maxpool2d_algo));

  // linear

  auto linear_algo = engine.make_algo("cifar_sparse_linear")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(6)
                         ->push_constant<LinearPushConstants>()
                         ->build();

  cached_algorithms.try_emplace("linear", std::move(linear_algo));
}

// ----------------------------------------------------------------------------
// Stage 1 (first conv2d)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_1(cifar_sparse::AppData& appdata) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 1, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_image_data),
                                  engine.get_buffer_info(appdata.u_conv1_values),
                                  engine.get_buffer_info(appdata.u_conv1_row_ptr),
                                  engine.get_buffer_info(appdata.u_conv1_col_idx),
                                  engine.get_buffer_info(appdata.u_conv1_bias),
                                  engine.get_buffer_info(appdata.u_conv1_output),
                              });

  algo->update_push_constant(Conv2dPushConstants{
      .input_height = kInputHeight,
      .input_width = kInputWidth,
      .weight_output_channels = 64,
      .weight_input_channels = kInputChannels,
      .weight_height = static_cast<uint32_t>(appdata.conv1_weights.rows),
      .weight_width = static_cast<uint32_t>(appdata.conv1_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  const uint32_t total_iterations = appdata.conv1_weights.rows;

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 2 (first maxpool2d)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_2(cifar_sparse::AppData& appdata) {
  constexpr auto output_height = (kInputHeight - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (kInputWidth - kPoolSize) / kPoolStride + 1;
  auto total_iterations = 64 * output_height * output_width;

  LOG_KERNEL(LogKernelType::kVK, 2, &appdata);

  auto algo = cached_algorithms.at("maxpool2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv1_output),
                                  engine.get_buffer_info(appdata.u_pool1_output),
                              });

  algo->update_push_constant(MaxpoolPushConstants{
      .input_channels = 64,
      .input_height = 32,
      .input_width = 32,
      .pool_size = kPoolSize,
      .stride = kPoolStride,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 3 (second conv2d)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_3(cifar_sparse::AppData& appdata) {
  const auto total_iterations = appdata.conv2_weights.rows;

  LOG_KERNEL(LogKernelType::kVK, 3, &appdata);

  auto algo = cached_algorithms.at("conv2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_pool1_output),
                                  engine.get_buffer_info(appdata.u_conv2_values),
                                  engine.get_buffer_info(appdata.u_conv2_row_ptr),
                                  engine.get_buffer_info(appdata.u_conv2_col_idx),
                                  engine.get_buffer_info(appdata.u_conv2_bias),
                                  engine.get_buffer_info(appdata.u_conv2_output),
                              });

  algo->update_push_constant(Conv2dPushConstants{
      .input_height = 16,
      .input_width = 16,
      .weight_output_channels = 192,
      .weight_input_channels = 64,
      .weight_height = static_cast<uint32_t>(appdata.conv2_weights.rows),
      .weight_width = static_cast<uint32_t>(appdata.conv2_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 4 (second maxpool2d)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_4(cifar_sparse::AppData& appdata) {
  constexpr auto input_channels = 192;
  constexpr auto input_height = 16;
  constexpr auto input_width = 16;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations = input_channels * output_height * output_width;

  LOG_KERNEL(LogKernelType::kVK, 4, &appdata);

  auto algo = cached_algorithms.at("maxpool2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv2_output),
                                  engine.get_buffer_info(appdata.u_pool2_output),
                              });

  algo->update_push_constant(MaxpoolPushConstants{
      .input_channels = input_channels,
      .input_height = input_height,
      .input_width = input_width,
      .pool_size = kPoolSize,
      .stride = kPoolStride,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 5 (third conv2d)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_5(cifar_sparse::AppData& appdata) {
  const auto total_iterations = appdata.conv3_weights.rows;

  LOG_KERNEL(LogKernelType::kVK, 5, &appdata);

  auto algo = cached_algorithms.at("conv2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_pool2_output),
                                  engine.get_buffer_info(appdata.u_conv3_values),
                                  engine.get_buffer_info(appdata.u_conv3_row_ptr),
                                  engine.get_buffer_info(appdata.u_conv3_col_idx),
                                  engine.get_buffer_info(appdata.u_conv3_bias),
                                  engine.get_buffer_info(appdata.u_conv3_output),
                              });

  algo->update_push_constant(Conv2dPushConstants{
      .input_height = 8,
      .input_width = 8,
      .weight_output_channels = 384,
      .weight_input_channels = 192,
      .weight_height = static_cast<uint32_t>(appdata.conv3_weights.rows),
      .weight_width = static_cast<uint32_t>(appdata.conv3_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 6 (fourth conv2d)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_6(cifar_sparse::AppData& appdata) {
  const auto total_iterations = appdata.conv4_weights.rows;

  LOG_KERNEL(LogKernelType::kVK, 6, &appdata);

  auto algo = cached_algorithms.at("conv2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv3_output),
                                  engine.get_buffer_info(appdata.u_conv4_values),
                                  engine.get_buffer_info(appdata.u_conv4_row_ptr),
                                  engine.get_buffer_info(appdata.u_conv4_col_idx),
                                  engine.get_buffer_info(appdata.u_conv4_bias),
                                  engine.get_buffer_info(appdata.u_conv4_output),
                              });

  algo->update_push_constant(Conv2dPushConstants{
      .input_height = 8,
      .input_width = 8,
      .weight_output_channels = 256,
      .weight_input_channels = 384,
      .weight_height = static_cast<uint32_t>(appdata.conv4_weights.rows),
      .weight_width = static_cast<uint32_t>(appdata.conv4_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 7 (fifth conv2d)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_7(cifar_sparse::AppData& appdata) {
  const auto total_iterations = appdata.conv5_weights.rows;

  LOG_KERNEL(LogKernelType::kVK, 7, &appdata);

  auto algo = cached_algorithms.at("conv2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv4_output),
                                  engine.get_buffer_info(appdata.u_conv5_values),
                                  engine.get_buffer_info(appdata.u_conv5_row_ptr),
                                  engine.get_buffer_info(appdata.u_conv5_col_idx),
                                  engine.get_buffer_info(appdata.u_conv5_bias),
                                  engine.get_buffer_info(appdata.u_conv5_output),
                              });

  algo->update_push_constant(Conv2dPushConstants{
      .input_height = 8,
      .input_width = 8,
      .weight_output_channels = 256,
      .weight_input_channels = 256,
      .weight_height = static_cast<uint32_t>(appdata.conv5_weights.rows),
      .weight_width = static_cast<uint32_t>(appdata.conv5_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 8 (third maxpool2d)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_8(cifar_sparse::AppData& appdata) {
  constexpr auto input_channels = 256;
  constexpr auto input_height = 8;
  constexpr auto input_width = 8;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations = input_channels * output_height * output_width;

  LOG_KERNEL(LogKernelType::kVK, 8, &appdata);

  auto algo = cached_algorithms.at("maxpool2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv5_output),
                                  engine.get_buffer_info(appdata.u_pool3_output),
                              });

  algo->update_push_constant(MaxpoolPushConstants{
      .input_channels = input_channels,
      .input_height = input_height,
      .input_width = input_width,
      .pool_size = kPoolSize,
      .stride = kPoolStride,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 9 (linear)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_9(cifar_sparse::AppData& appdata) {
  const auto total_iterations = appdata.linear_weights.rows;

  LOG_KERNEL(LogKernelType::kVK, 9, &appdata);

  auto algo = cached_algorithms.at("linear").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_pool3_output),
                                  engine.get_buffer_info(appdata.u_linear_values),
                                  engine.get_buffer_info(appdata.u_linear_row_ptr),
                                  engine.get_buffer_info(appdata.u_linear_col_idx),
                                  engine.get_buffer_info(appdata.u_linear_bias),
                                  engine.get_buffer_info(appdata.u_linear_output),
                              });

  algo->update_push_constant(LinearPushConstants{
      .weight_matrix_rows = static_cast<uint32_t>(appdata.linear_weights.rows),
      .weight_matrix_cols = static_cast<uint32_t>(appdata.linear_weights.cols),
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// V2
// ----------------------------------------------------------------------------

namespace v2 {

// // Push constants holding all kernel parameters.
// layout(push_constant) uniform Params {
//   int batch_size;    // Number of batches.
//   int in_channels;   // Input channels.
//   int in_height;     // Input height.
//   int in_width;      // Input width.
//   int out_channels;  // Output channels.
//   int out_height;    // Output height.
//   int out_width;     // Output width.
//   int kernel_size;   // Kernel size (assumed square: kernel_size x kernel_size).
//   int stride;        // Convolution stride.
//   int padding;       // Padding.
//   int relu;          // Flag for ReLU activation (nonzero true).
//   int bias_size;     // Bias vector size (0 if bias is not used).
// }
// params;

struct Conv2dPushConstants_v2 {
  int32_t batch_size;
  int32_t in_channels;
  int32_t in_height;
  int32_t in_width;
  int32_t out_channels;
  int32_t out_height;
  int32_t out_width;
  int32_t kernel_size;
  int32_t stride;
  int32_t padding;
  int32_t relu;
  int32_t bias_size;
};

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------

VulkanDispatcher::VulkanDispatcher() : engine(), seq(engine.make_seq()) {
  spdlog::debug("VulkanDispatcher::VulkanDispatcher(), Initializing VulkanDispatcher");

  // // Work-group size: 256 threads per work-group.
  // layout(local_size_x = 256) in;

  // // Bindings for the input and output data.
  // layout(std430, set = 0, binding = 0) readonly buffer InputBuffer {
  //   float input_data[];  // Input layout: [batch, in_channels, in_height, in_width]
  // };

  // layout(std430, set = 0, binding = 1) writeonly buffer OutputBuffer {
  //   float output_data[];  // Output layout: [batch, out_channels, out_height, out_width]
  // };

  // // Bindings for sparse weight tensors in CSR format.
  // layout(std430, set = 0, binding = 2) readonly buffer WeightValsBuffer {
  //   float weight_vals[];  // Nonzero weight values.
  // };

  // layout(std430, set = 0, binding = 3) readonly buffer WeightRowPtrBuffer {
  //   int weight_row_ptr[];  // Row pointers: length = out_channels + 1.
  // };

  // layout(std430, set = 0, binding = 4) readonly buffer WeightColIdxBuffer {
  //   int weight_col_idx[];  // Column indices (flat kernel index).
  // };

  // // Binding for bias vector (if available). If no bias is used, set bias_size to 0.
  // layout(std430, set = 0, binding = 5) readonly buffer BiasBuffer { float bias_data[]; };

  auto conv2d_algo = engine.make_algo("new_cifar_sparse_conv2d")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(6)
                         ->push_constant<Conv2dPushConstants_v2>()
                         ->build();

  cached_algorithms.try_emplace("conv2d", std::move(conv2d_algo));
}

void VulkanDispatcher::run_stage_1(cifar_sparse::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 1, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_input.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv1_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.conv1_sparse.values_pmr_vec()),
                                  engine.get_buffer_info(appdata.conv1_sparse.row_ptr_pmr_vec()),
                                  engine.get_buffer_info(appdata.conv1_sparse.col_idx_pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv1_b.pmr_vec()),
                              });

  const int batch_size = appdata.u_input.d0();   // Expected 128
  const int in_channels = appdata.u_input.d1();  // Expected 3 (RGB)
  const int in_height = appdata.u_input.d2();    // Expected 32
  const int in_width = appdata.u_input.d3();     // Expected 32

  const int out_channels = appdata.conv1_sparse.rows;  // Expected 16

  const int out_height = (in_height + 2 * kPadding - kKernelSize) / kStride + 1;
  const int out_width = (in_width + 2 * kPadding - kKernelSize) / kStride + 1;

  const int total_output = batch_size * out_channels * out_height * out_width;

  algo->update_push_constant(Conv2dPushConstants_v2{
      .batch_size = batch_size,
      .in_channels = in_channels,
      .in_height = in_height,
      .in_width = in_width,
      .out_channels = out_channels,
      .out_height = out_height,
      .out_width = out_width,
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
      .bias_size = appdata.u_conv1_b.size(),
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

}  // namespace cifar_sparse::vulkan
