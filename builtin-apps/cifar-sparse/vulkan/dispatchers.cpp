#include "dispatchers.hpp"

#include <cstdint>

#include "../../debug_logger.hpp"

namespace cifar_sparse::vulkan {

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

// // Push constant block to pass convolution parameters.
// layout(push_constant) uniform Params {
//   int batch_size;  // Number of batches.
//   int channels;    // Number of channels.
//   int in_height;   // Input height.
//   int in_width;    // Input width.
//   int out_height;  // Output height, computed as (in_height - pool_size) / stride + 1.
//   int out_width;   // Output width, computed as (in_width - pool_size) / stride + 1.
//   int pool_size;   // Size of the pooling window (assumed square).
//   int stride;      // Stride of the pooling operation.
// }
// params;

struct MaxpoolPushConstants_v2 {
  int32_t batch_size;
  int32_t channels;
  int32_t in_height;
  int32_t in_width;
  int32_t out_height;
  int32_t out_width;
  int32_t pool_size;
  int32_t stride;
};

// // Push constants for kernel parameters.
// layout(push_constant) uniform Params {
//   int batch_size;      // Number of samples in the batch.
//   int input_features;  // Number of features in each input sample.
//   int out_neurons;     // Number of output neurons (rows in the weight matrix).
// }
// params;

struct LinearPushConstants_v2 {
  int32_t batch_size;
  int32_t input_features;
  int32_t out_neurons;
};

// ----------------------------------------------------------------------------
// Constructor (v2)
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

  // // Configure work-group size (256 threads per workgroup).
  // layout(local_size_x = 256) in;

  // // Input data buffer: flattened tensor of shape [batch, channels, in_height, in_width]
  // layout(std430, set = 0, binding = 0) readonly buffer InputBuffer { float input_data[]; };

  // // Output data buffer: flattened tensor of shape [batch, channels, out_height, out_width]
  // layout(std430, set = 0, binding = 1) writeonly buffer OutputBuffer { float output_data[]; };

  auto maxpool_algo = engine.make_algo("new_cifar_sparse_maxpool")
                          ->work_group_size(256, 1, 1)
                          ->num_sets(1)
                          ->num_buffers(2)
                          ->push_constant<MaxpoolPushConstants_v2>()
                          ->build();

  cached_algorithms.try_emplace("maxpool", std::move(maxpool_algo));

  // // Work-group size: 256 invocations per workgroup.
  // layout(local_size_x = 256) in;

  // // Input data buffer: flattened tensor of shape [batch_size, input_features]
  // layout(std430, set = 0, binding = 0) readonly buffer InputBuffer { float input_data[]; };

  // // Output data buffer: flattened tensor of shape [batch_size, out_neurons]
  // layout(std430, set = 0, binding = 1) writeonly buffer OutputBuffer { float output_data[]; };

  // // Sparse weight values (nonzeros) for the CSR matrix (dimensions: [out_neurons,
  // input_features]) layout(std430, set = 0, binding = 2) readonly buffer WeightValsBuffer { float
  // weight_vals[]; };

  // // CSR row pointers: length = out_neurons + 1.
  // layout(std430, set = 0, binding = 3) readonly buffer WeightRowPtrBuffer { int weight_row_ptr[];
  // };

  // // CSR column indices: flat indices of nonzero weight locations (i.e. input feature indices).
  // layout(std430, set = 0, binding = 4) readonly buffer WeightColIdxBuffer { int weight_col_idx[];
  // };

  // // Bias vector: one element per output neuron.
  // layout(std430, set = 0, binding = 5) readonly buffer BiasBuffer { float bias_data[]; };

  auto linear_algo = engine.make_algo("new_cifar_sparse_linear")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(6)
                         ->push_constant<LinearPushConstants_v2>()
                         ->build();

  cached_algorithms.try_emplace("linear", std::move(linear_algo));
}

// ----------------------------------------------------------------------------
// Stage 1 (v2)
// ----------------------------------------------------------------------------

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
      .bias_size = appdata.u_conv1_b.d0(),
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
// Stage 2 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_2(cifar_sparse::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 2, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv1_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_pool1_out.pmr_vec()),
                              });

  // Extract dimensions from the convolution output NDArray4D.
  const int batch_size = appdata.u_conv1_out.d0();  // Expected: 128
  const int channels = appdata.u_conv1_out.d1();    // Expected: 16
  const int in_height = appdata.u_conv1_out.d2();   // Expected: 32
  const int in_width = appdata.u_conv1_out.d3();    // Expected: 32

  const int out_height = (in_height - kPoolSize) / kPoolStride + 1;
  const int out_width = (in_width - kPoolSize) / kPoolStride + 1;

  const int total_output = batch_size * channels * out_height * out_width;

  algo->update_push_constant(MaxpoolPushConstants_v2{
      .batch_size = batch_size,
      .channels = channels,
      .in_height = in_height,
      .in_width = in_width,
      .out_height = out_height,
      .out_width = out_width,
      .pool_size = kPoolSize,
      .stride = kPoolStride,
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
// Stage 3 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_3(cifar_sparse::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 3, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_pool1_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv2_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.conv2_sparse.values_pmr_vec()),
                                  engine.get_buffer_info(appdata.conv2_sparse.row_ptr_pmr_vec()),
                                  engine.get_buffer_info(appdata.conv2_sparse.col_idx_pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv2_b.pmr_vec()),
                              });

  const int batch_size = appdata.u_pool1_out.d0();
  const int in_channels = appdata.u_pool1_out.d1();
  const int in_height = appdata.u_pool1_out.d2();
  const int in_width = appdata.u_pool1_out.d3();

  const int out_channels = appdata.conv2_sparse.rows;

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
      .bias_size = appdata.u_conv2_b.d0(),
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
// Stage 4 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_4(cifar_sparse::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 4, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv2_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_pool2_out.pmr_vec()),
                              });

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv2_out.d0();
  const int channels = appdata.u_conv2_out.d1();
  const int in_height = appdata.u_conv2_out.d2();
  const int in_width = appdata.u_conv2_out.d3();

  const int out_height = (in_height - kPoolSize) / kPoolStride + 1;
  const int out_width = (in_width - kPoolSize) / kPoolStride + 1;

  const int total_output = batch_size * channels * out_height * out_width;

  algo->update_push_constant(MaxpoolPushConstants_v2{
      .batch_size = batch_size,
      .channels = channels,
      .in_height = in_height,
      .in_width = in_width,
      .out_height = out_height,
      .out_width = out_width,
      .pool_size = kPoolSize,
      .stride = kPoolStride,
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
// Stage 5 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_5(cifar_sparse::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 5, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_pool2_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv3_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.conv3_sparse.values_pmr_vec()),
                                  engine.get_buffer_info(appdata.conv3_sparse.row_ptr_pmr_vec()),
                                  engine.get_buffer_info(appdata.conv3_sparse.col_idx_pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv3_b.pmr_vec()),
                              });

  const int batch_size = appdata.u_pool2_out.d0();
  const int in_channels = appdata.u_pool2_out.d1();
  const int in_height = appdata.u_pool2_out.d2();
  const int in_width = appdata.u_pool2_out.d3();

  const int out_channels = appdata.conv3_sparse.rows;

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
      .bias_size = appdata.u_conv3_b.d0(),
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
// Stage 6 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_6(cifar_sparse::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 6, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv3_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv4_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.conv4_sparse.values_pmr_vec()),
                                  engine.get_buffer_info(appdata.conv4_sparse.row_ptr_pmr_vec()),
                                  engine.get_buffer_info(appdata.conv4_sparse.col_idx_pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv4_b.pmr_vec()),
                              });

  const int batch_size = appdata.u_conv3_out.d0();
  const int in_channels = appdata.u_conv3_out.d1();
  const int in_height = appdata.u_conv3_out.d2();
  const int in_width = appdata.u_conv3_out.d3();

  const int out_channels = appdata.conv4_sparse.rows;

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
      .bias_size = appdata.u_conv4_b.d0(),
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
// Stage 7 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_7(cifar_sparse::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("conv2d").get();

  LOG_KERNEL(LogKernelType::kVK, 7, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv4_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv5_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.conv5_sparse.values_pmr_vec()),
                                  engine.get_buffer_info(appdata.conv5_sparse.row_ptr_pmr_vec()),
                                  engine.get_buffer_info(appdata.conv5_sparse.col_idx_pmr_vec()),
                                  engine.get_buffer_info(appdata.u_conv5_b.pmr_vec()),
                              });

  const int batch_size = appdata.u_conv4_out.d0();
  const int in_channels = appdata.u_conv4_out.d1();
  const int in_height = appdata.u_conv4_out.d2();
  const int in_width = appdata.u_conv4_out.d3();

  const int out_channels = appdata.conv5_sparse.rows;

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
      .bias_size = appdata.u_conv5_b.d0(),
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
// Stage 8 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_8(cifar_sparse::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 8, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_conv5_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_pool3_out.pmr_vec()),
                              });

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv5_out.d0();
  const int channels = appdata.u_conv5_out.d1();
  const int in_height = appdata.u_conv5_out.d2();
  const int in_width = appdata.u_conv5_out.d3();

  const int out_height = (in_height - kPoolSize) / kPoolStride + 1;
  const int out_width = (in_width - kPoolSize) / kPoolStride + 1;

  const int total_output = batch_size * channels * out_height * out_width;

  algo->update_push_constant(MaxpoolPushConstants_v2{
      .batch_size = batch_size,
      .channels = channels,
      .in_height = in_height,
      .in_width = in_width,
      .out_height = out_height,
      .out_width = out_width,
      .pool_size = kPoolSize,
      .stride = kPoolStride,
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
// Stage 9 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_9(cifar_sparse::v2::AppData& appdata) {
  auto algo = cached_algorithms.at("linear").get();

  LOG_KERNEL(LogKernelType::kVK, 9, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_pool3_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.u_linear_out.pmr_vec()),
                                  engine.get_buffer_info(appdata.linear_sparse.values_pmr_vec()),
                                  engine.get_buffer_info(appdata.linear_sparse.row_ptr_pmr_vec()),
                                  engine.get_buffer_info(appdata.linear_sparse.col_idx_pmr_vec()),
                                  engine.get_buffer_info(appdata.u_linear_b.pmr_vec()),
                              });

  // Calculate flattened input size (total number of features per sample)
  const int batch_size = appdata.u_pool3_out.d0();
  const int channels = appdata.u_pool3_out.d1();
  const int height = appdata.u_pool3_out.d2();
  const int width = appdata.u_pool3_out.d3();
  const int input_features = channels * height * width;

  // Number of output neurons is the number of rows in the weight matrix
  const int out_neurons = appdata.linear_sparse.rows;

  const int total_output = batch_size * out_neurons;

  algo->update_push_constant(LinearPushConstants_v2{
      .batch_size = batch_size,
      .input_features = input_features,
      .out_neurons = out_neurons,
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
