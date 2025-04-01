#include "dispatchers.hpp"

#include <cstdint>

#include "../../debug_logger.hpp"

namespace cifar_sparse::vulkan {

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

struct MaxpoolPushConstants {
  uint32_t input_channels;
  uint32_t input_height;
  uint32_t input_width;
  uint32_t pool_size;
  uint32_t stride;
};

struct LinearPushConstants {
  uint32_t weight_matrix_rows;
  uint32_t weight_matrix_cols;
};

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
// Singleton Constructor
// ----------------------------------------------------------------------------

Singleton::Singleton() : engine(kiss_vk::Engine()), seq(engine.make_seq()) {
  spdlog::debug("Singleton instance created.");

  auto conv2d_algo = engine.make_algo("cifar_sparse_conv2d")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(6)
                         ->push_constant<Conv2dPushConstants>()
                         ->build();

  algorithms.try_emplace("conv2d", std::move(conv2d_algo));

  auto maxpool2d_algo = engine.make_algo("cifar_sparse_maxpool")
                            ->work_group_size(256, 1, 1)
                            ->num_sets(1)
                            ->num_buffers(2)
                            ->push_constant<MaxpoolPushConstants>()
                            ->build();

  algorithms.try_emplace("maxpool2d", std::move(maxpool2d_algo));

  auto linear_algo = engine.make_algo("cifar_sparse_linear")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(6)
                         ->push_constant<LinearPushConstants>()
                         ->build();

  algorithms.try_emplace("linear", std::move(linear_algo));
}

// ----------------------------------------------------------------------------
// Stage 1 (first conv2d)
// ----------------------------------------------------------------------------

void Singleton::process_stage_1(cifar_sparse::AppData &app_data) {
  const uint32_t total_iterations = app_data.conv1_weights.rows;

  LOG_KERNEL(LogKernelType::kVK, 1, &app_data);

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_image_data),
                                  engine.get_buffer_info(app_data.u_conv1_values),
                                  engine.get_buffer_info(app_data.u_conv1_row_ptr),
                                  engine.get_buffer_info(app_data.u_conv1_col_idx),
                                  engine.get_buffer_info(app_data.u_conv1_bias),
                                  engine.get_buffer_info(app_data.u_conv1_output),
                              });

  algo->update_push_constant(Conv2dPushConstants{
      .input_height = kInputHeight,
      .input_width = kInputWidth,
      .weight_output_channels = 64,
      .weight_input_channels = kInputChannels,
      .weight_height = static_cast<uint32_t>(app_data.conv1_weights.rows),
      .weight_width = static_cast<uint32_t>(app_data.conv1_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  for (auto batch = 0; batch < kNumBatches; ++batch) {
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
}

// ----------------------------------------------------------------------------
// Stage 2 (first maxpool2d)
// ----------------------------------------------------------------------------

void Singleton::process_stage_2(cifar_sparse::AppData &app_data) {
  constexpr auto output_height = (kInputHeight - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (kInputWidth - kPoolSize) / kPoolStride + 1;
  auto total_iterations = kInputChannels * output_height * output_width;

  LOG_KERNEL(LogKernelType::kVK, 2, &app_data);

  auto algo = algorithms.at("maxpool2d").get();

  //   algo->update_descriptor_sets({
  //       engine.get_buffer(app_data.u_conv1_output.data()),
  //       engine.get_buffer(app_data.u_pool1_output.data()),
  //   });

  //   algo->update_push_constants(MaxpoolPushConstants{
  //       .input_channels = 64,
  //       .input_height = 32,
  //       .input_width = 32,
  //       .pool_size = kPoolSize,
  //       .stride = kPoolStride,
  //   });

  //   seq->record_commands(algo, total_iterations);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_conv1_output),
                                  engine.get_buffer_info(app_data.u_pool1_output),
                              });

  algo->update_push_constant(MaxpoolPushConstants{
      .input_channels = 64,
      .input_height = 32,
      .input_width = 32,
      .pool_size = kPoolSize,
      .stride = kPoolStride,
  });

  for (auto batch = 0; batch < kNumBatches; ++batch) {
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
}

// ----------------------------------------------------------------------------
// Stage 3 (second conv2d)
// ----------------------------------------------------------------------------

void Singleton::process_stage_3(cifar_sparse::AppData &app_data) {
  const auto total_iterations = app_data.conv2_weights.rows;

  LOG_KERNEL(LogKernelType::kVK, 3, &app_data);

  auto algo = algorithms.at("conv2d").get();

  //   algo->update_descriptor_sets({
  //       engine.get_buffer(app_data.u_pool1_output.data()),
  //       engine.get_buffer(app_data.u_conv2_values.data()),
  //       engine.get_buffer(app_data.u_conv2_row_ptr.data()),
  //       engine.get_buffer(app_data.u_conv2_col_idx.data()),
  //       engine.get_buffer(app_data.u_conv2_bias.data()),
  //       engine.get_buffer(app_data.u_conv2_output.data()),
  //   });

  //   algo->update_push_constants(Conv2dPushConstants{
  //       .input_height = 16,
  //       .input_width = 16,
  //       .weight_output_channels = 192,
  //       .weight_input_channels = 64,
  //       .weight_height = static_cast<uint32_t>(app_data.conv2_weights.rows),
  //       .weight_width = static_cast<uint32_t>(app_data.conv2_weights.cols),
  //       .kernel_size = kKernelSize,
  //       .stride = kStride,
  //       .padding = kPadding,
  //       .relu = kRelu,
  //   });

  //   seq->record_commands(algo, total_iterations);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_pool1_output),
                                  engine.get_buffer_info(app_data.u_conv2_values),
                                  engine.get_buffer_info(app_data.u_conv2_row_ptr),
                                  engine.get_buffer_info(app_data.u_conv2_col_idx),
                                  engine.get_buffer_info(app_data.u_conv2_bias),
                                  engine.get_buffer_info(app_data.u_conv2_output),
                              });

  algo->update_push_constant(Conv2dPushConstants{
      .input_height = 16,
      .input_width = 16,
      .weight_output_channels = 192,
      .weight_input_channels = 64,
      .weight_height = static_cast<uint32_t>(app_data.conv2_weights.rows),
      .weight_width = static_cast<uint32_t>(app_data.conv2_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  for (auto batch = 0; batch < kNumBatches; ++batch) {
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
}

//----------------------------------------------------------------------------
// Stage 4 (second maxpool2d)
// ----------------------------------------------------------------------------

void Singleton::process_stage_4(cifar_sparse::AppData &app_data) {
  constexpr auto input_channels = 192;
  constexpr auto input_height = 16;
  constexpr auto input_width = 16;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations = input_channels * output_height * output_width;

  LOG_KERNEL(LogKernelType::kVK, 4, &app_data);

  auto algo = algorithms.at("maxpool2d").get();

  //   algo->update_descriptor_sets({
  //       engine.get_buffer(app_data.u_conv2_output.data()),
  //       engine.get_buffer(app_data.u_pool2_output.data()),
  //   });

  //   algo->update_push_constants(MaxpoolPushConstants{
  //       .input_channels = input_channels,
  //       .input_height = input_height,
  //       .input_width = input_width,
  //       .pool_size = kPoolSize,
  //       .stride = kPoolStride,
  //   });

  //   seq->record_commands(algo, total_iterations);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_conv2_output),
                                  engine.get_buffer_info(app_data.u_pool2_output),
                              });

  algo->update_push_constant(MaxpoolPushConstants{
      .input_channels = input_channels,
      .input_height = input_height,
      .input_width = input_width,
      .pool_size = kPoolSize,
      .stride = kPoolStride,
  });

  for (auto batch = 0; batch < kNumBatches; ++batch) {
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
}

// ----------------------------------------------------------------------------
// Stage 5 (third conv2d)
// ----------------------------------------------------------------------------

void Singleton::process_stage_5(cifar_sparse::AppData &app_data) {
  const auto total_iterations = app_data.conv3_weights.rows;

  LOG_KERNEL(LogKernelType::kVK, 5, &app_data);

  auto algo = algorithms.at("conv2d").get();

  //   algo->update_descriptor_sets({
  //       engine.get_buffer(app_data.u_pool2_output.data()),
  //       engine.get_buffer(app_data.u_conv3_values.data()),
  //       engine.get_buffer(app_data.u_conv3_row_ptr.data()),
  //       engine.get_buffer(app_data.u_conv3_col_idx.data()),
  //       engine.get_buffer(app_data.u_conv3_bias.data()),
  //       engine.get_buffer(app_data.u_conv3_output.data()),
  //   });

  //   algo->update_push_constants(Conv2dPushConstants{
  //       .input_height = 8,
  //       .input_width = 8,
  //       .weight_output_channels = 384,
  //       .weight_input_channels = 192,
  //       .weight_height = static_cast<uint32_t>(app_data.conv3_weights.rows),
  //       .weight_width = static_cast<uint32_t>(app_data.conv3_weights.cols),
  //       .kernel_size = kKernelSize,
  //       .stride = kStride,
  //       .padding = kPadding,
  //       .relu = kRelu,
  //   });

  //   seq->record_commands(algo, total_iterations);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_pool2_output),
                                  engine.get_buffer_info(app_data.u_conv3_values),
                                  engine.get_buffer_info(app_data.u_conv3_row_ptr),
                                  engine.get_buffer_info(app_data.u_conv3_col_idx),
                                  engine.get_buffer_info(app_data.u_conv3_bias),
                                  engine.get_buffer_info(app_data.u_conv3_output),
                              });

  algo->update_push_constant(Conv2dPushConstants{
      .input_height = 8,
      .input_width = 8,
      .weight_output_channels = 384,
      .weight_input_channels = 192,
      .weight_height = static_cast<uint32_t>(app_data.conv3_weights.rows),
      .weight_width = static_cast<uint32_t>(app_data.conv3_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  for (auto batch = 0; batch < kNumBatches; ++batch) {
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
}

// ----------------------------------------------------------------------------
// Stage 6 (fourth conv2d)
// ----------------------------------------------------------------------------

void Singleton::process_stage_6(cifar_sparse::AppData &app_data) {
  const auto total_iterations = app_data.conv4_weights.rows;

  LOG_KERNEL(LogKernelType::kVK, 6, &app_data);

  auto algo = algorithms.at("conv2d").get();

  //   algo->update_descriptor_sets({
  //       engine.get_buffer(app_data.u_conv3_output.data()),
  //       engine.get_buffer(app_data.u_conv4_values.data()),
  //       engine.get_buffer(app_data.u_conv4_row_ptr.data()),
  //       engine.get_buffer(app_data.u_conv4_col_idx.data()),
  //       engine.get_buffer(app_data.u_conv4_bias.data()),
  //       engine.get_buffer(app_data.u_conv4_output.data()),
  //   });

  //   algo->update_push_constants(Conv2dPushConstants{
  //       .input_height = 8,
  //       .input_width = 8,
  //       .weight_output_channels = 256,
  //       .weight_input_channels = 384,
  //       .weight_height = static_cast<uint32_t>(app_data.conv4_weights.rows),
  //       .weight_width = static_cast<uint32_t>(app_data.conv4_weights.cols),
  //       .kernel_size = kKernelSize,
  //       .stride = kStride,
  //       .padding = kPadding,
  //       .relu = kRelu,
  //   });

  //   seq->record_commands(algo, total_iterations);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_conv3_output),
                                  engine.get_buffer_info(app_data.u_conv4_values),
                                  engine.get_buffer_info(app_data.u_conv4_row_ptr),
                                  engine.get_buffer_info(app_data.u_conv4_col_idx),
                                  engine.get_buffer_info(app_data.u_conv4_bias),
                                  engine.get_buffer_info(app_data.u_conv4_output),
                              });

  algo->update_push_constant(Conv2dPushConstants{
      .input_height = 8,
      .input_width = 8,
      .weight_output_channels = 256,
      .weight_input_channels = 384,
      .weight_height = static_cast<uint32_t>(app_data.conv4_weights.rows),
      .weight_width = static_cast<uint32_t>(app_data.conv4_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  for (auto batch = 0; batch < kNumBatches; ++batch) {
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
}

// ----------------------------------------------------------------------------
// Stage 7 (fifth conv2d)
// ----------------------------------------------------------------------------

void Singleton::process_stage_7(cifar_sparse::AppData &app_data) {
  const auto total_iterations = app_data.conv5_weights.rows;

  LOG_KERNEL(LogKernelType::kVK, 7, &app_data);

  auto algo = algorithms.at("conv2d").get();

  //   algo->update_descriptor_sets({
  //       engine.get_buffer(app_data.u_conv4_output.data()),
  //       engine.get_buffer(app_data.u_conv5_values.data()),
  //       engine.get_buffer(app_data.u_conv5_row_ptr.data()),
  //       engine.get_buffer(app_data.u_conv5_col_idx.data()),
  //       engine.get_buffer(app_data.u_conv5_bias.data()),
  //       engine.get_buffer(app_data.u_conv5_output.data()),
  //   });

  //   algo->update_push_constants(Conv2dPushConstants{
  //       .input_height = 8,
  //       .input_width = 8,
  //       .weight_output_channels = 256,
  //       .weight_input_channels = 256,
  //       .weight_height = static_cast<uint32_t>(app_data.conv5_weights.rows),
  //       .weight_width = static_cast<uint32_t>(app_data.conv5_weights.cols),
  //       .kernel_size = kKernelSize,
  //       .stride = kStride,
  //       .padding = kPadding,
  //       .relu = kRelu,
  //   });

  //   seq->record_commands(algo, total_iterations);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_conv4_output),
                                  engine.get_buffer_info(app_data.u_conv5_values),
                                  engine.get_buffer_info(app_data.u_conv5_row_ptr),
                                  engine.get_buffer_info(app_data.u_conv5_col_idx),
                                  engine.get_buffer_info(app_data.u_conv5_bias),
                                  engine.get_buffer_info(app_data.u_conv5_output),
                              });

  algo->update_push_constant(Conv2dPushConstants{
      .input_height = 8,
      .input_width = 8,
      .weight_output_channels = 256,
      .weight_input_channels = 256,
      .weight_height = static_cast<uint32_t>(app_data.conv5_weights.rows),
      .weight_width = static_cast<uint32_t>(app_data.conv5_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  for (auto batch = 0; batch < kNumBatches; ++batch) {
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
}

// ----------------------------------------------------------------------------
// Stage 8 (third maxpool2d)
// ----------------------------------------------------------------------------

void Singleton::process_stage_8(cifar_sparse::AppData &app_data) {
  constexpr auto input_channels = 256;
  constexpr auto input_height = 8;
  constexpr auto input_width = 8;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations = input_channels * output_height * output_width;

  LOG_KERNEL(LogKernelType::kVK, 8, &app_data);

  auto algo = algorithms.at("maxpool2d").get();

  //   algo->update_descriptor_sets({
  //       engine.get_buffer(app_data.u_conv5_output.data()),
  //       engine.get_buffer(app_data.u_pool3_output.data()),
  //   });

  //   algo->update_push_constants(MaxpoolPushConstants{
  //       .input_channels = input_channels,
  //       .input_height = input_height,
  //       .input_width = input_width,
  //       .pool_size = kPoolSize,
  //       .stride = kPoolStride,
  //   });

  //   seq->record_commands(algo, total_iterations);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_conv5_output),
                                  engine.get_buffer_info(app_data.u_pool3_output),
                              });

  algo->update_push_constant(MaxpoolPushConstants{
      .input_channels = input_channels,
      .input_height = input_height,
      .input_width = input_width,
      .pool_size = kPoolSize,
      .stride = kPoolStride,
  });

  for (auto batch = 0; batch < kNumBatches; ++batch) {
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
}

// ----------------------------------------------------------------------------
// Stage 9 (linear)
// ----------------------------------------------------------------------------

void Singleton::process_stage_9(cifar_sparse::AppData &app_data) {
  const auto total_iterations = app_data.linear_weights.rows;

  LOG_KERNEL(LogKernelType::kVK, 9, &app_data);

  auto algo = algorithms.at("linear").get();

  //   algo->update_descriptor_sets({
  //       engine.get_buffer(app_data.u_pool3_output.data()),
  //       engine.get_buffer(app_data.u_linear_values.data()),
  //       engine.get_buffer(app_data.u_linear_row_ptr.data()),
  //       engine.get_buffer(app_data.u_linear_col_idx.data()),
  //       engine.get_buffer(app_data.u_linear_bias.data()),
  //       engine.get_buffer(app_data.u_linear_output.data()),
  //   });

  //   algo->update_push_constants(LinearPushConstants{
  //       .weight_matrix_rows = static_cast<uint32_t>(app_data.linear_weights.rows),
  //       .weight_matrix_cols = static_cast<uint32_t>(app_data.linear_weights.cols),
  //   });

  //   seq->record_commands(algo, total_iterations);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_pool3_output),
                                  engine.get_buffer_info(app_data.u_linear_values),
                                  engine.get_buffer_info(app_data.u_linear_row_ptr),
                                  engine.get_buffer_info(app_data.u_linear_col_idx),
                                  engine.get_buffer_info(app_data.u_linear_bias),
                                  engine.get_buffer_info(app_data.u_linear_output),
                              });

  algo->update_push_constant(LinearPushConstants{
      .weight_matrix_rows = static_cast<uint32_t>(app_data.linear_weights.rows),
      .weight_matrix_cols = static_cast<uint32_t>(app_data.linear_weights.cols),
  });

  for (auto batch = 0; batch < kNumBatches; ++batch) {
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
}

}  // namespace cifar_sparse::vulkan
