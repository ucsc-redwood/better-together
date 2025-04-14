#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "dispatcher.hpp"

// ----------------------------------------------------------------------------
// Helper functions
// ----------------------------------------------------------------------------

// Check if tensor contains non-zero values
bool has_non_zero_values(const float* data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (data[i] != 0.0f) {
      return true;
    }
  }
  return false;
}

// ----------------------------------------------------------------------------
// Basic Initialization
// ----------------------------------------------------------------------------

TEST(VulkanDispatcherTest, BasicInitialization) {
  cifar_dense::vulkan::VulkanDispatcher dispatcher;
  cifar_dense::AppDataBatch appdata(dispatcher.get_mr());

  // Test that dispatcher can be created and run stage 1
  EXPECT_NO_THROW(dispatcher.run_stage_1(appdata));
}

// ----------------------------------------------------------------------------
// Test individual stages
// ----------------------------------------------------------------------------

TEST(VulkanDispatcherTest, Stage1_Conv2d) {
  cifar_dense::vulkan::VulkanDispatcher dispatcher;
  cifar_dense::AppDataBatch appdata(dispatcher.get_mr());

  // Run conv2d operation
  dispatcher.run_stage_1(appdata);

  // Get the output tensor data
  const float* output_data = appdata.u_conv1_out.raw();
  const size_t output_size = appdata.u_conv1_out.size();

  // Check output shape matches expected dimensions
  const auto& out_shape = appdata.u_conv1_out.shape();
  EXPECT_EQ(out_shape[0], 128);  // batch size
  EXPECT_EQ(out_shape[1], 16);   // output channels
  EXPECT_EQ(out_shape[2], 32);   // output height
  EXPECT_EQ(out_shape[3], 32);   // output width

  EXPECT_TRUE(has_non_zero_values(output_data, output_size))
      << "Conv2d output should contain non-zero values";
}

TEST(VulkanDispatcherTest, Stage2_Maxpool) {
  cifar_dense::vulkan::VulkanDispatcher dispatcher;
  cifar_dense::AppDataBatch appdata(dispatcher.get_mr());

  // Need to run stage 1 first
  dispatcher.run_stage_1(appdata);
  // Then run maxpool operation
  dispatcher.run_stage_2(appdata);

  // Get the output tensor data
  const float* output_data = appdata.u_pool1_out.raw();
  const size_t output_size = appdata.u_pool1_out.size();

  // Check output shape matches expected dimensions
  const auto& out_shape = appdata.u_pool1_out.shape();
  EXPECT_EQ(out_shape[0], 128);  // batch size
  EXPECT_EQ(out_shape[1], 16);   // channels
  EXPECT_EQ(out_shape[2], 16);   // height
  EXPECT_EQ(out_shape[3], 16);   // width

  EXPECT_TRUE(has_non_zero_values(output_data, output_size))
      << "Maxpool output should contain non-zero values";
}

TEST(VulkanDispatcherTest, Stage3_Conv2d) {
  cifar_dense::vulkan::VulkanDispatcher dispatcher;
  cifar_dense::AppDataBatch appdata(dispatcher.get_mr());

  // Run previous stages
  dispatcher.run_stage_1(appdata);
  dispatcher.run_stage_2(appdata);
  // Then run second conv2d operation
  dispatcher.run_stage_3(appdata);

  // Get the output tensor data
  const float* output_data = appdata.u_conv2_out.raw();
  const size_t output_size = appdata.u_conv2_out.size();

  // Check output shape matches expected dimensions
  const auto& out_shape = appdata.u_conv2_out.shape();
  EXPECT_EQ(out_shape[0], 128);  // batch size
  EXPECT_EQ(out_shape[1], 32);   // output channels
  EXPECT_EQ(out_shape[2], 16);   // height
  EXPECT_EQ(out_shape[3], 16);   // width

  EXPECT_TRUE(has_non_zero_values(output_data, output_size))
      << "Second conv2d output should contain non-zero values";
}

TEST(VulkanDispatcherTest, Stage4_Maxpool) {
  cifar_dense::vulkan::VulkanDispatcher dispatcher;
  cifar_dense::AppDataBatch appdata(dispatcher.get_mr());

  // Run previous stages
  dispatcher.dispatch_multi_stage(appdata, 1, 3);
  // Then run second maxpool operation
  dispatcher.run_stage_4(appdata);

  // Get the output tensor data
  const float* output_data = appdata.u_pool2_out.raw();
  const size_t output_size = appdata.u_pool2_out.size();

  // Check output shape matches expected dimensions
  const auto& out_shape = appdata.u_pool2_out.shape();
  EXPECT_EQ(out_shape[0], 128);  // batch size
  EXPECT_EQ(out_shape[1], 32);   // channels
  EXPECT_EQ(out_shape[2], 8);    // height
  EXPECT_EQ(out_shape[3], 8);    // width

  EXPECT_TRUE(has_non_zero_values(output_data, output_size))
      << "Second maxpool output should contain non-zero values";
}

TEST(VulkanDispatcherTest, Stage5_Conv2d) {
  cifar_dense::vulkan::VulkanDispatcher dispatcher;
  cifar_dense::AppDataBatch appdata(dispatcher.get_mr());

  // Run previous stages
  dispatcher.dispatch_multi_stage(appdata, 1, 4);
  // Then run third conv2d operation
  dispatcher.run_stage_5(appdata);

  // Get the output tensor data
  const float* output_data = appdata.u_conv3_out.raw();
  const size_t output_size = appdata.u_conv3_out.size();

  // Check output shape matches expected dimensions
  const auto& out_shape = appdata.u_conv3_out.shape();
  EXPECT_EQ(out_shape[0], 128);  // batch size
  EXPECT_EQ(out_shape[1], 64);   // output channels
  EXPECT_EQ(out_shape[2], 8);    // height
  EXPECT_EQ(out_shape[3], 8);    // width

  EXPECT_TRUE(has_non_zero_values(output_data, output_size))
      << "Third conv2d output should contain non-zero values";
}

TEST(VulkanDispatcherTest, StageFinal_Linear) {
  cifar_dense::vulkan::VulkanDispatcher dispatcher;
  cifar_dense::AppDataBatch appdata(dispatcher.get_mr());

  // Run through all previous stages
  dispatcher.dispatch_multi_stage(appdata, 1, 8);
  // Then run the final linear operation
  dispatcher.run_stage_9(appdata);

  // Check final output shape
  const auto& out_shape = appdata.u_linear_out.shape();
  EXPECT_EQ(out_shape[0], 128);  // batch size
  EXPECT_EQ(out_shape[1], 10);   // output features (10 classes)

  // Check if output contains values (predictions)
  const float* output_data = appdata.u_linear_out.raw();
  const size_t output_size = appdata.u_linear_out.size();

  EXPECT_TRUE(has_non_zero_values(output_data, output_size))
      << "Linear layer output should contain non-zero values";

  // Print the first few predictions for visual inspection
  if (output_size > 0) {
    spdlog::info("Sample prediction values:");
    for (auto i = 0u; i < std::min(size_t(5), output_size); i++) {
      spdlog::info("Value at {}: {}", i, output_data[i]);
    }
  }
}

// ----------------------------------------------------------------------------
// Multi-stage Dispatch
// ----------------------------------------------------------------------------

TEST(VulkanDispatcherTest, MultiStageDispatch) {
  cifar_dense::vulkan::VulkanDispatcher dispatcher;
  cifar_dense::AppDataBatch appdata(dispatcher.get_mr());

  // Run the full neural network pipeline
  EXPECT_NO_THROW(dispatcher.dispatch_multi_stage(appdata, 1, 9));

  // Check final output shape
  const auto& out_shape = appdata.u_linear_out.shape();
  EXPECT_EQ(out_shape[0], 128);  // batch size
  EXPECT_EQ(out_shape[1], 10);   // output features (10 classes)

  // Check if output contains values (predictions)
  const float* output_data = appdata.u_linear_out.raw();
  const size_t output_size = appdata.u_linear_out.size();

  EXPECT_TRUE(has_non_zero_values(output_data, output_size))
      << "Final output should contain prediction values";

  // Print final predictions for a few samples
  cifar_dense::print_batch_predictions(appdata, 3);
}

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
