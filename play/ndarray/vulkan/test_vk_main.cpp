#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "dispatcher.hpp"

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
// Conv2d Output Non-Zero
// ----------------------------------------------------------------------------

TEST(VulkanDispatcherTest, Conv2dOutputNonZero) {
  cifar_dense::vulkan::VulkanDispatcher dispatcher;
  cifar_dense::AppDataBatch appdata(dispatcher.get_mr());

  // Run conv2d operation
  dispatcher.run_stage_1(appdata);

  // Get the output tensor data
  const float* output_data = appdata.u_conv1_out.raw();
  const size_t output_size = appdata.u_conv1_out.size();

  // Check if any value is non-zero
  bool has_non_zero = false;
  for (size_t i = 0; i < output_size; ++i) {
    if (output_data[i] != 0.0f) {
      has_non_zero = true;
      break;
    }
  }

  // Check output shape matches expected dimensions
  const auto& out_shape = appdata.u_conv1_out.shape();
  EXPECT_EQ(out_shape[0], 128);  // batch size
  EXPECT_EQ(out_shape[1], 16);   // output channels
  EXPECT_EQ(out_shape[2], 32);   // output height
  EXPECT_EQ(out_shape[3], 32);   // output width

  EXPECT_TRUE(has_non_zero) << "Conv2d output should contain non-zero values";
}

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::trace);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
