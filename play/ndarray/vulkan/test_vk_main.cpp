#include <gtest/gtest.h>

#include "dispatcher.hpp"

TEST(VulkanDispatcherTest, BasicInitialization) {
  cifar_dense::vulkan::VulkanDispatcher dispatcher;
  cifar_dense::AppDataBatch appdata;

  // Test that dispatcher can be created and run stage 1
  EXPECT_NO_THROW(dispatcher.run_stage_1(appdata));
}

TEST(VulkanDispatcherTest, Conv2dOutputShape) {
  cifar_dense::vulkan::VulkanDispatcher dispatcher;
  cifar_dense::AppDataBatch appdata;

  // Run conv2d operation
  dispatcher.run_stage_1(appdata);

  // Check output shape matches expected dimensions
  const auto& out_shape = appdata.u_conv1_out.shape();
  EXPECT_EQ(out_shape[0], 128);  // batch size
  EXPECT_EQ(out_shape[1], 16);   // output channels
  EXPECT_EQ(out_shape[2], 30);   // output height
  EXPECT_EQ(out_shape[3], 30);   // output width
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
