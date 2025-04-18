#include <gtest/gtest.h>

#include "../appdata.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// test appdata initialization
// ----------------------------------------------------------------------------

TEST(AppdataTest, Initialization) {
  cifar_dense::vulkan::VulkanDispatcher disp;
  cifar_dense::AppData appdata(disp.get_mr());

  EXPECT_EQ(appdata.u_conv1_w.d0(), 16);
  EXPECT_EQ(appdata.u_linear_w.d1(), 10);
}

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

TEST(Stage1Test, Basic) {
  cifar_dense::vulkan::VulkanDispatcher disp;
  cifar_dense::AppData appdata(disp.get_mr());

  // Run stage 1
  disp.run_stage_1(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv1_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv1_out.d1(), 16);
  EXPECT_EQ(appdata.u_conv1_out.d2(), 32);
  EXPECT_EQ(appdata.u_conv1_out.d3(), 32);

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_1(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 2
// ----------------------------------------------------------------------------

TEST(Stage2Test, Basic) {
  cifar_dense::vulkan::VulkanDispatcher disp;
  cifar_dense::AppData appdata(disp.get_mr());

  // Run stage 2
  disp.run_stage_2(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool1_out.d0(), 128);
  EXPECT_EQ(appdata.u_pool1_out.d1(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d2(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d3(), 16);

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_2(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 3
// ----------------------------------------------------------------------------

TEST(Stage3Test, Basic) {
  cifar_dense::vulkan::VulkanDispatcher disp;
  cifar_dense::AppData appdata(disp.get_mr());

  // Run stage 3
  disp.run_stage_3(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv2_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv2_out.d1(), 32);
  EXPECT_EQ(appdata.u_conv2_out.d2(), 16);
  EXPECT_EQ(appdata.u_conv2_out.d3(), 16);

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_3(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 4
// ----------------------------------------------------------------------------

TEST(Stage4Test, Basic) {
  cifar_dense::vulkan::VulkanDispatcher disp;
  cifar_dense::AppData appdata(disp.get_mr());

  // Run stage 4
  disp.run_stage_4(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool2_out.d0(), 128);
  EXPECT_EQ(appdata.u_pool2_out.d1(), 32);
  EXPECT_EQ(appdata.u_pool2_out.d2(), 8);
  EXPECT_EQ(appdata.u_pool2_out.d3(), 8);

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_4(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 5
// ----------------------------------------------------------------------------

TEST(Stage5Test, Basic) {
  cifar_dense::vulkan::VulkanDispatcher disp;
  cifar_dense::AppData appdata(disp.get_mr());

  // Run stage 5
  disp.run_stage_5(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv3_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv3_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv3_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv3_out.d3(), 8);

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_5(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 6
// ----------------------------------------------------------------------------

TEST(Stage6Test, Basic) {
  cifar_dense::vulkan::VulkanDispatcher disp;
  cifar_dense::AppData appdata(disp.get_mr());

  // Run stage 6
  disp.run_stage_6(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv4_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv4_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv4_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv4_out.d3(), 8);

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_6(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 7
// ----------------------------------------------------------------------------

TEST(Stage7Test, Basic) {
  cifar_dense::vulkan::VulkanDispatcher disp;
  cifar_dense::AppData appdata(disp.get_mr());

  // Run stage 7
  disp.run_stage_7(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv5_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv5_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv5_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv5_out.d3(), 8);

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_7(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 8
// ----------------------------------------------------------------------------

TEST(Stage8Test, Basic) {
  cifar_dense::vulkan::VulkanDispatcher disp;
  cifar_dense::AppData appdata(disp.get_mr());

  // Run stage 8
  disp.run_stage_8(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool3_out.d0(), 128);
  EXPECT_EQ(appdata.u_pool3_out.d1(), 64);
  EXPECT_EQ(appdata.u_pool3_out.d2(), 4);
  EXPECT_EQ(appdata.u_pool3_out.d3(), 4);

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_8(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 9
// ----------------------------------------------------------------------------

TEST(Stage9Test, Basic) {
  cifar_dense::vulkan::VulkanDispatcher disp;
  cifar_dense::AppData appdata(disp.get_mr());

  // Run stage 9
  disp.run_stage_9(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_linear_out.d0(), 128);
  EXPECT_EQ(appdata.u_linear_out.d1(), 10);

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_9(appdata));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
