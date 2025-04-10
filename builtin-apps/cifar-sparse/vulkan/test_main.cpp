#include <gtest/gtest.h>

#include "../sparse_appdata.hpp"
#include "builtin-apps/app.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// test appdata initialization
// ----------------------------------------------------------------------------

TEST(AppdataTest, Initialization) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;
  cifar_sparse::v2::AppData appdata(disp.get_mr());

  EXPECT_EQ(appdata.conv1_sparse.rows, 16);
  EXPECT_EQ(appdata.conv1_sparse.cols, 27);
}

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

TEST(Stage1Test, Basic) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;
  cifar_sparse::v2::AppData appdata(disp.get_mr());

  // Run stage 1
  disp.run_stage_1(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv1_out.d0(), 512);
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
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;
  cifar_sparse::v2::AppData appdata(disp.get_mr());

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

int main(int argc, char **argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
