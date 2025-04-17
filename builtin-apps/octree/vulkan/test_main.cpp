#include <gtest/gtest.h>
#include <libmorton/morton.h>
#include <spdlog/spdlog.h>

#include <memory>

#include "../../app.hpp"
#include "dispatchers.hpp"

TEST(VulkanOctreeTest, Test1) {
  octree::vulkan::VulkanDispatcher disp;
  octree::AppData appdata(disp.get_mr());

  EXPECT_NO_THROW(disp.run_stage_1(appdata));
}

TEST(VulkanOctreeTest, Test2) {
//   octree::vulkan::VulkanDispatcher disp;
//   octree::AppData appdata(disp.get_mr());

//   disp.run_stage_1(appdata);
//   EXPECT_NO_THROW(disp.run_stage_2(appdata));

//   bool is_sorted = true;
//   for (size_t i = 1; i < appdata.n; ++i) {
//     if (appdata.u_morton_codes_sorted[i] < appdata.u_morton_codes_sorted[i - 1]) {
//       is_sorted = false;
//       break;
//     }
//   }
//   EXPECT_TRUE(is_sorted);
}

int main(int argc, char** argv) {
  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Parse command-line arguments
  parse_args(argc, argv);

  // Set logging level
  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  // Run the tests
  return RUN_ALL_TESTS();
}
