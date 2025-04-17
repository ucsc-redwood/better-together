#include <gtest/gtest.h>
#include <libmorton/morton.h>
#include <spdlog/spdlog.h>

#include <memory>

#include "../../app.hpp"
#include "dispatchers.hpp"

TEST(VulkanOctreeTest, Test1) {
  octree::vulkan::VulkanDispatcher disp;
  octree::AppData appdata(disp.get_mr());

  disp.dispatch_multi_stage(appdata, 1, 1);

  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Verify some basic expectations after running stage 1
  // Morton codes should be computed for all input points
  bool all_non_zero = true;
  for (size_t i = 0; i < appdata.n; ++i) {
    if (appdata.u_morton_codes_alt[i] == 0) {
      all_non_zero = false;
      break;
    }
  }

  EXPECT_TRUE(all_non_zero) << "Some morton codes were not computed";
}

TEST(VulkanOctreeTest, Test2) {
  octree::vulkan::VulkanDispatcher disp;
  octree::AppData appdata(disp.get_mr());

  // Run stage 1 first to generate morton codes
  disp.dispatch_multi_stage(appdata, 1, 1);

  // Then run stage 2 to sort them
  disp.dispatch_multi_stage(appdata, 2, 2);

  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Verify that the morton codes are sorted
  for (size_t i = 1; i < appdata.n; ++i) {
    EXPECT_LE(appdata.u_morton_codes[i - 1], appdata.u_morton_codes[i])
        << "Morton codes are not sorted at index " << i;
  }
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
