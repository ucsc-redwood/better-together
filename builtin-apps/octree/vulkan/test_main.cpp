#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <memory>

#include "../../app.hpp"
#include "dispatchers.hpp"

#include <libmorton/morton.h>

TEST(VulkanOctreeTest, Test1) {
  octree::vulkan::VulkanDispatcher disp;

  auto appdata = std::make_unique<octree::AppData>(disp.get_mr());

  EXPECT_NO_THROW(disp.run_stage_1(*appdata));

  // check if morton codes are non-zero
  for (size_t i = 0; i < appdata->n; ++i) {
    EXPECT_NE(appdata->u_morton_codes[i], 0) << "Morton code at index " << i << " is zero";
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
