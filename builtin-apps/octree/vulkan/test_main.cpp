#include <gtest/gtest.h>
#include <libmorton/morton.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <memory_resource>

#include "../../app.hpp"
#include "../appdata.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// Test Vulkan Context Setup
// ----------------------------------------------------------------------------

class OctreeVulkanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up Vulkan context here if needed
    vk_dispatcher = std::make_unique<octree::vulkan::VulkanDispatcher>();
    mr = vk_dispatcher->get_mr();
  }

  void TearDown() override {
    // Clean up Vulkan context here if needed
    vk_dispatcher.reset();
  }

  std::unique_ptr<octree::vulkan::VulkanDispatcher> vk_dispatcher;
  std::pmr::memory_resource* mr;
};

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

TEST_F(OctreeVulkanTest, Stage1) {
  octree::AppData appdata(mr);

  // Run stage 1
  EXPECT_NO_THROW(vk_dispatcher->run_stage_1(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 2
// ----------------------------------------------------------------------------

TEST_F(OctreeVulkanTest, Stage2) {
  octree::AppData appdata(mr);

  // Run stage 2
  EXPECT_NO_THROW(vk_dispatcher->run_stage_2(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 3
// ----------------------------------------------------------------------------

TEST_F(OctreeVulkanTest, Stage3) {
  octree::AppData appdata(mr);

  // Run stage 3
  EXPECT_NO_THROW(vk_dispatcher->run_stage_3(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 4
// ----------------------------------------------------------------------------

TEST_F(OctreeVulkanTest, Stage4) {
  octree::AppData appdata(mr);

  // Run stage 4
  EXPECT_NO_THROW(vk_dispatcher->run_stage_4(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 5
// ----------------------------------------------------------------------------

TEST_F(OctreeVulkanTest, Stage5) {
  octree::AppData appdata(mr);

  // Run stage 5
  EXPECT_NO_THROW(vk_dispatcher->run_stage_5(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 6
// ----------------------------------------------------------------------------

TEST_F(OctreeVulkanTest, Stage6) {
  octree::AppData appdata(mr);

  // Run stage 6
  EXPECT_NO_THROW(vk_dispatcher->run_stage_6(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 7
// ----------------------------------------------------------------------------

TEST_F(OctreeVulkanTest, Stage7) {
  octree::AppData appdata(mr);

  // Run stage 7
  EXPECT_NO_THROW(vk_dispatcher->run_stage_7(appdata));
}

int main(int argc, char** argv) {
  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Parse command-line arguments
  parse_args(argc, argv);

  // Set logging level to off
  spdlog::set_level(spdlog::level::off);

  // Run the tests
  return RUN_ALL_TESTS();
}
