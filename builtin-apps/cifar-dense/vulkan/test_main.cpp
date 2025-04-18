#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <memory_resource>

#include "../../app.hpp"
#include "../appdata.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// Test Vulkan Context Setup
// ----------------------------------------------------------------------------

class CIFARDenseVulkanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up Vulkan context here if needed
    vk_dispatcher = std::make_unique<cifar_dense::vulkan::VulkanDispatcher>();
    mr = vk_dispatcher->get_mr();
  }

  void TearDown() override {
    // Clean up Vulkan context here if needed
    vk_dispatcher.reset();
  }

  std::unique_ptr<cifar_dense::vulkan::VulkanDispatcher> vk_dispatcher;
  std::pmr::memory_resource* mr;
};

// ----------------------------------------------------------------------------
// test Stage 1: Conv1
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage1) {
  cifar_dense::AppData appdata(mr);

  // Run stage 1
  EXPECT_NO_THROW(vk_dispatcher->run_stage_1(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 2: MaxPool1
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage2) {
  cifar_dense::AppData appdata(mr);

  // Run stage 2
  EXPECT_NO_THROW(vk_dispatcher->run_stage_2(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 3: Conv2
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage3) {
  cifar_dense::AppData appdata(mr);

  // Run stage 3
  EXPECT_NO_THROW(vk_dispatcher->run_stage_3(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 4: MaxPool2
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage4) {
  cifar_dense::AppData appdata(mr);

  // Run stage 4
  EXPECT_NO_THROW(vk_dispatcher->run_stage_4(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 5: Conv3
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage5) {
  cifar_dense::AppData appdata(mr);

  // Run stage 5
  EXPECT_NO_THROW(vk_dispatcher->run_stage_5(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 6: Conv4
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage6) {
  cifar_dense::AppData appdata(mr);

  // Run stage 6
  EXPECT_NO_THROW(vk_dispatcher->run_stage_6(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 7: Conv5
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage7) {
  cifar_dense::AppData appdata(mr);

  // Run stage 7
  EXPECT_NO_THROW(vk_dispatcher->run_stage_7(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 8: MaxPool3
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage8) {
  cifar_dense::AppData appdata(mr);

  // Run stage 8
  EXPECT_NO_THROW(vk_dispatcher->run_stage_8(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 9: Linear
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage9) {
  cifar_dense::AppData appdata(mr);

  // Run stage 9
  EXPECT_NO_THROW(vk_dispatcher->run_stage_9(appdata));
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
