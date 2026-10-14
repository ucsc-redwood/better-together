#include <gtest/gtest.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include "engine.hpp"

TEST(VulkanEngineTest, EngineInitialization) {
  // Test that engine can be initialized
  EXPECT_NO_THROW({ kiss_vk::Engine engine; });
}

TEST(VulkanEngineTest, SequenceCreation) {
  kiss_vk::Engine engine;

  // Test that sequence can be created
  EXPECT_NO_THROW({ auto seq = engine.make_seq(); });
}

// Device-init spec, "Fail loudly on a missing required feature": the check names exactly
// the feature the device lacks. Starts from the real iGPU's supported set, so it skips
// (like the differential suites) on a box without one.
TEST(VulkanDeviceFeatures, MissingShaderInt8IsReported) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;

  vk::PhysicalDeviceVulkan12Features supported{};
  vk::PhysicalDeviceFeatures2 features2{.pNext = &supported};
  engine.get_physical_device().getFeatures2(&features2);
  EXPECT_TRUE(kiss_vk::missing_required_features(supported).empty());

  supported.shaderInt8 = VK_FALSE;
  EXPECT_EQ(kiss_vk::missing_required_features(supported), std::vector<std::string>{"shaderInt8"});
}

int main(int argc, char** argv) {
  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Set logging level to off (SPDLOG_LEVEL=debug overrides, e.g. to see the enabled layers)
  spdlog::set_level(spdlog::level::off);
  spdlog::cfg::load_env_levels();

  // Run the tests
  return RUN_ALL_TESTS();
}