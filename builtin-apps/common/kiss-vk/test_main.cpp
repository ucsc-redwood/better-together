#include <gtest/gtest.h>
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

int main(int argc, char** argv) {
  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Set logging level to off
  spdlog::set_level(spdlog::level::off);

  // Run the tests
  return RUN_ALL_TESTS();
}