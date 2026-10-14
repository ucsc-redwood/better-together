#include <gtest/gtest.h>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/spdlog.h>

#include <set>

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

// Pipeline-config (design D5): the SPIR-V scan reads spec ids and workgroup size from
// committed *_spv.h. Hardware-free.
TEST(SpirvReflection, CommittedShaders) {
  const auto radix = kiss_vk::reflect_spirv(kiss_vk::load_shader_spirv("multi_radixsort"));
  EXPECT_EQ(radix.spec_ids, std::set<uint32_t>{kiss_vk::kSubgroupSizeSpecId});
  EXPECT_EQ(radix.local_size, (std::array<uint32_t, 3>{256, 1, 1}));
  for (const auto& id : radix.local_size_spec_id) EXPECT_FALSE(id.has_value());

  const auto edge = kiss_vk::reflect_spirv(kiss_vk::load_shader_spirv("tree_edge_count"));
  EXPECT_TRUE(edge.spec_ids.empty());
  EXPECT_EQ(edge.local_size, (std::array<uint32_t, 3>{512, 1, 1}));

  EXPECT_THROW((void)kiss_vk::reflect_spirv(std::vector<uint32_t>{1, 2, 3, 4, 5}),
               std::runtime_error);
}

// Pipeline-config spec, "Registry subgroup value is a cross-check only": a registry
// subgroup size that disagrees with the device-derived one warns, naming both values.
// Hardware-free: minipc's devices/minipc.json declares 64.
TEST(RegistrySubgroupCrossCheck, MismatchWarns) {
  auto sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(8);
  const auto logger = spdlog::default_logger();
  const auto old_level = logger->level();
  logger->sinks().push_back(sink);
  logger->set_level(spdlog::level::warn);
  const std::string old_device = g_device_id;
  g_device_id = "minipc";

  const bool mismatch_ok = check_vulkan_subgroup_size(16);
  const auto mismatch_logs = sink->last_formatted();
  const bool match_ok = check_vulkan_subgroup_size(64);
  const auto all_logs = sink->last_formatted();

  g_device_id = old_device;
  logger->set_level(old_level);
  logger->sinks().pop_back();

  EXPECT_FALSE(mismatch_ok);
  ASSERT_EQ(mismatch_logs.size(), 1u);
  EXPECT_NE(mismatch_logs[0].find("[warning]"), std::string::npos) << mismatch_logs[0];
  EXPECT_NE(mismatch_logs[0].find("gpu.subgroup_size 64"), std::string::npos) << mismatch_logs[0];
  EXPECT_NE(mismatch_logs[0].find("reports 16"), std::string::npos) << mismatch_logs[0];
  EXPECT_TRUE(match_ok);
  EXPECT_EQ(all_logs.size(), 1u);  // the match logged nothing
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