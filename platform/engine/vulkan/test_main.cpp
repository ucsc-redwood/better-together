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

  vk::PhysicalDeviceVulkan13Features supported_13{};
  vk::PhysicalDeviceVulkan12Features supported{.pNext = &supported_13};
  vk::PhysicalDeviceFeatures2 features2{.pNext = &supported};
  engine.get_physical_device().getFeatures2(&features2);
  EXPECT_TRUE(kiss_vk::missing_required_features(supported, supported_13).empty());

  supported.shaderInt8 = VK_FALSE;
  EXPECT_EQ(kiss_vk::missing_required_features(supported, supported_13),
            std::vector<std::string>{"shaderInt8"});

  // maintenance4: required for LocalSizeId (VUID-RuntimeSpirv-LocalSizeId-06434).
  supported.shaderInt8 = VK_TRUE;
  supported_13.maintenance4 = VK_FALSE;
  EXPECT_EQ(kiss_vk::missing_required_features(supported, supported_13),
            std::vector<std::string>{"maintenance4"});
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

// SPIR-V 1.6 `local_size_*_id`: OpExecutionModeId LocalSizeId over spec constants.
TEST(SpirvReflection, LocalSizeId) {
  const auto r = kiss_vk::reflect_spirv(kiss_vk::load_shader_spirv("engine_test_workgroup_size"));
  EXPECT_EQ(r.spec_ids, (std::set<uint32_t>{100, 101, 102}));
  EXPECT_EQ(r.local_size, (std::array<uint32_t, 3>{1, 1, 1}));  // the spec constants' defaults
  EXPECT_EQ(r.local_size_spec_id[0], 100u);
  EXPECT_EQ(r.local_size_spec_id[1], 101u);
  EXPECT_EQ(r.local_size_spec_id[2], 102u);
}

// Pipeline-config spec, "Declared workgroup size is applied to the shader".
namespace {
std::string build_error(const std::shared_ptr<kiss_vk::Algorithm>& algo) {
  try {
    (void)algo->build();
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}
std::string dims(const auto& s) { return fmt::format("({}, {}, {})", s[0], s[1], s[2]); }
}  // namespace

TEST(VulkanWorkGroupSize, DeclaredSizeReachesShader) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;
  auto seq = engine.make_seq();
  std::pmr::vector<uint32_t> out(4, 0, engine.get_mr());

  auto algo = engine.make_algo("engine_test_workgroup_size")
                  ->work_group_size(64, 2, 1)
                  ->num_buffers(1)
                  ->build();
  algo->update_descriptor_set(0, {engine.get_buffer_info(out)});

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_dispatch(seq->get_handle(), {1, 1, 1});
  seq->cmd_end();
  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();

  EXPECT_EQ(out[0], 128u);  // invocations in the one workgroup
  EXPECT_EQ(out[1], 64u);
  EXPECT_EQ(out[2], 2u);
  EXPECT_EQ(out[3], 1u);
}

TEST(VulkanWorkGroupSize, MismatchWithLiteralLocalSizeThrows) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;
  // tree_morton.comp: layout(local_size_x = 256) in;
  const auto err =
      build_error(engine.make_algo("tree_morton")->work_group_size(128, 1, 1)->num_buffers(2));
  EXPECT_NE(err.find("declared workgroup size (128, 1, 1)"), std::string::npos) << err;
  EXPECT_NE(err.find("LocalSize is (256, 1, 1)"), std::string::npos) << err;
}

TEST(VulkanWorkGroupSize, ExceedsDeviceLimitsThrows) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;
  const auto& limits = engine.compute_limits();
  const auto& max_wg = limits.max_work_group_size;

  const std::array<uint32_t, 3> too_wide{max_wg[0] + 1, 1, 1};
  auto err = build_error(engine.make_algo("engine_test_workgroup_size")
                             ->work_group_size(too_wide[0], too_wide[1], too_wide[2])
                             ->num_buffers(1));
  EXPECT_NE(err.find("workgroup size " + dims(too_wide)), std::string::npos) << err;
  EXPECT_NE(err.find("maxComputeWorkGroupSize " + dims(max_wg)), std::string::npos) << err;

  // Every dimension at its own limit: within maxComputeWorkGroupSize, but far over the
  // invocation limit on every fleet GPU.
  ASSERT_GT(uint64_t{max_wg[0]} * max_wg[1] * max_wg[2], limits.max_work_group_invocations);
  err = build_error(engine.make_algo("engine_test_workgroup_size")
                        ->work_group_size(max_wg[0], max_wg[1], max_wg[2])
                        ->num_buffers(1));
  EXPECT_NE(err.find("workgroup size " + dims(max_wg)), std::string::npos) << err;
  EXPECT_NE(
      err.find(fmt::format("maxComputeWorkGroupInvocations {}", limits.max_work_group_invocations)),
      std::string::npos)
      << err;
}

// A shader that declares constant_id = 0 (subgroup size) must be built
// subgroup_size_dependent(), and only such a shader may be: never run on the default.
TEST(VulkanSubgroupSize, DependentPipelinesMatchTheShader) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;
  auto err = build_error(engine.make_algo("multi_radixsort")
                             ->work_group_size(256, 1, 1)
                             ->num_buffers(3)
                             ->push_constant_size(16));
  EXPECT_NE(err.find("is not subgroup_size_dependent()"), std::string::npos) << err;

  err = build_error(engine.make_algo("tree_morton")
                        ->work_group_size(256, 1, 1)
                        ->subgroup_size_dependent()
                        ->num_buffers(2));
  EXPECT_NE(err.find("has no constant_id = 0"), std::string::npos) << err;

  EXPECT_NO_THROW((void)engine.make_algo("multi_radixsort")
                      ->work_group_size(256, 1, 1)
                      ->subgroup_size_dependent()
                      ->num_buffers(3)
                      ->push_constant_size(16)
                      ->build());
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
  EXPECT_NE(mismatch_logs[0].find("engine uses 16"), std::string::npos) << mismatch_logs[0];
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