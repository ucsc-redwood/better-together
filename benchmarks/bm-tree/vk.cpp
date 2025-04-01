#include <benchmark/benchmark.h>

#include "../argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/resources_path.hpp"
#include "builtin-apps/tree/vulkan/dispatchers.hpp"

#define PREPARE_DATA                                 \
  auto& vk = tree::vulkan::Singleton::getInstance(); \
  auto mr = vk.get_mr();                             \
  auto app_data = std::make_unique<tree::vulkan::VkAppData_Safe>(mr);

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

class VK_Tree : public benchmark::Fixture {};

BENCHMARK_DEFINE_F(VK_Tree, Baseline)
(benchmark::State& state) {
  PREPARE_DATA;

  for (auto _ : state) {
    vk.process_stage_1(*app_data);
    vk.process_stage_2(*app_data);
    vk.process_stage_3(*app_data);
    vk.process_stage_4(*app_data);
    vk.process_stage_5(*app_data);
    vk.process_stage_6(*app_data);
    vk.process_stage_7(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Baseline)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage1)
(benchmark::State& state) {
  PREPARE_DATA;

  for (auto _ : state) {
    vk.process_stage_1(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage1)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage2)
(benchmark::State& state) {
  PREPARE_DATA;

  vk.process_stage_1(*app_data);

  for (auto _ : state) {
    vk.process_stage_2(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage2)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage3)
(benchmark::State& state) {
  PREPARE_DATA;

  vk.process_stage_1(*app_data);
  vk.process_stage_2(*app_data);

  for (auto _ : state) {
    vk.process_stage_3(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage3)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage4)
(benchmark::State& state) {
  PREPARE_DATA;

  vk.process_stage_1(*app_data);
  vk.process_stage_2(*app_data);
  vk.process_stage_3(*app_data);

  for (auto _ : state) {
    vk.process_stage_4(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage4)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage5)
(benchmark::State& state) {
  PREPARE_DATA;

  vk.process_stage_1(*app_data);
  vk.process_stage_2(*app_data);
  vk.process_stage_3(*app_data);
  vk.process_stage_4(*app_data);

  for (auto _ : state) {
    vk.process_stage_5(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage5)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage6)
(benchmark::State& state) {
  PREPARE_DATA;

  vk.process_stage_1(*app_data);
  vk.process_stage_2(*app_data);
  vk.process_stage_3(*app_data);
  vk.process_stage_4(*app_data);
  vk.process_stage_5(*app_data);

  for (auto _ : state) {
    vk.process_stage_6(*app_data);
  }
}

// will cause a crash
BENCHMARK_REGISTER_F(VK_Tree, Stage6)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage7)
(benchmark::State& state) {
  PREPARE_DATA;

  vk.process_stage_1(*app_data);
  vk.process_stage_2(*app_data);
  vk.process_stage_3(*app_data);
  vk.process_stage_4(*app_data);
  vk.process_stage_5(*app_data);
  vk.process_stage_6(*app_data);

  for (auto _ : state) {
    vk.process_stage_7(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage7)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::off);

  const auto storage_location = helpers::get_benchmark_storage_location();
  const auto out_name = storage_location.string() + "/BM_Tree_VK_" + g_device_id + ".json";

  auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv, out_name);

  benchmark::Initialize(&new_argc, new_argv.data());
  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}