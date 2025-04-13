#include <benchmark/benchmark.h>

#include "../argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/resources_path.hpp"

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

class VK_CifarSparse : public benchmark::Fixture {};

BENCHMARK_DEFINE_F(VK_CifarSparse, Baseline)
(benchmark::State& state) {
  cifar_sparse::vulkan::VulkanDispatcher disp;
  cifar_sparse::AppData app_data(disp.get_mr());

  for (auto _ : state) {
    disp.run_stage_1(app_data);
    disp.run_stage_2(app_data);
    disp.run_stage_3(app_data);
    disp.run_stage_4(app_data);
    disp.run_stage_5(app_data);
    disp.run_stage_6(app_data);
    disp.run_stage_7(app_data);
    disp.run_stage_8(app_data);
    disp.run_stage_9(app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Baseline)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage1)
(benchmark::State& state) {
  cifar_sparse::vulkan::VulkanDispatcher disp;
  cifar_sparse::AppData app_data(disp.get_mr());

  disp.dispatch_multi_stage(app_data, 1, 1);

  for (auto _ : state) {
    disp.run_stage_1(app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage1)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage2)
(benchmark::State& state) {
  cifar_sparse::vulkan::VulkanDispatcher disp;
  cifar_sparse::AppData app_data(disp.get_mr());

  disp.dispatch_multi_stage(app_data, 1, 2);

  for (auto _ : state) {
    disp.run_stage_2(app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage2)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage3)
(benchmark::State& state) {
  cifar_sparse::vulkan::VulkanDispatcher disp;
  cifar_sparse::AppData app_data(disp.get_mr());

  disp.dispatch_multi_stage(app_data, 1, 3);

  for (auto _ : state) {
    disp.run_stage_3(app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage3)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage4)
(benchmark::State& state) {
  cifar_sparse::vulkan::VulkanDispatcher disp;
  cifar_sparse::AppData app_data(disp.get_mr());

  disp.dispatch_multi_stage(app_data, 1, 4);

  for (auto _ : state) {
    disp.run_stage_4(app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage4)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage5)
(benchmark::State& state) {
  cifar_sparse::vulkan::VulkanDispatcher disp;
  cifar_sparse::AppData app_data(disp.get_mr());

  disp.dispatch_multi_stage(app_data, 1, 5);

  for (auto _ : state) {
    disp.run_stage_5(app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage5)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage6)
(benchmark::State& state) {
  cifar_sparse::vulkan::VulkanDispatcher disp;
  cifar_sparse::AppData app_data(disp.get_mr());

  disp.dispatch_multi_stage(app_data, 1, 6);

  for (auto _ : state) {
    disp.run_stage_6(app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage6)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage7)
(benchmark::State& state) {
  cifar_sparse::vulkan::VulkanDispatcher disp;
  cifar_sparse::AppData app_data(disp.get_mr());

  disp.dispatch_multi_stage(app_data, 1, 7);

  for (auto _ : state) {
    disp.run_stage_7(app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage7)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 8
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage8)
(benchmark::State& state) {
  cifar_sparse::vulkan::VulkanDispatcher disp;
  cifar_sparse::AppData app_data(disp.get_mr());

  disp.dispatch_multi_stage(app_data, 1, 8);

  for (auto _ : state) {
    disp.run_stage_8(app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage8)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 9
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage9)
(benchmark::State& state) {
  cifar_sparse::vulkan::VulkanDispatcher disp;
  cifar_sparse::AppData app_data(disp.get_mr());

  disp.dispatch_multi_stage(app_data, 1, 9);

  for (auto _ : state) {
    disp.run_stage_9(app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage9)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------
int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::off);

  const auto storage_location = helpers::get_benchmark_storage_location();
  const auto out_name = storage_location.string() + "/BM_CifarSparse_VK_" + g_device_id + ".json";

  auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv, out_name);

  benchmark::Initialize(&new_argc, new_argv.data());
  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}