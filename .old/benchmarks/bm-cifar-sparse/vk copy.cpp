#include <benchmark/benchmark.h>

#include "../argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/sparse_appdata.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/resources_path.hpp"

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

static void run_baseline_pinned(cifar_sparse::AppData& app_data) {
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_1(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_2(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_3(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_4(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_5(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_6(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_7(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_8(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_9(app_data);
}

class VK_CifarSparse : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    auto mr = cifar_sparse::vulkan::Singleton::getInstance().get_mr();

    app_data = std::make_unique<cifar_sparse::AppData>(mr);

    // process the data once
    run_baseline_pinned(*app_data);
  }

  void TearDown(benchmark::State&) override { app_data.reset(); }

  std::unique_ptr<cifar_sparse::AppData> app_data;
};

BENCHMARK_DEFINE_F(VK_CifarSparse, Baseline)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_1(*app_data);
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_2(*app_data);
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_3(*app_data);
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_4(*app_data);
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_5(*app_data);
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_6(*app_data);
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_7(*app_data);
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_8(*app_data);
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_9(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Baseline)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage1)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_1(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage1)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage2)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_2(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage2)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage3)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_3(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage3)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage4)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_4(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage4)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage5)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_5(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage5)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage6)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_6(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage6)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage7)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_7(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage7)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 8
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage8)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_8(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage8)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 9
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage9)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_9(*app_data);
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