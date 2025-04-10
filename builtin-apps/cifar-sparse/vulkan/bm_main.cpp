#include <benchmark/benchmark.h>
#include <omp.h>

#include "../sparse_appdata.hpp"
#include "builtin-apps/app.hpp"
#include "dispatchers.hpp"

static void BM_run_OMP_baseline(benchmark::State& state) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;
  cifar_sparse::v2::AppData appdata(disp.get_mr());

  // Warm up the data structures to avoid first-run anomalies
  disp.run_stage_1(appdata);
  disp.run_stage_2(appdata);
  disp.run_stage_3(appdata);
  disp.run_stage_4(appdata);
  disp.run_stage_5(appdata);
  disp.run_stage_6(appdata);
  disp.run_stage_7(appdata);
  disp.run_stage_8(appdata);
  disp.run_stage_9(appdata);

  for (auto _ : state) {
    // print thread id
    disp.run_stage_1(appdata);
    disp.run_stage_2(appdata);
    disp.run_stage_3(appdata);
    disp.run_stage_4(appdata);
    disp.run_stage_5(appdata);
    disp.run_stage_6(appdata);
    disp.run_stage_7(appdata);
    disp.run_stage_8(appdata);
    disp.run_stage_9(appdata);
  }
}

BENCHMARK(BM_run_OMP_baseline)->Unit(benchmark::kMillisecond);

static void BM_run_OMP_stage(benchmark::State& state) {
  const int stage = state.range(0);

  cifar_sparse::vulkan::v2::VulkanDispatcher disp;
  cifar_sparse::v2::AppData appdata(disp.get_mr());

  // Ensure stage is valid
  if (stage < 1 || stage > 9) {
    state.SkipWithError("Invalid stage number");
    return;
  }

  // Warm up to prevent first-run anomalies
  disp.dispatch_multi_stage(appdata, 1, stage);

  for (auto _ : state) {
    disp.dispatch_multi_stage(appdata, stage, stage);
  }
}

BENCHMARK(BM_run_OMP_stage)->DenseRange(1, 9)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup
  return 0;
}
