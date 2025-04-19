#include <benchmark/benchmark.h>
#include <omp.h>

#include "builtin-apps/app.hpp"  // for 'g_big_cores', 'g_med_cores', 'g_lit_cores'
#include "const.hpp"

// ----------------------------------------------------------------------------
// Baseline: OMP
// ----------------------------------------------------------------------------

static void BM_baseline_omp(benchmark::State& state) {
  int n_threads = state.range(0);

  if ((size_t)n_threads > std::thread::hardware_concurrency()) {
    state.SkipWithMessage("n_threads is greater than the number of hardware threads");
    return;
  }

  DispatcherT disp;
  AppDataT appdata(disp.get_mr());

  for (auto _ : state) {
#pragma omp parallel num_threads(n_threads)
    {
      tree::omp::run_stage_1(appdata);
      tree::omp::run_stage_2(appdata);
      tree::omp::run_stage_3(appdata);
      tree::omp::run_stage_4(appdata);
      tree::omp::run_stage_5(appdata);
      tree::omp::run_stage_6(appdata);
      tree::omp::run_stage_7(appdata);
    }
  }
}

BENCHMARK(BM_baseline_omp)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------------------
// Baseline: Vulkan
// ----------------------------------------------------------------------------

static void BM_baseline_vulkan(benchmark::State& state) {
  DispatcherT disp;
  AppDataT appdata(disp.get_mr());

  // Warm up
  disp.run_stage_1(appdata);
  disp.run_stage_2(appdata);
  disp.run_stage_3(appdata);
  disp.run_stage_4(appdata);
  disp.run_stage_5(appdata);
  disp.run_stage_6(appdata);
  disp.run_stage_7(appdata);

  // Benchmark
  for (auto _ : state) {
    disp.run_stage_1(appdata);
    disp.run_stage_2(appdata);
    disp.run_stage_3(appdata);
    disp.run_stage_4(appdata);
    disp.run_stage_5(appdata);
    disp.run_stage_6(appdata);
    disp.run_stage_7(appdata);
  }
}

BENCHMARK(BM_baseline_vulkan)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

// e.g.,
// xmake r bm-baseline-tree-vk

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup
  return 0;
}
