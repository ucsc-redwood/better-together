#include <benchmark/benchmark.h>
#include <omp.h>

#include "builtin-apps/app.hpp"  // for 'g_big_cores', 'g_med_cores', 'g_lit_cores'
#include "const.hpp"

// ----------------------------------------------------------------------------
// Baseline: OMP
// ----------------------------------------------------------------------------

#define INIT_DATA_SINGLE \
  DispatcherT disp;      \
  AppDataT appdata(disp.get_mr());

static void BM_baseline_omp(benchmark::State& state) {
  INIT_DATA_SINGLE;

  // Warm up
  tree::omp::dispatch_multi_stage(appdata, 1, 7);

  // Benchmark
  for (auto _ : state) {
    tree::omp::dispatch_multi_stage(appdata, 1, 7);
  }
}

BENCHMARK(BM_baseline_omp)->Unit(benchmark::kMillisecond)->Name("OMP/Tree/Baseline");

// ----------------------------------------------------------------------------
// Baseline: OMP (Little Cores)
// ----------------------------------------------------------------------------

static void BM_baseline_omp_little_cores(benchmark::State& state) {
  INIT_DATA_SINGLE;

  if (!has_lit_cores()) {
    state.SkipWithMessage("No little cores");
    return;
  }

  // Warm up
  tree::omp::dispatch_multi_stage(LITTLE_CORES, appdata, 1, 7);

  // Benchmark
  for (auto _ : state) {
    tree::omp::dispatch_multi_stage(LITTLE_CORES, appdata, 1, 7);
  }
}

BENCHMARK(BM_baseline_omp_little_cores)
    ->Unit(benchmark::kMillisecond)
    ->Name("OMP/CIFAR-Sparse/Baseline/LittleCores");

// ----------------------------------------------------------------------------
// Baseline: OMP (Mid Cores)
// ----------------------------------------------------------------------------

static void BM_baseline_omp_mid_cores(benchmark::State& state) {
  INIT_DATA_SINGLE;

  if (!has_med_cores()) {
    state.SkipWithMessage("No mid cores");
    return;
  }

  // Warm up
  tree::omp::dispatch_multi_stage(MEDIUM_CORES, appdata, 1, 7);

  // Benchmark
  for (auto _ : state) {
    tree::omp::dispatch_multi_stage(MEDIUM_CORES, appdata, 1, 7);
  }
}

BENCHMARK(BM_baseline_omp_mid_cores)
    ->Unit(benchmark::kMillisecond)
    ->Name("OMP/CIFAR-Sparse/Baseline/MidCores");

// ----------------------------------------------------------------------------
// Baseline: OMP (Big Cores)
// ----------------------------------------------------------------------------

static void BM_baseline_omp_big_cores(benchmark::State& state) {
  INIT_DATA_SINGLE;

  if (!has_big_cores()) {
    state.SkipWithMessage("No big cores");
    return;
  }

  // Warm up
  tree::omp::dispatch_multi_stage(BIG_CORES, appdata, 1, 7);

  // Benchmark
  for (auto _ : state) {
    tree::omp::dispatch_multi_stage(BIG_CORES, appdata, 1, 7);
  }
}

BENCHMARK(BM_baseline_omp_big_cores)
    ->Unit(benchmark::kMillisecond)
    ->Name("OMP/CIFAR-Sparse/Baseline/BigCores");

// ----------------------------------------------------------------------------
// Baseline: Vulkan
// ----------------------------------------------------------------------------

static void BM_baseline_vk(benchmark::State& state) {
  INIT_DATA_SINGLE;

  // Warm up
  disp.dispatch_multi_stage(appdata, 1, 7);

  // Benchmark
  for (auto _ : state) {
    disp.dispatch_multi_stage(appdata, 1, 7);
  }
}

BENCHMARK(BM_baseline_vk)->Unit(benchmark::kMillisecond)->Name("VK/Tree/Baseline");

// ----------------------------------------------------------------------------
// Main
//
// Run with
// xmake r bm-baseline-tree-vk --device jetson
//
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup
  return 0;
}
