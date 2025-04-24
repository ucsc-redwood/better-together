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
  cifar_sparse::omp::dispatch_multi_stage(appdata, 1, 9);

  // Benchmark
  for (auto _ : state) {
    cifar_sparse::omp::dispatch_multi_stage(appdata, 1, 9);
  }
}

BENCHMARK(BM_baseline_omp)->Unit(benchmark::kMillisecond)->Name("OMP/CIFAR-Sparse/Baseline");

// ----------------------------------------------------------------------------
// Baseline: Vulkan
// ----------------------------------------------------------------------------

static void BM_baseline_vk(benchmark::State& state) {
  INIT_DATA_SINGLE;

  // Warm up
  disp.dispatch_multi_stage(appdata, 1, 9);

  // Benchmark
  for (auto _ : state) {
    disp.dispatch_multi_stage(appdata, 1, 9);
  }
}

BENCHMARK(BM_baseline_vk)->Unit(benchmark::kMillisecond)->Name("VK/CIFAR-Sparse/Baseline");

// ----------------------------------------------------------------------------
// Main
//
// Run with
// xmake r bm-baseline-cifar-sparse-vk --device jetson
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
