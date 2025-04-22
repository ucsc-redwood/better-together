#include <benchmark/benchmark.h>
#include <omp.h>

#include "builtin-apps/app.hpp"  // for 'g_big_cores', 'g_med_cores', 'g_lit_cores'
#include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
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
  AppDataT appdata(&disp.get_mr());

  // Warm up
  cifar_dense::omp::dispatch_multi_stage(appdata, 1, 9);

  // Benchmark
  for (auto _ : state) {
    cifar_dense::omp::dispatch_multi_stage(appdata, 1, 9);
  }
}

BENCHMARK(BM_baseline_omp)->DenseRange(1, 8)->Unit(benchmark::kMillisecond)->Name("OMP/Baseline");

// ----------------------------------------------------------------------------
// Baseline: CUDA
// ----------------------------------------------------------------------------

static void BM_baseline_cuda(benchmark::State& state) {
  DispatcherT disp;
  AppDataT appdata(&disp.get_mr());

  // Warm up
  disp.dispatch_multi_stage(appdata, 1, 9);

  // Benchmark
  for (auto _ : state) {
    disp.dispatch_multi_stage(appdata, 1, 9);
  }
}

BENCHMARK(BM_baseline_cuda)->Unit(benchmark::kMillisecond)->Name("CUDA/Baseline");

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
