#include <benchmark/benchmark.h>
#include <omp.h>

#include "builtin-apps/app.hpp"  // for 'g_big_cores', 'g_med_cores', 'g_lit_cores'
#include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
#include "const.hpp"

// ----------------------------------------------------------------------------
// Baseline: OMP
// ----------------------------------------------------------------------------

#define INIT_DATA_SINGLE \
  DispatcherT disp;      \
  AppDataT appdata(&disp.get_mr());

static void BM_baseline_omp(benchmark::State& state) {
  INIT_DATA_SINGLE;

  // Warm up
  cifar_dense::omp::dispatch_multi_stage(appdata, 1, 9);

  // Benchmark
  for (auto _ : state) {
    cifar_dense::omp::dispatch_multi_stage(appdata, 1, 9);
  }
}

BENCHMARK(BM_baseline_omp)->Unit(benchmark::kMillisecond)->Name("OMP/Baseline");

static void BM_baseline_omp_little(benchmark::State& state) {
  if (g_lit_cores.empty()) {
    state.SkipWithMessage("No little cores available");
    return;
  }

  INIT_DATA_SINGLE;

  // Warm up
  cifar_dense::omp::dispatch_multi_stage(appdata, 1, 9);

  // Benchmark
  for (auto _ : state) {
    cifar_dense::omp::dispatch_multi_stage(LITTLE_CORES, appdata, 1, 9);
  }
}

BENCHMARK(BM_baseline_omp_little)->Unit(benchmark::kMillisecond)->Name("OMP/Baseline/Little");

static void BM_baseline_omp_medium(benchmark::State& state) {
  if (g_med_cores.empty()) {
    state.SkipWithMessage("No medium cores available");
    return;
  }

  INIT_DATA_SINGLE;

  // Warm up
  cifar_dense::omp::dispatch_multi_stage(appdata, 1, 9);

  // Benchmark
  for (auto _ : state) {
    cifar_dense::omp::dispatch_multi_stage(MEDIUM_CORES, appdata, 1, 9);
  }
}

BENCHMARK(BM_baseline_omp_medium)->Unit(benchmark::kMillisecond)->Name("OMP/Baseline/Medium");

static void BM_baseline_omp_big(benchmark::State& state) {
  if (g_big_cores.empty()) {
    state.SkipWithMessage("No big cores available");
    return;
  }

  INIT_DATA_SINGLE;

  // Warm up
  cifar_dense::omp::dispatch_multi_stage(appdata, 1, 9);

  // Benchmark
  for (auto _ : state) {
    cifar_dense::omp::dispatch_multi_stage(BIG_CORES, appdata, 1, 9);
  }
}

BENCHMARK(BM_baseline_omp_big)->Unit(benchmark::kMillisecond)->Name("OMP/Baseline/Big");

// ----------------------------------------------------------------------------
// Baseline: CUDA
// ----------------------------------------------------------------------------

static void BM_baseline_cuda(benchmark::State& state) {
  INIT_DATA_SINGLE;

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
