#include <benchmark/benchmark.h>

#include "dispatchers.hpp"

constexpr auto kDefaultInputFile = "cifar10_images/img_00005.npy";

static void BM_Stage_1(benchmark::State& state) {
  const auto num_threads = state.range(0);

  AppData appdata(kDefaultInputFile);
  for (auto _ : state) {
    omp::dispatch_single_stage(appdata, 1, num_threads);
  }
}

BENCHMARK(BM_Stage_1)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();