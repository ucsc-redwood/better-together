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

static void BM_Stage_2(benchmark::State& state) {
  const auto num_threads = state.range(0);

  AppData appdata(kDefaultInputFile);
  for (auto _ : state) {
    omp::dispatch_single_stage(appdata, 2, num_threads);
  }
}

BENCHMARK(BM_Stage_2)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

static void BM_Stage_3(benchmark::State& state) {
  const auto num_threads = state.range(0);

  AppData appdata(kDefaultInputFile);
  for (auto _ : state) {
    omp::dispatch_single_stage(appdata, 3, num_threads);
  }
}

BENCHMARK(BM_Stage_3)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

static void BM_Stage_4(benchmark::State& state) {
  const auto num_threads = state.range(0);

  AppData appdata(kDefaultInputFile);
  for (auto _ : state) {
    omp::dispatch_single_stage(appdata, 4, num_threads);
  }
}

BENCHMARK(BM_Stage_4)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

static void BM_Stage_5(benchmark::State& state) {
  const auto num_threads = state.range(0);

  AppData appdata(kDefaultInputFile);
  for (auto _ : state) {
    omp::dispatch_single_stage(appdata, 5, num_threads);
  }
}

BENCHMARK(BM_Stage_5)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

static void BM_Stage_6(benchmark::State& state) {
  const auto num_threads = state.range(0);

  AppData appdata(kDefaultInputFile);
  for (auto _ : state) {
    omp::dispatch_single_stage(appdata, 6, num_threads);
  }
}

BENCHMARK(BM_Stage_6)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

static void BM_Stage_7(benchmark::State& state) {
  const auto num_threads = state.range(0);

  AppData appdata(kDefaultInputFile);
  for (auto _ : state) {
    omp::dispatch_single_stage(appdata, 7, num_threads);
  }
}

BENCHMARK(BM_Stage_7)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

static void BM_Stage_8(benchmark::State& state) {
  const auto num_threads = state.range(0);

  AppData appdata(kDefaultInputFile);
  for (auto _ : state) {
    omp::dispatch_single_stage(appdata, 8, num_threads);
  }
}

BENCHMARK(BM_Stage_8)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

static void BM_Stage_9(benchmark::State& state) {
  const auto num_threads = state.range(0);

  AppData appdata(kDefaultInputFile);
  for (auto _ : state) {
    omp::dispatch_single_stage(appdata, 9, num_threads);
  }
}

BENCHMARK(BM_Stage_9)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();