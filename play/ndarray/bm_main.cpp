#include <benchmark/benchmark.h>

#include "builtin-apps/app.hpp"
#include "dispatchers.hpp"

constexpr auto kDefaultInputFile = "cifar10_images/img_00005.npy";

// static void BM_Stage_1(benchmark::State& state) {
//   const auto num_threads = state.range(0);

//   cifar_dense::AppData appdata(kDefaultInputFile);
//   for (auto _ : state) {
//     omp::dispatch_single_stage(appdata, 1, num_threads);
//   }
// }

// BENCHMARK(BM_Stage_1)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

static void BM_Stage_1(benchmark::State& state) {
  const auto num_threads = state.range(0);

  cifar_dense::AppDataBatch appdata(kDefaultInputFile);
  for (auto _ : state) {
    omp::dispatch_multi_stage<ProcessorType::kLittleCore>(num_threads, appdata, 1, 1);
  }
}

BENCHMARK(BM_Stage_1)->DenseRange(1, 6)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  parse_args(argc, argv);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
