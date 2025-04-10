#include <benchmark/benchmark.h>
#include <omp.h>

#include "../sparse_appdata.hpp"
#include "builtin-apps/app.hpp"
#include "dispatchers.hpp"

static void BM_run_OMP_baseline(benchmark::State& state) {
  const int num_threads = state.range(0);

  auto mr = std::pmr::new_delete_resource();
  cifar_sparse::v2::AppData appdata(mr);

  // Warm up the data structures to avoid first-run anomalies
  cifar_sparse::omp::v2::dispatch_multi_stage_unrestricted(num_threads, appdata, 1, 9);

  for (auto _ : state) {
#pragma omp parallel num_threads(num_threads)
    {
      // print thread id
      cifar_sparse::omp::v2::run_stage_1(appdata);
      cifar_sparse::omp::v2::run_stage_2(appdata);
      cifar_sparse::omp::v2::run_stage_3(appdata);
      cifar_sparse::omp::v2::run_stage_4(appdata);
      cifar_sparse::omp::v2::run_stage_5(appdata);
      cifar_sparse::omp::v2::run_stage_6(appdata);
      cifar_sparse::omp::v2::run_stage_7(appdata);
      cifar_sparse::omp::v2::run_stage_8(appdata);
      cifar_sparse::omp::v2::run_stage_9(appdata);
    }
  }
}

BENCHMARK(BM_run_OMP_baseline)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

static void BM_run_OMP_stage(benchmark::State& state) {
  const int stage = state.range(0);
  const int num_threads = state.range(1);
  const ProcessorType core_type = static_cast<ProcessorType>(state.range(2));

  auto mr = std::pmr::new_delete_resource();
  cifar_sparse::v2::AppData appdata(mr);

  std::vector<int> cores_to_use;
  if (core_type == ProcessorType::kLittleCore) {
    cores_to_use = g_little_cores;
  } else if (core_type == ProcessorType::kMediumCore) {
    cores_to_use = g_medium_cores;
  } else {
    cores_to_use = g_big_cores;
  }

  if (num_threads > (static_cast<int>(cores_to_use.size()))) {
    state.SkipWithMessage("");
    return;
  }

  // Ensure stage is valid
  if (stage < 1 || stage > 9) {
    state.SkipWithError("Invalid stage number");
    return;
  }

  // Warm up to prevent first-run anomalies
  cifar_sparse::omp::v2::dispatch_multi_stage(cores_to_use, num_threads, appdata, 1, stage);

  for (auto _ : state) {
    cifar_sparse::omp::v2::dispatch_multi_stage(cores_to_use, num_threads, appdata, stage, stage);
  }
}

static void CustomArgs(benchmark::internal::Benchmark* b) {
  for (int stage = 1; stage <= 9; ++stage) {
    for (int num_threads = 1; num_threads <= 8; ++num_threads) {
      for (const ProcessorType core_type :
           {ProcessorType::kLittleCore, ProcessorType::kMediumCore, ProcessorType::kBigCore}) {
        b->Args({stage, num_threads, static_cast<int>(core_type)});
      }
    }
  }
}

BENCHMARK(BM_run_OMP_stage)->Apply(CustomArgs)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  parse_args(argc, argv);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup
  return 0;
}
