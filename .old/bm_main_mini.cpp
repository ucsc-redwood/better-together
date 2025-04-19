#include <benchmark/benchmark.h>
#include <omp.h>

#include "../omp/dispatchers.hpp"
#include "../sparse_appdata.hpp"
#include "builtin-apps/app.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// Mini Benchmark: Vk Baseline
// ----------------------------------------------------------------------------

static void BM_run_VK_baseline(benchmark::State& state) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;
  cifar_sparse::v2::AppData appdata(disp.get_mr());

  // Warm up the data structures to avoid first-run anomalies
  disp.dispatch_multi_stage(appdata, 1, 9);

  for (auto _ : state) {
    disp.dispatch_multi_stage(appdata, 1, 9);
  }
}

BENCHMARK(BM_run_VK_baseline)->Unit(benchmark::kMillisecond)->Name("VK/CifarSparse/Mini/Baseline");

// ----------------------------------------------------------------------------
// Mini Benchmark: Vk Stage
// ----------------------------------------------------------------------------

static void BM_run_VK_stage(benchmark::State& state) {
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

BENCHMARK(BM_run_VK_stage)
    ->DenseRange(1, 9)
    ->Unit(benchmark::kMillisecond)
    ->Name("VK/CifarSparse/Mini/Stage");

// ----------------------------------------------------------------------------
// Mini Benchmark: OMP Baseline
// ----------------------------------------------------------------------------

static void BM_run_OMP_baseline(benchmark::State& state) {
  const int num_threads = state.range(0);

  if (static_cast<size_t>(num_threads) > std::thread::hardware_concurrency()) {
    state.SkipWithMessage("");
    return;
  }

  auto mr = std::pmr::new_delete_resource();
  cifar_sparse::v2::AppData appdata(mr);

#pragma omp parallel
  {
    // Warm up the data structures to avoid first-run anomalies
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

  for (auto _ : state) {
#pragma omp parallel num_threads(num_threads)
    {
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

BENCHMARK(BM_run_OMP_baseline)
    ->DenseRange(1, 8)
    ->Unit(benchmark::kMillisecond)
    ->Name("OMP/CifarSparse/Mini/Baseline");

// ----------------------------------------------------------------------------
// Mini Benchmark: OMP Stage
// ----------------------------------------------------------------------------

static void BM_run_OMP_stage(benchmark::State& state) {
  const int stage = state.range(0);
  const ProcessorType core_type = static_cast<ProcessorType>(state.range(1));
  const int num_threads = state.range(2);

  auto mr = std::pmr::new_delete_resource();
  cifar_sparse::v2::AppData appdata(mr);

  std::vector<int> cores_to_use;
  if (core_type == ProcessorType::kLittleCore) {
    cores_to_use = g_lit_cores;
    state.SetLabel("Little, " + std::to_string(num_threads) + " cores");
  } else if (core_type == ProcessorType::kMediumCore) {
    cores_to_use = g_med_cores;
    state.SetLabel("Medium, " + std::to_string(num_threads) + " cores");
  } else {
    cores_to_use = g_big_cores;
    state.SetLabel("Big, " + std::to_string(num_threads) + " cores");
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
    for (const ProcessorType core_type :
         {ProcessorType::kLittleCore, ProcessorType::kMediumCore, ProcessorType::kBigCore}) {
      for (int num_threads = 1; num_threads <= 8; ++num_threads) {
        b->Args({stage, static_cast<int>(core_type), num_threads});
      }
    }
  }
}

BENCHMARK(BM_run_OMP_stage)
    ->Apply(CustomArgs)
    ->Unit(benchmark::kMillisecond)
    ->Name("OMP/CifarSparse/Mini/Stage");

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup
  return 0;
}
