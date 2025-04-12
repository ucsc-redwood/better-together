#include <benchmark/benchmark.h>
#include <omp.h>

#include "../omp/dispatchers.hpp"
#include "../sparse_appdata.hpp"
#include "builtin-apps/app.hpp"
#include "dispatchers.hpp"

// // ----------------------------------------------------------------------------
// // Realistic Benchmark: Vk Baseline
// // ----------------------------------------------------------------------------

// static void BM_run_VK_baseline(benchmark::State& state) {
//   cifar_sparse::vulkan::v2::VulkanDispatcher disp;
//   cifar_sparse::v2::AppData appdata(disp.get_mr());

//   // Warm up the data structures to avoid first-run anomalies
//   disp.dispatch_multi_stage(appdata, 1, 9);

//   for (auto _ : state) {
//     disp.dispatch_multi_stage(appdata, 1, 9);
//   }
// }

// BENCHMARK(BM_run_VK_baseline)->Unit(benchmark::kMillisecond)->Name("VK/CifarSparse/Baseline");

// // ----------------------------------------------------------------------------
// // Realistic Benchmark: Vk Stage
// // ----------------------------------------------------------------------------

// static void BM_run_VK_stage(benchmark::State& state) {
//   const int stage = state.range(0);

//   cifar_sparse::vulkan::v2::VulkanDispatcher disp;
//   cifar_sparse::v2::AppData appdata(disp.get_mr());

//   // Ensure stage is valid
//   if (stage < 1 || stage > 9) {
//     state.SkipWithError("Invalid stage number");
//     return;
//   }

//   // Warm up to prevent first-run anomalies
//   disp.dispatch_multi_stage(appdata, 1, stage);

//   for (auto _ : state) {
//     disp.dispatch_multi_stage(appdata, stage, stage);
//   }
// }

// BENCHMARK(BM_run_VK_stage)
//     ->DenseRange(1, 9)
//     ->Unit(benchmark::kMillisecond)
//     ->Name("VK/CifarSparse/Stage");

// ----------------------------------------------------------------------------
// Realistic Benchmark: OMP Stage
// ----------------------------------------------------------------------------

static void BM_run_OMP_stage(benchmark::State& state) {
  const int stage = state.range(0);
  const ProcessorType core_type = static_cast<ProcessorType>(state.range(1));

  auto mr = std::pmr::new_delete_resource();
  cifar_sparse::v2::AppData appdata(mr);

  std::vector<int> cores_to_use;
  if (core_type == ProcessorType::kLittleCore) {
    cores_to_use = g_little_cores;
    state.SetLabel("Little, " + std::to_string(cores_to_use.size()) + " cores");
  } else if (core_type == ProcessorType::kMediumCore) {
    cores_to_use = g_medium_cores;
    state.SetLabel("Medium, " + std::to_string(cores_to_use.size()) + " cores");
  } else {
    cores_to_use = g_big_cores;
    state.SetLabel("Big, " + std::to_string(cores_to_use.size()) + " cores");
  }

  const int num_threads = cores_to_use.size();

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
      b->Args({stage, static_cast<int>(core_type)});
    }
  }
}

BENCHMARK(BM_run_OMP_stage)
    ->Apply(CustomArgs)
    ->Unit(benchmark::kMillisecond)
    ->Name("OMP/CifarSparse/Stage");

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
