#include <benchmark/benchmark.h>
#include <omp.h>

#include "builtin-apps/app.hpp"  // for 'g_big_cores', 'g_medium_cores', 'g_little_cores'
#include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
#include "builtin-apps/cifar-dense/vulkan/dispatchers.hpp"

// ----------------------------------------------------------------------------
// Baseline: OMP
// ----------------------------------------------------------------------------

static void BM_baseline_omp(benchmark::State& state) {
  int n_threads = state.range(0);

  if ((size_t)n_threads > std::thread::hardware_concurrency()) {
    state.SkipWithMessage("n_threads is greater than the number of hardware threads");
    return;
  }

  cifar_dense::vulkan::VulkanDispatcher disp;
  cifar_dense::AppData appdata(disp.get_mr());

  // Warm up
  cifar_dense::omp::run_stage_1(appdata);

  for (auto _ : state) {
#pragma omp parallel num_threads(n_threads)
    {
      cifar_dense::omp::run_stage_1(appdata);
      cifar_dense::omp::run_stage_2(appdata);
      cifar_dense::omp::run_stage_3(appdata);
      cifar_dense::omp::run_stage_4(appdata);
      cifar_dense::omp::run_stage_5(appdata);
      cifar_dense::omp::run_stage_6(appdata);
      cifar_dense::omp::run_stage_7(appdata);
      cifar_dense::omp::run_stage_8(appdata);
      cifar_dense::omp::run_stage_9(appdata);
    }
  }
}

BENCHMARK(BM_baseline_omp)->DenseRange(1, 8)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------------------
// Baseline: Vulkan
// ----------------------------------------------------------------------------

static void BM_baseline_vulkan(benchmark::State& state) {
  cifar_dense::vulkan::VulkanDispatcher disp;
  cifar_dense::AppData appdata(disp.get_mr());

  // Warm up
  disp.dispatch_multi_stage(appdata, 1, 9);

  // Benchmark
  for (auto _ : state) {
    disp.dispatch_multi_stage(appdata, 1, 9);
  }
}

BENCHMARK(BM_baseline_vulkan)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

// e.g.,
// xmake r bm-baseline-cifar-dense-vk -l off

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup
  return 0;
}
