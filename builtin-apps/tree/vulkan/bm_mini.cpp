#include <benchmark/benchmark.h>

#include "../../app.hpp"
#include "dispatchers.hpp"

static void BM_InitializeTree(benchmark::State& state) {
  tree::vulkan::VulkanDispatcher disp;

  // warm up
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  for (auto _ : state) {
    tree::vulkan::VkAppData_Safe appdata(disp.get_mr());
  }
}

BENCHMARK(BM_InitializeTree)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
