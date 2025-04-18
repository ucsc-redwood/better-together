#include <benchmark/benchmark.h>

#include <memory_resource>

#include "../../app.hpp"
#include "../appdata.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------
// Stage 1: Conv1
// ----------------------------------------------------------------

static void BM_Stage1(benchmark::State& state) {
  auto vk_dispatcher = std::make_unique<cifar_sparse::vulkan::VulkanDispatcher>();
  auto mr = vk_dispatcher->get_mr();
  cifar_sparse::AppData appdata(mr);

  // warm up
  vk_dispatcher->run_stage_1(appdata);

  for (auto _ : state) {
    vk_dispatcher->run_stage_1(appdata);
  }
}

BENCHMARK(BM_Stage1)->Unit(benchmark::kMillisecond)->Name("Vulkan/CIFAR-Sparse/Stage1");

// ----------------------------------------------------------------
// Stage 2: MaxPool1
// ----------------------------------------------------------------

static void BM_Stage2(benchmark::State& state) {
  auto vk_dispatcher = std::make_unique<cifar_sparse::vulkan::VulkanDispatcher>();
  auto mr = vk_dispatcher->get_mr();
  cifar_sparse::AppData appdata(mr);

  // Run all previous stages
  vk_dispatcher->run_stage_1(appdata);

  // warm up
  vk_dispatcher->run_stage_2(appdata);

  for (auto _ : state) {
    vk_dispatcher->run_stage_2(appdata);
  }
}

BENCHMARK(BM_Stage2)->Unit(benchmark::kMillisecond)->Name("Vulkan/CIFAR-Sparse/Stage2");

// ----------------------------------------------------------------
// Stage 3: Conv2
// ----------------------------------------------------------------

static void BM_Stage3(benchmark::State& state) {
  auto vk_dispatcher = std::make_unique<cifar_sparse::vulkan::VulkanDispatcher>();
  auto mr = vk_dispatcher->get_mr();
  cifar_sparse::AppData appdata(mr);

  // Run all previous stages
  vk_dispatcher->run_stage_1(appdata);
  vk_dispatcher->run_stage_2(appdata);

  // warm up
  vk_dispatcher->run_stage_3(appdata);

  for (auto _ : state) {
    vk_dispatcher->run_stage_3(appdata);
  }
}

BENCHMARK(BM_Stage3)->Unit(benchmark::kMillisecond)->Name("Vulkan/CIFAR-Sparse/Stage3");

// ----------------------------------------------------------------
// Stage 4: MaxPool2
// ----------------------------------------------------------------

static void BM_Stage4(benchmark::State& state) {
  auto vk_dispatcher = std::make_unique<cifar_sparse::vulkan::VulkanDispatcher>();
  auto mr = vk_dispatcher->get_mr();
  cifar_sparse::AppData appdata(mr);

  // Run all previous stages
  vk_dispatcher->run_stage_1(appdata);
  vk_dispatcher->run_stage_2(appdata);
  vk_dispatcher->run_stage_3(appdata);

  // warm up
  vk_dispatcher->run_stage_4(appdata);

  for (auto _ : state) {
    vk_dispatcher->run_stage_4(appdata);
  }
}

BENCHMARK(BM_Stage4)->Unit(benchmark::kMillisecond)->Name("Vulkan/CIFAR-Sparse/Stage4");

// ----------------------------------------------------------------
// Stage 5: Conv3
// ----------------------------------------------------------------

static void BM_Stage5(benchmark::State& state) {
  auto vk_dispatcher = std::make_unique<cifar_sparse::vulkan::VulkanDispatcher>();
  auto mr = vk_dispatcher->get_mr();
  cifar_sparse::AppData appdata(mr);

  // Run all previous stages
  vk_dispatcher->run_stage_1(appdata);
  vk_dispatcher->run_stage_2(appdata);
  vk_dispatcher->run_stage_3(appdata);
  vk_dispatcher->run_stage_4(appdata);

  // warm up
  vk_dispatcher->run_stage_5(appdata);

  for (auto _ : state) {
    vk_dispatcher->run_stage_5(appdata);
  }
}

BENCHMARK(BM_Stage5)->Unit(benchmark::kMillisecond)->Name("Vulkan/CIFAR-Sparse/Stage5");

// ----------------------------------------------------------------
// Stage 6: Conv4
// ----------------------------------------------------------------

static void BM_Stage6(benchmark::State& state) {
  auto vk_dispatcher = std::make_unique<cifar_sparse::vulkan::VulkanDispatcher>();
  auto mr = vk_dispatcher->get_mr();
  cifar_sparse::AppData appdata(mr);

  // Run all previous stages
  vk_dispatcher->run_stage_1(appdata);
  vk_dispatcher->run_stage_2(appdata);
  vk_dispatcher->run_stage_3(appdata);
  vk_dispatcher->run_stage_4(appdata);
  vk_dispatcher->run_stage_5(appdata);

  // warm up
  vk_dispatcher->run_stage_6(appdata);

  for (auto _ : state) {
    vk_dispatcher->run_stage_6(appdata);
  }
}

BENCHMARK(BM_Stage6)->Unit(benchmark::kMillisecond)->Name("Vulkan/CIFAR-Sparse/Stage6");

// ----------------------------------------------------------------
// Stage 7: Conv5
// ----------------------------------------------------------------

static void BM_Stage7(benchmark::State& state) {
  auto vk_dispatcher = std::make_unique<cifar_sparse::vulkan::VulkanDispatcher>();
  auto mr = vk_dispatcher->get_mr();
  cifar_sparse::AppData appdata(mr);

  // Run all previous stages
  vk_dispatcher->run_stage_1(appdata);
  vk_dispatcher->run_stage_2(appdata);
  vk_dispatcher->run_stage_3(appdata);
  vk_dispatcher->run_stage_4(appdata);
  vk_dispatcher->run_stage_5(appdata);
  vk_dispatcher->run_stage_6(appdata);

  // warm up
  vk_dispatcher->run_stage_7(appdata);

  for (auto _ : state) {
    vk_dispatcher->run_stage_7(appdata);
  }
}

BENCHMARK(BM_Stage7)->Unit(benchmark::kMillisecond)->Name("Vulkan/CIFAR-Sparse/Stage7");

// ----------------------------------------------------------------
// Stage 8: MaxPool3
// ----------------------------------------------------------------

static void BM_Stage8(benchmark::State& state) {
  auto vk_dispatcher = std::make_unique<cifar_sparse::vulkan::VulkanDispatcher>();
  auto mr = vk_dispatcher->get_mr();
  cifar_sparse::AppData appdata(mr);

  // Run all previous stages
  vk_dispatcher->run_stage_1(appdata);
  vk_dispatcher->run_stage_2(appdata);
  vk_dispatcher->run_stage_3(appdata);
  vk_dispatcher->run_stage_4(appdata);
  vk_dispatcher->run_stage_5(appdata);
  vk_dispatcher->run_stage_6(appdata);
  vk_dispatcher->run_stage_7(appdata);

  // warm up
  vk_dispatcher->run_stage_8(appdata);

  for (auto _ : state) {
    vk_dispatcher->run_stage_8(appdata);
  }
}

BENCHMARK(BM_Stage8)->Unit(benchmark::kMillisecond)->Name("Vulkan/CIFAR-Sparse/Stage8");

// ----------------------------------------------------------------
// Stage 9: Linear
// ----------------------------------------------------------------

static void BM_Stage9(benchmark::State& state) {
  auto vk_dispatcher = std::make_unique<cifar_sparse::vulkan::VulkanDispatcher>();
  auto mr = vk_dispatcher->get_mr();
  cifar_sparse::AppData appdata(mr);

  // Run all previous stages
  vk_dispatcher->run_stage_1(appdata);
  vk_dispatcher->run_stage_2(appdata);
  vk_dispatcher->run_stage_3(appdata);
  vk_dispatcher->run_stage_4(appdata);
  vk_dispatcher->run_stage_5(appdata);
  vk_dispatcher->run_stage_6(appdata);
  vk_dispatcher->run_stage_7(appdata);
  vk_dispatcher->run_stage_8(appdata);

  // warm up
  vk_dispatcher->run_stage_9(appdata);

  for (auto _ : state) {
    vk_dispatcher->run_stage_9(appdata);
  }
}

BENCHMARK(BM_Stage9)->Unit(benchmark::kMillisecond)->Name("Vulkan/CIFAR-Sparse/Stage9");

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}