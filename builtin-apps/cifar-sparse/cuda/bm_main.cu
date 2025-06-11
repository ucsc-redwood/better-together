#include <benchmark/benchmark.h>

#include <memory_resource>

#include "../../app.hpp"
#include "../appdata.hpp"
#include "dispatchers.cuh"

// ----------------------------------------------------------------
// Stage 1: Conv1
// ----------------------------------------------------------------

static void BM_Stage1(benchmark::State& state) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // warm up
  disp.dispatch_stage(appdata, 1);

  for (auto _ : state) {
    disp.dispatch_stage(appdata, 1);
  }
}

BENCHMARK(BM_Stage1)->Unit(benchmark::kMillisecond)->Name("CUDA/CIFAR-Sparse/Stage1");

// ----------------------------------------------------------------
// Stage 2: MaxPool1
// ----------------------------------------------------------------

static void BM_Stage2(benchmark::State& state) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run all previous stages before benchmarking
  disp.dispatch_stage(appdata, 1);

  // warm up
  disp.dispatch_stage(appdata, 2);

  for (auto _ : state) {
    disp.dispatch_stage(appdata, 2);
  }
}

BENCHMARK(BM_Stage2)->Unit(benchmark::kMillisecond)->Name("CUDA/CIFAR-Sparse/Stage2");

// ----------------------------------------------------------------
// Stage 3: Conv2
// ----------------------------------------------------------------

static void BM_Stage3(benchmark::State& state) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run all previous stages before benchmarking
  disp.dispatch_stage(appdata, 1);
  disp.dispatch_stage(appdata, 2);

  // warm up
  disp.dispatch_stage(appdata, 3);

  for (auto _ : state) {
    disp.dispatch_stage(appdata, 3);
  }
}

BENCHMARK(BM_Stage3)->Unit(benchmark::kMillisecond)->Name("CUDA/CIFAR-Sparse/Stage3");

// ----------------------------------------------------------------
// Stage 4: MaxPool2
// ----------------------------------------------------------------

static void BM_Stage4(benchmark::State& state) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run all previous stages before benchmarking
  disp.dispatch_stage(appdata, 1);
  disp.dispatch_stage(appdata, 2);
  disp.dispatch_stage(appdata, 3);

  // warm up
  disp.dispatch_stage(appdata, 4);

  for (auto _ : state) {
    disp.dispatch_stage(appdata, 4);
  }
}

BENCHMARK(BM_Stage4)->Unit(benchmark::kMillisecond)->Name("CUDA/CIFAR-Sparse/Stage4");

// ----------------------------------------------------------------
// Stage 5: Conv3
// ----------------------------------------------------------------

static void BM_Stage5(benchmark::State& state) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run all previous stages before benchmarking
  disp.dispatch_stage(appdata, 1);
  disp.dispatch_stage(appdata, 2);
  disp.dispatch_stage(appdata, 3);
  disp.dispatch_stage(appdata, 4);

  // warm up
  disp.dispatch_stage(appdata, 5);

  for (auto _ : state) {
    disp.dispatch_stage(appdata, 5);
  }
}

BENCHMARK(BM_Stage5)->Unit(benchmark::kMillisecond)->Name("CUDA/CIFAR-Sparse/Stage5");

// ----------------------------------------------------------------
// Stage 6: Conv4
// ----------------------------------------------------------------

static void BM_Stage6(benchmark::State& state) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run all previous stages before benchmarking
  disp.dispatch_stage(appdata, 1);
  disp.dispatch_stage(appdata, 2);
  disp.dispatch_stage(appdata, 3);
  disp.dispatch_stage(appdata, 4);
  disp.dispatch_stage(appdata, 5);

  // warm up
  disp.dispatch_stage(appdata, 6);

  for (auto _ : state) {
    disp.dispatch_stage(appdata, 6);
  }
}

BENCHMARK(BM_Stage6)->Unit(benchmark::kMillisecond)->Name("CUDA/CIFAR-Sparse/Stage6");

// ----------------------------------------------------------------
// Stage 7: Conv5
// ----------------------------------------------------------------

static void BM_Stage7(benchmark::State& state) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run all previous stages before benchmarking
  disp.dispatch_stage(appdata, 1);
  disp.dispatch_stage(appdata, 2);
  disp.dispatch_stage(appdata, 3);
  disp.dispatch_stage(appdata, 4);
  disp.dispatch_stage(appdata, 5);
  disp.dispatch_stage(appdata, 6);

  // warm up
  disp.dispatch_stage(appdata, 7);

  for (auto _ : state) {
    disp.dispatch_stage(appdata, 7);
  }
}

BENCHMARK(BM_Stage7)->Unit(benchmark::kMillisecond)->Name("CUDA/CIFAR-Sparse/Stage7");

// ----------------------------------------------------------------
// Stage 8: MaxPool3
// ----------------------------------------------------------------

static void BM_Stage8(benchmark::State& state) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run all previous stages before benchmarking
  disp.dispatch_stage(appdata, 1);
  disp.dispatch_stage(appdata, 2);
  disp.dispatch_stage(appdata, 3);
  disp.dispatch_stage(appdata, 4);
  disp.dispatch_stage(appdata, 5);
  disp.dispatch_stage(appdata, 6);
  disp.dispatch_stage(appdata, 7);

  // warm up
  disp.dispatch_stage(appdata, 8);

  for (auto _ : state) {
    disp.dispatch_stage(appdata, 8);
  }
}

BENCHMARK(BM_Stage8)->Unit(benchmark::kMillisecond)->Name("CUDA/CIFAR-Sparse/Stage8");

// ----------------------------------------------------------------
// Stage 9: Linear
// ----------------------------------------------------------------

static void BM_Stage9(benchmark::State& state) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run all previous stages before benchmarking
  disp.dispatch_stage(appdata, 1);
  disp.dispatch_stage(appdata, 2);
  disp.dispatch_stage(appdata, 3);
  disp.dispatch_stage(appdata, 4);
  disp.dispatch_stage(appdata, 5);
  disp.dispatch_stage(appdata, 6);
  disp.dispatch_stage(appdata, 7);
  disp.dispatch_stage(appdata, 8);

  // warm up
  disp.dispatch_stage(appdata, 9);

  for (auto _ : state) {
    disp.dispatch_stage(appdata, 9);
  }
}

BENCHMARK(BM_Stage9)->Unit(benchmark::kMillisecond)->Name("CUDA/CIFAR-Sparse/Stage9");

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}