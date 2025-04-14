#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include "benchmarks/argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/cu_bench_helper.cuh"
#include "cuda/dispatchers.cuh"

// ----------------------------------------------------------------
// Global config
// ----------------------------------------------------------------

bool g_flush_l2_cache = false;

__global__ void warmup_kernel() {
  // Do nothing
}

void warmup() {
  warmup_kernel<<<1, 1>>>();
  CheckCuda(cudaDeviceSynchronize());
}

#define PREPARE_DATA                                    \
  cuda::CudaDispatcher<cuda::CudaManagedResource> disp; \
  cifar_dense::AppDataBatch batched_appdata(&disp.get_mr())

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

class CUDA_CifarDense : public benchmark::Fixture {};

BENCHMARK_DEFINE_F(CUDA_CifarDense, Baseline)
(benchmark::State& state) {
  PREPARE_DATA;

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);

    disp.dispatch_multi_stage(batched_appdata, 1, 9);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Baseline)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage1)
(benchmark::State& state) {
  PREPARE_DATA;

  // previous steps + warmup
  disp.dispatch_multi_stage(batched_appdata, 1, 1);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);

    disp.run_stage_1_async(batched_appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage1)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage2)
(benchmark::State& state) {
  PREPARE_DATA;

  // previous steps + warmup
  disp.dispatch_multi_stage(batched_appdata, 1, 2);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);

    disp.run_stage_2_async(batched_appdata);
  }
}
BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage2)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage3)
(benchmark::State& state) {
  PREPARE_DATA;

  // previous steps + warmup
  disp.dispatch_multi_stage(batched_appdata, 1, 3);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);

    disp.run_stage_3_async(batched_appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage3)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage4)
(benchmark::State& state) {
  PREPARE_DATA;

  // previous steps + warmup
  disp.dispatch_multi_stage(batched_appdata, 1, 4);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);

    disp.run_stage_4_async(batched_appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage4)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage5)
(benchmark::State& state) {
  PREPARE_DATA;

  // previous steps + warmup
  disp.dispatch_multi_stage(batched_appdata, 1, 5);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);

    disp.run_stage_5_async(batched_appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage5)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage6)
(benchmark::State& state) {
  PREPARE_DATA;

  // previous steps + warmup
  disp.dispatch_multi_stage(batched_appdata, 1, 6);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);

    disp.run_stage_6_async(batched_appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage6)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage7)
(benchmark::State& state) {
  PREPARE_DATA;

  // previous steps + warmup
  disp.dispatch_multi_stage(batched_appdata, 1, 7);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);

    disp.run_stage_7_async(batched_appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage7)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 8
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage8)
(benchmark::State& state) {
  PREPARE_DATA;

  // previous steps + warmup
  disp.dispatch_multi_stage(batched_appdata, 1, 8);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);

    disp.run_stage_8_async(batched_appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage8)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 9
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage9)
(benchmark::State& state) {
  PREPARE_DATA;

  // previous steps + warmup
  disp.dispatch_multi_stage(batched_appdata, 1, 9);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);

    disp.run_stage_9_async(batched_appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage9)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN

  app.add_option("--flush-l2-cache", g_flush_l2_cache, "Flush L2 cache");

  PARSE_ARGS_END

  spdlog::set_level(spdlog::level::off);

  // Where to save the results json file?
  const auto storage_location = helpers::get_benchmark_storage_location();
  const auto out_name = storage_location.string() + "/BM_CifarDense_CUDA_" + g_device_id + ".json";

  // Sanitize the arguments to pass to Google Benchmark
  auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv, out_name);

  benchmark::Initialize(&new_argc, new_argv.data());
  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;

  warmup();

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
