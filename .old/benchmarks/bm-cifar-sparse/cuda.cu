#include <benchmark/benchmark.h>

#include "../argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/cuda/dispatchers.cuh"
#include "builtin-apps/cifar-sparse/sparse_appdata.hpp"
#include "builtin-apps/common/cuda/cu_bench_helper.cuh"
#include "builtin-apps/common/cuda/helpers.cuh"
#include "builtin-apps/common/cuda/manager.cuh"
#include "builtin-apps/resources_path.hpp"

#define PREPARE_DATA                 \
  cuda::CudaManager mgr;             \
  auto mr = &mgr.get_mr();           \
  cifar_sparse::AppData appdata(mr); \
  CheckCuda(cudaDeviceSynchronize());

// ----------------------------------------------------------------
// Global config
// ----------------------------------------------------------------

bool g_flush_l2_cache = false;

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

class CUDA_CifarSparse : public benchmark::Fixture {};

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Baseline)
(benchmark::State& state) {
  PREPARE_DATA;

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);
    cifar_sparse::cuda::process_stage_1(appdata);
    cifar_sparse::cuda::process_stage_2(appdata);
    cifar_sparse::cuda::process_stage_3(appdata);
    cifar_sparse::cuda::process_stage_4(appdata);
    cifar_sparse::cuda::process_stage_5(appdata);
    cifar_sparse::cuda::process_stage_6(appdata);
    cifar_sparse::cuda::process_stage_7(appdata);
    cifar_sparse::cuda::process_stage_8(appdata);
    cifar_sparse::cuda::process_stage_9(appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Baseline)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage1)
(benchmark::State& state) {
  PREPARE_DATA;

  // previous steps + warmup
  cifar_sparse::cuda::process_stage_1(appdata);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);
    cifar_sparse::cuda::process_stage_1(appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage1)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage2)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_sparse::cuda::process_stage_1(appdata);
  cifar_sparse::cuda::process_stage_2(appdata);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);
    cifar_sparse::cuda::process_stage_2(appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage2)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage3)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_sparse::cuda::process_stage_1(appdata);
  cifar_sparse::cuda::process_stage_2(appdata);
  cifar_sparse::cuda::process_stage_3(appdata);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);
    cifar_sparse::cuda::process_stage_3(appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage3)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage4)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_sparse::cuda::process_stage_1(appdata);
  cifar_sparse::cuda::process_stage_2(appdata);
  cifar_sparse::cuda::process_stage_3(appdata);
  cifar_sparse::cuda::process_stage_4(appdata);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);
    cifar_sparse::cuda::process_stage_4(appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage4)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage5)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_sparse::cuda::process_stage_1(appdata);
  cifar_sparse::cuda::process_stage_2(appdata);
  cifar_sparse::cuda::process_stage_3(appdata);
  cifar_sparse::cuda::process_stage_4(appdata);
  cifar_sparse::cuda::process_stage_5(appdata);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);
    cifar_sparse::cuda::process_stage_5(appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage5)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage6)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_sparse::cuda::process_stage_1(appdata);
  cifar_sparse::cuda::process_stage_2(appdata);
  cifar_sparse::cuda::process_stage_3(appdata);
  cifar_sparse::cuda::process_stage_4(appdata);
  cifar_sparse::cuda::process_stage_5(appdata);
  cifar_sparse::cuda::process_stage_6(appdata);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);
    cifar_sparse::cuda::process_stage_6(appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage6)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage7)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_sparse::cuda::process_stage_1(appdata);
  cifar_sparse::cuda::process_stage_2(appdata);
  cifar_sparse::cuda::process_stage_3(appdata);
  cifar_sparse::cuda::process_stage_4(appdata);
  cifar_sparse::cuda::process_stage_5(appdata);
  cifar_sparse::cuda::process_stage_6(appdata);
  cifar_sparse::cuda::process_stage_7(appdata);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);
    cifar_sparse::cuda::process_stage_7(appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage7)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 8
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage8)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_sparse::cuda::process_stage_1(appdata);
  cifar_sparse::cuda::process_stage_2(appdata);
  cifar_sparse::cuda::process_stage_3(appdata);
  cifar_sparse::cuda::process_stage_4(appdata);
  cifar_sparse::cuda::process_stage_5(appdata);
  cifar_sparse::cuda::process_stage_6(appdata);
  cifar_sparse::cuda::process_stage_7(appdata);
  cifar_sparse::cuda::process_stage_8(appdata);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);
    cifar_sparse::cuda::process_stage_8(appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage8)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 9
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage9)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_sparse::cuda::process_stage_1(appdata);
  cifar_sparse::cuda::process_stage_2(appdata);
  cifar_sparse::cuda::process_stage_3(appdata);
  cifar_sparse::cuda::process_stage_4(appdata);
  cifar_sparse::cuda::process_stage_5(appdata);
  cifar_sparse::cuda::process_stage_6(appdata);
  cifar_sparse::cuda::process_stage_7(appdata);
  cifar_sparse::cuda::process_stage_8(appdata);
  cifar_sparse::cuda::process_stage_9(appdata);
  CheckCuda(cudaDeviceSynchronize());

  for (auto _ : state) {
    CudaEventTimer timer(state, g_flush_l2_cache);
    cifar_sparse::cuda::process_stage_9(appdata);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage9)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN

  app.add_option("--flush-l2-cache", g_flush_l2_cache, "Flush L2 cache");

  PARSE_ARGS_END

  spdlog::set_level(spdlog::level::off);

  // Where to save the results json file?
  const auto storage_location = helpers::get_benchmark_storage_location();
  const auto out_name = storage_location.string() + "/BM_CifarSparse_CUDA_" + g_device_id + ".json";

  // Sanitize the arguments to pass to Google Benchmark
  auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv, out_name);

  benchmark::Initialize(&new_argc, new_argv.data());
  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}