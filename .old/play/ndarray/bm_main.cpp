#include <benchmark/benchmark.h>

#include <cstddef>
#include <vector>

#include "benchmarks/argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "omp/dispatchers.hpp"

void FlushCache() {
  // Choose a buffer size significantly larger than the L3 cache (10 MB in this case).
  constexpr size_t size = 10 * 1024 * 1024;  // 10 MB
  std::vector<char> buffer(size, 1);

  // Use a volatile variable to prevent the compiler from optimizing away the loop.
  volatile char sink = 0;

  // Iterate over the buffer in steps (assume cache line size is 64 bytes).
  for (size_t i = 0; i < size; i += 64) {
    sink += buffer[i];
  }
}

// ------------------------------------------------------------
// Baseline benchmarks
// ------------------------------------------------------------

void register_baseline_benchmark() {
  std::string benchmark_name = "OMP_CifarDense/Baseline";

  benchmark::RegisterBenchmark(
      benchmark_name.c_str(),
      [](benchmark::State& state) {
        const auto n_threads = state.range(0);
        cifar_dense::AppDataBatch batched_appdata(std::pmr::new_delete_resource());
        for (auto _ : state) {
          state.PauseTiming();
          FlushCache();
          state.ResumeTiming();

          omp::dispatch_multi_stage_unrestricted(n_threads, batched_appdata, 1, 9);

          benchmark::ClobberMemory();
        }
      })
      ->DenseRange(1, std::thread::hardware_concurrency())
      ->Unit(benchmark::kMillisecond);
}

// ------------------------------------------------------------
// Stage benchmarks
// ------------------------------------------------------------

const char* name_of_processor_type(ProcessorType pt) {
  switch (pt) {
    case ProcessorType::kLittleCore:
      return "little";
    case ProcessorType::kMediumCore:
      return "medium";
    case ProcessorType::kBigCore:
      return "big";
  }
}

template <int Stage, ProcessorType PT>
void register_stage_benchmark(const std::vector<int>& cores) {
  for (size_t i = 1; i <= cores.size(); ++i) {
    std::string benchmark_name =
        "OMP_CifarDense/Stage" + std::to_string(Stage) + "_" + name_of_processor_type(PT);

    benchmark::RegisterBenchmark(
        benchmark_name.c_str(),
        [i, cores](benchmark::State& state) {
          cifar_dense::AppDataBatch batched_appdata(std::pmr::new_delete_resource());
          for (auto _ : state) {
            state.PauseTiming();
            FlushCache();
            state.ResumeTiming();

            int num_threads = static_cast<int>(i);
            omp::dispatch_multi_stage(Stage, Stage, PT, num_threads, batched_appdata);

            benchmark::ClobberMemory();
          }
        })
        ->Arg(i)
        ->Unit(benchmark::kMillisecond);
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  register_baseline_benchmark();

#define REGISTER_STAGE(STAGE)                                                  \
  register_stage_benchmark<STAGE, ProcessorType::kLittleCore>(g_little_cores); \
  register_stage_benchmark<STAGE, ProcessorType::kMediumCore>(g_medium_cores); \
  register_stage_benchmark<STAGE, ProcessorType::kBigCore>(g_big_cores)

  REGISTER_STAGE(1);
  REGISTER_STAGE(2);
  REGISTER_STAGE(3);
  REGISTER_STAGE(4);
  REGISTER_STAGE(5);
  REGISTER_STAGE(6);
  REGISTER_STAGE(7);
  REGISTER_STAGE(8);
  REGISTER_STAGE(9);

  // Where to save the results json file?
  const auto storage_location = helpers::get_benchmark_storage_location();
  const auto out_name = storage_location.string() + "/BM_CifarDense_OMP_" + g_device_id + ".json";

  // Sanitize the arguments to pass to Google Benchmark
  auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv, out_name);

  benchmark::Initialize(&new_argc, new_argv.data());
  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
