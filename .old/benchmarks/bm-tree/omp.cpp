#include <benchmark/benchmark.h>

#include <thread>

#include "../argc_argv_sanitizer.hpp"
#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/resources_path.hpp"
#include "builtin-apps/tree/omp/dispatchers.hpp"
// #include "builtin-apps/tree/tree_appdata.hpp"

// ------------------------------------------------------------
// Global variables
// ------------------------------------------------------------

static void run_baseline_unrestricted(tree::SafeAppData& appdata, const int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    tree::omp::process_stage_1(appdata);
    tree::omp::process_stage_2(appdata);
    tree::omp::process_stage_3(appdata);
    tree::omp::process_stage_4(appdata);
    tree::omp::process_stage_5(appdata);
    tree::omp::process_stage_6(appdata);
    tree::omp::process_stage_7(appdata);
  }
}

class OMP_Tree : public benchmark::Fixture {
 protected:
  std::unique_ptr<tree::SafeAppData> appdata_ptr;

  void SetUp(const ::benchmark::State&) override {
    appdata_ptr = std::make_unique<tree::SafeAppData>(std::pmr::new_delete_resource());

    const auto stage_2_threads = std::thread::hardware_concurrency();

    run_baseline_unrestricted(*appdata_ptr, stage_2_threads);
  }

  void TearDown(const ::benchmark::State&) override { appdata_ptr.reset(); }
};

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(OMP_Tree, Baseline)
(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    run_baseline_unrestricted(*appdata_ptr, n_threads);
  }
}

BENCHMARK_REGISTER_F(OMP_Tree, Baseline)
    ->DenseRange(1, std::thread::hardware_concurrency())
    ->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stages
// ----------------------------------------------------------------

template <int stage, ProcessorType processor_type>
  requires(stage >= 1 && stage <= 9)
void run_stage(tree::SafeAppData& appdata, const int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    // Bind to core if needed:
    if constexpr (processor_type == ProcessorType::kLittleCore) {
      bind_thread_to_cores(g_lit_cores);
    } else if constexpr (processor_type == ProcessorType::kMediumCore) {
      bind_thread_to_cores(g_med_cores);
    } else if constexpr (processor_type == ProcessorType::kBigCore) {
      bind_thread_to_cores(g_big_cores);
    } else {
      assert(false);
    }

    tree::omp::run_stage<stage>(appdata);
  }
}

// Template for stage benchmarks
template <int stage, ProcessorType processor_type>
class StageFixture : public OMP_Tree {
 public:
  void BenchmarkCase(benchmark::State& state) {
    const auto n_threads = state.range(0);
    for (auto _ : state) {
      run_stage<stage, processor_type>(*appdata_ptr, n_threads);
    }
  }
  BENCHMARK_PRIVATE_DECLARE(StageFixture);
};

// Helper to get processor type name
constexpr const char* get_processor_name(ProcessorType type) {
  switch (type) {
    case ProcessorType::kLittleCore:
      return "little";
    case ProcessorType::kMediumCore:
      return "medium";
    case ProcessorType::kBigCore:
      return "big";
    default:
      return "unknown";
  }
}

// ----------------------------------------------------------------
// Template for registering stage benchmarks
// ----------------------------------------------------------------

template <int stage, ProcessorType proc_type>
void RegisterStageBenchmark() {
  const char* proc_name = get_processor_name(proc_type);

  size_t n_cores = 0;
  if constexpr (proc_type == ProcessorType::kLittleCore) {
    n_cores = g_lit_cores.size();
  } else if constexpr (proc_type == ProcessorType::kMediumCore) {
    n_cores = g_med_cores.size();
  } else if constexpr (proc_type == ProcessorType::kBigCore) {
    n_cores = g_big_cores.size();
  }

  for (size_t i = 1; i <= n_cores; ++i) {
    auto* benchmark = new StageFixture<stage, proc_type>();
    std::string name = "OMP_Tree/Stage" + std::to_string(stage) + "_" + proc_name;

    ::benchmark::internal::RegisterBenchmarkInternal(benchmark)->Arg(i)->Name(name)->Unit(
        benchmark::kMillisecond);
  }
}

// ----------------------------------------------------------------
// Register all stage benchmarks for a processor type
// ----------------------------------------------------------------

template <ProcessorType proc_type>
void RegisterAllStagesForProcessor() {
  RegisterStageBenchmark<1, proc_type>();
  RegisterStageBenchmark<2, proc_type>();
  RegisterStageBenchmark<3, proc_type>();
  RegisterStageBenchmark<4, proc_type>();
  RegisterStageBenchmark<5, proc_type>();
  RegisterStageBenchmark<6, proc_type>();
  RegisterStageBenchmark<7, proc_type>();
}

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  // Register benchmarks for all processor types
  RegisterAllStagesForProcessor<ProcessorType::kLittleCore>();
  RegisterAllStagesForProcessor<ProcessorType::kMediumCore>();
  RegisterAllStagesForProcessor<ProcessorType::kBigCore>();

  // Where to save the results json file?
  const auto storage_location = helpers::get_benchmark_storage_location();
  const auto out_name = storage_location.string() + "/BM_Tree_OMP_" + g_device_id + ".json";

  // Sanitize the arguments to pass to Google Benchmark
  auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv, out_name);

  // Initialize Google Benchmark and run benchmarks
  benchmark::Initialize(&new_argc, new_argv.data());
  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
