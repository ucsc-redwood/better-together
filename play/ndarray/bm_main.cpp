#include <benchmark/benchmark.h>

#include "builtin-apps/app.hpp"
#include "dispatchers.hpp"

constexpr auto kDefaultInputFile = "cifar10_images/img_00005.npy";

// static void BM_Stage_1(benchmark::State& state) {
//   const auto num_threads = state.range(0);

//   cifar_dense::AppDataBatch appdata(kDefaultInputFile);
//   for (auto _ : state) {
//     omp::dispatch_multi_stage<ProcessorType::kLittleCore>(num_threads, appdata, 1, 1);
//   }
// }

// BENCHMARK(BM_Stage_1)->DenseRange(1, 6)->Unit(benchmark::kMillisecond);

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

    benchmark::RegisterBenchmark(benchmark_name.c_str(),
                                 [i, cores](benchmark::State& state) {
                                   cifar_dense::AppDataBatch appdata(kDefaultInputFile);
                                   for (auto _ : state) {
                                     int num_threads = static_cast<int>(i);
                                     omp::dispatch_multi_stage<PT>(
                                         num_threads, appdata, Stage, Stage);
                                   }
                                 })
        ->Arg(i)
        ->Unit(benchmark::kMillisecond);
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

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

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
