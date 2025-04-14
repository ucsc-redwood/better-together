#include <benchmark/benchmark.h>

#include <memory_resource>

#include "../../app.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

static void BM_Stage1(benchmark::State& state) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // warm up
  tree::omp::run_stage_1(appdata);

  for (auto _ : state) {
    tree::omp::run_stage_1(appdata);
  }
}

BENCHMARK(BM_Stage1)->Unit(benchmark::kMillisecond)->Name("OMP/Tree/Stage1");

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

static void BM_Stage2(benchmark::State& state) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // warm up
  tree::omp::run_stage_2(appdata);

  for (auto _ : state) {
    tree::omp::run_stage_2(appdata);
  }
}

BENCHMARK(BM_Stage2)->Unit(benchmark::kMillisecond)->Name("OMP/Tree/Stage2");

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

static void BM_Stage3(benchmark::State& state) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // warm up
  tree::omp::run_stage_3(appdata);

  for (auto _ : state) {
    tree::omp::run_stage_3(appdata);
  }
}

BENCHMARK(BM_Stage3)->Unit(benchmark::kMillisecond)->Name("OMP/Tree/Stage3");

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

static void BM_Stage4(benchmark::State& state) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // warm up
  tree::omp::run_stage_4(appdata);

  for (auto _ : state) {
    tree::omp::run_stage_4(appdata);
  }
}

BENCHMARK(BM_Stage4)->Unit(benchmark::kMillisecond)->Name("OMP/Tree/Stage4");

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

static void BM_Stage5(benchmark::State& state) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // warm up
  tree::omp::run_stage_5(appdata);

  for (auto _ : state) {
    tree::omp::run_stage_5(appdata);
  }
}

BENCHMARK(BM_Stage5)->Unit(benchmark::kMillisecond)->Name("OMP/Tree/Stage5");

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

static void BM_Stage6(benchmark::State& state) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // warm up
  tree::omp::run_stage_6(appdata);

  for (auto _ : state) {
    tree::omp::run_stage_6(appdata);
  }
}

BENCHMARK(BM_Stage6)->Unit(benchmark::kMillisecond)->Name("OMP/Tree/Stage6");

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

static void BM_Stage7(benchmark::State& state) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // warm up
  tree::omp::run_stage_7(appdata);

  for (auto _ : state) {
    tree::omp::run_stage_7(appdata);
  }
}

BENCHMARK(BM_Stage7)->Unit(benchmark::kMillisecond)->Name("OMP/Tree/Stage7");

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
