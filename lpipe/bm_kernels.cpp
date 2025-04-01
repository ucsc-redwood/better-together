#include <benchmark/benchmark.h>

#include "kernels.hpp"

static void BM_ProcessTaskStageA(benchmark::State& state) {
  Task task = new_task(640 * 480);
  for (auto _ : state) {
    process_task_stage_A(task);
  }
}

BENCHMARK(BM_ProcessTaskStageA)->Unit(benchmark::kMillisecond);

static void BM_ProcessTaskStageB(benchmark::State& state) {
  Task task = new_task(640 * 480);
  for (auto _ : state) {
    process_task_stage_B(task);
  }
}

BENCHMARK(BM_ProcessTaskStageB)->Unit(benchmark::kMillisecond);

static void BM_ProcessTaskStageC(benchmark::State& state) {
  Task task = new_task(640 * 480);
  for (auto _ : state) {
    process_task_stage_C(task);
  }
}

BENCHMARK(BM_ProcessTaskStageC)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
