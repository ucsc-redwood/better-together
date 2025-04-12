#include <benchmark/benchmark.h>
#include <omp.h>

#include "../omp/dispatchers.hpp"
#include "../sparse_appdata.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "builtin-apps/pipeline/worker.hpp"
#include "dispatchers.hpp"

using MyTask = Task<cifar_sparse::v2::AppData>;

#define SETUP_CORES_AND_TASKS()                                               \
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;                            \
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;                    \
  SPSCQueue<MyTask*, kPoolSize> free_task_pool;                               \
  for (size_t i = 0; i < kPoolSize; ++i) {                                    \
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr())); \
    free_task_pool.enqueue(preallocated_tasks.back().get());                  \
  }

// ----------------------------------------------------------------------------
// Vk Baseline
// ----------------------------------------------------------------------------

static void BM_run_VK_baseline(benchmark::State& state) {
  SETUP_CORES_AND_TASKS();

  SPSCQueue<MyTask*, kPoolSize> q_0;
  SPSCQueue<MyTask*, kPoolSize> q_1;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread<MyTask>, std::ref(free_task_pool), std::ref(q_0), [&](MyTask& task) {
          disp.dispatch_multi_stage(task.appdata, 1, 9);
        });

    auto t1 = std::thread(
        worker_thread<MyTask>, std::ref(q_0), std::ref(free_task_pool), [&](MyTask& _) {});

    t0.join();
    t1.join();
  }
}

BENCHMARK(BM_run_VK_baseline)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5)
    ->Name("VK/CifarSparse/Baseline");

// ----------------------------------------------------------------------------
// Vk Stage
// ----------------------------------------------------------------------------

static void BM_run_VK_stage(benchmark::State& state) {
  SETUP_CORES_AND_TASKS();

  const int stage = state.range(0);

  if (stage < 1 || stage > 9) {
    state.SkipWithError("Invalid stage number");
    return;
  }

  SPSCQueue<MyTask*, kPoolSize> q_0;
  SPSCQueue<MyTask*, kPoolSize> q_1;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread<MyTask>, std::ref(free_task_pool), std::ref(q_0), [&](MyTask& task) {
          disp.dispatch_multi_stage(task.appdata, stage, stage);
        });

    auto t1 = std::thread(
        worker_thread<MyTask>, std::ref(q_0), std::ref(free_task_pool), [&](MyTask& _) {});

    t0.join();
    t1.join();
  }
}

BENCHMARK(BM_run_VK_stage)
    ->Unit(benchmark::kMillisecond)
    ->DenseRange(1, 9)
    ->Iterations(1)
    ->Repetitions(5)
    ->Name("VK/CifarSparse/Stage");

// ----------------------------------------------------------------------------
// Omp Stage
// ----------------------------------------------------------------------------

static void BM_run_OMP_stage(benchmark::State& state) {
  const int stage = state.range(0);
  const ProcessorType core_type = static_cast<ProcessorType>(state.range(1));

  std::vector<int> cores_to_use;
  int num_threads;

  if (core_type == ProcessorType::kLittleCore) {
    cores_to_use = g_little_cores;
    num_threads = g_little_cores.size();
  } else if (core_type == ProcessorType::kMediumCore) {
    cores_to_use = g_medium_cores;
    num_threads = g_medium_cores.size();
  } else {
    cores_to_use = g_big_cores;
    num_threads = g_big_cores.size();
  }

  if (num_threads > (static_cast<int>(cores_to_use.size()))) {
    state.SkipWithMessage("");
    return;
  }

  // Ensure stage is valid
  if (stage < 1 || stage > 9) {
    state.SkipWithError("Invalid stage number");
    return;
  }

  SETUP_CORES_AND_TASKS();

  SPSCQueue<MyTask*, kPoolSize> q_0;
  SPSCQueue<MyTask*, kPoolSize> q_1;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread<MyTask>, std::ref(free_task_pool), std::ref(q_0), [&](MyTask& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              cores_to_use, num_threads, task.appdata, stage, stage);
        });

    auto t1 = std::thread(
        worker_thread<MyTask>, std::ref(q_0), std::ref(free_task_pool), [&](MyTask& _) {});

    t0.join();
    t1.join();
  }
}

static void CustomArgs(benchmark::internal::Benchmark* b) {
  for (int stage = 1; stage <= 9; ++stage) {
    for (const ProcessorType core_type :
         {ProcessorType::kLittleCore, ProcessorType::kMediumCore, ProcessorType::kBigCore}) {
      b->Args({stage, static_cast<int>(core_type)});
    }
  }
}

BENCHMARK(BM_run_OMP_stage)
    ->Apply(CustomArgs)
    ->Iterations(1)
    ->Repetitions(5)
    ->Unit(benchmark::kMillisecond)
    ->Name("OMP/CifarSparse/Stage");

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup

  return 0;
}
