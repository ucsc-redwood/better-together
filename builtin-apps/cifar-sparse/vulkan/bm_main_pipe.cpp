#include <benchmark/benchmark.h>
#include <omp.h>

#include "../omp/dispatchers.hpp"
#include "../sparse_appdata.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "builtin-apps/pipeline/worker.hpp"
#include "dispatchers.hpp"

// // ----------------------------------------------------------------------------
// // Realistic Benchmark: Vk Baseline
// // ----------------------------------------------------------------------------

// static void BM_run_VK_baseline(benchmark::State& state) {
//   cifar_sparse::vulkan::v2::VulkanDispatcher disp;
//   cifar_sparse::v2::AppData appdata(disp.get_mr());

//   // Warm up the data structures to avoid first-run anomalies
//   disp.dispatch_multi_stage(appdata, 1, 9);

//   for (auto _ : state) {
//     disp.dispatch_multi_stage(appdata, 1, 9);
//   }
// }

// BENCHMARK(BM_run_VK_baseline)->Unit(benchmark::kMillisecond)->Name("VK/CifarSparse/Baseline");

// // ----------------------------------------------------------------------------
// // Realistic Benchmark: Vk Stage
// // ----------------------------------------------------------------------------

// static void BM_run_VK_stage(benchmark::State& state) {
//   const int stage = state.range(0);

//   cifar_sparse::vulkan::v2::VulkanDispatcher disp;
//   cifar_sparse::v2::AppData appdata(disp.get_mr());

//   // Ensure stage is valid
//   if (stage < 1 || stage > 9) {
//     state.SkipWithError("Invalid stage number");
//     return;
//   }

//   // Warm up to prevent first-run anomalies
//   disp.dispatch_multi_stage(appdata, 1, stage);

//   for (auto _ : state) {
//     disp.dispatch_multi_stage(appdata, stage, stage);
//   }
// }

// BENCHMARK(BM_run_VK_stage)
//     ->DenseRange(1, 9)
//     ->Unit(benchmark::kMillisecond)
//     ->Name("VK/CifarSparse/Stage");

// ----------------------------------------------------------------------------
// Realistic Benchmark: OMP Stage
// ----------------------------------------------------------------------------

using MyTask = Task<cifar_sparse::v2::AppData>;

static void BM_run_OMP_stage_without_full(benchmark::State& state) {
  const int stage = state.range(0);
  const ProcessorType core_type = static_cast<ProcessorType>(state.range(1));

  auto mr = std::pmr::new_delete_resource();
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;
  SPSCQueue<MyTask*, kPoolSize> free_task_pool;
  for (size_t i = 0; i < kPoolSize; ++i) {
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(mr));
    free_task_pool.enqueue(preallocated_tasks.back().get());
  }

  std::vector<int> cores_to_use;
  if (core_type == ProcessorType::kLittleCore) {
    cores_to_use = g_little_cores;
    state.SetLabel("Little, " + std::to_string(cores_to_use.size()) + " cores");
  } else if (core_type == ProcessorType::kMediumCore) {
    cores_to_use = g_medium_cores;
    state.SetLabel("Medium, " + std::to_string(cores_to_use.size()) + " cores");
  } else {
    cores_to_use = g_big_cores;
    state.SetLabel("Big, " + std::to_string(cores_to_use.size()) + " cores");
  }

  const int num_threads = cores_to_use.size();

  // Ensure stage is valid
  if (stage < 1 || stage > 9) {
    state.SkipWithError("Invalid stage number");
    return;
  }

  for (auto _ : state) {
    auto t0 = std::thread(worker_thread<MyTask>,
                          std::ref(free_task_pool),
                          std::ref(free_task_pool),
                          [&](MyTask& task) {
                            cifar_sparse::omp::v2::dispatch_multi_stage(
                                cores_to_use, num_threads, task.appdata, stage, stage);
                          });

    t0.join();
  }
}

using MyTask = Task<cifar_sparse::v2::AppData>;

static void BM_run_OMP_stage_full(benchmark::State& state) {
  const int stage = state.range(0);
  const ProcessorType core_type = static_cast<ProcessorType>(state.range(1));

  auto mr = std::pmr::new_delete_resource();
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;
  SPSCQueue<MyTask*, kPoolSize> free_task_pool;
  for (size_t i = 0; i < kPoolSize; ++i) {
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(mr));
    free_task_pool.enqueue(preallocated_tasks.back().get());
  }

  // The cores we want to measure

  std::vector<int> cores_to_use;
  if (core_type == ProcessorType::kLittleCore) {
    cores_to_use = g_little_cores;
    state.SetLabel("Little, " + std::to_string(cores_to_use.size()) + " cores");
  } else if (core_type == ProcessorType::kMediumCore) {
    cores_to_use = g_medium_cores;
    state.SetLabel("Medium, " + std::to_string(cores_to_use.size()) + " cores");
  } else {
    cores_to_use = g_big_cores;
    state.SetLabel("Big, " + std::to_string(cores_to_use.size()) + " cores");
  }

  const int num_threads = cores_to_use.size();

  // find the counter part of the cores

  std::vector<int> cores_to_use_counter_part_a;
  std::vector<int> cores_to_use_counter_part_b;
  int num_threads_counter_part_a;
  int num_threads_counter_part_b;

  if (core_type == ProcessorType::kLittleCore) {
    cores_to_use_counter_part_a = g_medium_cores;
    cores_to_use_counter_part_b = g_big_cores;
    num_threads_counter_part_a = cores_to_use_counter_part_a.size();
    num_threads_counter_part_b = cores_to_use_counter_part_b.size();
  } else if (core_type == ProcessorType::kMediumCore) {
    cores_to_use_counter_part_a = g_little_cores;
    cores_to_use_counter_part_b = g_big_cores;
    num_threads_counter_part_a = cores_to_use_counter_part_a.size();
    num_threads_counter_part_b = cores_to_use_counter_part_b.size();
  } else {
    cores_to_use_counter_part_a = g_little_cores;
    cores_to_use_counter_part_b = g_medium_cores;
    num_threads_counter_part_a = cores_to_use_counter_part_a.size();
    num_threads_counter_part_b = cores_to_use_counter_part_b.size();
  }

  SPSCQueue<MyTask*, kPoolSize> q_0_1;
  SPSCQueue<MyTask*, kPoolSize> q_1_2;

  // Ensure stage is valid
  if (stage < 1 || stage > 9) {
    state.SkipWithError("Invalid stage number");
    return;
  }

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread<MyTask>, std::ref(free_task_pool), std::ref(q_0_1), [&](MyTask& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              cores_to_use, num_threads, task.appdata, stage, stage);
        });

    auto t1 =
        std::thread(worker_thread<MyTask>, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              cores_to_use_counter_part_a, num_threads_counter_part_a, task.appdata, stage, stage);
        });

    auto t2 = std::thread(
        worker_thread<MyTask>, std::ref(q_1_2), std::ref(free_task_pool), [&](MyTask& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              cores_to_use_counter_part_b, num_threads_counter_part_b, task.appdata, stage, stage);
        });

    t0.join();
    t1.join();
    t2.join();
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

BENCHMARK(BM_run_OMP_stage_without_full)
    ->Apply(CustomArgs)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5)
    ->Name("OMP/CifarSparse/Stage/Non-Full");

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup
  return 0;
}
