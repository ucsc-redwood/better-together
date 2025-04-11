#include <benchmark/benchmark.h>
#include <omp.h>

#include "../omp/dispatchers.hpp"
#include "../sparse_appdata.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"
#include "dispatchers.hpp"

struct Task {
  static uint32_t uid_counter;
  uint32_t uid;

  // ----------------------------------
  cifar_sparse::v2::AppData appdata;
  // ----------------------------------

  explicit Task(std::pmr::memory_resource* mr) : appdata(mr) { uid = uid_counter++; }

  void reset() {
    uid = uid_counter++;
    appdata.reset();
  }
};

uint32_t Task::uid_counter = 0;

using ProcessFunction = std::function<void(Task&)>;

constexpr size_t kPoolSize = 32;
constexpr size_t kNumToProcess = 100;

void worker_thread_normal(SPSCQueue<Task*, kPoolSize>& in_queue,
                          SPSCQueue<Task*, kPoolSize>& out_queue,
                          ProcessFunction process_function) {
  for (size_t _ = 0; _ < kNumToProcess; ++_) {
    Task* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    // Process the task (timing is handled in the provided process function)
    process_function(*task);

    // Forward processed task
    while (!out_queue.enqueue(std::move(task))) {
      std::this_thread::yield();
    }
  }
}

// ----------------------------------------------------------------------------
// Vk Baseline
// ----------------------------------------------------------------------------

static void BM_run_VK_baseline(benchmark::State& state) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;

  std::vector<std::unique_ptr<Task>> preallocated_tasks;
  SPSCQueue<Task*, kPoolSize> free_task_pool;
  SPSCQueue<Task*, kPoolSize> q_0;
  SPSCQueue<Task*, kPoolSize> q_1;

  // Init once
  for (size_t i = 0; i < kPoolSize; ++i) {
    spdlog::info("Init task {}", i);
    preallocated_tasks.emplace_back(std::make_unique<Task>(disp.get_mr()));
    free_task_pool.enqueue(preallocated_tasks.back().get());
  }

  for (auto _ : state) {
    auto t0 =
        std::thread(worker_thread_normal, std::ref(free_task_pool), std::ref(q_0), [&](Task& task) {
          disp.dispatch_multi_stage(task.appdata, 1, 9);
        });

    auto t1 =
        std::thread(worker_thread_normal, std::ref(q_0), std::ref(free_task_pool), [&](Task& _) {
          // busy wait for 20ms
          auto start = std::chrono::high_resolution_clock::now();
          while (std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - start)
                     .count() < 10) {
            // Busy wait
          }
        });

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
  const int stage = state.range(0);

  // Ensure stage is valid
  if (stage < 1 || stage > 9) {
    state.SkipWithError("Invalid stage number");
    return;
  }

  cifar_sparse::vulkan::v2::VulkanDispatcher disp;

  std::vector<std::unique_ptr<Task>> preallocated_tasks;
  SPSCQueue<Task*, kPoolSize> free_task_pool;
  SPSCQueue<Task*, kPoolSize> q_0;
  SPSCQueue<Task*, kPoolSize> q_1;

  // Init once
  for (size_t i = 0; i < kPoolSize; ++i) {
    spdlog::info("Init task {}", i);
    preallocated_tasks.emplace_back(std::make_unique<Task>(disp.get_mr()));
    free_task_pool.enqueue(preallocated_tasks.back().get());
  }

  for (auto _ : state) {
    auto t0 =
        std::thread(worker_thread_normal, std::ref(free_task_pool), std::ref(q_0), [&](Task& task) {
          disp.dispatch_multi_stage(task.appdata, stage, stage);
        });

    auto t1 =
        std::thread(worker_thread_normal, std::ref(q_0), std::ref(free_task_pool), [&](Task& _) {
          // // busy wait for 20ms
          // auto start = std::chrono::high_resolution_clock::now();
          // while (std::chrono::duration_cast<std::chrono::milliseconds>(
          //            std::chrono::high_resolution_clock::now() - start)
          //            .count() < 10) {
          //   // Busy wait
          // }
        });

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

  cifar_sparse::vulkan::v2::VulkanDispatcher disp;

  std::vector<std::unique_ptr<Task>> preallocated_tasks;
  SPSCQueue<Task*, kPoolSize> free_task_pool;
  SPSCQueue<Task*, kPoolSize> q_0;
  SPSCQueue<Task*, kPoolSize> q_1;

  // Init once
  for (size_t i = 0; i < kPoolSize; ++i) {
    spdlog::info("Init task {}", i);
    preallocated_tasks.emplace_back(std::make_unique<Task>(disp.get_mr()));
    free_task_pool.enqueue(preallocated_tasks.back().get());
  }

  for (auto _ : state) {
    auto t0 =
        std::thread(worker_thread_normal, std::ref(free_task_pool), std::ref(q_0), [&](Task& task) {
          // disp.dispatch_multi_stage(task.appdata, stage, stage);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              cores_to_use, num_threads, task.appdata, stage, stage);
        });

    auto t1 =
        std::thread(worker_thread_normal, std::ref(q_0), std::ref(free_task_pool), [&](Task& _) {
          // // busy wait for 20ms
          // auto start = std::chrono::high_resolution_clock::now();
          // while (std::chrono::duration_cast<std::chrono::milliseconds>(
          //            std::chrono::high_resolution_clock::now() - start)
          //            .count() < 10) {
          //   // Busy wait
          // }
        });

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
