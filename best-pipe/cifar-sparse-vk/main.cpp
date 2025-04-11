#include <benchmark/benchmark.h>
#include <omp.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"

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
// Omp Stage
// ----------------------------------------------------------------------------

// static void BM_run_OMP_stage(benchmark::State& state) {
//   const ProcessorType core_type = static_cast<ProcessorType>(state.range(1));

//   std::vector<int> cores_to_use;
//   int num_threads;

//   if (core_type == ProcessorType::kLittleCore) {
//     cores_to_use = g_little_cores;
//     num_threads = g_little_cores.size();
//   } else if (core_type == ProcessorType::kMediumCore) {
//     cores_to_use = g_medium_cores;
//     num_threads = g_medium_cores.size();
//   } else {
//     cores_to_use = g_big_cores;
//     num_threads = g_big_cores.size();
//   }

//   if (num_threads > (static_cast<int>(cores_to_use.size()))) {
//     state.SkipWithMessage("");
//     return;
//   }

//   cifar_sparse::vulkan::v2::VulkanDispatcher disp;

//   std::vector<std::unique_ptr<Task>> preallocated_tasks;
//   SPSCQueue<Task*, kPoolSize> free_task_pool;

//   SPSCQueue<Task*, kPoolSize> q_0_1;
//   SPSCQueue<Task*, kPoolSize> q_1_2;
//   SPSCQueue<Task*, kPoolSize> q_2_3;

//   // Init once
//   for (size_t i = 0; i < kPoolSize; ++i) {
//     spdlog::info("Init task {}", i);
//     preallocated_tasks.emplace_back(std::make_unique<Task>(disp.get_mr()));
//     free_task_pool.enqueue(preallocated_tasks.back().get());
//   }

//   // Schedule 1:
//   //   Chunk 1: Stages [1] → medium (2 threads)
//   //   Chunk 2: Stages [2, 3, 4] → gpu_vulkan (1 thread)
//   //   Chunk 3: Stages [5] → little (4 threads)
//   //   Chunk 4: Stages [6, 7, 8, 9] → big (2 threads)

//   //   Chunk times: ['607.00', '1101.00', '737.00', '815.70']
//   //   Max chunk time: 1101.00
//   //   GPU speedup: 1.77x
//   //   Load balance ratio: 0.55
//   //   Load imbalance: 35.06%
//   //   Time variance: 32786.0919

//   for (auto _ : state) {
//     auto t0 = std::thread(
//         worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(
//               cores_to_use, num_threads, task.appdata, 1, 1);
//         });

//     auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task)
//     {
//       disp.dispatch_multi_stage(task.appdata, 2, 4);
//     });

//     auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task)
//     {
//       cifar_sparse::omp::v2::dispatch_multi_stage(cores_to_use, num_threads, task.appdata, 5, 5);
//     });

//     auto t3 = std::thread(
//         worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(
//               cores_to_use, num_threads, task.appdata, 6, 9);
//         });

//     t0.join();
//     t1.join();
//     t2.join();
//     t3.join();
//   }
// }

// BENCHMARK(BM_run_OMP_stage)->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(5);

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != "3A021JEHN02756") {
    return 0;
  }

  //   benchmark::Initialize(&argc, argv);
  //   benchmark::RunSpecifiedBenchmarks();
  //   benchmark::Shutdown();  // Ensure proper cleanup

  cifar_sparse::vulkan::v2::VulkanDispatcher disp;

  std::vector<std::unique_ptr<Task>> preallocated_tasks;
  SPSCQueue<Task*, kPoolSize> free_task_pool;

  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  // Init once
  for (size_t i = 0; i < kPoolSize; ++i) {
    spdlog::info("Init task {}", i);
    preallocated_tasks.emplace_back(std::make_unique<Task>(disp.get_mr()));
    free_task_pool.enqueue(preallocated_tasks.back().get());
  }

  // Schedule 1:
  //   Chunk 1: Stages [1] → medium (2 threads)
  //   Chunk 2: Stages [2, 3, 4] → gpu_vulkan (1 thread)
  //   Chunk 3: Stages [5] → little (4 threads)
  //   Chunk 4: Stages [6, 7, 8, 9] → big (2 threads)

  //   Chunk times: ['607.00', '1101.00', '737.00', '815.70']
  //   Max chunk time: 1101.00
  //   GPU speedup: 1.77x
  //   Load balance ratio: 0.55
  //   Load imbalance: 35.06%
  //   Time variance: 32786.0919

  auto start_time = std::chrono::high_resolution_clock::now();

  auto t0 =
      std::thread(worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
        cifar_sparse::omp::v2::dispatch_multi_stage(
            g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
      });

  auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
    disp.dispatch_multi_stage(task.appdata, 2, 4);
  });

  auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task) {
    cifar_sparse::omp::v2::dispatch_multi_stage(
        g_little_cores, g_little_cores.size(), task.appdata, 5, 5);
  });

  auto t3 =
      std::thread(worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
        cifar_sparse::omp::v2::dispatch_multi_stage(
            g_big_cores, g_big_cores.size(), task.appdata, 6, 9);
      });

  t0.join();
  t1.join();
  t2.join();
  t3.join();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  spdlog::info("Duration: {} ms", duration.count());

  return 0;
}
