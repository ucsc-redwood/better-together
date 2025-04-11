#include <benchmark/benchmark.h>
#include <omp.h>

#include "../sparse_appdata.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"
#include "dispatchers.hpp"
// #include "../omp/dispatch_multi_stage.hpp"

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

void worker_thread(SPSCQueue<Task*, kPoolSize>& in_queue,
                   SPSCQueue<Task*, kPoolSize>& out_queue,
                   ProcessFunction process_function,
                   std::atomic<bool>& running) {
  while (running.load(std::memory_order_relaxed)) {
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

void process_task(Task& task) { spdlog::info("Processing task {}", task.uid); }

static void BM_run_OMP_baseline(benchmark::State& state) {
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

  std::atomic<size_t> tasks_processed{0};

  for (auto _ : state) {
    std::atomic<bool> running{true};

    auto t0 = std::thread(
        worker_thread,
        std::ref(free_task_pool),
        std::ref(q_0),
        [&](Task& task) { disp.dispatch_multi_stage(task.appdata, 1, 9); },
        std::ref(running));

    auto t1 = std::thread(
        worker_thread,
        std::ref(q_0),
        std::ref(free_task_pool),
        [&](Task& task) {
          // busy wait for 20ms
          auto start = std::chrono::high_resolution_clock::now();
          while (std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - start)
                     .count() < 20) {
            // Busy wait
          }
          tasks_processed.fetch_add(1, std::memory_order_relaxed);
        },
        std::ref(running));

    // run for 5 seconds
    std::this_thread::sleep_for(std::chrono::seconds(5));
    running.store(false);

    t0.join();
    t1.join();

    state.SetItemsProcessed(tasks_processed.load());
  }
}

BENCHMARK(BM_run_OMP_baseline)->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(5);

// constexpr size_t kPoolSize = 20;

// void init_task_pool(SPSCQueue<Task*, 32>& free_task_pool, std::pmr::memory_resource* mr) {
//   for (size_t i = 0; i < kPoolSize; ++i) {
//     free_task_pool.enqueue(new Task(mr));
//   }
// }

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup

  // cifar_sparse::vulkan::v2::VulkanDispatcher disp;

  // std::vector<std::unique_ptr<Task>> preallocated_tasks;
  // SPSCQueue<Task*, kPoolSize> free_task_pool;
  // SPSCQueue<Task*, kPoolSize> q_0;
  // SPSCQueue<Task*, kPoolSize> q_1;

  // // Init once
  // for (size_t i = 0; i < kPoolSize; ++i) {
  //   spdlog::info("Init task {}", i);
  //   preallocated_tasks.emplace_back(std::make_unique<Task>(disp.get_mr()));
  //   free_task_pool.enqueue(preallocated_tasks.back().get());
  // }

  // auto t0 = std::thread(worker_thread, std::ref(free_task_pool), std::ref(q_0), [](Task& task) {
  //   std::cout << "[A] Processing task " << task.uid << std::endl;

  //   auto start = std::chrono::high_resolution_clock::now();
  //   while (std::chrono::duration_cast<std::chrono::milliseconds>(
  //              std::chrono::high_resolution_clock::now() - start)
  //              .count() < 20) {
  //     // Busy wait
  //   }
  // });
  // auto t1 = std::thread(worker_thread, std::ref(q_0), std::ref(free_task_pool), [](Task& task) {
  //   std::cout << "[B] Processing task " << task.uid << std::endl;

  //   auto start = std::chrono::high_resolution_clock::now();
  //   while (std::chrono::duration_cast<std::chrono::milliseconds>(
  //              std::chrono::high_resolution_clock::now() - start)
  //              .count() < 20) {
  //     // Busy wait
  //   }
  // });

  // t0.join();
  // t1.join();

  return 0;
}
