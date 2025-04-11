#include <benchmark/benchmark.h>
#include <omp.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/config_reader.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"

struct Record {
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
};

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

  std::array<Record, 4> records;

  void start_record(const int chunk_id) {
    records[chunk_id].start_time = std::chrono::high_resolution_clock::now();
  }

  void end_record(const int chunk_id) {
    records[chunk_id].end_time = std::chrono::high_resolution_clock::now();
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

// // ----------------------------------------------------------------------------
// // Omp Stage
// // ----------------------------------------------------------------------------

#define SETUP_CORES_AND_TASKS(state)                                        \
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;                          \
  std::vector<std::unique_ptr<Task>> preallocated_tasks;                    \
  SPSCQueue<Task*, kPoolSize> free_task_pool;                               \
  for (size_t i = 0; i < kPoolSize; ++i) {                                  \
    preallocated_tasks.emplace_back(std::make_unique<Task>(disp.get_mr())); \
    free_task_pool.enqueue(preallocated_tasks.back().get());                \
  }

// ----------------------------------------------------------------------------
// Specialized Schedules
// ----------------------------------------------------------------------------

void dump_records(SPSCQueue<Task*, kPoolSize>& free_task_pool) {
  // Print the records, pop all from free_task_pool

  while (!free_task_pool.empty()) {
    Task* task = nullptr;
    while (!free_task_pool.dequeue(task)) {
      std::this_thread::yield();
    }

    std::cout << "Task " << task->uid << ":\n";
    for (size_t i = 0; i < 4; ++i) {
      std::cout << "  Chunk " << i << ":\n";
      std::cout << "    Start: "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       task->records[i].start_time.time_since_epoch())
                       .count()
                << " us\n";
      std::cout << "    End: "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       task->records[i].end_time.time_since_epoch())
                       .count()
                << " us\n";
      std::cout << "    Duration: "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       task->records[i].end_time - task->records[i].start_time)
                       .count()
                << " us\n";
    }
  }
}

// Schedule best:
//   Chunk 1: Stages [1, 2, 3, 4] → gpu_vulkan
//   Chunk 2: Stage [5] → medium (g_medium_cores)
//   Chunk 3: Stages [6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_best(benchmark::State& state) {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          task.start_record(0);
          disp.dispatch_multi_stage(task.appdata, 1, 4);
          task.end_record(0);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      task.start_record(1);
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_medium_cores, g_medium_cores.size(), task.appdata, 5, 5);
      task.end_record(1);
    });

    auto t2 = std::thread(
        worker_thread_normal, std::ref(q_1_2), std::ref(free_task_pool), [&](Task& task) {
          task.start_record(2);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 6, 9);
          task.end_record(2);
        });

    t0.join();
    t1.join();
    t2.join();
  }

  // Print the records, pop all from free_task_pool

  // while (!free_task_pool.empty()) {
  //   Task* task = nullptr;
  //   while (!free_task_pool.dequeue(task)) {
  //     std::this_thread::yield();
  //   }

  //   std::cout << "Task " << task->uid << ":\n";
  //   for (size_t i = 0; i < 4; ++i) {
  //     std::cout << "  Chunk " << i << ":\n";
  //     std::cout << "    Start: "
  //               << std::chrono::duration_cast<std::chrono::microseconds>(
  //                      task->records[i].start_time.time_since_epoch())
  //                      .count()
  //               << " us\n";
  //     std::cout << "    End: "
  //               << std::chrono::duration_cast<std::chrono::microseconds>(
  //                      task->records[i].end_time.time_since_epoch())
  //                      .count()
  //               << " us\n";
  //     std::cout << "    Duration: "
  //               << std::chrono::duration_cast<std::chrono::microseconds>(
  //                      task->records[i].end_time - task->records[i].start_time)
  //                      .count()
  //               << " us\n";
  //   }
  // }

  // for (size_t i = 0; i < kPoolSize; ++i) {
  //   std::cout << "Task " << i << ":\n";
  //   for (size_t j = 0; j < 4; ++j) {
  //     std::cout << "  Chunk " << j << ": " <<
  //     std::chrono::duration_cast<std::chrono::microseconds>(records[j].end_time -
  //     records[j].start_time).count() << " us\n";
  //   }
  // }
}
BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_best)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// Schedule 1:
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stage [5] → little (g_little_cores)
//   Chunk 4: Stages [6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_1(benchmark::State& state) {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          task.start_record(0);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
          task.end_record(0);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      task.start_record(1);
      disp.dispatch_multi_stage(task.appdata, 2, 4);
      task.end_record(1);
    });

    auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task) {
      task.start_record(2);
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_little_cores, g_little_cores.size(), task.appdata, 5, 5);
      task.end_record(2);
    });

    auto t3 = std::thread(
        worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
          task.start_record(3);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 6, 9);
          task.end_record(3);
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  // Print the records, pop all from free_task_pool

  while (!free_task_pool.empty()) {
    Task* task = nullptr;
    while (!free_task_pool.dequeue(task)) {
      std::this_thread::yield();
    }

    std::cout << "Task " << task->uid << ":\n";
    for (size_t i = 0; i < 4; ++i) {
      std::cout << "  Chunk " << i << ":\n";
      std::cout << "    Start: "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       task->records[i].start_time.time_since_epoch())
                       .count()
                << " us\n";
      std::cout << "    End: "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       task->records[i].end_time.time_since_epoch())
                       .count()
                << " us\n";
      std::cout << "    Duration: "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       task->records[i].end_time - task->records[i].start_time)
                       .count()
                << " us\n";
    }
  }
}
BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// Schedule 2:
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stage [5] → little (g_little_cores)
//   Chunk 4: Stages [6, 7, 8, 9] → medium (g_medium_cores)
static void BM_pipe_cifar_sparse_vk_schedule_2(benchmark::State& state) {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          task.start_record(0);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
          task.end_record(0);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      task.start_record(1);
      disp.dispatch_multi_stage(task.appdata, 2, 4);
      task.end_record(1);
    });

    auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task) {
      task.start_record(2);
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_little_cores, g_little_cores.size(), task.appdata, 5, 5);
      task.end_record(2);
    });

    auto t3 = std::thread(
        worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
          task.start_record(3);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 6, 9);
          task.end_record(3);
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  dump_records(free_task_pool);
}
BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// Schedule 3:
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7] → big (g_big_cores)
//   Chunk 4: Stages [8, 9] → little (g_little_cores)
static void BM_pipe_cifar_sparse_vk_schedule_3(benchmark::State& state) {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          task.start_record(0);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
          task.end_record(0);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      task.start_record(1);
      disp.dispatch_multi_stage(task.appdata, 2, 4);
      task.end_record(1);
    });

    auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task) {
      task.start_record(2);
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_big_cores, g_big_cores.size(), task.appdata, 5, 7);
      task.end_record(2);
    });

    auto t3 = std::thread(
        worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
          task.start_record(3);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_little_cores, g_little_cores.size(), task.appdata, 8, 9);
          task.end_record(3);
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  dump_records(free_task_pool);
}
BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// Schedule 4:
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7] → medium (g_medium_cores)
//   Chunk 4: Stages [8, 9] → little (g_little_cores)
static void BM_pipe_cifar_sparse_vk_schedule_4(benchmark::State& state) {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          task.start_record(0);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
          task.end_record(0);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      task.start_record(1);
      disp.dispatch_multi_stage(task.appdata, 2, 4);
      task.end_record(1);
    });

    auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task) {
      task.start_record(2);
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_medium_cores, g_medium_cores.size(), task.appdata, 5, 7);
      task.end_record(2);
    });

    auto t3 = std::thread(
        worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
          task.start_record(3);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_little_cores, g_little_cores.size(), task.appdata, 8, 9);
          task.end_record(3);
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  dump_records(free_task_pool);
}
BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_4)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// Schedule 5:
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8] → big (g_big_cores)
//   Chunk 4: Stage [9] → little (g_little_cores)
static void BM_pipe_cifar_sparse_vk_schedule_5(benchmark::State& state) {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          task.start_record(0);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
          task.end_record(0);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      task.start_record(1);
      disp.dispatch_multi_stage(task.appdata, 2, 4);
      task.end_record(1);
    });

    auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task) {
      task.start_record(2);
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_big_cores, g_big_cores.size(), task.appdata, 5, 8);
      task.end_record(2);
    });

    auto t3 = std::thread(
        worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
          task.start_record(3);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_little_cores, g_little_cores.size(), task.appdata, 9, 9);
          task.end_record(3);
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  dump_records(free_task_pool);
}
BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_5)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// Schedule 6:
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8] → medium (g_medium_cores)
//   Chunk 4: Stage [9] → little (g_little_cores)
static void BM_pipe_cifar_sparse_vk_schedule_6(benchmark::State& state) {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          task.start_record(0);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
          task.end_record(0);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      task.start_record(1);
      disp.dispatch_multi_stage(task.appdata, 2, 4);
      task.end_record(1);
    });

    auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task) {
      task.start_record(2);
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_medium_cores, g_medium_cores.size(), task.appdata, 5, 8);
      task.end_record(2);
    });

    auto t3 = std::thread(
        worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
          task.start_record(3);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_little_cores, g_little_cores.size(), task.appdata, 9, 9);
          task.end_record(3);
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  dump_records(free_task_pool);
}
BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_6)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// Schedule 7 (3 chunks):
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_7(benchmark::State& state) {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          task.start_record(0);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
          task.end_record(0);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      task.start_record(1);
      disp.dispatch_multi_stage(task.appdata, 2, 4);
      task.end_record(1);
    });

    auto t2 = std::thread(
        worker_thread_normal, std::ref(q_1_2), std::ref(free_task_pool), [&](Task& task) {
          task.start_record(2);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 5, 9);
          task.end_record(2);
        });

    t0.join();
    t1.join();
    t2.join();
  }

  dump_records(free_task_pool);
}
BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_7)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// Schedule 8 (3 chunks):
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8, 9] → medium (g_medium_cores)
static void BM_pipe_cifar_sparse_vk_schedule_8(benchmark::State& state) {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          task.start_record(0);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
          task.end_record(0);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      task.start_record(1);
      disp.dispatch_multi_stage(task.appdata, 2, 4);
      task.end_record(1);
    });

    auto t2 = std::thread(
        worker_thread_normal, std::ref(q_1_2), std::ref(free_task_pool), [&](Task& task) {
          task.start_record(2);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 5, 9);
          task.end_record(2);
        });

    t0.join();
    t1.join();
    t2.join();
  }

  dump_records(free_task_pool);
}
BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_8)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// Schedule 9 (3 chunks, same as Schedule 7):
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_9(benchmark::State& state) {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          task.start_record(0);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
          task.end_record(0);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      task.start_record(1);
      disp.dispatch_multi_stage(task.appdata, 2, 4);
      task.end_record(1);
    });

    auto t2 = std::thread(
        worker_thread_normal, std::ref(q_1_2), std::ref(free_task_pool), [&](Task& task) {
          task.start_record(2);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 5, 9);
          task.end_record(2);
        });

    t0.join();
    t1.join();
    t2.join();
  }

  dump_records(free_task_pool);
}
BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_9)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// Schedule 10 (3 chunks, same as Schedule 8):
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8, 9] → medium (g_medium_cores)
static void BM_pipe_cifar_sparse_vk_schedule_10(benchmark::State& state) {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  for (auto _ : state) {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          task.start_record(0);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
          task.end_record(0);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      task.start_record(1);
      disp.dispatch_multi_stage(task.appdata, 2, 4);
      task.end_record(1);
    });

    auto t2 = std::thread(
        worker_thread_normal, std::ref(q_1_2), std::ref(free_task_pool), [&](Task& task) {
          task.start_record(2);
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 5, 9);
          task.end_record(2);
        });

    t0.join();
    t1.join();
    t2.join();
  }

  dump_records(free_task_pool);
}
BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_10)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != "3A021JEHN02756") {
    return 0;
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup

  return 0;
}
