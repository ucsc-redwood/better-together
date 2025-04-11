#include <omp.h>

#include <deque>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/config_reader.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"

struct Record {
  ProcessorType processed_by;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
};

// A global manager to store execution records for all tasks
struct TaskRecords {
  // Store all records in sequence for simpler lookup
  std::vector<std::array<Record, 4>> records;

  void ensure_capacity(size_t count) {
    if (records.size() < count) {
      records.resize(count);
    }
  }

  void store_record(uint32_t task_id, int chunk_id, const Record& record) {
    if (task_id >= records.size()) {
      records.resize(task_id + 1);
    }
    records[task_id][chunk_id] = record;
  }

  void clear() { records.clear(); }

  void dump_all() {
    std::cout << "Total tasks processed: " << records.size() << std::endl;

    for (size_t task_id = 0; task_id < records.size(); task_id++) {
      std::cout << "Task " << task_id << ":\n";
      for (size_t chunk_id = 0; chunk_id < 4; ++chunk_id) {
        const auto& record = records[task_id][chunk_id];
        std::cout << "  Chunk " << chunk_id << ":\n";

        // Only print if the record has timing data
        if (record.start_time.time_since_epoch().count() > 0) {
          std::cout << "    Start: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(
                           record.start_time.time_since_epoch())
                           .count()
                    << " us\n";
          std::cout << "    End: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(
                           record.end_time.time_since_epoch())
                           .count()
                    << " us\n";
          std::cout << "    Duration: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(record.end_time -
                                                                             record.start_time)
                           .count()
                    << " us\n";
        } else {
          std::cout << "    (No data for this chunk)\n";
        }
      }
    }
  }
};

// Global task records
TaskRecords g_task_records;

struct Task {
  uint32_t task_id;  // Unique ID for this task, assigned sequentially 0...N

  // ----------------------------------
  cifar_sparse::v2::AppData appdata;
  // ----------------------------------

  explicit Task(std::pmr::memory_resource* mr, uint32_t id) : task_id(id), appdata(mr) {}

  void reset() { appdata.reset(); }

  std::array<Record, 4> records;

  void set_processed_by(const int chunk_id, const ProcessorType processor_type) {
    records[chunk_id].processed_by = processor_type;
  }

  void start_record(const int chunk_id) {
    records[chunk_id].start_time = std::chrono::high_resolution_clock::now();
  }

  void end_record(const int chunk_id) {
    records[chunk_id].end_time = std::chrono::high_resolution_clock::now();
    // Store in global record storage
    g_task_records.store_record(task_id, chunk_id, records[chunk_id]);
  }
};

using ProcessFunction = std::function<void(Task&)>;

constexpr size_t kPoolSize = 32;
constexpr size_t kNumToProcess = 100;

// For tracking task execution - shared by all worker threads
struct TaskExecutionTracker {
  std::atomic<size_t> next_task_id{0};

  // Get the next task ID to process
  size_t get_next_task_id() { return next_task_id.fetch_add(1); }

  void reset() { next_task_id.store(0); }
};

TaskExecutionTracker g_tracker;

void worker_thread_normal(SPSCQueue<Task*, kPoolSize>& in_queue,
                          SPSCQueue<Task*, kPoolSize>& out_queue,
                          ProcessFunction process_function) {
  for (size_t _ = 0; _ < kNumToProcess; ++_) {
    Task* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    // Assign this execution a unique task ID
    uint32_t current_task_id = g_tracker.get_next_task_id();
    task->task_id = current_task_id;

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

#define SETUP_CORES_AND_TASKS(state)                                           \
  g_task_records.clear();                                                      \
  g_task_records.ensure_capacity(kNumToProcess);                               \
  g_tracker.reset();                                                           \
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;                             \
  std::vector<std::unique_ptr<Task>> preallocated_tasks;                       \
  SPSCQueue<Task*, kPoolSize> free_task_pool;                                  \
  for (size_t i = 0; i < kPoolSize; ++i) {                                     \
    preallocated_tasks.emplace_back(std::make_unique<Task>(disp.get_mr(), i)); \
    free_task_pool.enqueue(preallocated_tasks.back().get());                   \
  }

// ----------------------------------------------------------------------------
// Specialized Schedules
// ----------------------------------------------------------------------------

void dump_records(SPSCQueue<Task*, kPoolSize>& free_task_pool) {
  // Dump all collected records
  g_task_records.dump_all();

  // Empty the free_task_pool without processing
  Task* task = nullptr;
  while (free_task_pool.dequeue(task)) {
    // Just drain the queue
  }
}

// Schedule best:
//   Chunk 1: Stages [1, 2, 3, 4] → gpu_vulkan
//   Chunk 2: Stage [5] → medium (g_medium_cores)
//   Chunk 3: Stages [6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_best() {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  {
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

  dump_records(free_task_pool);
}

// Schedule 1:
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stage [5] → little (g_little_cores)
//   Chunk 4: Stages [6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_1() {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  {
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
  dump_records(free_task_pool);
}

// Schedule 2:
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stage [5] → little (g_little_cores)
//   Chunk 4: Stages [6, 7, 8, 9] → medium (g_medium_cores)
static void BM_pipe_cifar_sparse_vk_schedule_2() {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  {
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

// Schedule 3:
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7] → big (g_big_cores)
//   Chunk 4: Stages [8, 9] → little (g_little_cores)
static void BM_pipe_cifar_sparse_vk_schedule_3() {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  {
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

// Schedule 4:
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7] → medium (g_medium_cores)
//   Chunk 4: Stages [8, 9] → little (g_little_cores)
static void BM_pipe_cifar_sparse_vk_schedule_4() {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  {
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

// Schedule 5:
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8] → big (g_big_cores)
//   Chunk 4: Stage [9] → little (g_little_cores)
static void BM_pipe_cifar_sparse_vk_schedule_5() {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  {
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

// Schedule 6:
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8] → medium (g_medium_cores)
//   Chunk 4: Stage [9] → little (g_little_cores)
static void BM_pipe_cifar_sparse_vk_schedule_6() {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  {
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

// Schedule 7 (3 chunks):
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_7() {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  {
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

// Schedule 8 (3 chunks):
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8, 9] → medium (g_medium_cores)
static void BM_pipe_cifar_sparse_vk_schedule_8() {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  {
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

// Schedule 9 (3 chunks, same as Schedule 7):
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_9() {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  {
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

// Schedule 10 (3 chunks, same as Schedule 8):
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8, 9] → medium (g_medium_cores)
static void BM_pipe_cifar_sparse_vk_schedule_10() {
  SETUP_CORES_AND_TASKS(state);
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  {
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

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != "3A021JEHN02756") {
    return 0;
  }

  return 0;
}
