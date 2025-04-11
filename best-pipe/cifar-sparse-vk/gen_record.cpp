#include <omp.h>

#include <deque>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/conf.hpp"
#include "builtin-apps/config_reader.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"

constexpr size_t kPoolSize = 32;
constexpr size_t kNumToProcess = 100;

// ----------------------------------------------------------------------------
// Record
// ----------------------------------------------------------------------------

struct Record {
  ProcessorType processed_by;
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
};

struct RecordManager {
  std::array<std::array<Record, 4>, kNumToProcess> records;

  void setup(const int chunk_id, const ProcessorType processed_by) {
    for (size_t i = 0; i < kNumToProcess; ++i) {
      records[i][chunk_id].processed_by = processed_by;
    }
  }

  void start_tick(const uint32_t processing_id, const int chunk_id) {
    records[processing_id][chunk_id].start = std::chrono::high_resolution_clock::now();
  }

  void end_tick(const uint32_t processing_id, const int chunk_id) {
    records[processing_id][chunk_id].end = std::chrono::high_resolution_clock::now();
  }

  void dump_records() {
    for (size_t i = 0; i < kNumToProcess; ++i) {
      std::cout << "Task " << i << ":\n";
      for (size_t j = 0; j < 4; ++j) {
        std::cout << "  Chunk " << j << ":\n";
        std::cout << "    Start: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         records[i][j].start.time_since_epoch())
                         .count()
                  << " us\n";
        std::cout << "    End: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         records[i][j].end.time_since_epoch())
                         .count()
                  << " us\n";
        std::cout << "    Duration: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(records[i][j].end -
                                                                           records[i][j].start)
                         .count()
                  << " us\n";
      }
    }
  }
};

RecordManager g_record_manager;

// ----------------------------------------------------------------------------
// Task
// ----------------------------------------------------------------------------

struct Task {
  static uint32_t uid_counter;
  uint32_t task_id;  // Unique ID for this task, assigned sequentially 0...N

  // ----------------------------------
  cifar_sparse::v2::AppData appdata;
  // ----------------------------------

  explicit Task(std::pmr::memory_resource* mr) : task_id(uid_counter++), appdata(mr) {}

  void reset() {
    // appdata.reset();
    task_id = uid_counter++;
  }

  void start_record(const int chunk_id) { g_record_manager.start_tick(task_id, chunk_id); }

  void end_record(const int chunk_id) { g_record_manager.end_tick(task_id, chunk_id); }
};

uint32_t Task::uid_counter = 0;

// ----------------------------------------------------------------------------
// Worker Thread
// ----------------------------------------------------------------------------

using ProcessFunction = std::function<void(Task&)>;

void worker_thread_normal(const int chunk_id,
                          SPSCQueue<Task*, kPoolSize>& in_queue,
                          SPSCQueue<Task*, kPoolSize>& out_queue,
                          ProcessFunction process_function) {
  for (size_t processing_id = 0; processing_id < kNumToProcess; ++processing_id) {
    Task* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    // Process the task (timing is handled in the provided process function)
    g_record_manager.start_tick(processing_id, chunk_id);
    process_function(*task);
    g_record_manager.end_tick(processing_id, chunk_id);

    // Forward processed task
    while (!out_queue.enqueue(std::move(task))) {
      std::this_thread::yield();
    }
  }
}

// ----------------------------------------------------------------------------
// Omp Stage
// ----------------------------------------------------------------------------

#define SETUP_CORES_AND_TASKS()                                             \
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

// Schedule best:
//   Chunk 1: Stages [1, 2, 3, 4] → gpu_vulkan
//   Chunk 2: Stage [5] → medium (g_medium_cores)
//   Chunk 3: Stages [6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_best() {
  SETUP_CORES_AND_TASKS();
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  {
    g_record_manager.setup(0, ProcessorType::kVulkan);
    g_record_manager.setup(1, ProcessorType::kMediumCore);
    g_record_manager.setup(2, ProcessorType::kBigCore);

    auto t0 = std::thread(
        worker_thread_normal, 0, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          disp.dispatch_multi_stage(task.appdata, 1, 4);
        });

    auto t1 =
        std::thread(worker_thread_normal, 1, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 5, 5);
        });

    auto t2 = std::thread(
        worker_thread_normal, 2, std::ref(q_1_2), std::ref(free_task_pool), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 6, 9);
        });

    t0.join();
    t1.join();
    t2.join();
  }

  g_record_manager.dump_records();
}

// Schedule 1:
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stage [5] → little (g_little_cores)
//   Chunk 4: Stages [6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_1() {
  SETUP_CORES_AND_TASKS();
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
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

    auto t3 = std::thread(
        worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 6, 9);
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

// Schedule 2:
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stage [5] → little (g_little_cores)
//   Chunk 4: Stages [6, 7, 8, 9] → medium (g_medium_cores)
static void BM_pipe_cifar_sparse_vk_schedule_2() {
  SETUP_CORES_AND_TASKS();
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      disp.dispatch_multi_stage(task.appdata, 2, 4);
    });

    auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task) {
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_little_cores, g_little_cores.size(), task.appdata, 5, 5);
    });

    auto t3 = std::thread(
        worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 6, 9);
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

// Schedule 3:
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7] → big (g_big_cores)
//   Chunk 4: Stages [8, 9] → little (g_little_cores)
static void BM_pipe_cifar_sparse_vk_schedule_3() {
  SETUP_CORES_AND_TASKS();
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      disp.dispatch_multi_stage(task.appdata, 2, 4);
    });

    auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task) {
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_big_cores, g_big_cores.size(), task.appdata, 5, 7);
    });

    auto t3 = std::thread(
        worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_little_cores, g_little_cores.size(), task.appdata, 8, 9);
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

// Schedule 4:
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7] → medium (g_medium_cores)
//   Chunk 4: Stages [8, 9] → little (g_little_cores)
static void BM_pipe_cifar_sparse_vk_schedule_4() {
  SETUP_CORES_AND_TASKS();
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      disp.dispatch_multi_stage(task.appdata, 2, 4);
    });

    auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task) {
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_medium_cores, g_medium_cores.size(), task.appdata, 5, 7);
    });

    auto t3 = std::thread(
        worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_little_cores, g_little_cores.size(), task.appdata, 8, 9);
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

// Schedule 5:
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8] → big (g_big_cores)
//   Chunk 4: Stage [9] → little (g_little_cores)
static void BM_pipe_cifar_sparse_vk_schedule_5() {
  SETUP_CORES_AND_TASKS();
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      disp.dispatch_multi_stage(task.appdata, 2, 4);
    });

    auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task) {
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_big_cores, g_big_cores.size(), task.appdata, 5, 8);
    });

    auto t3 = std::thread(
        worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_little_cores, g_little_cores.size(), task.appdata, 9, 9);
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

// Schedule 6:
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8] → medium (g_medium_cores)
//   Chunk 4: Stage [9] → little (g_little_cores)
static void BM_pipe_cifar_sparse_vk_schedule_6() {
  SETUP_CORES_AND_TASKS();
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;
  SPSCQueue<Task*, kPoolSize> q_2_3;

  {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      disp.dispatch_multi_stage(task.appdata, 2, 4);
    });

    auto t2 = std::thread(worker_thread_normal, std::ref(q_1_2), std::ref(q_2_3), [&](Task& task) {
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_medium_cores, g_medium_cores.size(), task.appdata, 5, 8);
    });

    auto t3 = std::thread(
        worker_thread_normal, std::ref(q_2_3), std::ref(free_task_pool), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_little_cores, g_little_cores.size(), task.appdata, 9, 9);
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

// Schedule 7 (3 chunks):
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_7() {
  SETUP_CORES_AND_TASKS();
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      disp.dispatch_multi_stage(task.appdata, 2, 4);
    });

    auto t2 = std::thread(
        worker_thread_normal, std::ref(q_1_2), std::ref(free_task_pool), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 5, 9);
        });

    t0.join();
    t1.join();
    t2.join();
  }
}

// Schedule 8 (3 chunks):
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8, 9] → medium (g_medium_cores)
static void BM_pipe_cifar_sparse_vk_schedule_8() {
  SETUP_CORES_AND_TASKS();
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      disp.dispatch_multi_stage(task.appdata, 2, 4);
    });

    auto t2 = std::thread(
        worker_thread_normal, std::ref(q_1_2), std::ref(free_task_pool), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 5, 9);
        });

    t0.join();
    t1.join();
    t2.join();
  }
}

// Schedule 9 (3 chunks, same as Schedule 7):
//   Chunk 1: Stage [1] → medium (g_medium_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_9() {
  SETUP_CORES_AND_TASKS();
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      disp.dispatch_multi_stage(task.appdata, 2, 4);
    });

    auto t2 = std::thread(
        worker_thread_normal, std::ref(q_1_2), std::ref(free_task_pool), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 5, 9);
        });

    t0.join();
    t1.join();
    t2.join();
  }
}

// Schedule 10 (3 chunks, same as Schedule 8):
//   Chunk 1: Stage [1] → big (g_big_cores)
//   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
//   Chunk 3: Stages [5, 6, 7, 8, 9] → medium (g_medium_cores)
static void BM_pipe_cifar_sparse_vk_schedule_10() {
  SETUP_CORES_AND_TASKS();
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      disp.dispatch_multi_stage(task.appdata, 2, 4);
    });

    auto t2 = std::thread(
        worker_thread_normal, std::ref(q_1_2), std::ref(free_task_pool), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_medium_cores, g_medium_cores.size(), task.appdata, 5, 9);
        });

    t0.join();
    t1.join();
    t2.join();
  }
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN

  int schedule_id = 0;
  app.add_option("--schedule", schedule_id, "Schedule to run")->required();

  PARSE_ARGS_END

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != "3A021JEHN02756") {
    return 0;
  }

  switch (schedule_id) {
    case 1:
      BM_pipe_cifar_sparse_vk_schedule_1();
      break;
    case 2:
      BM_pipe_cifar_sparse_vk_schedule_2();
      break;
    case 3:
      BM_pipe_cifar_sparse_vk_schedule_3();
      break;
    case 4:
      BM_pipe_cifar_sparse_vk_schedule_4();
      break;
    case 5:
      BM_pipe_cifar_sparse_vk_schedule_5();
      break;
    case 6:
      BM_pipe_cifar_sparse_vk_schedule_6();
      break;
    case 7:
      BM_pipe_cifar_sparse_vk_schedule_7();
      break;
    case 8:
      BM_pipe_cifar_sparse_vk_schedule_8();
      break;
    case 9:
      BM_pipe_cifar_sparse_vk_schedule_9();
      break;
    case 10:
      BM_pipe_cifar_sparse_vk_schedule_10();
      break;
    default:
      BM_pipe_cifar_sparse_vk_schedule_best();
  }

  return 0;
}
