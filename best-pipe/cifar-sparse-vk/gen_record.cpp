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

struct Task {
  uint32_t task_id;  // Unique ID for this task, assigned sequentially 0...N

  // ----------------------------------
  cifar_sparse::v2::AppData appdata;
  // ----------------------------------

  explicit Task(std::pmr::memory_resource* mr, uint32_t id) : task_id(id), appdata(mr) {}

  void reset() { appdata.reset(); }
};

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

#define SETUP_CORES_AND_TASKS()                                                \
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

// Schedule best:
//   Chunk 1: Stages [1, 2, 3, 4] → gpu_vulkan
//   Chunk 2: Stage [5] → medium (g_medium_cores)
//   Chunk 3: Stages [6, 7, 8, 9] → big (g_big_cores)
static void BM_pipe_cifar_sparse_vk_schedule_best() {
  SETUP_CORES_AND_TASKS();
  SPSCQueue<Task*, kPoolSize> q_0_1;
  SPSCQueue<Task*, kPoolSize> q_1_2;

  {
    auto t0 = std::thread(
        worker_thread_normal, std::ref(free_task_pool), std::ref(q_0_1), [&](Task& task) {
          disp.dispatch_multi_stage(task.appdata, 1, 4);
        });

    auto t1 = std::thread(worker_thread_normal, std::ref(q_0_1), std::ref(q_1_2), [&](Task& task) {
      cifar_sparse::omp::v2::dispatch_multi_stage(
          g_medium_cores, g_medium_cores.size(), task.appdata, 5, 5);
    });

    auto t2 = std::thread(
        worker_thread_normal, std::ref(q_1_2), std::ref(free_task_pool), [&](Task& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, 6, 9);
        });

    t0.join();
    t1.join();
    t2.join();
  }
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
