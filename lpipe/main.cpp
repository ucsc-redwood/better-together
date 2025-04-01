#include <omp.h>

#include <cassert>
#include <cstddef>

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"

// ------------------------------------------------------------------------------------------------
#include "kernels.hpp"
#include "spsc_queue.hpp"
#include "task.hpp"

constexpr size_t kNumTasks = 100;

template <ProcessorType PT>
void debug_print(const Task& task) {
#pragma omp critical
  {
    auto num_threads = omp_get_num_threads();

    if constexpr (PT == ProcessorType::kLittleCore) {
      spdlog::debug(
          "[L{:d}] core processed task {} on core {:d}", num_threads, task.uid, sched_getcpu());
    } else if constexpr (PT == ProcessorType::kMediumCore) {
      spdlog::debug(
          "[M{:d}] core processed task {} on core {:d}", num_threads, task.uid, sched_getcpu());
    } else if constexpr (PT == ProcessorType::kBigCore) {
      spdlog::debug(
          "[B{:d}] core processed task {} on core {:d}", num_threads, task.uid, sched_getcpu());
    }
  }
}

template <ProcessorType PT>
static void worker_thread(const size_t num_threads,
                          SPSCQueue<Task, 1024>& in_queue,
                          SPSCQueue<Task, 1024>* out_queue,
                          std::function<void(Task&)> process_function) {
  for (size_t i = 0; i < kNumTasks; ++i) {
    Task task;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

#pragma omp parallel num_threads(num_threads)
    {
      if constexpr (PT == ProcessorType::kLittleCore) {
        bind_thread_to_cores(g_little_cores);
      } else if constexpr (PT == ProcessorType::kMediumCore) {
        bind_thread_to_cores(g_medium_cores);
      } else if constexpr (PT == ProcessorType::kBigCore) {
        bind_thread_to_cores(g_big_cores);
      }

      // ----------------------------------
      //   debug_print<PT>(task);
      process_function(task);
      // ----------------------------------
    }

    // Forward processed task
    if (out_queue) {
      out_queue->enqueue(std::move(task));
    }
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  SPSCQueue<Task> q_0_1;
  SPSCQueue<Task> q_1_2;
  SPSCQueue<Task> q_2_3;

  // Master thread pushing tasks
  for (size_t i = 0; i < kNumTasks; ++i) {
    q_0_1.enqueue(new_task(640 * 480));
  }

  // ------------------------------------------------------------------------------------------------

  {
    auto start = std::chrono::high_resolution_clock::now();

    std::thread t1(worker_thread<ProcessorType::kLittleCore>,
                   g_little_cores.size(),
                   std::ref(q_0_1),
                   &q_1_2,
                   process_task_stage_A);
    std::thread t2(worker_thread<ProcessorType::kMediumCore>,
                   g_medium_cores.size(),
                   std::ref(q_1_2),
                   &q_2_3,
                   process_task_stage_B);
    std::thread t3(worker_thread<ProcessorType::kBigCore>,
                   g_big_cores.size(),
                   std::ref(q_2_3),
                   nullptr,
                   process_task_stage_C);

    t1.join();
    t2.join();
    t3.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    spdlog::info(
        "Time taken by tasks: {} ms, average: {} ", duration.count(), duration.count() / kNumTasks);
  }

  // ------------------------------------------------------------------------------------------------

  spdlog::info("Done");
  return 0;
}