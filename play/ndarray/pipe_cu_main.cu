#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "cuda/dispatchers.cuh"
#include "omp/dispatchers.hpp"
#include "spsc_queue.hpp"

constexpr size_t kNumTasks = 100;

struct Task {
  static uint32_t uid_counter;
  uint32_t uid;

  // ----------------------------------
  cifar_dense::AppDataBatch appdata;

  explicit Task() : appdata(std::pmr::new_delete_resource()) { uid = uid_counter++; }
  // ----------------------------------
};

uint32_t Task::uid_counter = 0;

// template <ProcessorType PT>
static void omp_worker_thread(const std::vector<int>& cores_to_use,
                              const size_t num_threads,
                              SPSCQueue<Task*, 1024>& in_queue,
                              SPSCQueue<Task*, 1024>* out_queue
                              // std::function<void(Task&)> process_function
) {
  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    // tmp
    auto start_stage = 1;
    auto end_stage = 9;

#pragma omp parallel num_threads(num_threads)
    {
      bind_thread_to_cores(cores_to_use);

      for (int stage = start_stage; stage <= end_stage; stage++) {
        omp::dispatch_fns_batch[stage - 1](task->appdata);
      }
    }

    // Forward processed task
    if (out_queue) {
      out_queue->enqueue(std::move(task));
    }
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  SPSCQueue<Task*> q_0_1;
  SPSCQueue<Task*> q_1_2;

  // Master thread pushing tasks
  for (size_t i = 0; i < kNumTasks; ++i) {
    q_0_1.enqueue(new Task());
  }

  // ------------------------------------------------------------------------------------------------

  {
    auto start = std::chrono::high_resolution_clock::now();

    std::thread t1(
        omp_worker_thread, g_little_cores, g_little_cores.size(), std::ref(q_0_1), &q_1_2);

    t1.join();
    // t2.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    spdlog::info(
        "Time taken by tasks: {} ms, average: {} ", duration.count(), duration.count() / kNumTasks);
  }

  // ------------------------------------------------------------------------------------------------

  spdlog::info("Done");
  return 0;
}
