#include <functional>

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "cuda/dispatchers.cuh"
#include "omp/dispatchers.hpp"
#include "spsc_queue.hpp"

constexpr size_t kNumTasks = 100;

// Global CUDA objects to ensure proper lifetime and thread visibility
cuda::CudaManager g_cuda_mgr;
cuda::DeviceModelData* g_device_model_data = nullptr;

struct Task {
  static uint32_t uid_counter;
  uint32_t uid;

  // ----------------------------------
  cifar_dense::AppDataBatch appdata;

  //   // just reference
  //   const cuda::CudaManager& mgr;
  //   const cuda::DeviceModelData& d_model_data;

  explicit Task() : appdata(std::pmr::new_delete_resource()) { uid = uid_counter++; }
  // ----------------------------------
};

uint32_t Task::uid_counter = 0;

using ProcessFunction = std::function<void(Task&)>;

static void worker_thread(SPSCQueue<Task*, 1024>& in_queue,
                          SPSCQueue<Task*, 1024>* out_queue,
                          ProcessFunction process_function) {
  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    process_function(*task);

    // Forward processed task
    if (out_queue) {
      out_queue->enqueue(std::move(task));
    }
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  SPSCQueue<Task*> q_0_1;
  SPSCQueue<Task*> q_1_2;

  // Initialize global device model data
  g_device_model_data = new cuda::DeviceModelData(cifar_dense::AppDataBatch::get_model());

  // Master thread pushing tasks
  for (size_t i = 0; i < kNumTasks; ++i) {
    q_0_1.enqueue(new Task());
  }

  // ------------------------------------------------------------------------------------------------

  {
    auto start = std::chrono::high_resolution_clock::now();

    std::thread t1(worker_thread, std::ref(q_0_1), &q_1_2, [&](Task& task) {
      omp::dispatch_multi_stage(g_little_cores, g_little_cores.size(), task.appdata, 1, 4);
    });

    std::thread t2(worker_thread, std::ref(q_1_2), nullptr, [&](Task& task) {
      cuda::dispatch_multi_stage(task.appdata, *g_device_model_data, 5, 9, g_cuda_mgr);
    });

    t1.join();
    t2.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    spdlog::info(
        "Time taken by tasks: {} ms, average: {} ", duration.count(), duration.count() / kNumTasks);
  }

  // Cleanup
  delete g_device_model_data;

  // ------------------------------------------------------------------------------------------------

  spdlog::info("Done");
  return 0;
}
