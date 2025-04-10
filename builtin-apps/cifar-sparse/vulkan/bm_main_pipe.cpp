#include <benchmark/benchmark.h>
#include <omp.h>

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
};

uint32_t Task::uid_counter = 0;

using ProcessFunction = std::function<void(Task&)>;

constexpr size_t kNumTasks = 10;

void worker_thread(SPSCQueue<Task*, 1024>& in_queue,
                   SPSCQueue<Task*, 1024>* out_queue,
                   ProcessFunction process_function) {
  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    // Process the task (timing is handled in the provided process function)
    process_function(*task);

    // Forward processed task
    if (out_queue) {
      out_queue->enqueue(std::move(task));
    }
  }
}

static void BM_run_OMP_baseline(benchmark::State& state) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;

  SPSCQueue<Task*, 1024> q_0;
  SPSCQueue<Task*, 1024> q_1;

  for (size_t i = 0; i < kNumTasks; ++i) {
    q_0.enqueue(new Task(disp.get_mr()));
  }

  for (auto _ : state) {
    worker_thread(q_0, &q_1, [&](Task& task) { disp.dispatch_multi_stage(task.appdata, 1, 9); });
  }
}

BENCHMARK(BM_run_OMP_baseline)->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(5);

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup

  // cifar_sparse::vulkan::v2::VulkanDispatcher disp;

  // SPSCQueue<Task*, 1024> q_0;
  // SPSCQueue<Task*, 1024> q_1;

  // for (size_t i = 0; i < kNumTasks; ++i) {
  //   q_0.enqueue(new Task(disp.get_mr()));
  // }

  // worker_thread(q_0, &q_1, [&](Task& task) { disp.dispatch_multi_stage(task.appdata, 1, 9); });

  return 0;
}
