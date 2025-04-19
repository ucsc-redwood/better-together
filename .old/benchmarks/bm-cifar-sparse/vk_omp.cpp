#include <benchmark/benchmark.h>

#include "../argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/conf.hpp"
#include "builtin-apps/resources_path.hpp"
#include "builtin-apps/spsc_queue.hpp"
#include "spdlog/common.h"

constexpr size_t kNumTasks = 10;

struct Task {
  static uint32_t uid_counter;
  uint32_t uid;

  // ----------------------------------
  cifar_sparse::AppData app_data;

  explicit Task(std::pmr::memory_resource* mr) : app_data(mr) { uid = uid_counter++; }
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

    // Process the task (timing is handled in the provided process function)
    process_function(*task);

    // Forward processed task
    if (out_queue) {
      out_queue->enqueue(std::move(task));
    }
  }
}

static void measured_worker_thread(benchmark::State& state,
                                   SPSCQueue<Task*, 1024>& in_queue,
                                   SPSCQueue<Task*, 1024>* out_queue,
                                   ProcessFunction process_function) {
  state.PauseTiming();
  std::vector<std::chrono::duration<double>> durations;
  durations.reserve(kNumTasks);
  state.ResumeTiming();

  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    // Process the task (timing is handled in the provided process function)
    auto start = std::chrono::high_resolution_clock::now();
    process_function(*task);
    auto end = std::chrono::high_resolution_clock::now();
    durations.push_back(end - start);

    // Forward processed task
    if (out_queue) {
      out_queue->enqueue(std::move(task));
    }
  }
  state.PauseTiming();
  double product = 1.0;
  for (const auto& duration : durations) {
    product *= duration.count();
  }
  double geomean = std::pow(product, 1.0 / durations.size());
  state.counters["geomean"] = geomean;
  state.ResumeTiming();
}

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

#define PREPARE_DATA                           \
  cifar_sparse::vulkan::VulkanDispatcher disp; \
  SPSCQueue<Task*, 1024> in_queue;             \
  SPSCQueue<Task*, 1024> out_queue;            \
  for (size_t i = 0; i < kNumTasks; ++i) {     \
    Task* task = new Task(disp.get_mr());      \
    in_queue.enqueue(std::move(task));         \
  }

class VK_CifarSparse : public benchmark::Fixture {};

BENCHMARK_DEFINE_F(VK_CifarSparse, Baseline)
(benchmark::State& state) {
  PREPARE_DATA;

  for (auto _ : state) {
    worker_thread(in_queue, &out_queue, [&](Task& task) {
      cifar_sparse::omp::dispatch_multi_stage_unrestricted(
          std::thread::hardware_concurrency(), task.app_data, 1, 9);
    });
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Baseline)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------
// Stages
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage1)
(benchmark::State& state) {
  PREPARE_DATA;

  auto cores = g_lit_cores;
  auto num_threads = cores.size();

  for (auto _ : state) {
    std::thread t1([&]() {
      measured_worker_thread(state, in_queue, &out_queue, [&](Task& task) {
        cifar_sparse::omp::dispatch_multi_stage(cores, num_threads, task.app_data, 1, 1);
      });
    });
    std::thread t2([&]() {
      worker_thread(in_queue, &out_queue, [&](Task& task) {
        cifar_sparse::omp::dispatch_multi_stage(g_big_cores, num_threads, task.app_data, 2, 3);
      });
    });

    t1.join();
    t2.join();
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------
int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  // const auto storage_location = helpers::get_benchmark_storage_location();
  // const auto out_name = storage_location.string() + "/BM_CifarSparse_VK_" + g_device_id +
  // ".json";

  // auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv, out_name);

  benchmark::Initialize(&argc, argv);
  // if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
