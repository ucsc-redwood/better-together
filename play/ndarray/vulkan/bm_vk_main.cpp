#include <benchmark/benchmark.h>

#include <cstddef>
#include <functional>

#include "../omp/dispatchers.hpp"
#include "../spsc_queue.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/config_reader.hpp"
#include "dispatcher.hpp"

constexpr size_t kNumTasks = 100;

struct Task {
  static uint32_t uid_counter;
  uint32_t uid;

  // ----------------------------------
  cifar_dense::AppDataBatch appdata;

  explicit Task(std::pmr::memory_resource* mr) : appdata(mr) { uid = uid_counter++; }
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
  std::vector<uint64_t> timings;
  timings.reserve(kNumTasks);
  state.ResumeTiming();

  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    // Process the task (timing is handled in the provided process function)
    const auto start = std::chrono::high_resolution_clock::now();
    process_function(*task);
    const auto end = std::chrono::high_resolution_clock::now();
    timings.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    // Forward processed task
    if (out_queue) {
      out_queue->enqueue(std::move(task));
    }
  }

  // After processing all tasks, compute summary statistics.
  state.PauseTiming();
  uint64_t total_time = 0;
  double log_sum = 0.0;
  for (const auto& timing : timings) {
    total_time += timing;
    log_sum += std::log(static_cast<double>(timing));
  }
  const auto avg_time = total_time / kNumTasks;
  const auto geomean = std::exp(log_sum / kNumTasks);
  state.counters["avg"] = avg_time;
  state.counters["geomean"] = geomean;
  state.ResumeTiming();
}

static void BM_run_OMP_stage_1(benchmark::State& state) {
  const ProcessorType core_type = static_cast<ProcessorType>(state.range(0));
  const int num_threads = state.range(1);

  constexpr size_t kNumTasks = 100;

  cifar_dense::vulkan::VulkanDispatcher disp;

  SPSCQueue<Task*, 1024> in_queue;
  SPSCQueue<Task*, 1024> out_queue;

  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = new Task(disp.get_mr());
    in_queue.enqueue(std::move(task));
  }

  for (auto _ : state) {
    worker_thread(in_queue, &out_queue, [&](Task& task) {
      omp::dispatch_multi_stage(1, 1, core_type, num_threads, task.appdata);
    });
  }
}

static void BM_run_OMP_stage_1_concurrent(benchmark::State& state) {
  const ProcessorType core_type = static_cast<ProcessorType>(state.range(0));
  const int num_threads = state.range(1);

  constexpr size_t kNumTasks = 100;

  cifar_dense::vulkan::VulkanDispatcher disp;

  // Create a pipeline with three queues
  SPSCQueue<Task*, 1024> q_0;  // Input queue
  SPSCQueue<Task*, 1024> q_1;  // Intermediate queue
  SPSCQueue<Task*, 1024> q_2;  // Output queue

  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = new Task(disp.get_mr());
    q_0.enqueue(std::move(task));
  }

  for (auto _ : state) {
    std::thread t1([&]() {
      // Thread 1: Read from q_0, write to q_1
      measured_worker_thread(state, q_0, &q_1, [&](Task& task) {
        omp::dispatch_multi_stage(1, 1, core_type, num_threads, task.appdata);
      });
    });

    std::thread t2([&]() {
      // Thread 2: Read from q_1, write to q_2
      worker_thread(q_1, &q_2, [&](Task& task) {
        disp.dispatch_multi_stage(task.appdata, 2, 4);
      });
    });

    t1.join();
    t2.join();
  }
}
[[nodiscard]] inline const std::string processor_type_name(const ProcessorType type) {
  switch (type) {
    case ProcessorType::kLittleCore:
      return "Little";
    case ProcessorType::kMediumCore:
      return "Medium";
    case ProcessorType::kBigCore:
      return "Big";
    default:
      return "Unknown";
  }
}

void register_stage_benchmark() {
  for (const auto core_type :
       {ProcessorType::kLittleCore, ProcessorType::kMediumCore, ProcessorType::kBigCore}) {
    const auto num_threads = [&]() {
      if (core_type == ProcessorType::kLittleCore) {
        return g_little_cores.size();
      } else if (core_type == ProcessorType::kMediumCore) {
        return g_medium_cores.size();
      } else {
        return g_big_cores.size();
      }
    }();
    
    // Skip registering benchmarks if there are no cores of this type
    if (num_threads == 0) {
      continue;
    }

    const auto name = "BM_run_OMP_stage_1_concurrent/" + processor_type_name(core_type) + "/" +
                      std::to_string(num_threads);
    benchmark::RegisterBenchmark(name.c_str(), BM_run_OMP_stage_1_concurrent)
      ->Args({static_cast<int>(core_type), static_cast<int>(num_threads)})
      ->Unit(benchmark::kMillisecond)
      ->Iterations(1)
      ->Repetitions(5);
  }
}

void register_stage_benchmark_normal() {
  for (const auto core_type :
       {ProcessorType::kLittleCore, ProcessorType::kMediumCore, ProcessorType::kBigCore}) {
    const auto num_threads = [&]() {
      if (core_type == ProcessorType::kLittleCore) {
        return g_little_cores.size();
      } else if (core_type == ProcessorType::kMediumCore) {
        return g_medium_cores.size();
      } else {
        return g_big_cores.size();
      }
    }();
    
    // Skip registering benchmarks if there are no cores of this type
    if (num_threads == 0) {
      continue;
    }

    const auto name =
      "BM_run_OMP_stage_1/" + processor_type_name(core_type) + "/" + std::to_string(num_threads);
    benchmark::RegisterBenchmark(name.c_str(), BM_run_OMP_stage_1)
      ->Args({static_cast<int>(core_type), static_cast<int>(num_threads)})
      ->Unit(benchmark::kMillisecond)
      ->Iterations(1)
      ->Repetitions(5);
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  // ------------------------------------------------------------------------------------------------

  register_stage_benchmark();
  register_stage_benchmark_normal();

  // ------------------------------------------------------------------------------------------------

  benchmark::Initialize(&argc, argv);
  //   if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
