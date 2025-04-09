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

  // ----------------------------------
  // Timer start
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  // ----------------------------------

  void start_timer() { start_time = std::chrono::high_resolution_clock::now(); }
  void end_timer() { end_time = std::chrono::high_resolution_clock::now(); }
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

    task->start_timer();
    process_function(*task);
    task->end_timer();

    // Forward processed task
    if (out_queue) {
      out_queue->enqueue(std::move(task));
    }
  }
}

static void process_chunk(const ChunkConfig& config,
                          SPSCQueue<Task*>& in_queue,
                          SPSCQueue<Task*>& out_queue,
                          cifar_dense::vulkan::VulkanDispatcher& disp) {
  if (config.exec_model == ExecutionModel::kOMP) {
    // OMP
    worker_thread(in_queue, &out_queue, [&](Task& task) {
      omp::dispatch_multi_stage(config.start_stage,
                                config.end_stage,
                                config.proc_type.value(),
                                config.num_threads.value(),
                                task.appdata);
    });

  } else if (config.exec_model == ExecutionModel::kGPU) {
    // GPU
    worker_thread(in_queue, &out_queue, [&](Task& task) {
      disp.dispatch_multi_stage(task.appdata, config.start_stage, config.end_stage);
    });

  } else {
    throw std::runtime_error("Invalid execution model");
  }
}

// static void BM_run_stage_1(benchmark::State& state) {
//   cifar_dense::vulkan::VulkanDispatcher disp;
//   cifar_dense::AppDataBatch appdata(disp.get_mr());

//   for (auto _ : state) {
//     disp.run_stage_1(appdata);
//   }
// }

// BENCHMARK(BM_run_stage_1)->Unit(benchmark::kMillisecond);

// static void BM_run_stage_1_with_queue(benchmark::State& state) {
//   constexpr size_t kNumTasks = 100;

//   cifar_dense::vulkan::VulkanDispatcher disp;

//   SPSCQueue<Task*, 1024> in_queue;
//   SPSCQueue<Task*, 1024> out_queue;

//   for (size_t i = 0; i < kNumTasks; ++i) {
//     in_queue.enqueue(new Task(disp.get_mr()));
//   }

//   for (auto _ : state) {
//     worker_thread(in_queue, &out_queue, [&](Task& task) { disp.run_stage_1(task.appdata); });
//   }
// }

// BENCHMARK(BM_run_stage_1_with_queue)->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(5);

static void BM_run_VK_stage_1_concurrent(benchmark::State& state) {
  constexpr size_t kNumTasks = 100;

  cifar_dense::vulkan::VulkanDispatcher disp;

  SPSCQueue<Task*, 1024> in_queue;
  SPSCQueue<Task*, 1024> out_queue;

  for (size_t i = 0; i < kNumTasks; ++i) {
    in_queue.enqueue(new Task(disp.get_mr()));
  }

  for (auto _ : state) {
    worker_thread(in_queue, &out_queue, [&](Task& task) { disp.run_stage_1(task.appdata); });
  }

  //   // Now we should have an list of tasks in out_queue, print them
  //   Task* task = nullptr;
  //   while (out_queue.dequeue(task)) {
  //     std::cout << "Task " << task->uid << " took "
  //               << std::chrono::duration_cast<std::chrono::milliseconds>(task->end_time -
  //                                                                        task->start_time)
  //                      .count()
  //               << " ms" << std::endl;
  //   }
}

// BENCHMARK(BM_run_VK_stage_1_concurrent)
//     ->Unit(benchmark::kMillisecond)
//     ->Iterations(1)
//     ->Repetitions(5);

static void BM_run_OMP_stage_1(benchmark::State& state) {
  const ProcessorType core_type = static_cast<ProcessorType>(state.range(0));
  const int num_threads = state.range(1);

  constexpr size_t kNumTasks = 100;

  cifar_dense::vulkan::VulkanDispatcher disp;

  SPSCQueue<Task*, 1024> in_queue;
  SPSCQueue<Task*, 1024> out_queue;

  for (size_t i = 0; i < kNumTasks; ++i) {
    in_queue.enqueue(new Task(disp.get_mr()));
  }

  for (auto _ : state) {
    worker_thread(in_queue, &out_queue, [&](Task& task) {
      omp::dispatch_multi_stage(1, 1, core_type, num_threads, task.appdata);
    });
  }

  //   // Now we should have an list of tasks in out_queue, print them
  //   Task* task = nullptr;
  //   while (out_queue.dequeue(task)) {
  //     std::cout << "Task " << task->uid << " took "
  //               << std::chrono::duration_cast<std::chrono::milliseconds>(task->end_time -
  //                                                                        task->start_time)
  //                      .count()
  //               << " ms" << std::endl;
  //   }
}

// BENCHMARK(BM_run_OMP_stage_1)
//     ->Args({static_cast<int>(ProcessorType::kLittleCore), 4})
//     ->Args({static_cast<int>(ProcessorType::kBigCore), 2})
//     ->Unit(benchmark::kMillisecond)
//     ->Iterations(1)
//     ->Repetitions(5);

// static void execute_schedule(const std::vector<ChunkConfig>& chunk_configs) {
//   cifar_dense::vulkan::VulkanDispatcher disp;

//   std::vector<SPSCQueue<Task*>> concur_qs(chunk_configs.size() + 1);

//   // Setting up the input queue
//   for (size_t i = 0; i < kNumTasks; ++i) {
//     concur_qs[0].enqueue(new Task(disp.get_mr()));
//   }

//   std::vector<std::thread> threads;

//   // Benchmark start
//   const auto start = std::chrono::high_resolution_clock::now();

//   for (size_t i = 0; i < chunk_configs.size(); ++i) {
//     threads.emplace_back(
//         [&, i]() { process_chunk(chunk_configs[i], concur_qs[i], concur_qs[i + 1], disp); });
//   }

//   for (auto& t : threads) {
//     t.join();
//   }

//   const auto end = std::chrono::high_resolution_clock::now();
//   const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//   const double avg_task_time = duration.count() / static_cast<double>(kNumTasks);
//   std::cout << "Time per task: " << avg_task_time << " ms" << std::endl;
// }

void register_stage_benchmark() {
  for (size_t num_thread = 1; num_thread <= g_little_cores.size(); ++num_thread) {
    const auto name = "BM_run_OMP_stage_1/Little/" + std::to_string(num_thread);
    benchmark::RegisterBenchmark(name.c_str(), BM_run_OMP_stage_1)
        ->Args({static_cast<int>(ProcessorType::kLittleCore), static_cast<int>(num_thread)})
        ->Unit(benchmark::kMillisecond)
        ->Iterations(1)
        ->Repetitions(5);
  }
}

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;

  //   std::string schedule_file_path;
  //   app.add_option("-f,--file", schedule_file_path, "Schedule file path")->required();

  PARSE_ARGS_END;

  // ------------------------------------------------------------------------------------------------

  //   const auto chunks = readChunksFromJson(fs::path(schedule_file_path));
  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  //   execute_schedule(chunks);
  register_stage_benchmark();

  // ------------------------------------------------------------------------------------------------

  benchmark::Initialize(&argc, argv);
  //   if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
