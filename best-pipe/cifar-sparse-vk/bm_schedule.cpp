#include <benchmark/benchmark.h>
#include <omp.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "builtin-apps/pipeline/worker.hpp"

using MyTask = Task<cifar_sparse::v2::AppData>;

// ----------------------------------------------------------------------------
// Specialized Schedules
// ----------------------------------------------------------------------------

// std::string g_schedule_base_url;  // default 'kDefaultScheduleBaseDir'
// int g_schedule_id;

// static void BM_pipe_cifar_sparse_vk_schedule_best(benchmark::State& state) {
//   cifar_sparse::vulkan::v2::VulkanDispatcher disp;
//   std::vector<std::unique_ptr<MyTask>> preallocated_tasks;

//   // --------------------------------------------------------------------------

//   const auto schedule_json = fetch_json_from_url(
//       make_full_url(g_schedule_base_url, g_device_id, "CifarSparse", g_schedule_id));

//   spdlog::info("Schedule JSON: {}", schedule_json.dump(4));

//   const auto chunk_configs = readChunksFromJson(schedule_json);

//   std::vector<SPSCQueue<MyTask*, kPoolSize>> concur_qs(chunk_configs.size() + 1);
//   std::vector<std::thread> threads;

//   for (size_t i = 0; i < kPoolSize; ++i) {
//     preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr()));
//     concur_qs[0].enqueue(preallocated_tasks.back().get());
//   }

//   for (auto _ : state) {
//     for (size_t i = 0; i < chunk_configs.size(); ++i) {
//       threads.emplace_back([&, i]() {
//         worker_thread<MyTask>(
//             std::ref(concur_qs[i]), std::ref(concur_qs[i + 1]), [&](MyTask& task) {
//               if (chunk_configs[i].exec_model == ExecutionModel::kVulkan) {
//                 // GPU/Vulkan
//                 disp.dispatch_multi_stage(
//                     task.appdata, chunk_configs[i].start_stage, chunk_configs[i].end_stage);
//               } else {
//                 // CPU/OpenMP
//                 cifar_sparse::omp::v2::dispatch_multi_stage(
//                     g_medium_cores, g_medium_cores.size(), task.appdata, 5, 5);
//               }
//             });
//       });
//     }

//     for (auto& t : threads) {
//       t.join();
//     }
//   }
//   // --------------------------------------------------------------------------
// }

// static void BM_pipe_cifar_sparse_vk_schedule_best(benchmark::State& state) {
//   cifar_sparse::vulkan::v2::VulkanDispatcher disp;
//   std::vector<std::unique_ptr<MyTask>> preallocated_tasks;
//   SPSCQueue<MyTask*, kPoolSize> free_task_pool;
//   for (size_t i = 0; i < kPoolSize; ++i) {
//     preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr()));
//     free_task_pool.enqueue(preallocated_tasks.back().get());
//   }

//   // --------------------------------------------------------------------------

//   // Optimal chunk time: 577227/50000
//   // Stage 1: core type Big with time 7.89115
//   // Stage 2: core type GPU with time 4.95056
//   // Stage 3: core type GPU with time 4.36582
//   // Stage 4: core type Medium with time 6.49812
//   // Stage 5: core type Medium with time 1.66993
//   // Stage 6: core type Medium with time 1.69134
//   // Stage 7: core type Medium with time 1.68515
//   // Stage 8: core type Little with time 6.29989
//   // Stage 9: core type Little with time 0.182615

//   // const auto chunk_configs = readChunksFromJson(schedule_json);
//   std::vector<ChunkConfig> chunk_configs = {
//       {ExecutionModel::kOMP, 1, 1, ProcessorType::kBigCore, 2},
//       {ExecutionModel::kGPU, 2, 3, ProcessorType::kVulkan},
//       {ExecutionModel::kOMP, 4, 7, ProcessorType::kMediumCore, 2},
//       {ExecutionModel::kOMP, 8, 9, ProcessorType::kLittleCore, 4},
//   };

//   SPSCQueue<MyTask*, kPoolSize> q_0_1;
//   SPSCQueue<MyTask*, kPoolSize> q_1_2;
//   SPSCQueue<MyTask*, kPoolSize> q_2_3;

//   for (auto _ : state) {
//     auto t0 = std::thread(
//         worker_thread<MyTask>, std::ref(free_task_pool), std::ref(q_0_1), [&](MyTask& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(
//               g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
//         });

//     auto t1 =
//         std::thread(worker_thread<MyTask>, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
//           disp.dispatch_multi_stage(task.appdata, 2, 3);
//         });

//     auto t2 =
//         std::thread(worker_thread<MyTask>, std::ref(q_1_2), std::ref(q_2_3), [&](MyTask& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(
//               g_medium_cores, g_medium_cores.size(), task.appdata, 4, 7);
//         });

//     auto t3 = std::thread(
//         worker_thread<MyTask>, std::ref(q_2_3), std::ref(free_task_pool), [&](MyTask& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(
//               g_little_cores, g_little_cores.size(), task.appdata, 8, 9);
//         });

//     t0.join();
//     t1.join();
//     t2.join();
//     t3.join();
//   }

//   // --------------------------------------------------------------------------
// }

namespace device_3A021JEHN02756 {

#define SETUP_DATA                                                            \
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;                            \
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;                    \
  SPSCQueue<MyTask*, kPoolSize> q_0;                                          \
  SPSCQueue<MyTask*, kPoolSize> q_1;                                          \
  SPSCQueue<MyTask*, kPoolSize> q_2;                                          \
  SPSCQueue<MyTask*, kPoolSize> q_3;                                          \
  for (size_t i = 0; i < kPoolSize; ++i) {                                    \
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr())); \
    q_0.enqueue(preallocated_tasks.back().get());                             \
  }

template <typename QueueT>
[[nodiscard]]
std::thread create_thread(
    QueueT& in, QueueT& out, const std::vector<int>& cores, const int start, const int end) {
  return std::thread(
      worker_thread<MyTask>, std::ref(in), std::ref(out), [&cores, start, end](MyTask& task) {
        cifar_sparse::omp::v2::dispatch_multi_stage(cores, cores.size(), task.appdata, start, end);
      });
}

template <typename QueueT>
[[nodiscard]]
std::thread create_thread(QueueT& in,
                          QueueT& out,
                          cifar_sparse::vulkan::v2::VulkanDispatcher& disp,
                          const int start,
                          const int end) {
  return std::thread(
      worker_thread<MyTask>, std::ref(in), std::ref(out), [&disp, start, end](MyTask& task) {
        disp.dispatch_multi_stage(task.appdata, start, end);
      });
}

// ----------------------------------------------------------------------------
// Schedule 0
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2]
// Medium = [3, 4, 5, 6]
// Little = [7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_0(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 3);
    auto t2 = create_thread(q_2, q_3, g_medium_cores, 4, 7);
    auto t3 = create_thread(q_3, q_0, g_little_cores, 8, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_0)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 1
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2]
// Medium = [3, 4, 5, 6]
// Little = [7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_1(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 3);
    auto t2 = create_thread(q_2, q_3, g_medium_cores, 4, 7);
    auto t3 = create_thread(q_3, q_0, g_little_cores, 8, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 2
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2]
// Little = [3]
// Medium = [4, 5, 6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_2(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 3);
    auto t2 = create_thread(q_2, q_3, g_little_cores, 4, 4);
    auto t3 = create_thread(q_3, q_0, g_medium_cores, 5, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 3
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3]
// Medium = [4, 5, 6]
// Little = [7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_3(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 4);
    auto t2 = create_thread(q_2, q_3, g_medium_cores, 5, 7);
    auto t3 = create_thread(q_3, q_0, g_little_cores, 8, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 4
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3]
// Medium = [4, 5, 6, 7]
// Little = [8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_4(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 4);
    auto t2 = create_thread(q_2, q_3, g_medium_cores, 5, 8);
    auto t3 = create_thread(q_3, q_0, g_little_cores, 9, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_4)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 5
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3]
// Medium = [4, 5, 6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_5(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 4);
    auto t2 = create_thread(q_2, q_0, g_medium_cores, 5, 9);

    t0.join();
    t1.join();
    t2.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_5)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 6
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3]
// Little = [4]
// Medium = [5, 6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_6(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 4);
    auto t2 = create_thread(q_2, q_3, g_little_cores, 5, 5);
    auto t3 = create_thread(q_3, q_0, g_medium_cores, 6, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_6)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 7
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3]
// Little = [4, 5]
// Medium = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_7(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 4);
    auto t2 = create_thread(q_2, q_3, g_little_cores, 5, 6);
    auto t3 = create_thread(q_3, q_0, g_medium_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_7)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 8
// ----------------------------------------------------------------------------
// Stage assignments:
// Medium = [0]
// GPU = [1, 2, 3]
// Little = [4]
// Big = [5, 6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_8(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_medium_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 4);
    auto t2 = create_thread(q_2, q_3, g_little_cores, 5, 5);
    auto t3 = create_thread(q_3, q_0, g_big_cores, 6, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_8)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 9
// ----------------------------------------------------------------------------
// Stage assignments:
// Medium = [0]
// GPU = [1, 2, 3]
// Big = [4, 5, 6]
// Little = [7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_9(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_medium_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 4);
    auto t2 = create_thread(q_2, q_3, g_big_cores, 5, 7);
    auto t3 = create_thread(q_3, q_0, g_little_cores, 8, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_9)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 10
// ----------------------------------------------------------------------------
// Stage assignments:
// Medium = [0]
// GPU = [1, 2, 3]
// Little = [4, 5]
// Big = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_10(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_medium_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 4);
    auto t2 = create_thread(q_2, q_3, g_little_cores, 5, 6);
    auto t3 = create_thread(q_3, q_0, g_big_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_10)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

}  // namespace device_3A021JEHN02756

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != "3A021JEHN02756") {
    return 0;
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup
  return 0;
}
