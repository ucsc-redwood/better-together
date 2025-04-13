#include <omp.h>

#include "../omp/dispatchers.hpp"
#include "../sparse_appdata.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/conf.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "builtin-apps/pipeline/worker.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// Realistic Benchmark: OMP Stage
// ----------------------------------------------------------------------------

using MyTask = Task<cifar_sparse::v2::AppData>;

// static void BM_run_OMP_stage_full(const int stage_to_measure) {
//   cifar_sparse::vulkan::v2::VulkanDispatcher disp;
//   std::vector<std::unique_ptr<MyTask>> preallocated_tasks;
//   SPSCQueue<MyTask*, kPoolSize> free_task_pool;
//   for (size_t i = 0; i < kPoolSize; ++i) {
//     preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr()));
//     free_task_pool.enqueue(preallocated_tasks.back().get());
//   }

//   SPSCQueue<MyTask*, kPoolSize> q_0_1;
//   SPSCQueue<MyTask*, kPoolSize> q_1_2;
//   SPSCQueue<MyTask*, kPoolSize> q_2_3;

//   // Main benchmark
//   {
//     // Only setup the records once
//     RecordManager::instance().setup(100,
//                                     {std::make_pair(0, ProcessorType::kLittleCore),
//                                      std::make_pair(1, ProcessorType::kMediumCore),
//                                      std::make_pair(2, ProcessorType::kBigCore),
//                                      std::make_pair(3, ProcessorType::kVulkan)});

//     // little core
//     auto t0 = std::thread(worker_thread_record<MyTask>,
//                           0,
//                           std::ref(free_task_pool),
//                           std::ref(q_0_1),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(g_little_cores,
//                                                                         g_little_cores.size(),
//                                                                         task.appdata,
//                                                                         stage_to_measure,
//                                                                         stage_to_measure);
//                           });

//     // medium core
//     auto t1 = std::thread(
//         worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(g_medium_cores,
//                                                       g_medium_cores.size(),
//                                                       task.appdata,
//                                                       stage_to_measure,
//                                                       stage_to_measure);
//         });

//     // big core
//     auto t2 = std::thread(
//         worker_thread_record<MyTask>, 2, std::ref(q_1_2), std::ref(q_2_3), [&](MyTask& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(
//               g_big_cores, g_big_cores.size(), task.appdata, stage_to_measure, stage_to_measure);
//         });

//     // vulkan core
//     auto t3 =
//         std::thread(worker_thread_record<MyTask>,
//                     3,
//                     std::ref(q_2_3),
//                     std::ref(free_task_pool),
//                     [&](MyTask& task) {
//                       disp.dispatch_multi_stage(task.appdata, stage_to_measure,
//                       stage_to_measure);
//                     });

//     t0.join();
//     t1.join();
//     t2.join();
//     t3.join();

//     RecordManager::instance().print_processor_type_stats(ProcessorType::kLittleCore);
//     RecordManager::instance().print_processor_type_stats(ProcessorType::kMediumCore);
//     RecordManager::instance().print_processor_type_stats(ProcessorType::kBigCore);
//     RecordManager::instance().print_processor_type_stats(ProcessorType::kVulkan);
//   }
// }

static void BM_run_OMP_stage_full(const int stage_to_measure) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;
  SPSCQueue<MyTask*, kPoolSize> free_task_pool;
  for (size_t i = 0; i < kPoolSize; ++i) {
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr()));
    free_task_pool.enqueue(preallocated_tasks.back().get());
  }

  SPSCQueue<MyTask*, kPoolSize> q_0_1;
  SPSCQueue<MyTask*, kPoolSize> q_1_2;
  SPSCQueue<MyTask*, kPoolSize> q_2_3;

  // Main benchmark
  {
    // Only setup the records once
    RecordManager::instance().setup(100,
                                    {std::make_pair(0, ProcessorType::kVulkan),
                                     std::make_pair(1, ProcessorType::kBigCore),
                                     std::make_pair(2, ProcessorType::kMediumCore),
                                     std::make_pair(3, ProcessorType::kLittleCore)});

    // vulkan core
    auto t0 =
        std::thread(worker_thread_record<MyTask>,
                    0,
                    std::ref(q_2_3),
                    std::ref(free_task_pool),
                    [&](MyTask& task) {
                      disp.dispatch_multi_stage(task.appdata, stage_to_measure, stage_to_measure);
                    });

    // big core
    auto t1 = std::thread(
        worker_thread_record<MyTask>, 1, std::ref(q_1_2), std::ref(q_2_3), [&](MyTask& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, stage_to_measure, stage_to_measure);
        });

    // medium core
    auto t2 = std::thread(
        worker_thread_record<MyTask>, 2, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(g_medium_cores,
                                                      g_medium_cores.size(),
                                                      task.appdata,
                                                      stage_to_measure,
                                                      stage_to_measure);
        });

    // little core
    auto t3 = std::thread(worker_thread_record<MyTask>,
                          3,
                          std::ref(free_task_pool),
                          std::ref(q_0_1),
                          [&](MyTask& task) {
                            cifar_sparse::omp::v2::dispatch_multi_stage(g_little_cores,
                                                                        g_little_cores.size(),
                                                                        task.appdata,
                                                                        stage_to_measure,
                                                                        stage_to_measure);
                          });

    t0.join();
    t1.join();
    t2.join();
    t3.join();

    RecordManager::instance().print_processor_type_stats(ProcessorType::kVulkan);
    RecordManager::instance().print_processor_type_stats(ProcessorType::kBigCore);
    RecordManager::instance().print_processor_type_stats(ProcessorType::kMediumCore);
    RecordManager::instance().print_processor_type_stats(ProcessorType::kLittleCore);
  }
}

static void BM_run_OMP_stage_non_full(const int stage_to_measure,
                                      const ProcessorType core_type_to_measure) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;
  SPSCQueue<MyTask*, kPoolSize> free_task_pool;
  for (size_t i = 0; i < kPoolSize; ++i) {
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr()));
    free_task_pool.enqueue(preallocated_tasks.back().get());
  }

  SPSCQueue<MyTask*, kPoolSize> q_0_1;
  SPSCQueue<MyTask*, kPoolSize> q_1_2;
  SPSCQueue<MyTask*, kPoolSize> q_2_3;

  // find cores
  std::vector<int> cores_to_use;
  ProcessorType counter_part_a;
  ProcessorType counter_part_b;
  ProcessorType counter_part_c;

  if (core_type_to_measure == ProcessorType::kLittleCore) {
    cores_to_use = g_little_cores;
    counter_part_a = ProcessorType::kMediumCore;
    counter_part_b = ProcessorType::kBigCore;
    counter_part_c = ProcessorType::kVulkan;
  } else if (core_type_to_measure == ProcessorType::kMediumCore) {
    cores_to_use = g_medium_cores;
    counter_part_a = ProcessorType::kLittleCore;
    counter_part_b = ProcessorType::kBigCore;
    counter_part_c = ProcessorType::kVulkan;
  } else if (core_type_to_measure == ProcessorType::kBigCore) {
    cores_to_use = g_big_cores;
    counter_part_a = ProcessorType::kLittleCore;
    counter_part_b = ProcessorType::kMediumCore;
    counter_part_c = ProcessorType::kVulkan;
  } else if (core_type_to_measure == ProcessorType::kVulkan) {
    counter_part_a = ProcessorType::kLittleCore;
    counter_part_b = ProcessorType::kMediumCore;
    counter_part_c = ProcessorType::kBigCore;
  } else {
    throw "asdasdasdasdasdasdasda";
  }

  // Main benchmark
  {
    // Only setup the records once
    RecordManager::instance().setup(100,
                                    {std::make_pair(0, core_type_to_measure),
                                     std::make_pair(1, counter_part_a),
                                     std::make_pair(2, counter_part_b),
                                     std::make_pair(3, counter_part_c)});

    //
    auto t0 =
        std::thread(worker_thread_record<MyTask>,
                    0,
                    std::ref(free_task_pool),
                    std::ref(q_0_1),
                    [&](MyTask& task) {
                      if (core_type_to_measure == ProcessorType::kVulkan) {
                        disp.dispatch_multi_stage(task.appdata, stage_to_measure, stage_to_measure);
                      } else {
                        cifar_sparse::omp::v2::dispatch_multi_stage(cores_to_use,
                                                                    cores_to_use.size(),
                                                                    task.appdata,
                                                                    stage_to_measure,
                                                                    stage_to_measure);
                      }
                    });

    // medium core
    auto t1 = std::thread(
        worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& _) {});

    // big core
    auto t2 = std::thread(
        worker_thread_record<MyTask>, 2, std::ref(q_1_2), std::ref(q_2_3), [&](MyTask& _) {});

    // vulkan core
    auto t3 = std::thread(
        worker_thread_record<MyTask>, 3, std::ref(q_2_3), std::ref(free_task_pool), [&](MyTask& _) {
        });

    t0.join();
    t1.join();
    t2.join();
    t3.join();

    RecordManager::instance().print_processor_type_stats(core_type_to_measure);
  }
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN

  int stage = 0;
  app.add_option("--stage", stage, "Stage to run")->required();

  bool full = false;
  app.add_flag("--full", full, "Run fully occupied benchmark");

  PARSE_ARGS_END

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != "3A021JEHN02756") {
    return 0;
  }

  if (full) {
    BM_run_OMP_stage_full(stage);
  } else {
    BM_run_OMP_stage_non_full(stage, ProcessorType::kLittleCore);
    BM_run_OMP_stage_non_full(stage, ProcessorType::kMediumCore);
    BM_run_OMP_stage_non_full(stage, ProcessorType::kBigCore);
    BM_run_OMP_stage_non_full(stage, ProcessorType::kVulkan);
  }

  return 0;
}
