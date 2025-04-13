#include <omp.h>

#include <vector>

#include "builtin-apps/app.hpp"
#include "common.hpp"

// ----------------------------------------------------------------------------
// Specialized Schedules
// ----------------------------------------------------------------------------

namespace device_3A021JEHN02756 {

// ----------------------------------------------------------------------------
// Schedule 0
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2]
// Medium = [3, 4, 5, 6]
// Little = [7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_best() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kBigCore},
                                        {1, ProcessorType::kVulkan},
                                        {2, ProcessorType::kMediumCore},
                                        {3, ProcessorType::kLittleCore},
                                    });
    auto t0 = create_thread_record(0, q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 3);
    auto t2 = create_thread_record(2, q_2, q_3, g_medium_cores, 4, 7);
    auto t3 = create_thread_record(3, q_3, q_0, g_little_cores, 8, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  RecordManager::instance().dump_records();
}

}  // namespace device_3A021JEHN02756

// // Schedule 1:
// //   Chunk 1: Stage [1] → medium (g_medium_cores)
// //   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
// //   Chunk 3: Stage [5] → little (g_little_cores)
// //   Chunk 4: Stages [6, 7, 8, 9] → big (g_big_cores)
// static void BM_pipe_cifar_sparse_vk_schedule_1() {
//   SETUP_CORES_AND_TASKS();
//   SPSCQueue<MyTask*, kPoolSize> q_0_1;
//   SPSCQueue<MyTask*, kPoolSize> q_1_2;
//   SPSCQueue<MyTask*, kPoolSize> q_2_3;

//   {
//     RecordManager::instance().setup(kNumToProcess,
//                                     {
//                                         {0, ProcessorType::kMediumCore},
//                                         {1, ProcessorType::kVulkan},
//                                         {2, ProcessorType::kLittleCore},
//                                         {3, ProcessorType::kBigCore},
//                                     });

//     auto t0 = std::thread(worker_thread_record<MyTask>,
//                           0,
//                           std::ref(free_task_pool),
//                           std::ref(q_0_1),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
//                           });

//     auto t1 = std::thread(
//         worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
//           disp.dispatch_multi_stage(task.appdata, 2, 4);
//         });

//     auto t2 = std::thread(
//         worker_thread_record<MyTask>, 2, std::ref(q_1_2), std::ref(q_2_3), [&](MyTask& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(
//               g_little_cores, g_little_cores.size(), task.appdata, 5, 5);
//         });

//     auto t3 = std::thread(worker_thread_record<MyTask>,
//                           3,
//                           std::ref(q_2_3),
//                           std::ref(free_task_pool),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_big_cores, g_big_cores.size(), task.appdata, 6, 9);
//                           });

//     t0.join();
//     t1.join();
//     t2.join();
//     t3.join();
//   }

//   RecordManager::instance().dump_records();
// }

// // Schedule 2:
// //   Chunk 1: Stage [1] → big (g_big_cores)
// //   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
// //   Chunk 3: Stage [5] → little (g_little_cores)
// //   Chunk 4: Stages [6, 7, 8, 9] → medium (g_medium_cores)
// static void BM_pipe_cifar_sparse_vk_schedule_2() {
//   SETUP_CORES_AND_TASKS();
//   SPSCQueue<MyTask*, kPoolSize> q_0_1;
//   SPSCQueue<MyTask*, kPoolSize> q_1_2;
//   SPSCQueue<MyTask*, kPoolSize> q_2_3;

//   {
//     RecordManager::instance().setup(kNumToProcess,
//                                     {
//                                         {0, ProcessorType::kBigCore},
//                                         {1, ProcessorType::kVulkan},
//                                         {2, ProcessorType::kLittleCore},
//                                         {3, ProcessorType::kMediumCore},
//                                     });

//     auto t0 = std::thread(worker_thread_record<MyTask>,
//                           0,
//                           std::ref(free_task_pool),
//                           std::ref(q_0_1),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
//                           });

//     auto t1 = std::thread(
//         worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
//           disp.dispatch_multi_stage(task.appdata, 2, 4);
//         });

//     auto t2 = std::thread(
//         worker_thread_record<MyTask>, 2, std::ref(q_1_2), std::ref(q_2_3), [&](MyTask& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(
//               g_little_cores, g_little_cores.size(), task.appdata, 5, 5);
//         });

//     auto t3 = std::thread(worker_thread_record<MyTask>,
//                           3,
//                           std::ref(q_2_3),
//                           std::ref(free_task_pool),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_medium_cores, g_medium_cores.size(), task.appdata, 6, 9);
//                           });

//     t0.join();
//     t1.join();
//     t2.join();
//     t3.join();
//   }

//   RecordManager::instance().dump_records();
// }

// // Schedule 3:
// //   Chunk 1: Stage [1] → medium (g_medium_cores)
// //   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
// //   Chunk 3: Stages [5, 6, 7] → big (g_big_cores)
// //   Chunk 4: Stages [8, 9] → little (g_little_cores)
// static void BM_pipe_cifar_sparse_vk_schedule_3() {
//   SETUP_CORES_AND_TASKS();
//   SPSCQueue<MyTask*, kPoolSize> q_0_1;
//   SPSCQueue<MyTask*, kPoolSize> q_1_2;
//   SPSCQueue<MyTask*, kPoolSize> q_2_3;

//   {
//     RecordManager::instance().setup(kNumToProcess,
//                                     {
//                                         {0, ProcessorType::kMediumCore},
//                                         {1, ProcessorType::kVulkan},
//                                         {2, ProcessorType::kBigCore},
//                                         {3, ProcessorType::kLittleCore},
//                                     });

//     auto t0 = std::thread(worker_thread_record<MyTask>,
//                           0,
//                           std::ref(free_task_pool),
//                           std::ref(q_0_1),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
//                           });

//     auto t1 = std::thread(
//         worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
//           disp.dispatch_multi_stage(task.appdata, 2, 4);
//         });

//     auto t2 = std::thread(
//         worker_thread_record<MyTask>, 2, std::ref(q_1_2), std::ref(q_2_3), [&](MyTask& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(
//               g_big_cores, g_big_cores.size(), task.appdata, 5, 7);
//         });

//     auto t3 = std::thread(worker_thread_record<MyTask>,
//                           3,
//                           std::ref(q_2_3),
//                           std::ref(free_task_pool),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_little_cores, g_little_cores.size(), task.appdata, 8, 9);
//                           });

//     t0.join();
//     t1.join();
//     t2.join();
//     t3.join();
//   }

//   RecordManager::instance().dump_records();
// }

// // Schedule 4:
// //   Chunk 1: Stage [1] → big (g_big_cores)
// //   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
// //   Chunk 3: Stages [5, 6, 7] → medium (g_medium_cores)
// //   Chunk 4: Stages [8, 9] → little (g_little_cores)
// static void BM_pipe_cifar_sparse_vk_schedule_4() {
//   SETUP_CORES_AND_TASKS();
//   SPSCQueue<MyTask*, kPoolSize> q_0_1;
//   SPSCQueue<MyTask*, kPoolSize> q_1_2;
//   SPSCQueue<MyTask*, kPoolSize> q_2_3;

//   {
//     RecordManager::instance().setup(kNumToProcess,
//                                     {
//                                         {0, ProcessorType::kBigCore},
//                                         {1, ProcessorType::kVulkan},
//                                         {2, ProcessorType::kMediumCore},
//                                         {3, ProcessorType::kLittleCore},
//                                     });

//     auto t0 = std::thread(worker_thread_record<MyTask>,
//                           0,
//                           std::ref(free_task_pool),
//                           std::ref(q_0_1),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
//                           });

//     auto t1 = std::thread(
//         worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
//           disp.dispatch_multi_stage(task.appdata, 2, 4);
//         });

//     auto t2 = std::thread(
//         worker_thread_record<MyTask>, 2, std::ref(q_1_2), std::ref(q_2_3), [&](MyTask& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(
//               g_medium_cores, g_medium_cores.size(), task.appdata, 5, 7);
//         });

//     auto t3 = std::thread(worker_thread_record<MyTask>,
//                           3,
//                           std::ref(q_2_3),
//                           std::ref(free_task_pool),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_little_cores, g_little_cores.size(), task.appdata, 8, 9);
//                           });

//     t0.join();
//     t1.join();
//     t2.join();
//     t3.join();
//   }

//   RecordManager::instance().dump_records();
// }

// // Schedule 5:
// //   Chunk 1: Stage [1] → medium (g_medium_cores)
// //   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
// //   Chunk 3: Stages [5, 6, 7, 8] → big (g_big_cores)
// //   Chunk 4: Stage [9] → little (g_little_cores)
// static void BM_pipe_cifar_sparse_vk_schedule_5() {
//   SETUP_CORES_AND_TASKS();
//   SPSCQueue<MyTask*, kPoolSize> q_0_1;
//   SPSCQueue<MyTask*, kPoolSize> q_1_2;
//   SPSCQueue<MyTask*, kPoolSize> q_2_3;

//   {
//     RecordManager::instance().setup(kNumToProcess,
//                                     {
//                                         {0, ProcessorType::kMediumCore},
//                                         {1, ProcessorType::kVulkan},
//                                         {2, ProcessorType::kBigCore},
//                                         {3, ProcessorType::kLittleCore},
//                                     });

//     auto t0 = std::thread(worker_thread_record<MyTask>,
//                           0,
//                           std::ref(free_task_pool),
//                           std::ref(q_0_1),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
//                           });

//     auto t1 = std::thread(
//         worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
//           disp.dispatch_multi_stage(task.appdata, 2, 4);
//         });

//     auto t2 = std::thread(
//         worker_thread_record<MyTask>, 2, std::ref(q_1_2), std::ref(q_2_3), [&](MyTask& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(
//               g_big_cores, g_big_cores.size(), task.appdata, 5, 8);
//         });

//     auto t3 = std::thread(worker_thread_record<MyTask>,
//                           3,
//                           std::ref(q_2_3),
//                           std::ref(free_task_pool),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_little_cores, g_little_cores.size(), task.appdata, 9, 9);
//                           });

//     t0.join();
//     t1.join();
//     t2.join();
//     t3.join();
//   }

//   RecordManager::instance().dump_records();
// }

// // Schedule 6:
// //   Chunk 1: Stage [1] → big (g_big_cores)
// //   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
// //   Chunk 3: Stages [5, 6, 7, 8] → medium (g_medium_cores)
// //   Chunk 4: Stage [9] → little (g_little_cores)
// static void BM_pipe_cifar_sparse_vk_schedule_6() {
//   SETUP_CORES_AND_TASKS();
//   SPSCQueue<MyTask*, kPoolSize> q_0_1;
//   SPSCQueue<MyTask*, kPoolSize> q_1_2;
//   SPSCQueue<MyTask*, kPoolSize> q_2_3;

//   {
//     RecordManager::instance().setup(kNumToProcess,
//                                     {
//                                         {0, ProcessorType::kBigCore},
//                                         {1, ProcessorType::kVulkan},
//                                         {2, ProcessorType::kMediumCore},
//                                         {3, ProcessorType::kLittleCore},
//                                     });

//     auto t0 = std::thread(worker_thread_record<MyTask>,
//                           0,
//                           std::ref(free_task_pool),
//                           std::ref(q_0_1),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
//                           });

//     auto t1 = std::thread(
//         worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
//           disp.dispatch_multi_stage(task.appdata, 2, 4);
//         });

//     auto t2 = std::thread(
//         worker_thread_record<MyTask>, 2, std::ref(q_1_2), std::ref(q_2_3), [&](MyTask& task) {
//           cifar_sparse::omp::v2::dispatch_multi_stage(
//               g_medium_cores, g_medium_cores.size(), task.appdata, 5, 8);
//         });

//     auto t3 = std::thread(worker_thread_record<MyTask>,
//                           3,
//                           std::ref(q_2_3),
//                           std::ref(free_task_pool),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_little_cores, g_little_cores.size(), task.appdata, 9, 9);
//                           });

//     t0.join();
//     t1.join();
//     t2.join();
//     t3.join();
//   }

//   RecordManager::instance().dump_records();
// }

// // Schedule 7 (3 chunks):
// //   Chunk 1: Stage [1] → medium (g_medium_cores)
// //   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
// //   Chunk 3: Stages [5, 6, 7, 8, 9] → big (g_big_cores)
// static void BM_pipe_cifar_sparse_vk_schedule_7() {
//   SETUP_CORES_AND_TASKS();
//   SPSCQueue<MyTask*, kPoolSize> q_0_1;
//   SPSCQueue<MyTask*, kPoolSize> q_1_2;

//   {
//     RecordManager::instance().setup(kNumToProcess,
//                                     {
//                                         {0, ProcessorType::kMediumCore},
//                                         {1, ProcessorType::kVulkan},
//                                         {2, ProcessorType::kBigCore},
//                                     });

//     auto t0 = std::thread(worker_thread_record<MyTask>,
//                           0,
//                           std::ref(free_task_pool),
//                           std::ref(q_0_1),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
//                           });

//     auto t1 = std::thread(
//         worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
//           disp.dispatch_multi_stage(task.appdata, 2, 4);
//         });

//     auto t2 = std::thread(worker_thread_record<MyTask>,
//                           2,
//                           std::ref(q_1_2),
//                           std::ref(free_task_pool),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_big_cores, g_big_cores.size(), task.appdata, 5, 9);
//                           });

//     t0.join();
//     t1.join();
//     t2.join();
//   }

//   RecordManager::instance().dump_records();
// }

// // Schedule 8 (3 chunks):
// //   Chunk 1: Stage [1] → big (g_big_cores)
// //   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
// //   Chunk 3: Stages [5, 6, 7, 8, 9] → medium (g_medium_cores)
// static void BM_pipe_cifar_sparse_vk_schedule_8() {
//   SETUP_CORES_AND_TASKS();
//   SPSCQueue<MyTask*, kPoolSize> q_0_1;
//   SPSCQueue<MyTask*, kPoolSize> q_1_2;

//   {
//     RecordManager::instance().setup(kNumToProcess,
//                                     {
//                                         {0, ProcessorType::kBigCore},
//                                         {1, ProcessorType::kVulkan},
//                                         {2, ProcessorType::kMediumCore},
//                                     });

//     auto t0 = std::thread(worker_thread_record<MyTask>,
//                           0,
//                           std::ref(free_task_pool),
//                           std::ref(q_0_1),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
//                           });

//     auto t1 = std::thread(
//         worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
//           disp.dispatch_multi_stage(task.appdata, 2, 4);
//         });

//     auto t2 = std::thread(worker_thread_record<MyTask>,
//                           2,
//                           std::ref(q_1_2),
//                           std::ref(free_task_pool),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_medium_cores, g_medium_cores.size(), task.appdata, 5, 9);
//                           });

//     t0.join();
//     t1.join();
//     t2.join();
//   }

//   RecordManager::instance().dump_records();
// }

// // Schedule 9 (3 chunks, same as Schedule 7):
// //   Chunk 1: Stage [1] → medium (g_medium_cores)
// //   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
// //   Chunk 3: Stages [5, 6, 7, 8, 9] → big (g_big_cores)
// static void BM_pipe_cifar_sparse_vk_schedule_9() {
//   SETUP_CORES_AND_TASKS();
//   SPSCQueue<MyTask*, kPoolSize> q_0_1;
//   SPSCQueue<MyTask*, kPoolSize> q_1_2;

//   {
//     RecordManager::instance().setup(kNumToProcess,
//                                     {
//                                         {0, ProcessorType::kMediumCore},
//                                         {1, ProcessorType::kVulkan},
//                                         {2, ProcessorType::kBigCore},
//                                     });

//     auto t0 = std::thread(worker_thread_record<MyTask>,
//                           0,
//                           std::ref(free_task_pool),
//                           std::ref(q_0_1),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_medium_cores, g_medium_cores.size(), task.appdata, 1, 1);
//                           });

//     auto t1 = std::thread(
//         worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
//           disp.dispatch_multi_stage(task.appdata, 2, 4);
//         });

//     auto t2 = std::thread(worker_thread_record<MyTask>,
//                           2,
//                           std::ref(q_1_2),
//                           std::ref(free_task_pool),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_big_cores, g_big_cores.size(), task.appdata, 5, 9);
//                           });

//     t0.join();
//     t1.join();
//     t2.join();
//   }

//   RecordManager::instance().dump_records();
// }

// // Schedule 10 (3 chunks, same as Schedule 8):
// //   Chunk 1: Stage [1] → big (g_big_cores)
// //   Chunk 2: Stages [2, 3, 4] → gpu_vulkan
// //   Chunk 3: Stages [5, 6, 7, 8, 9] → medium (g_medium_cores)
// static void BM_pipe_cifar_sparse_vk_schedule_10() {
//   SETUP_CORES_AND_TASKS();
//   SPSCQueue<MyTask*, kPoolSize> q_0_1;
//   SPSCQueue<MyTask*, kPoolSize> q_1_2;

//   {
//     RecordManager::instance().setup(kNumToProcess,
//                                     {
//                                         {0, ProcessorType::kBigCore},
//                                         {1, ProcessorType::kVulkan},
//                                         {2, ProcessorType::kMediumCore},
//                                     });

//     auto t0 = std::thread(worker_thread_record<MyTask>,
//                           0,
//                           std::ref(free_task_pool),
//                           std::ref(q_0_1),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_big_cores, g_big_cores.size(), task.appdata, 1, 1);
//                           });

//     auto t1 = std::thread(
//         worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
//           disp.dispatch_multi_stage(task.appdata, 2, 4);
//         });

//     auto t2 = std::thread(worker_thread_record<MyTask>,
//                           2,
//                           std::ref(q_1_2),
//                           std::ref(free_task_pool),
//                           [&](MyTask& task) {
//                             cifar_sparse::omp::v2::dispatch_multi_stage(
//                                 g_medium_cores, g_medium_cores.size(), task.appdata, 5, 9);
//                           });

//     t0.join();
//     t1.join();
//     t2.join();
//   }

//   RecordManager::instance().dump_records();
// }

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
    // case 1:
    //   BM_pipe_cifar_sparse_vk_schedule_1();
    //   break;
    // case 2:
    //   BM_pipe_cifar_sparse_vk_schedule_2();
    //   break;
    // case 3:
    //   BM_pipe_cifar_sparse_vk_schedule_3();
    //   break;
    // case 4:
    //   BM_pipe_cifar_sparse_vk_schedule_4();
    //   break;
    // case 5:
    //   BM_pipe_cifar_sparse_vk_schedule_5();
    //   break;
    // case 6:
    //   BM_pipe_cifar_sparse_vk_schedule_6();
    //   break;
    // case 7:
    //   BM_pipe_cifar_sparse_vk_schedule_7();
    //   break;
    // case 8:
    //   BM_pipe_cifar_sparse_vk_schedule_8();
    //   break;
    // case 9:
    //   BM_pipe_cifar_sparse_vk_schedule_9();
    //   break;
    // case 10:
    //   BM_pipe_cifar_sparse_vk_schedule_10();
    //   break;
    default:
      BM_pipe_cifar_sparse_vk_schedule_best();
  }

  return 0;
}
