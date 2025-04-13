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

// ----------------------------------------------------------------------------
// Schedule 1
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2]
// Medium = [3, 4, 5, 6]
// Little = [7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_1() {
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

// ----------------------------------------------------------------------------
// Schedule 2
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2]
// Little = [3]
// Medium = [4, 5, 6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_2() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kBigCore},
                                        {1, ProcessorType::kVulkan},
                                        {2, ProcessorType::kLittleCore},
                                        {3, ProcessorType::kMediumCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 3);
    auto t2 = create_thread_record(2, q_2, q_3, g_little_cores, 4, 4);
    auto t3 = create_thread_record(3, q_3, q_0, g_medium_cores, 5, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  RecordManager::instance().dump_records();
}

// ----------------------------------------------------------------------------
// Schedule 3
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3]
// Medium = [4, 5, 6]
// Little = [7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_3() {
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
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 4);
    auto t2 = create_thread_record(2, q_2, q_3, g_medium_cores, 5, 7);
    auto t3 = create_thread_record(3, q_3, q_0, g_little_cores, 8, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  RecordManager::instance().dump_records();
}

// ----------------------------------------------------------------------------
// Schedule 4
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3]
// Medium = [4, 5, 6, 7]
// Little = [8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_4() {
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
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 4);
    auto t2 = create_thread_record(2, q_2, q_3, g_medium_cores, 5, 8);
    auto t3 = create_thread_record(3, q_3, q_0, g_little_cores, 9, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  RecordManager::instance().dump_records();
}

// ----------------------------------------------------------------------------
// Schedule 5
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3]
// Medium = [4, 5, 6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_5() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kBigCore},
                                        {1, ProcessorType::kVulkan},
                                        {2, ProcessorType::kMediumCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 4);
    auto t2 = create_thread_record(2, q_2, q_0, g_medium_cores, 5, 9);

    t0.join();
    t1.join();
    t2.join();
  }

  RecordManager::instance().dump_records();
}

// ----------------------------------------------------------------------------
// Schedule 6
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3]
// Little = [4]
// Medium = [5, 6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_6() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kBigCore},
                                        {1, ProcessorType::kVulkan},
                                        {2, ProcessorType::kLittleCore},
                                        {3, ProcessorType::kMediumCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 4);
    auto t2 = create_thread_record(2, q_2, q_3, g_little_cores, 5, 5);
    auto t3 = create_thread_record(3, q_3, q_0, g_medium_cores, 6, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  RecordManager::instance().dump_records();
}

// ----------------------------------------------------------------------------
// Schedule 7
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3]
// Little = [4, 5]
// Medium = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_7() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kBigCore},
                                        {1, ProcessorType::kVulkan},
                                        {2, ProcessorType::kLittleCore},
                                        {3, ProcessorType::kMediumCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 4);
    auto t2 = create_thread_record(2, q_2, q_3, g_little_cores, 5, 6);
    auto t3 = create_thread_record(3, q_3, q_0, g_medium_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  RecordManager::instance().dump_records();
}

// ----------------------------------------------------------------------------
// Schedule 8
// ----------------------------------------------------------------------------
// Stage assignments:
// Medium = [0]
// GPU = [1, 2, 3]
// Little = [4]
// Big = [5, 6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_8() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kMediumCore},
                                        {1, ProcessorType::kVulkan},
                                        {2, ProcessorType::kLittleCore},
                                        {3, ProcessorType::kBigCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, g_medium_cores, 1, 1);
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 4);
    auto t2 = create_thread_record(2, q_2, q_3, g_little_cores, 5, 5);
    auto t3 = create_thread_record(3, q_3, q_0, g_big_cores, 6, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  RecordManager::instance().dump_records();
}

// ----------------------------------------------------------------------------
// Schedule 9
// ----------------------------------------------------------------------------
// Stage assignments:
// Medium = [0]
// GPU = [1, 2, 3]
// Big = [4, 5, 6]
// Little = [7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_9() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kMediumCore},
                                        {1, ProcessorType::kVulkan},
                                        {2, ProcessorType::kBigCore},
                                        {3, ProcessorType::kLittleCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, g_medium_cores, 1, 1);
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 4);
    auto t2 = create_thread_record(2, q_2, q_3, g_big_cores, 5, 7);
    auto t3 = create_thread_record(3, q_3, q_0, g_little_cores, 8, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  RecordManager::instance().dump_records();
}

// ----------------------------------------------------------------------------
// Schedule 10
// ----------------------------------------------------------------------------
// Stage assignments:
// Medium = [0]
// GPU = [1, 2, 3]
// Little = [4, 5]
// Big = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_10() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kMediumCore},
                                        {1, ProcessorType::kVulkan},
                                        {2, ProcessorType::kLittleCore},
                                        {3, ProcessorType::kBigCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, g_medium_cores, 1, 1);
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 4);
    auto t2 = create_thread_record(2, q_2, q_3, g_little_cores, 5, 6);
    auto t3 = create_thread_record(3, q_3, q_0, g_big_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  RecordManager::instance().dump_records();
}

}  // namespace device_3A021JEHN02756

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
    case 1:
      device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_1();
      break;
    case 2:
      device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_2();
      break;
    case 3:
      device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_3();
      break;
    case 4:
      device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_4();
      break;
    case 5:
      device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_5();
      break;
    case 6:
      device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_6();
      break;
    case 7:
      device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_7();
      break;
    case 8:
      device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_8();
      break;
    case 9:
      device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_9();
      break;
    case 10:
      device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_10();
      break;
    default:
      device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_best();
  }

  return 0;
}
