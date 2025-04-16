#include <omp.h>

#include "builtin-apps/app.hpp"
#include "common.hpp"

// ----------------------------------------------------------------------------
// Specialized Schedules
// ----------------------------------------------------------------------------

struct Chunk {
  ProcessorType core;
  std::vector<int> indices;  // e.g., {0} or {1, 2, 3, 4, 5}
};

struct Schedule {
  std::vector<Chunk> chunks;

  [[nodiscard]] size_t n_chunks() const { return chunks.size(); }

  void setup_record_manager() const {
    std::vector<std::pair<int, ProcessorType>> chunk_records;

    for (size_t i = 0; i < chunks.size(); ++i) {
      chunk_records.push_back(std::make_pair(i, chunks[i].core));
    }

    RecordManager::instance().setup(kNumToProcess, chunk_records);
  }

  void print() const {
    constexpr const char* kProcessorTypeNames[] = {"L", "M", "B", "V"};

    for (const auto& chunk : chunks) {
      std::cout << "[" << kProcessorTypeNames[static_cast<int>(chunk.core)] << "] ";
      for (const auto& index : chunk.indices) {
        std::cout << index << " ";
      }
      std::cout << std::endl;
    }
  }
};

std::vector<int>& get_cores_by_type(const ProcessorType core_type) {
  switch (core_type) {
    case ProcessorType::kLittleCore:
      return g_little_cores;
    case ProcessorType::kMediumCore:
      return g_medium_cores;
    case ProcessorType::kBigCore:
      return g_big_cores;
    default:
      throw std::invalid_argument("Invalid core type");
  }
}

// ----------------------------------------------------------------------------
// Schedule Auto
// ----------------------------------------------------------------------------
using QueueT = SPSCQueue<MyTask*, kPoolSize>;

static void BM_pipe_cifar_sparse_vk_schedule_auto(const Schedule schedule) {
  auto n_chunks = schedule.n_chunks();

  cifar_sparse::vulkan::v2::VulkanDispatcher disp;
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;
  std::vector<QueueT> queues(n_chunks);
  for (size_t i = 0; i < kPoolSize; ++i) {
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr()));
    queues[0].enqueue(preallocated_tasks.back().get());
  }

  {
    std::vector<std::thread> threads;

    schedule.setup_record_manager();

    for (size_t i = 0; i < n_chunks; ++i) {
      QueueT& q_in = queues[i];
      QueueT& q_out = queues[(i + 1) % n_chunks];

      const int start = schedule.chunks[i].indices.front() + 1;
      const int end = schedule.chunks[i].indices.back() + 1;

      const ProcessorType pt = schedule.chunks[i].core;

      if (pt == ProcessorType::kVulkan) {
        threads.emplace_back(create_thread_record(i, q_in, q_out, disp, start, end));
      } else {
        threads.emplace_back(
            create_thread_record(i, q_in, q_out, get_cores_by_type(pt), start, end));
      }
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  RecordManager::instance().dump_records();
}

namespace device_3A021JEHN02756 {

// ----------------------------------------------------------------------------
// Schedule 1
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3, 4, 5]
// Medium = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_1() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kBigCore},
                                        {1, ProcessorType::kVulkan},
                                        {2, ProcessorType::kMediumCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 6);
    auto t2 = create_thread_record(2, q_2, q_0, g_medium_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
  }

  RecordManager::instance().dump_records();
}

// ----------------------------------------------------------------------------
// Schedule 2
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3, 4, 5]
// Medium = [6, 7]
// Little = [8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_2() {
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
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 6);
    auto t2 = create_thread_record(2, q_2, q_3, g_medium_cores, 7, 8);
    auto t3 = create_thread_record(3, q_3, q_0, g_little_cores, 9, 9);

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
// GPU = [1, 2, 3, 4]
// Little = [5]
// Medium = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_3() {
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
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 5);
    auto t2 = create_thread_record(2, q_2, q_3, g_little_cores, 6, 6);
    auto t3 = create_thread_record(3, q_3, q_0, g_medium_cores, 7, 9);

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
// Medium = [0]
// GPU = [1, 2, 3, 4, 5]
// Big = [6, 7]
// Little = [8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_4() {
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
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 6);
    auto t2 = create_thread_record(2, q_2, q_3, g_big_cores, 7, 8);
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
// Medium = [0]
// GPU = [1, 2, 3, 4]
// Little = [5]
// Big = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_5() {
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
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 5);
    auto t2 = create_thread_record(2, q_2, q_3, g_little_cores, 6, 6);
    auto t3 = create_thread_record(3, q_3, q_0, g_big_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }

  RecordManager::instance().dump_records();
}

// ----------------------------------------------------------------------------
// Schedule 6
// ----------------------------------------------------------------------------
// Stage assignments:
// Medium = [0]
// GPU = [1, 2, 3, 4, 5]
// Big = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_6() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kMediumCore},
                                        {1, ProcessorType::kVulkan},
                                        {2, ProcessorType::kBigCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, g_medium_cores, 1, 1);
    auto t1 = create_thread_record(1, q_1, q_2, disp, 2, 6);
    auto t2 = create_thread_record(2, q_2, q_0, g_big_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
  }

  RecordManager::instance().dump_records();
}

// ----------------------------------------------------------------------------
// Schedule 7
// ----------------------------------------------------------------------------
// Stage assignments:
// GPU = [0, 1, 2, 3]
// Big = [4]
// Little = [5]
// Medium = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_7() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kVulkan},
                                        {1, ProcessorType::kBigCore},
                                        {2, ProcessorType::kLittleCore},
                                        {3, ProcessorType::kMediumCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, disp, 1, 4);
    auto t1 = create_thread_record(1, q_1, q_2, g_big_cores, 5, 5);
    auto t2 = create_thread_record(2, q_2, q_3, g_little_cores, 6, 6);
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
// GPU = [0, 1, 2, 3]
// Medium = [4]
// Little = [5]
// Big = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_8() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kVulkan},
                                        {1, ProcessorType::kMediumCore},
                                        {2, ProcessorType::kLittleCore},
                                        {3, ProcessorType::kBigCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, disp, 1, 4);
    auto t1 = create_thread_record(1, q_1, q_2, g_medium_cores, 5, 5);
    auto t2 = create_thread_record(2, q_2, q_3, g_little_cores, 6, 6);
    auto t3 = create_thread_record(3, q_3, q_0, g_big_cores, 7, 9);

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
// GPU = [0, 1, 2, 3]
// Little = [4]
// Medium = [5, 6]
// Big = [7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_9() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kVulkan},
                                        {1, ProcessorType::kLittleCore},
                                        {2, ProcessorType::kMediumCore},
                                        {3, ProcessorType::kBigCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, disp, 1, 4);
    auto t1 = create_thread_record(1, q_1, q_2, g_little_cores, 5, 5);
    auto t2 = create_thread_record(2, q_2, q_3, g_medium_cores, 6, 7);
    auto t3 = create_thread_record(3, q_3, q_0, g_big_cores, 8, 9);

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
// GPU = [0, 1, 2, 3]
// Medium = [4, 5, 6]
// Big = [7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_10() {
  SETUP_DATA;

  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kVulkan},
                                        {1, ProcessorType::kMediumCore},
                                        {2, ProcessorType::kBigCore},
                                    });

    auto t0 = create_thread_record(0, q_0, q_1, disp, 1, 4);
    auto t1 = create_thread_record(1, q_1, q_2, g_medium_cores, 5, 7);
    auto t2 = create_thread_record(2, q_2, q_0, g_big_cores, 8, 9);

    t0.join();
    t1.join();
    t2.join();
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

  Schedule schedule;
  schedule.chunks = {
      {.core = ProcessorType::kBigCore, .indices = {0}},
      {.core = ProcessorType::kVulkan, .indices = {1, 2, 3, 4, 5}},
      {.core = ProcessorType::kMediumCore, .indices = {6, 7, 8}},
  };

  BM_pipe_cifar_sparse_vk_schedule_auto(schedule);

  // switch (schedule_id) {
  //   case 1:
  //     device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_1();
  //     break;
  //   case 2:
  //     device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_2();
  //     break;
  //   case 3:
  //     device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_3();
  //     break;
  //   case 4:
  //     device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_4();
  //     break;
  //   case 5:
  //     device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_5();
  //     break;
  //   case 6:
  //     device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_6();
  //     break;
  //   case 7:
  //     device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_7();
  //     break;
  //   case 8:
  //     device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_8();
  //     break;
  //   case 9:
  //     device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_9();
  //     break;
  //   case 10:
  //     device_3A021JEHN02756::BM_pipe_cifar_sparse_vk_schedule_10();
  //     break;
  //   default:
  //     std::cerr << "Invalid schedule ID. Please choose a schedule between 1 and 10." <<
  //     std::endl; return 1;
  // }

  return 0;
}
