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

  // Initialize the dispatcher and queues
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;
  std::vector<QueueT> queues(n_chunks);
  for (size_t i = 0; i < kPoolSize; ++i) {
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr()));
    queues[0].enqueue(preallocated_tasks.back().get());
  }

  // Run the schedule
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

  std::cout << "### Python Begin ###" << std::endl;

  RecordManager::instance().dump_records_for_python();

  schedule.print();

  std::cout << "### Python End ###" << std::endl;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

#define BIG(...)                                                \
  {                                                             \
    .core = ProcessorType::kBigCore, .indices = { __VA_ARGS__ } \
  }
#define MED(...)                                                   \
  {                                                                \
    .core = ProcessorType::kMediumCore, .indices = { __VA_ARGS__ } \
  }
#define LIT(...)                                                   \
  {                                                                \
    .core = ProcessorType::kLittleCore, .indices = { __VA_ARGS__ } \
  }
#define GPU(...)                                               \
  {                                                            \
    .core = ProcessorType::kVulkan, .indices = { __VA_ARGS__ } \
  }

#define SCHEDULE(...)         \
  {                           \
    .chunks = { __VA_ARGS__ } \
  }

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN

  int schedule_id = 0;
  app.add_option("--schedule", schedule_id, "Schedule to run")->required();

  PARSE_ARGS_END

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != "3A021JEHN02756") {
    return 0;
  }

  assert(schedule_id >= 1 && schedule_id <= 10);

  // clang-format off
  std::array<Schedule, 10> schedules = {{
      SCHEDULE(
          BIG(0),
          GPU(1, 2, 3, 4, 5),
          MED(6, 7, 8)
      ),
      SCHEDULE(
          BIG(0),
          GPU(1, 2, 3, 4, 5),
          MED(6, 7),
          LIT(8)
      ),
      SCHEDULE(
          BIG(0),
          GPU(1, 2, 3, 4),
          LIT(5),
          MED(6, 7, 8)
      ),
      SCHEDULE(
          MED(0),
          GPU(1, 2, 3, 4, 5),
          BIG(6, 7),
          LIT(8)
      ),
      SCHEDULE(
          MED(0),
          GPU(1, 2, 3, 4),
          LIT(5),
          BIG(6, 7, 8)
      ),
      SCHEDULE(
          MED(0),
          GPU(1, 2, 3, 4, 5),
          BIG(6, 7, 8)
      ),
      SCHEDULE(
          GPU(0, 1, 2, 3),
          BIG(4),
          LIT(5),
          MED(6, 7, 8)
      ),
      SCHEDULE(
          GPU(0, 1, 2, 3),
          MED(4),
          LIT(5),
          BIG(6, 7, 8)
      ),
      SCHEDULE(
          GPU(0, 1, 2, 3),
          LIT(4),
          MED(5, 6),
          BIG(7, 8)
      ),
      SCHEDULE(
          GPU(0, 1, 2, 3),
          MED(4, 5, 6),
          BIG(7, 8)
      ),
  }};
  // clang-format on

  Schedule schedule = schedules[schedule_id - 1];

  BM_pipe_cifar_sparse_vk_schedule_auto(schedule);

  return 0;
}
