#pragma once

#include "builtin-apps/app.hpp"
#include "builtin-apps/pipeline/record.hpp"
#include "builtin-apps/pipeline/worker.hpp"

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

static std::vector<int>& get_cores_by_type(const ProcessorType core_type) {
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
// Device 3A021JEHN02756
// ----------------------------------------------------------------------------

namespace device_3A021JEHN02756 {
// clang-format off
  static std::array<Schedule, 10> schedules = {{
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
}  // namespace device_3A021JEHN02756

// ----------------------------------------------------------------------------
// Device 9b034f1b
// ----------------------------------------------------------------------------

namespace device_9b034f1b {}

// ----------------------------------------------------------------------------
// Device jetson
// ----------------------------------------------------------------------------

namespace device_jetson {}

namespace device_jetsonlowpower {}