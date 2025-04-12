#pragma once

#include <array>
#include <cassert>
#include <iostream>

#include "../conf.hpp"

// Read the ARM virtual count register (CNTVCT_EL0)
static inline uint64_t now_cycles() {
  uint64_t cycles;
  asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
  return cycles;
}

// Read the counter frequency from CNTFRQ_EL0 for conversion if needed
static inline uint32_t get_counter_frequency() {
  uint32_t freq;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
  return freq;
}

struct Record {
  ProcessorType processed_by;
  uint64_t start;  // Cycle counter when starting
  uint64_t end;    // Cycle counter when finishing
};

class RecordManager {
 public:
  static RecordManager& instance() {
    static RecordManager inst;
    return inst;
  }

  /**
   * @brief Sets up the record manager with the specified number of tasks and processing chunks
   *
   * This method initializes the record manager with a fixed number of tasks to process
   * and configures which processor type (Vulkan, Medium Core, or Big Core) will handle
   * each processing chunk. Each task can be processed by up to 4 different chunks.
   *
   * @param num_to_process The total number of tasks to be processed
   * @param chunks A list of pairs specifying chunk ID and processor type
   *
   * @example
   * // Setup for a pipeline with 3 chunks:
   * // - Chunk 0: Vulkan GPU processing
   * // - Chunk 1: Medium core CPU processing
   * // - Chunk 2: Big core CPU processing
   * RecordManager::instance().setup(100, {
   *     {0, ProcessorType::kVulkan},
   *     {1, ProcessorType::kMediumCore},
   *     {2, ProcessorType::kBigCore}
   * });
   */
  void setup(const int num_to_process,
             const std::initializer_list<std::pair<int, ProcessorType>>& chunks) {
    records_.resize(num_to_process);

    for (const auto& [chunk_id, processed_by] : chunks) {
      assert(chunk_id < 4);
      assert(processed_by == ProcessorType::kVulkan || processed_by == ProcessorType::kMediumCore ||
             processed_by == ProcessorType::kBigCore);

      for (auto& record : records_) {
        record[chunk_id].processed_by = processed_by;
      }
    }
  }

  void start_tick(const uint32_t processing_id, const int chunk_id) {
    records_[processing_id][chunk_id].start = now_cycles();
  }

  void end_tick(const uint32_t processing_id, const int chunk_id) {
    records_[processing_id][chunk_id].end = now_cycles();
  }
  void dump_records() const {
    for (size_t i = 0; i < records_.size(); ++i) {
      std::cout << "Task " << i << ":\n";
      for (size_t j = 0; j < 4; ++j) {
        auto& rec = records_[i][j];

        std::cout << "  Chunk " << j << " (" << processor_type_to_string(rec.processed_by)
                  << "):\n";
        std::cout << "    Start: " << rec.start << " cycles\n";
        std::cout << "    End: " << rec.end << " cycles\n";

        const auto cycles_elapsed = rec.end - rec.start;
        const auto counter_frequency = get_counter_frequency();
        const auto miliseconds_elapsed =
            static_cast<double>(cycles_elapsed) * 1e3 / counter_frequency;
        std::cout << "    Duration: " << miliseconds_elapsed << " ms\n";
      }
    }
  }

  // write a function, given a ProcessorType, return the total time, average time, geomean, median,
  // min, max, standard deviation, cv
  void print_processor_type_stats(const ProcessorType& processor_type) const {

  }

 private:
  RecordManager() = default;
  std::vector<std::array<Record, 4>> records_;

  // write a helper function to convert cycles to microseconds
  static inline double cycles_to_microseconds(const uint64_t cycles, const uint32_t frequency) {
    return static_cast<double>(cycles) * 1e6 / frequency;
  }

  static inline double cycles_to_milliseconds(const uint64_t cycles, const uint32_t frequency) {
    return static_cast<double>(cycles) * 1e3 / frequency;
  }

  static inline std::string processor_type_to_string(const ProcessorType& processor_type) {
    switch (processor_type) {
      case ProcessorType::kVulkan:
        return "GPU";
      case ProcessorType::kMediumCore:
        return "Medium";
      case ProcessorType::kBigCore:
        return "Big";
      case ProcessorType::kLittleCore:
        return "Little";
      default:
        return "Unknown";
    }
  }
};

// struct Record {
//   ProcessorType processed_by;
//   std::chrono::time_point<std::chrono::high_resolution_clock> start;
//   std::chrono::time_point<std::chrono::high_resolution_clock> end;
// };

// class RecordManager {
//  public:
//   static RecordManager& instance() {
//     static RecordManager inst;
//     return inst;
//   }

//   /**
//    * @brief Sets up the record manager with the specified number of tasks and processing chunks
//    *
//    * This method initializes the record manager with a fixed number of tasks to process
//    * and configures which processor type (Vulkan, Medium Core, or Big Core) will handle
//    * each processing chunk. Each task can be processed by up to 4 different chunks.
//    *
//    * @param num_to_process The total number of tasks to be processed
//    * @param chunks A list of pairs specifying chunk ID and processor type
//    *
//    * @example
//    * // Setup for a pipeline with 3 chunks:
//    * // - Chunk 0: Vulkan GPU processing
//    * // - Chunk 1: Medium core CPU processing
//    * // - Chunk 2: Big core CPU processing
//    * RecordManager::instance().setup(100, {
//    *     {0, ProcessorType::kVulkan},
//    *     {1, ProcessorType::kMediumCore},
//    *     {2, ProcessorType::kBigCore}
//    * });
//    */
//   void setup(const int num_to_process,
//              const std::initializer_list<std::pair<int, ProcessorType>>& chunks) {
//     records_.resize(num_to_process);

//     for (const auto& [chunk_id, processed_by] : chunks) {
//       assert(chunk_id < 4);
//       assert(processed_by == ProcessorType::kVulkan || processed_by == ProcessorType::kMediumCore
//       ||
//              processed_by == ProcessorType::kBigCore);

//       for (auto& record : records_) {
//         record[chunk_id].processed_by = processed_by;
//       }
//     }
//   }

//   void start_tick(const uint32_t processing_id, const int chunk_id) {
//     records_[processing_id][chunk_id].start = std::chrono::high_resolution_clock::now();
//   }

//   void end_tick(const uint32_t processing_id, const int chunk_id) {
//     records_[processing_id][chunk_id].end = std::chrono::high_resolution_clock::now();
//   }

//   void dump_records() const {
//     for (size_t i = 0; i < records_.size(); ++i) {
//       std::cout << "Task " << i << ":\n";
//       for (size_t j = 0; j < 4; ++j) {
//         std::cout << "  Chunk " << j << ":\n";
//         std::cout << "    Start: "
//                   << std::chrono::duration_cast<std::chrono::microseconds>(
//                          records_[i][j].start.time_since_epoch())
//                          .count()
//                   << " us\n";
//         std::cout << "    End: "
//                   << std::chrono::duration_cast<std::chrono::microseconds>(
//                          records_[i][j].end.time_since_epoch())
//                          .count()
//                   << " us\n";
//         std::cout << "    Duration: "
//                   << std::chrono::duration_cast<std::chrono::microseconds>(records_[i][j].end -
//                                                                            records_[i][j].start)
//                          .count()
//                   << " us\n";
//       }
//     }
//   }

//  private:
//   RecordManager() = default;
//   std::vector<std::array<Record, 4>> records_;
// };
