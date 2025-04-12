#pragma once

#include <array>
#include <cassert>
#include <chrono>
#include <iostream>

#include "../conf.hpp"

struct Record {
  ProcessorType processed_by;
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
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
    records_[processing_id][chunk_id].start = std::chrono::high_resolution_clock::now();
  }

  void end_tick(const uint32_t processing_id, const int chunk_id) {
    records_[processing_id][chunk_id].end = std::chrono::high_resolution_clock::now();
  }

  void dump_records() const {
    for (size_t i = 0; i < records_.size(); ++i) {
      std::cout << "Task " << i << ":\n";
      for (size_t j = 0; j < 4; ++j) {
        std::cout << "  Chunk " << j << ":\n";
        std::cout << "    Start: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         records_[i][j].start.time_since_epoch())
                         .count()
                  << " us\n";
        std::cout << "    End: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         records_[i][j].end.time_since_epoch())
                         .count()
                  << " us\n";
        std::cout << "    Duration: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(records_[i][j].end -
                                                                           records_[i][j].start)
                         .count()
                  << " us\n";
      }
    }
  }

 private:
  RecordManager() = default;
  std::vector<std::array<Record, 4>> records_;
};
