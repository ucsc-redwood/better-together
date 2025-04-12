#pragma once

#include <array>
#include <chrono>
#include <iostream>

#include "../conf.hpp"

constexpr size_t kNumToProcess = 100;

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

  void setup(const int chunk_id, const ProcessorType processed_by) {
    for (size_t i = 0; i < kNumToProcess; ++i) {
      records_[i][chunk_id].processed_by = processed_by;
    }
  }

  void start_tick(const uint32_t processing_id, const int chunk_id) {
    records_[processing_id][chunk_id].start = std::chrono::high_resolution_clock::now();
  }

  void end_tick(const uint32_t processing_id, const int chunk_id) {
    records_[processing_id][chunk_id].end = std::chrono::high_resolution_clock::now();
  }

  void dump_records() const {
    for (size_t i = 0; i < kNumToProcess; ++i) {
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
  std::array<std::array<Record, 4>, kNumToProcess> records_;
};
