#pragma once

#include <iostream>
#include <optional>
#include <vector>

#include "../conf.hpp"

// ---------------------------------------------------------------------
// Execution model
// ---------------------------------------------------------------------

enum class ExecutionModel { kOMP, kVulkan, kCuda };

// Define the configuration and execution model.
// This is the configuration for a single chunk.
struct ChunkConfig {
  ExecutionModel exec_model;  // e.g., kOMP, kVulkan, kCuda
  int start_stage;
  int end_stage;
  std::optional<ProcessorType> cpu_proc_type;  // e.g., kLittleCore, kMediumCore, kBigCore
};

// TODO: add CUDA
static inline ProcessorType get_processor_type_from_chunk_config(const ChunkConfig& chunk_config) {
  if (chunk_config.exec_model == ExecutionModel::kOMP) {
    return chunk_config.cpu_proc_type.value();
  }

  return ProcessorType::kVulkan;
}

struct Schedule {
  std::vector<ChunkConfig> chunks;

  [[nodiscard]] size_t n_chunks() const { return chunks.size(); }

  [[nodiscard]] size_t start_stage(const size_t chunk_id) const {
    return chunks[chunk_id].start_stage;
  }

  [[nodiscard]] size_t end_stage(const size_t chunk_id) const { return chunks[chunk_id].end_stage; }

  void print(const size_t id) const {
    std::cout << "Schedule " << id << ":\n";
    const std::string indent = "  ";

    for (size_t i = 0; i < chunks.size(); ++i) {
      const auto& chunk = chunks[i];

      // Get processor type string
      std::string cpu_proc_type;
      if (chunk.exec_model == ExecutionModel::kOMP) {
        switch (chunk.cpu_proc_type.value()) {
          case ProcessorType::kLittleCore:
            cpu_proc_type = "Little";
            break;
          case ProcessorType::kMediumCore:
            cpu_proc_type = "Medium";
            break;
          case ProcessorType::kBigCore:
            cpu_proc_type = "Big   ";
            break;
          default:
            cpu_proc_type = "?";
        }
      }

      // Print chunk header with execution model and processor type
      std::cout << indent << "Chunk " << i << " [";
      switch (chunk.exec_model) {
        case ExecutionModel::kOMP:
          std::cout << "OMP/" << cpu_proc_type;
          break;
        case ExecutionModel::kVulkan:
          std::cout << "Vulkan    ";
          break;
        case ExecutionModel::kCuda:
          std::cout << "CUDA      ";
          break;
      }
      std::cout << "]: ";

      // Print stages included in this chunk
      for (int stage = chunk.start_stage; stage <= chunk.end_stage; ++stage) {
        std::cout << stage;
        if (stage != chunk.end_stage) {
          std::cout << ", ";
        }
      }
      std::cout << std::endl;
    }
  }
};
