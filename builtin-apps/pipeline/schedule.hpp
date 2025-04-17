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

  void print() const {
    for (size_t i = 0; i < chunks.size(); ++i) {
      const auto& chunk = chunks[i];

      // Get processor type string
      std::string proc_type;
      if (chunk.exec_model == ExecutionModel::kOMP) {
        switch (chunk.cpu_proc_type.value()) {
          case ProcessorType::kLittleCore:
            proc_type = "Little";
            break;
          case ProcessorType::kMediumCore:
            proc_type = "Medium";
            break;
          case ProcessorType::kBigCore:
            proc_type = "Big   ";
            break;
          default:
            proc_type = "?";
        }
      } else if (chunk.exec_model == ExecutionModel::kVulkan) {
        proc_type = "Vulkan";
      } else if (chunk.exec_model == ExecutionModel::kCuda) {
        proc_type = "Cuda  ";
      }

      // Print chunk header with execution model and processor type
      std::cout << "Chunk " << i << " [";
      switch (chunk.exec_model) {
        case ExecutionModel::kOMP:
          std::cout << "OMP/" << proc_type;
          break;
        case ExecutionModel::kVulkan:
          std::cout << "Vulkan";
          break;
        case ExecutionModel::kCuda:
          std::cout << "CUDA";
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

// #include <iostream>

// #include "../app.hpp"

// struct Chunk {
//   ProcessorType core;
//   std::vector<int> indices;  // e.g., {0} or {1, 2, 3, 4, 5}
// };

// struct Schedule {
//   std::vector<Chunk> chunks;

//   [[nodiscard]] size_t n_chunks() const { return chunks.size(); }

//   void print() const {
//     constexpr const char* kProcessorTypeNames[] = {"L", "M", "B", "V"};

//     for (const auto& chunk : chunks) {
//       std::cout << "[" << kProcessorTypeNames[static_cast<int>(chunk.core)] << "] ";
//       for (const auto& index : chunk.indices) {
//         std::cout << index << " ";
//       }
//       std::cout << std::endl;
//     }
//   }
// };

// inline std::vector<int>& get_cores_by_type(const ProcessorType core_type) {
//   switch (core_type) {
//     case ProcessorType::kLittleCore:
//       return g_little_cores;
//     case ProcessorType::kMediumCore:
//       return g_medium_cores;
//     case ProcessorType::kBigCore:
//       return g_big_cores;
//     default:
//       throw std::invalid_argument("Invalid core type");
//   }
// }
