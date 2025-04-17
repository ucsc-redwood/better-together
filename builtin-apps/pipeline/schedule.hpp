#pragma once

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
static inline ProcessorType get_processor_type(const ChunkConfig& chunk_config) {
  if (chunk_config.exec_model == ExecutionModel::kOMP) {
    return chunk_config.cpu_proc_type.value();
  }

  return ProcessorType::kVulkan;
}

struct Schedule {
  std::vector<ChunkConfig> chunks;
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
