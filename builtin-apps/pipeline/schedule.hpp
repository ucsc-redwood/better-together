#pragma once

#include <iostream>

#include "../app.hpp"

struct Chunk {
  ProcessorType core;
  std::vector<int> indices;  // e.g., {0} or {1, 2, 3, 4, 5}
};

struct Schedule {
  std::vector<Chunk> chunks;

  [[nodiscard]] size_t n_chunks() const { return chunks.size(); }

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

inline std::vector<int>& get_cores_by_type(const ProcessorType core_type) {
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
