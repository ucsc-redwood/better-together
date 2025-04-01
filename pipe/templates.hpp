#pragma once

#include <memory_resource>
#include <vector>

template <typename AppDataType>
[[nodiscard]] std::vector<AppDataType> init_appdata(std::pmr::memory_resource *mr,
                                                    const size_t num_tasks) {
  std::vector<AppDataType> all_data;
  all_data.reserve(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    all_data.emplace_back(mr);  // Each has big vectors
  }
  return all_data;
}
