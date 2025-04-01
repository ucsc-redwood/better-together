#pragma once

#include <algorithm>
#include <iostream>
#include <unordered_map>

#include "dense_appdata.hpp"

namespace cifar_dense {

[[nodiscard]] inline int arg_max(const float* ptr) {
  const auto max_index =
      std::distance(ptr, std::ranges::max_element(ptr, ptr + cifar_dense::kLinearOutFeatures));

  return max_index;
}

inline void print_prediction(const int max_index) {
  static const std::unordered_map<int, std::string_view> class_names{{0, "airplanes"},
                                                                     {1, "cars"},
                                                                     {2, "birds"},
                                                                     {3, "cats"},
                                                                     {4, "deer"},
                                                                     {5, "dogs"},
                                                                     {6, "frogs"},
                                                                     {7, "horses"},
                                                                     {8, "ships"},
                                                                     {9, "trucks"}};

  std::cout << "Predicted Image: ";
  std::cout << (class_names.contains(max_index) ? class_names.at(max_index) : "Unknown");
  std::cout << std::endl;
}

}  // namespace cifar_dense