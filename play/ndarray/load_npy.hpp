#pragma once

#include <cnpy.h>

#include <cstring>
#include <filesystem>

#include "builtin-apps/resources_path.hpp"
#include "ndarray.hpp"

namespace fs = std::filesystem;

// Helper to load an NDArray<ND> from a .npy file
template <size_t ND>
[[nodiscard]] NDArray<ND> load_from_npy(const std::string& filename,
                                        const std::array<size_t, ND>& expected_shape) {
  static const fs::path weights_dir = helpers::get_resource_base_path() / "weights_npy";

  // Load the file using cnpy
  cnpy::NpyArray npy_data = cnpy::npy_load(weights_dir / filename);
  float* raw_data = npy_data.data<float>();

  // Create NDArray with the user-supplied shape
  NDArray<ND> arr(expected_shape);

  // Optional: check if the .npy shape matches expected_shape
  // npy_data.shape is a vector<size_t>, so compare dimension-by-dimension:
  if (npy_data.shape.size() != ND) {
    throw std::runtime_error("Dimension mismatch in " + filename);
  }
  for (size_t i = 0; i < ND; ++i) {
    if (npy_data.shape[i] != expected_shape[i]) {
      throw std::runtime_error(
          "Shape mismatch in " + filename + ": expected " + std::to_string(expected_shape[i]) +
          " but got " + std::to_string(npy_data.shape[i]) + " on dimension " + std::to_string(i));
    }
  }

  // Compute total number of elements
  size_t total_elems = 1;
  for (auto dim : expected_shape) {
    total_elems *= dim;
  }

  // Copy from npy_data → arr.raw() in one shot
  std::memcpy(arr.raw(), raw_data, total_elems * sizeof(float));

  return arr;
}

template <size_t ND>
void load_from_npy_to_raw(const std::string& filename, NDArray<ND>& arr) {
  static const fs::path weights_dir = helpers::get_resource_base_path() / "weights_npy";

  // check arr's shape matches expected_shape
  const auto expected_shape = arr.shape();

  // Load the file using cnpy
  cnpy::NpyArray npy_data = cnpy::npy_load(weights_dir / filename);
  float* raw_data = npy_data.data<float>();

  // Optional: check if the .npy shape matches expected_shape
  // npy_data.shape is a vector<size_t>, so compare dimension-by-dimension:
  if (npy_data.shape.size() != ND) {
    throw std::runtime_error("Dimension mismatch in " + filename);
  }
  for (size_t i = 0; i < ND; ++i) {
    if (npy_data.shape[i] != expected_shape[i]) {
      throw std::runtime_error(
          "Shape mismatch in " + filename + ": expected " + std::to_string(expected_shape[i]) +
          " but got " + std::to_string(npy_data.shape[i]) + " on dimension " + std::to_string(i));
    }
  }

  // Compute total number of elements
  size_t total_elems = 1;
  for (auto dim : expected_shape) {
    total_elems *= dim;
  }

  // Copy from npy_data → arr.raw() in one shot
  std::memcpy(arr.raw(), raw_data, total_elems * sizeof(float));
}