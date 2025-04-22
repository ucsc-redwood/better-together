#pragma once

#include <cnpy.h>

#include <cstring>
#include <filesystem>

#include "builtin-apps/resources_path.hpp"
#include "ndarray.hpp"

namespace fs = std::filesystem;

namespace npy_loader {

// Helper function to resolve path to the npy file
inline fs::path resolve_npy_path(const std::string& filename) {
  return helpers::get_resource_base_path() / filename;
}

// Load 1D array from npy file
inline bool load_npy_to_ndarray(const std::string& filename, Ndarray1D& ndarray) {
  try {
    cnpy::NpyArray arr = cnpy::npy_load(resolve_npy_path(filename).string());

    // Check if loaded array is 1D
    if (arr.shape.size() != 1) {
      return false;
    }

    // Check dimensions match
    if (arr.shape[0] != static_cast<size_t>(ndarray.d0())) {
      return false;
    }

    // Check data type
    if (arr.word_size != sizeof(float)) {
      return false;
    }

    // Copy data
    std::memcpy(ndarray.data(), arr.data<float>(), ndarray.total_size() * sizeof(float));
    return true;
  } catch (const std::exception& e) {
    return false;
  }
}

// Load 2D array from npy file
inline bool load_npy_to_ndarray(const std::string& filename, Ndarray2D& ndarray) {
  try {
    cnpy::NpyArray arr = cnpy::npy_load(resolve_npy_path(filename).string());

    // Check if loaded array is 2D
    if (arr.shape.size() != 2) {
      return false;
    }

    // Check dimensions match
    if (arr.shape[0] != static_cast<size_t>(ndarray.d0()) ||
        arr.shape[1] != static_cast<size_t>(ndarray.d1())) {
      return false;
    }

    // Check data type
    if (arr.word_size != sizeof(float)) {
      return false;
    }

    // Copy data
    std::memcpy(ndarray.data(), arr.data<float>(), ndarray.total_size() * sizeof(float));
    return true;
  } catch (const std::exception& e) {
    return false;
  }
}

// Load 4D array from npy file
inline bool load_npy_to_ndarray(const std::string& filename, Ndarray4D& ndarray) {
  try {
    cnpy::NpyArray arr = cnpy::npy_load(resolve_npy_path(filename).string());

    // Check if loaded array is 4D
    if (arr.shape.size() != 4) {
      return false;
    }

    // Check dimensions match
    if (arr.shape[0] != static_cast<size_t>(ndarray.d0()) ||
        arr.shape[1] != static_cast<size_t>(ndarray.d1()) ||
        arr.shape[2] != static_cast<size_t>(ndarray.d2()) ||
        arr.shape[3] != static_cast<size_t>(ndarray.d3())) {
      return false;
    }

    // Check data type
    if (arr.word_size != sizeof(float)) {
      return false;
    }

    // Copy data
    std::memcpy(ndarray.data(), arr.data<float>(), ndarray.total_size() * sizeof(float));
    return true;
  } catch (const std::exception& e) {
    return false;
  }
}

}  // namespace npy_loader
