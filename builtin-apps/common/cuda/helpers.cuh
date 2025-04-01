#pragma once

#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>

// ----------------------------------------------------------------------------
// Math
// ----------------------------------------------------------------------------

constexpr size_t div_up(const size_t a, const size_t b) { return (a + b - 1) / b; }

// ----------------------------------------------------------------------------
// Helper function to handle CUDA errors
// ----------------------------------------------------------------------------

#define CheckCuda(call)                                                                    \
  do {                                                                                     \
    cudaError_t err = call;                                                                \
    if (err != cudaSuccess) {                                                              \
      spdlog::error(                                                                       \
          "CUDA error in {} at line {}: {}", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)

// ----------------------------------------------------------------------------
// Simplify launch parameters
// Need to define TOTAL_ITER (e.g., 'total_iter' = 10000), and then write some
// number for BLOCK_SIZE (e.g., 256)
// ----------------------------------------------------------------------------

#define SETUP_DEFAULT_LAUNCH_PARAMS(TOTAL_ITER, BLOCK_SIZE)     \
  static const auto block_dim = dim3{BLOCK_SIZE, 1, 1};         \
  static const auto grid_dim = div_up(TOTAL_ITER, block_dim.x); \
  static constexpr auto shared_mem = 0;
