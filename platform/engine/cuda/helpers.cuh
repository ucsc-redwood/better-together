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
// Post-launch error check
// ----------------------------------------------------------------------------
// A kernel launch returns no value, so a launch-time error (bad grid/block/shared-mem
// config) only surfaces at the NEXT CUDA call — getting mis-attributed to a later
// stage. cudaGetLastError() right after the launch attributes it to THIS launch and is
// cheap (no sync). In debug builds we also synchronize to surface in-kernel execution
// errors at the same site.
namespace cuda {
inline void check_cuda_launch(const char* name, const char* file, int line) {
  cudaError_t err = cudaGetLastError();
#ifndef NDEBUG
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
#endif
  if (err != cudaSuccess) {
    spdlog::error("CUDA launch error ({}) at {}:{}: {}", name, file, line, cudaGetErrorString(err));
    exit(1);
  }
}
}  // namespace cuda
#define CheckCudaLaunch(name) ::cuda::check_cuda_launch(name, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// Simplify launch parameters
// Need to define TOTAL_ITER (e.g., 'total_iter' = 10000), and then write some
// number for BLOCK_SIZE (e.g., 256)
// ----------------------------------------------------------------------------

#define SETUP_DEFAULT_LAUNCH_PARAMS(TOTAL_ITER, BLOCK_SIZE)     \
  static const auto block_dim = dim3{BLOCK_SIZE, 1, 1};         \
  static const auto grid_dim = div_up(TOTAL_ITER, block_dim.x); \
  static constexpr auto shared_mem = 0;
