#pragma once

namespace cuda {

namespace kernels {

// ============================================================================
// Kernel entry points
// ============================================================================

template <typename T>
__global__ void k_PrefixSumLocal(const T* u_input, T* u_output, int n, T* u_auxiliary = nullptr);

template <typename T>
__global__ void k_SingleBlockExclusiveScan(const T* u_input, T* u_output, int n);

template <typename T>
__global__ void k_MakeGlobalPrefixSum(const T* u_local_sums,
                                      const T* u_auxiliary_summed,
                                      T* u_global_sums,
                                      int n);

}  // namespace kernels

}  // namespace cuda
