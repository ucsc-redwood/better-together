#include "06_prefix_sum.cuh"
#include "agents/prefix_sum_agent.cuh"

namespace cuda {

namespace kernels {

// ============================================================================
// Kernel entry points (template's implementation)
// ============================================================================

template <typename T>
__global__ void k_PrefixSumLocal(const T* u_input, T* u_output, const int n, T* u_auxiliary) {
  using TempStorage = typename agents::PrefixSumAgent<T>::TempStorage_LoadScanStore;
  __shared__ TempStorage temp_storage;

  // Process tiles
  agents::PrefixSumAgent<T> agent(n);
  agent.Process_LocalPrefixSums(temp_storage, u_input, u_output, u_auxiliary);
}

template <typename T>
__global__ void k_SingleBlockExclusiveScan(const T* u_input, T* u_output, const int n) {
  using TempStorage = typename agents::PrefixSumAgent<T>::TempStorage_LoadScanStore;
  __shared__ TempStorage temp_storage;

  // Process tiles
  agents::PrefixSumAgent<T> agent(n);
  agent.Process_SingleBlockExclusiveScan(temp_storage, u_input, u_output);
}

template <typename T>
__global__ void k_MakeGlobalPrefixSum(const T* u_local_sums,
                                      const T* u_auxiliary_summed,
                                      T* u_global_sums,
                                      const int n) {
  using TempStorage = typename agents::PrefixSumAgent<T>::TempStorage_LoadStore;
  __shared__ TempStorage temp_storage;

  // Process tiles
  agents::PrefixSumAgent<T> agent(n);
  agent.Process_GlobalPrefixSum(temp_storage, u_local_sums, u_auxiliary_summed, u_global_sums);
}

// ============================================================================
// Instantiations (int and unsigned int)
// ============================================================================

template __global__ void k_PrefixSumLocal(const int* u_input,
                                          int* u_output,
                                          int n,
                                          int* u_auxiliary);

template __global__ void k_SingleBlockExclusiveScan(const int* u_input, int* u_output, int n);

template __global__ void k_MakeGlobalPrefixSum(const int* u_local_sums,
                                               const int* u_auxiliary_summed,
                                               int* u_global_sums,
                                               int n);

template __global__ void k_PrefixSumLocal(const unsigned int* u_input,
                                          unsigned int* u_output,
                                          int n,
                                          unsigned int* u_auxiliary);

template __global__ void k_SingleBlockExclusiveScan(const unsigned int* u_input,
                                                    unsigned int* u_output,
                                                    int n);

template __global__ void k_MakeGlobalPrefixSum(const unsigned int* u_local_sums,
                                               const unsigned int* u_auxiliary_summed,
                                               unsigned int* u_global_sums,
                                               int n);

}  // namespace kernels

}  // namespace cuda
