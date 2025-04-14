#include "05_edge_count.cuh"

namespace cuda {

namespace kernels {

__device__ __forceinline__ void process_edge_count_i(const int i,
                                                     const uint8_t* prefix_n,
                                                     const int* parents,
                                                     int* edge_count) {
  const auto my_depth = prefix_n[i] / 3;
  const auto parent_depth = prefix_n[parents[i]] / 3;
  edge_count[i] = my_depth - parent_depth;
}

__global__ void k_EdgeCount(const uint8_t* prefix_n,
                            const int* parents,
                            int* edge_count,
                            int n_brt_nodes) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;
  for (auto i = idx; i < n_brt_nodes; i += stride) {
    process_edge_count_i(i, prefix_n, parents, edge_count);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    edge_count[0] = 0;
  }
}

}  // namespace kernels

}  // namespace cuda
