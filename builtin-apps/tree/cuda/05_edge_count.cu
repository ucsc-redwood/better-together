#include "05_edge_count.cuh"

namespace cuda {

namespace kernels {

__device__ __forceinline__ void process_edge_count_i(const int i,
                                                     const uint8_t* prefix_n,
                                                     const int* parents,
                                                     int* edge_count) {
  // Octant depth = (prefix_n - 1)/3, NOT prefix_n/3 (prefix_n = clz(xor)-1 over a
  // 30-bit morton in a 32-bit word over-counts by 1; plain /3 rounds up an octant
  // at octant boundaries -> children mis-rooted one cell to the side). Matches
  // OMP func_edge.hpp / Vulkan tree_build_octree.comp.
  const auto my_depth = (static_cast<int>(prefix_n[i]) - 1) / 3;
  const auto parent_depth = (static_cast<int>(prefix_n[parents[i]]) - 1) / 3;
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
    // brt node 0 (radix-tree root) contributes exactly ONE octree node: the
    // full-domain ROOT (cell = range). Was 0 -> no root node, and the cross-brt
    // parent walk degenerated onto octree node 0 (deepest cell). Matches OMP/Vulkan.
    edge_count[0] = 1;
  }
}

}  // namespace kernels

}  // namespace cuda
