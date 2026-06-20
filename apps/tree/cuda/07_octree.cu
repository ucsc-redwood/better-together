#include <device_launch_parameters.h>

#include "07_octree.cuh"
#include "func_morton.cuh"
namespace cuda {

namespace kernels {

constexpr auto morton_bits = 30;

// Geometric octree depth of a radix node = (prefix_n - 1)/3 (see OMP
// func_octree.hpp oct_depth() / func_edge.hpp for why, not prefix_n/3).
__device__ __forceinline__ int oct_depth(const int prefix_n) { return (prefix_n - 1) / 3; }

__device__ __forceinline__ void set_child(const int node_idx,
                                          int (*u_children)[8],
                                          int* u_child_node_mask,
                                          const unsigned int which_child,
                                          const int oct_idx) {
  u_children[node_idx][which_child] = oct_idx;
  // Several brt nodes target the same octree node concurrently; a plain |= is a
  // racy read-modify-write that drops bits. Match the OMP (#pragma omp atomic)
  // and Vulkan (atomicOr) golden so the mask is order-independent. (BUGS-FOUND §8)
  atomicOr(&u_child_node_mask[node_idx], 1 << which_child);
}

__device__ __forceinline__ void set_leaf(const int node_idx,
                                         int (*u_children)[8],
                                         int* u_child_leaf_mask,
                                         const unsigned int which_child,
                                         const int leaf_idx) {
  u_children[node_idx][which_child] = leaf_idx;
  u_child_leaf_mask[node_idx] &= ~(1 << which_child);
}

// processing for index 'i'
__device__ __forceinline__ void process_oct_node(const int i /*brt node index*/,
                                                 // --------------------------
                                                 int (*oct_children)[8],
                                                 glm::vec4* oct_corner,
                                                 float* oct_cell_size,
                                                 int* oct_child_node_mask,
                                                 // --------------------------
                                                 const int* edge_offsets,
                                                 const int* edge_counts,
                                                 const uint32_t* morton_codes,
                                                 const uint8_t* rt_prefix_n,
                                                 const int* rt_parents,
                                                 const float min_coord,
                                                 const float range) {
  // For octrees, it starts at 'offset[x]', and the numbers is decided by the
  // 'count[i]'. You can imagine something like:
  // brt[0] contains oct nodes [0, 3] (4 total)
  // brt[1] contains oct nodes [4, 4] (1 total)
  // brt[2] contains oct nodes [5, 6] (2 total) ...
  // edge_offsets is an INCLUSIVE prefix sum (offset[i] includes count[i]); the
  // first octree node of brt node i is the EXCLUSIVE prefix sum
  // offset[i] - count[i] (see func_octree.hpp / tree_diff_oracle.hpp).
  // just a constant
  const auto root_level = oct_depth(rt_prefix_n[0]);

  // brt node 0 (radix-tree root) owns exactly ONE octree node: the full-domain
  // ROOT (cell = range). Write its geometry and return (it has no octree parent).
  if (i == 0) {
    const auto root_oct_idx = edge_offsets[0] - edge_counts[0];
    const auto root_prefix = morton_codes[0] >> (morton_bits - (3 * root_level))
                                                    << (morton_bits - (3 * root_level));
    morton32_to_xyz(&oct_corner[root_oct_idx], root_prefix, min_coord, range);
    oct_cell_size[root_oct_idx] = range;  // range / 2^(root_level - root_level)
    return;
  }

  auto oct_idx = edge_offsets[i] - edge_counts[i];
  const auto n_new_nodes = edge_counts[i];

  // for each new node,
  // (1) create their cornor/cell size
  // (2) attach them to their parent
  for (auto j = 0; j < n_new_nodes - 1; ++j) {
    const auto level = oct_depth(rt_prefix_n[i]) - j;  // every new node has a level

    const auto node_prefix = morton_codes[i] >> (morton_bits - (3 * level));
    const auto which_child = node_prefix & 0b111;
    const auto parent = oct_idx + 1;

    // set the parent's child to the current octnode
    set_child(parent, oct_children, oct_child_node_mask, which_child, oct_idx);

    // compute the corner of the current octnode
    morton32_to_xyz(
        &oct_corner[oct_idx], node_prefix << (morton_bits - (3 * level)), min_coord, range);

    // each cell is half the size of the level above it
    oct_cell_size[oct_idx] = range / static_cast<float>(1 << (level - root_level));

    // go to the next octnode (parent)
    oct_idx = parent;
  }

  if (n_new_nodes > 0) {
    auto rt_parent = rt_parents[i];

    auto counter = 0;
    while (edge_counts[rt_parent] == 0) {
      rt_parent = rt_parents[rt_parent];

      ++counter;
      if (counter > 30) {
        // 64 / 3
        break;
      }
    }

    const auto oct_parent = edge_offsets[rt_parent] - edge_counts[rt_parent];
    const auto top_level = oct_depth(rt_prefix_n[i]) - n_new_nodes + 1;
    const auto top_node_prefix = morton_codes[i] >> (morton_bits - (3 * top_level));

    const auto which_child = top_node_prefix & 0b111;

    set_child(oct_parent, oct_children, oct_child_node_mask, which_child, oct_idx);

    morton32_to_xyz(
        &oct_corner[oct_idx], top_node_prefix << (morton_bits - (3 * top_level)), min_coord, range);

    oct_cell_size[oct_idx] = range / static_cast<float>(1 << (top_level - root_level));
  }
}

__device__ __forceinline__ void process_link_leaf(const int i /*brt node index*/,
                                                  // --------------------------
                                                  int (*oct_children)[8],
                                                  int* oct_child_leaf_mask,
                                                  // --------------------------
                                                  const int* edge_offsets,
                                                  const int* edge_counts,
                                                  const uint32_t* morton_codes,
                                                  const uint8_t* rt_has_leaf_left,
                                                  const uint8_t* rt_has_leaf_right,
                                                  const uint8_t* rt_prefix_n,
                                                  const int* rt_parents,
                                                  const int* rt_left_child) {
  if (rt_has_leaf_left[i]) {
    const auto leaf_idx = rt_left_child[i];
    const auto leaf_level = oct_depth(rt_prefix_n[i]) + 1;
    const auto leaf_prefix = morton_codes[leaf_idx] >> (morton_bits - (3 * leaf_level));
    const auto child_idx = leaf_prefix & 0b111;

    // walk up the radix tree until finding a node which contributes an octnode
    auto counter = 0;
    auto rt_node = i;
    while (edge_counts[rt_node] == 0) {
      rt_node = rt_parents[rt_node];

      // add a way to break out of the loop in case of infinite loop
      ++counter;
      if (counter > 30) {
        break;
      }
    }

    // the lowest octnode in the string contributed by rt_node will be the
    // lowest index
    const auto bottom_oct_idx = edge_offsets[rt_node] - edge_counts[rt_node];
    set_leaf(bottom_oct_idx, oct_children, oct_child_leaf_mask, child_idx, leaf_idx);
  }
  if (rt_has_leaf_right[i]) {
    const auto leaf_idx = rt_left_child[i] + 1;
    const auto leaf_level = oct_depth(rt_prefix_n[i]) + 1;
    const auto leaf_prefix = morton_codes[leaf_idx] >> (morton_bits - (3 * leaf_level));
    const auto child_idx = leaf_prefix & 0b111;
    auto rt_node = i;
    while (edge_counts[rt_node] == 0) {
      rt_node = rt_parents[rt_node];
    }

    // the lowest octnode in the string contributed by rt_node will be the
    // lowest index
    const auto bottom_oct_idx = edge_offsets[rt_node] - edge_counts[rt_node];
    set_leaf(bottom_oct_idx, oct_children, oct_child_leaf_mask, child_idx, leaf_idx);
  }
}

__global__ void k_MakeOctNodes(int (*oct_children)[8],
                               glm::vec4* oct_corner,
                               float* oct_cell_size,
                               int* oct_child_node_mask,
                               const int* edge_offsets,  // prefix sum
                               const int* edge_counts,   // edge count
                               const unsigned int* codes,
                               const uint8_t* rt_prefix_n,
                               const int* rt_parents,
                               const float min_coord,
                               const float range,
                               const int n_brt_nodes) {
  // brt node 0 now contributes the full-domain ROOT octree node (edge_count[0]
  // == 1) and process_oct_node writes its geometry, so it MUST be processed --
  // the i==0 skip is removed. Matches OMP/Vulkan (start at 0).
  const auto n = static_cast<unsigned>(n_brt_nodes);
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  // all threads participate in the main work
  // i < N
  for (auto i = idx; i < n; i += stride) {
    process_oct_node(static_cast<int>(i),
                     oct_children,
                     oct_corner,
                     oct_cell_size,
                     oct_child_node_mask,
                     edge_offsets,
                     edge_counts,
                     codes,
                     rt_prefix_n,
                     rt_parents,
                     min_coord,
                     range);
  }
}

__global__ void k_LinkLeafNodes(int (*oct_children)[8],
                                int* oct_child_leaf_mask,
                                const int* edge_offsets,
                                const int* edge_counts,
                                const unsigned int* codes,
                                const uint8_t* rt_has_leaf_left,
                                const uint8_t* rt_has_leaf_right,
                                const uint8_t* rt_prefix_n,
                                const int* rt_parents,
                                const int* rt_left_child,
                                const int n_brt_nodes) {
  const auto n = static_cast<unsigned int>(n_brt_nodes);
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  // all threads participate in the main work
  // i < N
  for (auto i = idx; i < n; i += stride) {
    process_link_leaf(static_cast<int>(i),
                      oct_children,
                      oct_child_leaf_mask,
                      edge_offsets,
                      edge_counts,
                      codes,
                      rt_has_leaf_left,
                      rt_has_leaf_right,
                      rt_prefix_n,
                      rt_parents,
                      rt_left_child);
  }
}

}  // namespace kernels

}  // namespace cuda
