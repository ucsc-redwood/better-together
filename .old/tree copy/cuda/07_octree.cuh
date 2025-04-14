#pragma once

#include <glm/glm.hpp>

namespace cuda {

namespace kernels {

__global__ void k_MakeOctNodes(
    // --- begin octree parameters (outputs)
    int (*oct_children)[8],
    glm::vec4* oct_corner,
    float* oct_cell_size,
    int* oct_child_node_mask,
    // --- end octree parameters
    const int* edge_offsets,
    const int* edge_counts,
    const unsigned int* codes,
    const uint8_t* rt_prefix_n,
    const int* rt_parents,
    float min_coord,
    float range,
    int n_brt_nodes);

__global__ void k_LinkLeafNodes(
    // --- begin octree parameters (outputs)
    int (*oct_children)[8],
    int* oct_child_leaf_mask,
    // --- end octree parameters
    const int* edge_offsets,
    const int* edge_counts,
    const unsigned int* codes,
    const uint8_t* rt_has_leaf_left,
    const uint8_t* rt_has_leaf_right,
    const uint8_t* rt_prefix_n,
    const int* rt_parents,
    const int* rt_left_child,
    int n_brt_nodes);

}  // namespace kernels

}  // namespace cuda
