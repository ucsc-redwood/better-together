#pragma once

namespace cuda {

namespace kernels {

__global__ void k_BuildRadixTree(int n_unique,
                                 const unsigned int* codes,
                                 uint8_t* prefix_n,
                                 uint8_t* has_leaf_left,
                                 uint8_t* has_leaf_right,
                                 int* left_child,
                                 int* parent);

}  // namespace kernels

}  // namespace cuda
