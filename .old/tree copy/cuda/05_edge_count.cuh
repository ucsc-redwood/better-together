#pragma once

#include <cstdint>

namespace cuda {

namespace kernels {

__global__ void k_EdgeCount(const uint8_t* prefix_n,
                            const int* parents,
                            int* edge_count,
                            int n_brt_nodes);

}  // namespace kernels

}  // namespace cuda
