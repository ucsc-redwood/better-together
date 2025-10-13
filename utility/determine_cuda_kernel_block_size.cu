#include <cuda_runtime.h>

#include <iostream>

#include "builtin-apps/cifar-dense/cuda/all_kernels.cuh"
#include "builtin-apps/cifar-sparse/cuda/all_kernels.cuh"
#include "builtin-apps/common/cuda/helpers.cuh"
#include "builtin-apps/tree/cuda/01_morton.cuh"
#include "builtin-apps/tree/cuda/04_radix_tree.cuh"
#include "builtin-apps/tree/cuda/05_edge_count.cuh"
#include "builtin-apps/tree/cuda/07_octree.cuh"

template <class T>
[[nodiscard]] int determineBlockSize(T func) {
  int block_size = 1;
  int min_grid_size = 1;
  CheckCuda(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, func));
  return block_size;
}

int main() {
  std::cout << "cifar_dense::cuda::conv2d Block size: "
            << determineBlockSize(cifar_dense::cuda::conv2d) << std::endl;
  std::cout << "cifar_dense::cuda::maxpool2d Block size: "
            << determineBlockSize(cifar_dense::cuda::maxpool2d) << std::endl;
  std::cout << "cifar_dense::cuda::linear Block size: "
            << determineBlockSize(cifar_dense::cuda::linear) << std::endl;

  std::cout << "cifar_sparse::cuda::conv2d Block size: "
            << determineBlockSize(cifar_sparse::cuda::conv2d) << std::endl;
  std::cout << "cifar_sparse::cuda::maxpool2d Block size: "
            << determineBlockSize(cifar_sparse::cuda::maxpool2d) << std::endl;
  std::cout << "cifar_sparse::cuda::linear Block size: "
            << determineBlockSize(cifar_sparse::cuda::linear) << std::endl;

  std::cout << "tree::cuda::k_ComputeMortonCode Block size: "
            << determineBlockSize(::cuda::kernels::k_ComputeMortonCode) << std::endl;
  std::cout << "tree::cuda::k_BuildRadixTree Block size: "
            << determineBlockSize(::cuda::kernels::k_BuildRadixTree) << std::endl;
  std::cout << "tree::cuda::k_EdgeCount Block size: "
            << determineBlockSize(::cuda::kernels::k_EdgeCount) << std::endl;
  std::cout << "tree::cuda::k_MakeOctNodes Block size: "
            << determineBlockSize(::cuda::kernels::k_MakeOctNodes) << std::endl;
  std::cout << "tree::cuda::k_LinkLeafNodes Block size: "
            << determineBlockSize(::cuda::kernels::k_LinkLeafNodes) << std::endl;

  return 0;
}