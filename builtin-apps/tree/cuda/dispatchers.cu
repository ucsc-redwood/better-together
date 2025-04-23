#include <spdlog/spdlog.h>

#include <cub/cub.cuh>
#include <cub/util_math.cuh>

#include "../../common/cuda/helpers.cuh"
#include "../../debug_logger.hpp"
#include "01_morton.cuh"
#include "04_radix_tree.cuh"
#include "05_edge_count.cuh"
#include "07_octree.cuh"
#include "dispatchers.cuh"

namespace tree::cuda {

//---------------------------------------------------------------------
// Globals, constants and aliases
//---------------------------------------------------------------------

cub::CachingDeviceAllocator g_allocator(true);  // Caching allocator for device memory

constexpr bool kSync = false;

// ----------------------------------------------------------------------------
// Stage 1 (input -> morton code)
// ----------------------------------------------------------------------------

void CudaDispatcher::run_stage_1_async(tree::SafeAppData &app_data) {
  // constexpr auto block_size = 256;
  // const auto grid_size = cub::DivideAndRoundUp(app_data.get_n_input(), block_size);
  // constexpr auto s_mem = 0;

  const auto total_iterations = app_data.get_n_input();
  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 768);

  LOG_KERNEL(LogKernelType::kCUDA, 1, &app_data);

  ::cuda::kernels::k_ComputeMortonCode<<<grid_dim, block_dim, shared_mem>>>(
      app_data.u_input_points_s0.data(),
      app_data.u_morton_keys_s1_out.data(),
      app_data.get_n_input(),
      tree::kMinCoord,
      tree::kRange);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

// ----------------------------------------------------------------------------
// Stage 2 (sort) (morton code -> sorted morton code)
// ----------------------------------------------------------------------------

void CudaDispatcher::run_stage_2_async(tree::SafeAppData &app_data) {
  uint32_t *d_keys_in = app_data.u_morton_keys_s1.data();
  uint32_t *d_keys_out = app_data.u_morton_keys_sorted_s2_out.data();
  uint32_t num_items = app_data.get_n_input();

  LOG_KERNEL(LogKernelType::kCUDA, 2, &app_data);

  // Get temporary storage size
  // assert(app_data.cuda_temp_storage.has_value());

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);

  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Sort data
  cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);

  CubDebugExit(cudaDeviceSynchronize());

  CubDebugExit(cudaFree(d_temp_storage));

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

// ----------------------------------------------------------------------------
// Stage 3 (unique) (sorted morton code -> unique sorted morton code)
// ----------------------------------------------------------------------------

void CudaDispatcher::run_stage_3_async(tree::SafeAppData &app_data) {
  uint32_t *d_in = app_data.u_morton_keys_sorted_s2.data();
  uint32_t *d_out = app_data.u_morton_keys_unique_s3_out.data();
  uint32_t num_items = app_data.get_n_input();

  LOG_KERNEL(LogKernelType::kCUDA, 3, &app_data);

  // Allocate temporary storage
  // assert(app_data.cuda_temp_storage.has_value());

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  CubDebugExit(cub::DeviceSelect::Unique(d_temp_storage,
                                         temp_storage_bytes,
                                         d_in,
                                         d_out,
                                         app_data.u_num_selected_out.data(),
                                         num_items));

  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Run
  CubDebugExit(cub::DeviceSelect::Unique(d_temp_storage,
                                         temp_storage_bytes,
                                         d_in,
                                         d_out,
                                         app_data.u_num_selected_out.data(),
                                         num_items));

  CubDebugExit(cudaDeviceSynchronize());

  // -------- host --------------
  const auto n_unique = app_data.u_num_selected_out[0];
  app_data.set_n_unique(n_unique);
  app_data.set_n_brt_nodes(n_unique - 1);
  // ----------------------------

  CubDebugExit(cudaFree(d_temp_storage));

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

// ----------------------------------------------------------------------------
// Stage 4 (build tree) (unique sorted morton code -> tree nodes)
// ----------------------------------------------------------------------------

void CudaDispatcher::run_stage_4_async(tree::SafeAppData &app_data) {
  const auto total_iterations = app_data.get_n_unique();
  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 768);

  LOG_KERNEL(LogKernelType::kCUDA, 4, &app_data);

  ::cuda::kernels::k_BuildRadixTree<<<grid_dim, block_dim, shared_mem>>>(
      app_data.get_n_unique(),
      app_data.u_morton_keys_unique_s3.data(),
      app_data.u_brt_prefix_n_s4_out.data(),
      app_data.u_brt_has_leaf_left_s4_out.data(),
      app_data.u_brt_has_leaf_right_s4_out.data(),
      app_data.u_brt_left_child_s4_out.data(),
      app_data.u_brt_parents_s4_out.data());

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

// ----------------------------------------------------------------------------
// Stage 5 (edge count) (tree nodes -> edge count)
// ----------------------------------------------------------------------------

void CudaDispatcher::run_stage_5_async(tree::SafeAppData &app_data) {
  // constexpr auto gridDim = 16;
  // constexpr auto blockDim = 512;
  // constexpr auto sharedMem = 0;

  const auto total_iterations = app_data.get_n_brt_nodes();
  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 768);

  LOG_KERNEL(LogKernelType::kCUDA, 5, &app_data);

  ::cuda::kernels::k_EdgeCount<<<grid_dim, block_dim, shared_mem>>>(
      app_data.u_brt_prefix_n_s4.data(),
      app_data.u_brt_parents_s4.data(),
      app_data.u_edge_count_s5_out.data(),
      app_data.get_n_brt_nodes());

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

// ----------------------------------------------------------------------------
// Stage 6 (edge offset) (edge count -> edge offset)
// ----------------------------------------------------------------------------

void CudaDispatcher::run_stage_6_async(tree::SafeAppData &app_data) {
  LOG_KERNEL(LogKernelType::kCUDA, 6, &app_data);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::InclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                app_data.u_edge_count_s5.data(),
                                app_data.u_edge_offset_s6_out.data(),
                                app_data.get_n_brt_nodes());

  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Perform prefix sum (inclusive scan)
  cub::DeviceScan::InclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                app_data.u_edge_count_s5.data(),
                                app_data.u_edge_offset_s6_out.data(),
                                app_data.get_n_brt_nodes());

  CubDebugExit(cudaDeviceSynchronize());

  // -------- host --------------
  app_data.set_n_octree_nodes(app_data.u_edge_offset_s6_out[app_data.get_n_brt_nodes() - 1]);
  // ----------------------------

  CubDebugExit(cudaFree(d_temp_storage));

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

// ----------------------------------------------------------------------------
// Stage 7 (octree) (everything above -> octree)
// ----------------------------------------------------------------------------

void CudaDispatcher::run_stage_7_async(tree::SafeAppData &app_data) {
  // constexpr auto gridDim = 16;
  // constexpr auto blockDim = 512;
  // constexpr auto sharedMem = 0;

  const auto total_iterations = app_data.get_n_brt_nodes();
  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 768);

  LOG_KERNEL(LogKernelType::kCUDA, 7, &app_data);

  // First kernel: Create the octree structure
  ::cuda::kernels::k_MakeOctNodes<<<grid_dim, block_dim, shared_mem>>>(
      reinterpret_cast<int(*)[8]>(app_data.u_oct_children_s7_out.data()),
      app_data.u_oct_corner_s7_out.data(),
      app_data.u_oct_cell_size_s7_out.data(),
      app_data.u_oct_child_node_mask_s7_out.data(),
      app_data.u_edge_offset_s6.data(),
      app_data.u_edge_count_s5.data(),
      app_data.u_morton_keys_unique_s3.data(),
      app_data.u_brt_prefix_n_s4.data(),
      app_data.u_brt_parents_s4.data(),
      tree::kMinCoord,
      tree::kRange,
      app_data.get_n_brt_nodes());

  // CubDebugExit(cudaDeviceSynchronize());

  // Second kernel: Link leaf nodes
  ::cuda::kernels::k_LinkLeafNodes<<<grid_dim, block_dim, shared_mem>>>(
      reinterpret_cast<int(*)[8]>(app_data.u_oct_children_s7_out.data()),
      app_data.u_oct_child_leaf_mask_s7_out.data(),
      app_data.u_edge_offset_s6.data(),
      app_data.u_edge_count_s5.data(),
      app_data.u_morton_keys_unique_s3.data(),
      app_data.u_brt_has_leaf_left_s4.data(),
      app_data.u_brt_has_leaf_right_s4.data(),
      app_data.u_brt_prefix_n_s4.data(),
      app_data.u_brt_parents_s4.data(),
      app_data.u_brt_left_child_s4.data(),
      app_data.get_n_brt_nodes());

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

}  // namespace tree::cuda
