// This order of includes is important! Do not change it!
// clang-format off
#include "platform/util/debug_logger.hpp"
#include "dispatchers.cuh"

#include <cub/cub.cuh>
// clang-format on

#include "01_morton.cuh"
#include "04_radix_tree.cuh"
#include "05_edge_count.cuh"
#include "07_octree.cuh"
#include "platform/engine/cuda/helpers.cuh"

namespace tree::cuda {

// Grow-only device scratch for the cub temp-storage calls (stages 2/3/6). Replaces
// cub::CachingDeviceAllocator, which CUDA 13 / CCCL 3 removed. Reuse is safe: at most
// one CUDA chunk dispatches at a time (the runtime rejects concurrent GPU chunks and
// the profiler never runs two GPU users), and reuse by the next cub call on the same
// stream is stream-ordered. Never freed -- a few MB held for the process lifetime,
// same as the old allocator's cache.
namespace {
void* ensure_temp_storage(size_t bytes) {
  static void* ptr = nullptr;
  static size_t capacity = 0;
  if (bytes > capacity) {
    if (ptr != nullptr) CheckCuda(cudaFree(ptr));
    CheckCuda(cudaMalloc(&ptr, bytes));
    capacity = bytes;
  }
  return ptr;
}
}  // namespace

constexpr bool kSync = false;

void CudaDispatcher::run_stage_1_async(tree::SafeAppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 1, &appdata);

  const auto total_iterations = appdata.get_n_input();
  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 768);

  ::cuda::kernels::k_ComputeMortonCode<<<grid_dim, block_dim, shared_mem>>>(
      appdata.u_input_points_s0.data(),
      appdata.u_morton_keys_s1_out.data(),
      appdata.get_n_input(),
      tree::kMinCoord,
      tree::kRange);
  CheckCudaLaunch("k_ComputeMortonCode");

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_2_async(tree::SafeAppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 2, &appdata);

  const uint32_t* d_keys_in = appdata.u_morton_keys_s1.data();
  uint32_t* d_keys_out = appdata.u_morton_keys_sorted_s2_out.data();
  uint32_t num_items = appdata.get_n_input();

  // Get temporary storage size
  // assert(appdata.cuda_temp_storage.has_value());

  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);

  d_temp_storage = ensure_temp_storage(temp_storage_bytes);

  // Sort data
  cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);

  // Keep the stage synchronous on completion (host-side readers and the profiler
  // rely on it; the scratch itself needs no sync -- reuse is stream-ordered).
  CheckCuda(cudaDeviceSynchronize());

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_3_async(tree::SafeAppData& appdata) {
  const uint32_t* d_in = appdata.u_morton_keys_sorted_s2.data();
  uint32_t* d_out = appdata.u_morton_keys_unique_s3_out.data();
  uint32_t num_items = appdata.get_n_input();

  LOG_KERNEL(LogKernelType::kCUDA, 3, &appdata);

  // Allocate temporary storage
  // assert(appdata.cuda_temp_storage.has_value());

  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  CheckCuda(cub::DeviceSelect::Unique(d_temp_storage,
                                      temp_storage_bytes,
                                      d_in,
                                      d_out,
                                      appdata.u_num_selected_out.data(),
                                      num_items));

  d_temp_storage = ensure_temp_storage(temp_storage_bytes);

  // Run
  CheckCuda(cub::DeviceSelect::Unique(d_temp_storage,
                                      temp_storage_bytes,
                                      d_in,
                                      d_out,
                                      appdata.u_num_selected_out.data(),
                                      num_items));

  CheckCuda(cudaDeviceSynchronize());

  // -------- host --------------
  const auto n_unique = appdata.u_num_selected_out[0];
  appdata.set_n_unique(n_unique);
  appdata.set_n_brt_nodes(n_unique - 1);
  // ----------------------------

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_4_async(tree::SafeAppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 4, &appdata);

  const auto total_iterations = appdata.get_n_unique();
  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 768);

  ::cuda::kernels::k_BuildRadixTree<<<grid_dim, block_dim, shared_mem>>>(
      appdata.get_n_unique(),
      appdata.u_morton_keys_unique_s3.data(),
      appdata.u_brt_prefix_n_s4_out.data(),
      appdata.u_brt_has_leaf_left_s4_out.data(),
      appdata.u_brt_has_leaf_right_s4_out.data(),
      appdata.u_brt_left_child_s4_out.data(),
      appdata.u_brt_parents_s4_out.data());
  CheckCudaLaunch("k_BuildRadixTree");

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_5_async(tree::SafeAppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 5, &appdata);

  const auto total_iterations = appdata.get_n_brt_nodes();
  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 768);

  ::cuda::kernels::k_EdgeCount<<<grid_dim, block_dim, shared_mem>>>(
      appdata.u_brt_prefix_n_s4.data(),
      appdata.u_brt_parents_s4.data(),
      appdata.u_edge_count_s5_out.data(),
      appdata.get_n_brt_nodes());
  CheckCudaLaunch("k_EdgeCount");

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_6_async(tree::SafeAppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 6, &appdata);

  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::InclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                appdata.u_edge_count_s5.data(),
                                appdata.u_edge_offset_s6_out.data(),
                                appdata.get_n_brt_nodes());

  d_temp_storage = ensure_temp_storage(temp_storage_bytes);

  // Perform prefix sum (inclusive scan)
  cub::DeviceScan::InclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                appdata.u_edge_count_s5.data(),
                                appdata.u_edge_offset_s6_out.data(),
                                appdata.get_n_brt_nodes());

  CheckCuda(cudaDeviceSynchronize());

  // -------- host --------------
  appdata.set_n_octree_nodes(appdata.u_edge_offset_s6_out[appdata.get_n_brt_nodes() - 1]);
  // ----------------------------

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_7_async(tree::SafeAppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 7, &appdata);

  const auto total_iterations = appdata.get_n_brt_nodes();
  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 768);

  // First kernel: Create the octree structure
  ::cuda::kernels::k_MakeOctNodes<<<grid_dim, block_dim, shared_mem>>>(
      reinterpret_cast<int (*)[8]>(appdata.u_oct_children_s7_out.data()),
      appdata.u_oct_corner_s7_out.data(),
      appdata.u_oct_cell_size_s7_out.data(),
      appdata.u_oct_child_node_mask_s7_out.data(),
      appdata.u_edge_offset_s6.data(),
      appdata.u_edge_count_s5.data(),
      appdata.u_morton_keys_unique_s3.data(),
      appdata.u_brt_prefix_n_s4.data(),
      appdata.u_brt_parents_s4.data(),
      tree::kMinCoord,
      tree::kRange,
      appdata.get_n_brt_nodes());
  CheckCudaLaunch("k_MakeOctNodes");

  // CubDebugExit(cudaDeviceSynchronize());

  // Second kernel: Link leaf nodes
  ::cuda::kernels::k_LinkLeafNodes<<<grid_dim, block_dim, shared_mem>>>(
      reinterpret_cast<int (*)[8]>(appdata.u_oct_children_s7_out.data()),
      appdata.u_oct_child_leaf_mask_s7_out.data(),
      appdata.u_edge_offset_s6.data(),
      appdata.u_edge_count_s5.data(),
      appdata.u_morton_keys_unique_s3.data(),
      appdata.u_brt_has_leaf_left_s4.data(),
      appdata.u_brt_has_leaf_right_s4.data(),
      appdata.u_brt_prefix_n_s4.data(),
      appdata.u_brt_parents_s4.data(),
      appdata.u_brt_left_child_s4.data(),
      appdata.get_n_brt_nodes());
  CheckCudaLaunch("k_LinkLeafNodes");

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

}  // namespace tree::cuda
