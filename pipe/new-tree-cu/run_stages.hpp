// #pragma once

// #include "builtin-apps/affinity.hpp"
// #include "builtin-apps/app.hpp"
// #include "builtin-apps/common/cuda/manager.cuh"
// #include "builtin-apps/tree/cuda/dispatchers.cuh"
// #include "builtin-apps/tree/omp/dispatchers.hpp"

// template <int N>
// concept AllowedStage = (N >= 1 && N <= 7);

// // ---------------------------------------------------------------------
// // CPU stages
// // ---------------------------------------------------------------------

// namespace omp {

// constexpr std::array<void (*)(tree::SafeAppData &), 7> cpu_stages = {
//     tree::omp::process_stage_1,
//     tree::omp::process_stage_2,
//     tree::omp::process_stage_3,
//     tree::omp::process_stage_4,
//     tree::omp::process_stage_5,
//     tree::omp::process_stage_6,
//     tree::omp::process_stage_7,
// };

// template <int Start, int End, ProcessorType PT, int NThreads>
//   requires AllowedStage<Start> && AllowedStage<End> && (Start <= End)
// void run_multiple_stages(tree::SafeAppData &data, cuda::CudaManager &) {
// #pragma omp parallel num_threads(NThreads)
//   {
//     // Bind to core
//     if constexpr (PT == ProcessorType::kLittleCore) {
//       bind_thread_to_cores(g_little_cores);
//     } else if constexpr (PT == ProcessorType::kMediumCore) {
//       bind_thread_to_cores(g_medium_cores);
//     } else if constexpr (PT == ProcessorType::kBigCore) {
//       bind_thread_to_cores(g_big_cores);
//     }

//     for (int s = Start; s <= End; ++s) {
//       cpu_stages[s - 1](data);
//     }
//   }
// }

// }  // namespace omp

// // ---------------------------------------------------------------------
// // GPU stages
// // ---------------------------------------------------------------------

// // -----------------------------------------------------------------------------------------------
// // | Stage | Buffer Name                  | Allocated Size               | Real Data Used          |
// // |-------|------------------------------|------------------------------|-------------------------|
// // | 1     | u_input_points_s0            | n_input                      | n_input                 |
// // | 1     | u_morton_keys_s1             | n_input                      | n_input                 |
// // | 2     | u_morton_keys_sorted_s2      | n_input                      | n_input                 |
// // | 3     | u_morton_keys_unique_s3      | n_input                      | n_unique                |
// // | 4     | u_brt_prefix_n_s4            | n_input                      | n_brt_nodes             |
// // | 4     | u_brt_has_leaf_left_s4       | n_input                      | n_brt_nodes             |
// // | 4     | u_brt_has_leaf_right_s4      | n_input                      | n_brt_nodes             |
// // | 4     | u_brt_left_child_s4          | n_input                      | n_brt_nodes             |
// // | 4     | u_brt_parents_s4             | n_input                      | n_brt_nodes             |
// // | 5     | u_edge_count_s5              | n_input                      | n_brt_nodes             |
// // | 6     | u_edge_offset_s6             | n_input                      | n_brt_nodes             |
// // | 7     | u_oct_corner_s7              | n_input * 0.6f               | n_octree_nodes          |
// // | 7     | u_oct_cell_size_s7           | n_input * 0.6f               | n_octree_nodes          |
// // | 7     | u_oct_child_node_mask_s7     | n_input * 0.6f               | n_octree_nodes          |
// // | 7     | u_oct_child_leaf_mask_s7     | n_input * 0.6f               | n_octree_nodes          |
// // | 7     | u_oct_children_s7            | 8 * n_input * 0.6f           | 8 * n_octree_nodes      |
// // ------------------------------------------------------------------------------------------------

// namespace cuda {

// #define CudaAttachSingle(ptr) \
//   (cudaStreamAttachMemAsync(mgr.get_stream(), ptr, 0, cudaMemAttachSingle))
// #define CudaAttachHost(ptr) (cudaStreamAttachMemAsync(mgr.get_stream(), ptr, 0, cudaMemAttachHost))

// constexpr std::array<void (*)(tree::SafeAppData &), 7> gpu_stages = {
//     tree::cuda::process_stage_1,
//     tree::cuda::process_stage_2,
//     tree::cuda::process_stage_3,
//     tree::cuda::process_stage_4,
//     tree::cuda::process_stage_5,
//     tree::cuda::process_stage_6,
//     tree::cuda::process_stage_7,
// };

// template <int Start, int End>
//   requires AllowedStage<Start> && AllowedStage<End> && (Start <= End)
// void run_multiple_stages(tree::SafeAppData &data, cuda::CudaManager &mgr) {
//   CudaAttachSingle(data.u_input_points_s0.data());
//   CudaAttachSingle(data.u_morton_keys_s1.data());
//   CudaAttachSingle(data.u_morton_keys_sorted_s2.data());
//   CudaAttachSingle(data.u_morton_keys_unique_s3.data());
//   CudaAttachSingle(data.u_brt_prefix_n_s4.data());
//   CudaAttachSingle(data.u_brt_has_leaf_left_s4.data());
//   CudaAttachSingle(data.u_brt_has_leaf_right_s4.data());
//   CudaAttachSingle(data.u_brt_left_child_s4.data());
//   CudaAttachSingle(data.u_brt_parents_s4.data());
//   CudaAttachSingle(data.u_edge_count_s5.data());
//   CudaAttachSingle(data.u_edge_offset_s6.data());
//   CudaAttachSingle(data.u_oct_corner_s7.data());
//   CudaAttachSingle(data.u_oct_cell_size_s7.data());
//   CudaAttachSingle(data.u_oct_child_node_mask_s7.data());
//   CudaAttachSingle(data.u_oct_child_leaf_mask_s7.data());
//   CudaAttachSingle(data.u_oct_children_s7.data());

//   for (int s = Start; s <= End; ++s) {
//     gpu_stages[s - 1](data);
//   }

//   CheckCuda(cudaStreamSynchronize(mgr.get_stream()));

//   CudaAttachHost(data.u_input_points_s0.data());
//   CudaAttachHost(data.u_morton_keys_s1.data());
//   CudaAttachHost(data.u_morton_keys_sorted_s2.data());
//   CudaAttachHost(data.u_morton_keys_unique_s3.data());
//   CudaAttachHost(data.u_brt_prefix_n_s4.data());
//   CudaAttachHost(data.u_brt_has_leaf_left_s4.data());
//   CudaAttachHost(data.u_brt_has_leaf_right_s4.data());
//   CudaAttachHost(data.u_brt_left_child_s4.data());
//   CudaAttachHost(data.u_brt_parents_s4.data());
//   CudaAttachHost(data.u_edge_count_s5.data());
//   CudaAttachHost(data.u_edge_offset_s6.data());
//   CudaAttachHost(data.u_oct_corner_s7.data());
//   CudaAttachHost(data.u_oct_cell_size_s7.data());
//   CudaAttachHost(data.u_oct_child_node_mask_s7.data());
//   CudaAttachHost(data.u_oct_child_leaf_mask_s7.data());
//   CudaAttachHost(data.u_oct_children_s7.data());
// }

// }  // namespace cuda
