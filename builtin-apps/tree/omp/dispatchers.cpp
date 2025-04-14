#include "dispatchers.hpp"

#include <numeric>
#include <thread>

#include "../../debug_logger.hpp"
#include "func_brt.hpp"
#include "func_edge.hpp"
#include "func_morton.hpp"
#include "func_octree.hpp"
#include "func_sort.hpp"
#include "temp_storage.hpp"

namespace tree::omp {

// ---------------------------------------------------------------------
// Stage 1 (xyz -> morton)
// ----------------------------------------------------------------------------

void run_stage_1(tree::SafeAppData &appdata) {
  const int start = 0;
  const int end = appdata.get_n_input();

  LOG_KERNEL(LogKernelType::kOMP, 1, &appdata);

#pragma omp for
  for (int i = start; i < end; ++i) {
    appdata.u_morton_keys_s1[i] =
        xyz_to_morton32(appdata.u_input_points_s0[i], tree::kMinCoord, tree::kRange);
  }
}

// ----------------------------------------------------------------------------
// Stage 2 (morton -> sorted morton)
// ----------------------------------------------------------------------------

void run_stage_2(tree::SafeAppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 1, &appdata);

  // ----------------------------------------------------------------------------
  // Parallel sort version
  // ----------------------------------------------------------------------------

  const auto num_threads = omp_get_num_threads();
  const int tid = omp_get_thread_num();
  omp::parallel_sort(appdata.u_morton_keys_s1, appdata.u_morton_keys_sorted_s2, tid, num_threads);
}

// ----------------------------------------------------------------------------
// Stage 3 (sorted morton -> unique morton)
// ----------------------------------------------------------------------------

void run_stage_3(tree::SafeAppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 3, &appdata);

  const auto last = std::unique_copy(appdata.u_morton_keys_sorted_s2.data(),
                                     appdata.u_morton_keys_sorted_s2.data() + appdata.get_n_input(),
                                     appdata.u_morton_keys_unique_s3.data());
  const auto n_unique = std::distance(appdata.u_morton_keys_unique_s3.data(), last);

  appdata.set_n_unique(n_unique);
  appdata.set_n_brt_nodes(n_unique - 1);
}

// ----------------------------------------------------------------------------
// Stage 4 (unique morton -> brt)
// ----------------------------------------------------------------------------

void run_stage_4(tree::SafeAppData &appdata) {
  const int start = 0;
  const int end = appdata.get_n_unique();

  LOG_KERNEL(LogKernelType::kOMP, 4, &appdata);

#pragma omp for
  for (int i = start; i < end; ++i) {
    process_radix_tree_i(i,
                         appdata.get_n_brt_nodes(),
                         appdata.u_morton_keys_unique_s3.data(),
                         appdata.u_brt_prefix_n_s4.data(),
                         appdata.u_brt_has_leaf_left_s4.data(),
                         appdata.u_brt_has_leaf_right_s4.data(),
                         appdata.u_brt_left_child_s4.data(),
                         appdata.u_brt_parents_s4.data());
  }
}

// ----------------------------------------------------------------------------
// Stage 5 (brt -> edge count)
// ----------------------------------------------------------------------------

void run_stage_5(tree::SafeAppData &appdata) {
  const int start = 0;
  const int end = appdata.get_n_brt_nodes();

  LOG_KERNEL(LogKernelType::kOMP, 5, &appdata);

#pragma omp for
  for (int i = start; i < end; ++i) {
    process_edge_count_i(i,
                         appdata.u_brt_prefix_n_s4.data(),
                         appdata.u_brt_parents_s4.data(),
                         appdata.u_edge_count_s5.data());
  }
}

// ----------------------------------------------------------------------------
// Stage 6 (edge count -> edge offset)
// ----------------------------------------------------------------------------

void run_stage_6(tree::SafeAppData &appdata) {
  const int start = 0;
  const int end = appdata.get_n_brt_nodes();

  LOG_KERNEL(LogKernelType::kOMP, 6, &appdata);

  std::partial_sum(appdata.u_edge_count_s5.data() + start,
                   appdata.u_edge_count_s5.data() + end,
                   appdata.u_edge_offset_s6.data() + start);

  // num_octree node is the result of the partial sum
  const auto num_octree_nodes = appdata.u_edge_offset_s6[end - 1];

  appdata.set_n_octree_nodes(num_octree_nodes);
}

// ----------------------------------------------------------------------------
// Stage 7 (everything -> octree)
// ----------------------------------------------------------------------------

void run_stage_7(tree::SafeAppData &appdata) {
  // note: 1 here, skipping root
  const int start = 1;
  const int end = appdata.get_n_octree_nodes();

  LOG_KERNEL(LogKernelType::kOMP, 7, &appdata);

#pragma omp for
  for (int i = start; i < end; ++i) {
    process_oct_node(i,
                     reinterpret_cast<int(*)[8]>(appdata.u_oct_children_s7.data()),
                     appdata.u_oct_corner_s7.data(),
                     appdata.u_oct_cell_size_s7.data(),
                     appdata.u_oct_child_node_mask_s7.data(),
                     appdata.u_edge_offset_s6.data(),
                     appdata.u_edge_count_s5.data(),
                     appdata.u_morton_keys_unique_s3.data(),
                     appdata.u_brt_prefix_n_s4.data(),
                     appdata.u_brt_parents_s4.data(),
                     tree::kMinCoord,
                     tree::kRange);

    process_link_leaf(i,
                      reinterpret_cast<int(*)[8]>(appdata.u_oct_children_s7_out.data()),
                      appdata.u_oct_child_leaf_mask_s7_out.data(),
                      appdata.u_edge_offset_s6.data(),
                      appdata.u_edge_count_s5.data(),
                      appdata.u_morton_keys_unique_s3.data(),
                      appdata.u_brt_has_leaf_left_s4.data(),
                      appdata.u_brt_has_leaf_right_s4.data(),
                      appdata.u_brt_prefix_n_s4.data(),
                      appdata.u_brt_parents_s4.data(),
                      appdata.u_brt_left_child_s4.data());
  }
}

}  // namespace tree::omp
