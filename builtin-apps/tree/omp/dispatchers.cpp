#include "dispatchers.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>

#include "../../debug_logger.hpp"
#include "func_brt.hpp"
#include "func_edge.hpp"
#include "func_morton.hpp"
#include "func_octree.hpp"
#include "func_sort.hpp"

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
    appdata.u_morton_keys_s1_out[i] =
        xyz_to_morton32(appdata.u_input_points_s0[i], tree::kMinCoord, tree::kRange);
  }
}

// ----------------------------------------------------------------------------
// Stage 2 (morton -> sorted morton)
// ----------------------------------------------------------------------------

void run_stage_2(tree::SafeAppData &appdata) {
  std::copy(appdata.u_morton_keys_s1.begin(),
            appdata.u_morton_keys_s1.end(),
            appdata.u_morton_keys_s1_out.begin());
            
  std::sort(appdata.u_morton_keys_s1_out.begin(),
            appdata.u_morton_keys_s1_out.end());
  bool is_sorted = std::is_sorted(appdata.u_morton_keys_s1_out.begin(),
                              appdata.u_morton_keys_s1_out.end());
  // print frist 10 elements
  // for (int i = 0; i < 10 && i < appdata.get_n_input(); ++i) {
  //   std::cout << appdata.u_morton_keys_s1_out[i] << std::endl;
  // }
  if (!is_sorted) {
    spdlog::error("Morton keys are not sorted after stage 2!");
    throw std::runtime_error("Morton keys are not sorted after stage 2!");
  } else {
    spdlog::info("Morton keys are sorted after stage 2.");
  }
  // LOG_KERNEL(LogKernelType::kOMP, 1, &appdata);

  // ----------------------------------------------------------------------------
  // Parallel sort version
  // ----------------------------------------------------------------------------

  // const auto num_threads = omp_get_num_threads();
  // const int tid = omp_get_thread_num();
  // omp::parallel_sort(
  //     appdata.u_morton_keys_s1_out, appdata.u_morton_keys_sorted_s2_out, tid, num_threads);
}

// ----------------------------------------------------------------------------
// Stage 3 (sorted morton -> unique morton)
// ----------------------------------------------------------------------------

void run_stage_3(tree::SafeAppData &appdata) {
  LOG_KERNEL(LogKernelType::kOMP, 3, &appdata);

  const auto from = appdata.u_morton_keys_sorted_s2.data();
  auto to = appdata.u_morton_keys_unique_s3_out.data();

  const auto last = std::unique_copy(from, from + appdata.get_n_input(), to);
  const auto n_unique = std::distance(to, last);

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
    v1::process_radix_tree_i(i,
                             appdata.get_n_brt_nodes(),
                             appdata.u_morton_keys_unique_s3.data(),
                             appdata.u_brt_prefix_n_s4_out.data(),
                             appdata.u_brt_has_leaf_left_s4_out.data(),
                             appdata.u_brt_has_leaf_right_s4_out.data(),
                             appdata.u_brt_left_child_s4_out.data(),
                             appdata.u_brt_parents_s4_out.data());
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
    v1::process_edge_count_i(i,
                             appdata.u_brt_prefix_n_s4.data(),
                             appdata.u_brt_parents_s4.data(),
                             appdata.u_edge_count_s5_out.data());
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
                   appdata.u_edge_offset_s6_out.data() + start);

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
                     reinterpret_cast<int(*)[8]>(appdata.u_oct_children_s7_out.data()),
                     appdata.u_oct_corner_s7_out.data(),
                     appdata.u_oct_cell_size_s7_out.data(),
                     appdata.u_oct_child_node_mask_s7_out.data(),
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
