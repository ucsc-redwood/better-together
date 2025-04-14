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

// // ----------------------------------------------------------------------------
// // Stage 1 (xyz -> morton)
// // ----------------------------------------------------------------------------

// void process_stage_1(tree::AppData &appdata) {
//   const int start = 0;
//   const int end = appdata.get_n_input();

//   LOG_KERNEL(LogKernelType::kOMP, 1, &appdata);

// #pragma omp for
//   for (int i = start; i < end; ++i) {
//     appdata.u_morton_keys_s1[i] =
//         xyz_to_morton32(appdata.u_input_points_s0[i], tree::kMinCoord, tree::kRange);
//   }
// }

// // ----------------------------------------------------------------------------
// // Stage 2 (morton -> sorted morton)
// // ----------------------------------------------------------------------------

// void process_stage_2(tree::AppData &appdata) {
//   const auto num_threads = omp_get_num_threads();
//   // const auto num_buckets = num_threads;

//   LOG_KERNEL(LogKernelType::kOMP, 2, &appdata);

//   // ----------------------------------------------------------------------------
//   // Merge sort version
//   // ----------------------------------------------------------------------------

//   const int tid = omp_get_thread_num();
//   omp::parallel_sort(appdata.u_morton_keys_s1, appdata.u_morton_keys_sorted_s2, tid,
//   num_threads);

//   // omp::RadixSortTemp<uint32_t> temp_storage(appdata.get_n_input(), num_threads);

//   // parallel_radix_sort<uint32_t>(
//   //     appdata.u_morton_keys_s1, appdata.u_morton_keys_sorted_s2, temp_storage);

//   // omp::TmpStorage temp_storage;

//   // // const auto n_threads = omp_get_num_threads();
//   // const auto n_threads = std::thread::hardware_concurrency();
//   // const auto n_buckets = n_threads;

//   // // spdlog::info("---num_threads: {}", n_threads);
//   // // spdlog::info("--- num_buckets: {}", n_buckets);

//   // temp_storage.allocate(n_buckets, n_threads);

//   // // assert(temp_storage.is_allocated());

//   // bucket_sort(appdata.u_morton_keys_s1.data(),
//   //             appdata.u_morton_keys_sorted_s2.data(),
//   //             temp_storage.global_n_elem(),
//   //             temp_storage.global_starting_position(),
//   //             temp_storage.buckets(),
//   //             appdata.get_n_input(),
//   //             num_buckets,
//   //             num_threads);

//   // // by this point, 'u_morton_keys_sorted_s2' is sorted
// }

// // ----------------------------------------------------------------------------
// // Stage 3 (sorted morton -> unique morton)
// // ----------------------------------------------------------------------------

// void process_stage_3(tree::AppData &appdata) {
//   LOG_KERNEL(LogKernelType::kOMP, 3, &appdata);

//   const auto last = std::unique_copy(appdata.u_morton_keys_sorted_s2.data(),
//                                      appdata.u_morton_keys_sorted_s2.data() +
//                                      appdata.get_n_input(),
//                                      appdata.u_morton_keys_unique_s3.data());
//   const auto n_unique = std::distance(appdata.u_morton_keys_unique_s3.data(), last);

//   appdata.set_n_unique(n_unique);
//   appdata.set_n_brt_nodes(n_unique - 1);
// }

// // ----------------------------------------------------------------------------
// // Stage 4 (unique morton -> brt)
// // ----------------------------------------------------------------------------

// void process_stage_4(tree::AppData &appdata) {
//   const int start = 0;
//   const int end = appdata.get_n_unique();

//   LOG_KERNEL(LogKernelType::kOMP, 4, &appdata);

// #pragma omp for
//   for (int i = start; i < end; ++i) {
//     process_radix_tree_i(i,
//                          appdata.get_n_brt_nodes(),
//                          appdata.u_morton_keys_unique_s3.data(),
//                          appdata.u_brt_prefix_n_s4.data(),
//                          appdata.u_brt_has_leaf_left_s4.data(),
//                          appdata.u_brt_has_leaf_right_s4.data(),
//                          appdata.u_brt_left_child_s4.data(),
//                          appdata.u_brt_parents_s4.data());
//   }
// }

// // ----------------------------------------------------------------------------
// // Stage 5 (brt -> edge count)
// // ----------------------------------------------------------------------------

// void process_stage_5(tree::AppData &appdata) {
//   const int start = 0;
//   const int end = appdata.get_n_brt_nodes();

//   LOG_KERNEL(LogKernelType::kOMP, 5, &appdata);

//   for (int i = start; i < end; ++i) {
//     process_edge_count_i(i,
//                          appdata.u_brt_prefix_n_s4.data(),
//                          appdata.u_brt_parents_s4.data(),
//                          appdata.u_edge_count_s5.data());
//   }
// }

// // ----------------------------------------------------------------------------
// // Stage 6 (edge count -> edge offset)
// // ----------------------------------------------------------------------------

// void process_stage_6(tree::AppData &appdata) {
//   const int start = 0;
//   const int end = appdata.get_n_brt_nodes();

//   LOG_KERNEL(LogKernelType::kOMP, 6, &appdata);

//   std::partial_sum(appdata.u_edge_count_s5.data() + start,
//                    appdata.u_edge_count_s5.data() + end,
//                    appdata.u_edge_offset_s6.data() + start);

//   // num_octree node is the result of the partial sum
//   const auto num_octree_nodes = appdata.u_edge_offset_s6[end - 1];

//   appdata.set_n_octree_nodes(num_octree_nodes);
// }

// // ----------------------------------------------------------------------------
// // Stage 7 (everything -> octree)
// // ----------------------------------------------------------------------------

// void process_stage_7(tree::AppData &appdata) {
//   // note: 1 here, skipping root
//   const int start = 1;
//   const int end = appdata.get_n_octree_nodes();

//   LOG_KERNEL(LogKernelType::kOMP, 7, &appdata);

// #pragma omp for
//   for (int i = start; i < end; ++i) {
//     process_oct_node(i,
//                      reinterpret_cast<int(*)[8]>(appdata.u_oct_children_s7.data()),
//                      appdata.u_oct_corner_s7.data(),
//                      appdata.u_oct_cell_size_s7.data(),
//                      appdata.u_oct_child_node_mask_s7.data(),
//                      appdata.u_edge_offset_s6.data(),
//                      appdata.u_edge_count_s5.data(),
//                      appdata.u_morton_keys_unique_s3.data(),
//                      appdata.u_brt_prefix_n_s4.data(),
//                      appdata.u_brt_parents_s4.data(),
//                      tree::kMinCoord,
//                      tree::kRange);
//   }
// }

// ----------------------------------------------------------------------------
// Stage 1 (xyz -> morton)
// ----------------------------------------------------------------------------

void process_stage_1(tree::SafeAppData &appdata) {
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

void process_stage_2(tree::SafeAppData &appdata) {
  // const auto num_threads = omp_get_num_threads();
  // const auto num_buckets = num_threads;

  // LOG_KERNEL(LogKernelType::kOMP, 2, &appdata);

  // omp::TmpStorage temp_storage;
  // temp_storage.allocate(num_buckets, num_threads);
  // assert(temp_storage.is_allocated());

  // bucket_sort(appdata.u_morton_keys_s1.data(),
  //             appdata.u_morton_keys_sorted_s2.data(),
  //             temp_storage.global_n_elem(),
  //             temp_storage.global_starting_position(),
  //             temp_storage.buckets(),
  //             appdata.get_n_input(),
  //             num_buckets,
  //             num_threads);

  // // by this point, 'u_morton_keys_sorted_s2' is sorted

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

void process_stage_3(tree::SafeAppData &appdata) {
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

void process_stage_4(tree::SafeAppData &appdata) {
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

void process_stage_5(tree::SafeAppData &appdata) {
  const int start = 0;
  const int end = appdata.get_n_brt_nodes();

  LOG_KERNEL(LogKernelType::kOMP, 5, &appdata);

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

void process_stage_6(tree::SafeAppData &appdata) {
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

void process_stage_7(tree::SafeAppData &appdata) {
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
  }
}

}  // namespace tree::omp
