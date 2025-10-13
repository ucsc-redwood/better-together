#include "dispatchers.hpp"

#include <cstdint>
#include <numeric>

#include "../../app.hpp"
#include "../../debug_logger.hpp"

namespace tree::vulkan {

struct LocalPushConstants {
  uint32_t n_elements;
};

struct GlobalPushConstants {
  uint32_t n_blocks;
};

// --------------------------------------------------------------------------
// Stage 1 - Morton code computation
// --------------------------------------------------------------------------

struct MortonPushConstants {
  uint32_t n;
  float min_coord;
  float range;
};

// --------------------------------------------------------------------------
// Stage 2-3 - Radix Sort and Unique
// --------------------------------------------------------------------------

struct InputSizePushConstantsUnsigned {
  uint32_t n;
};

struct FindDupsPushConstants {
  int32_t n;
};

// // layout(push_constant) uniform PushConstants {
// //   uint n_logical_blocks;  // Number of logical blocks requested
// //   uint n;                 // Total number of elements in the array
// //   uint width;             // Current width of each sorted subsequence
// //   uint num_pairs;         // Number of pairs to merge
// // }
// // pc;
// struct MergeSortPushConstants {
//   uint32_t n_logical_blocks;
//   uint32_t n;
//   uint32_t width;
//   uint32_t num_pairs;
// };

struct PrefixSumPushConstants {
  uint32_t inputSize;
};

struct MoveDupsPushConstants {
  uint32_t n;
};

// --------------------------------------------------------------------------
// Stage 4 - Build Radix Tree
// --------------------------------------------------------------------------

struct RadixTreePushConstants {
  int32_t n;
};

// --------------------------------------------------------------------------
// Stage 5 - Edge Count
// --------------------------------------------------------------------------

struct EdgeCountPushConstants {
  int32_t n_brt_nodes;
};

// --------------------------------------------------------------------------
// Stage 7 - Build Octree
// --------------------------------------------------------------------------

struct OctreePushConstants {
  float min_coord;
  float range;
  int32_t n_brt_nodes;
};

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------

// SHADER_ENTRY(tree_build_octree),
// SHADER_ENTRY(tree_build_radix_tree),
// SHADER_ENTRY(tree_edge_count),
// SHADER_ENTRY(tree_find_dups),
// SHADER_ENTRY(tree_merge_sort),
// SHADER_ENTRY(tree_morton),
// SHADER_ENTRY(tree_move_dups),
// SHADER_ENTRY(tree_naive_prefix_sum),
//
VulkanDispatcher::VulkanDispatcher() : engine(), seq(engine.make_seq()) {
  spdlog::info("VulkanDispatcher instance created.");

  auto morton_algo = engine.make_algo("tree_morton")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(2)
                         ->push_constant<MortonPushConstants>()
                         ->build();

  cached_algorithms.try_emplace("morton", std::move(morton_algo));

  auto radixsort_algo =
      engine.make_algo("tmp_single_radixsort_warp" + std::to_string(get_vulkan_warp_size()))
          ->work_group_size(256, 1, 1)
          ->num_sets(1)
          ->num_buffers(2)
          ->push_constant<InputSizePushConstantsUnsigned>()
          ->build();

  cached_algorithms.try_emplace("radixsort", std::move(radixsort_algo));

  auto find_dups_algo = engine.make_algo("tree_find_dups")
                            ->work_group_size(256, 1, 1)
                            ->num_sets(1)
                            ->num_buffers(2)
                            ->push_constant<FindDupsPushConstants>()
                            ->build();

  cached_algorithms.try_emplace("find_dups", std::move(find_dups_algo));

  auto prefix_sum_algo = engine.make_algo("tree_naive_prefix_sum")
                             ->work_group_size(1, 1, 1)
                             ->num_sets(1)
                             ->num_buffers(2)
                             ->push_constant<PrefixSumPushConstants>()
                             ->build();

  cached_algorithms.try_emplace("prefix_sum", std::move(prefix_sum_algo));

  auto move_dups_algo = engine.make_algo("tree_move_dups")
                            ->work_group_size(256, 1, 1)
                            ->num_sets(1)
                            ->num_buffers(3)
                            ->push_constant<MoveDupsPushConstants>()
                            ->build();

  cached_algorithms.try_emplace("move_dups", std::move(move_dups_algo));

  auto build_radix_tree_algo = engine.make_algo("tree_build_radix_tree")
                                   ->work_group_size(256, 1, 1)
                                   ->num_sets(1)
                                   ->num_buffers(6)
                                   ->push_constant<RadixTreePushConstants>()
                                   ->build();

  cached_algorithms.try_emplace("build_radix_tree", std::move(build_radix_tree_algo));

  auto edge_count_algo = engine.make_algo("tree_edge_count")
                             ->work_group_size(512, 1, 1)  // Edge count uses 512 threads
                             ->num_sets(1)
                             ->num_buffers(3)
                             ->push_constant<EdgeCountPushConstants>()
                             ->build();

  cached_algorithms.try_emplace("edge_count", std::move(edge_count_algo));

  auto build_octree_algo = engine.make_algo("tree_build_octree")
                               ->work_group_size(256, 1, 1)
                               ->num_sets(1)
                               ->num_buffers(13)
                               ->push_constant<OctreePushConstants>()
                               ->build();

  cached_algorithms.try_emplace("build_octree", std::move(build_octree_algo));
}

// ----------------------------------------------------------------------------
// Stage 1 (Input -> Morton)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_1(VkAppData_Safe &appdata) {
  LOG_KERNEL(LogKernelType::kVK, 1, &appdata);

  auto algo = cached_algorithms.at("morton").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_input_points_s0),
                                  engine.get_buffer_info(appdata.u_morton_keys_s1_out),
                              });

  algo->update_push_constant(MortonPushConstants{
      .n = static_cast<uint32_t>(appdata.get_n_input()),
      .min_coord = tree::kMinCoord,
      .range = tree::kRange,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(
      seq->get_handle(),
      {static_cast<uint32_t>(kiss_vk::div_ceil(appdata.get_n_input(), 256)), 1, 1});
  seq->cmd_end();

  seq->reset_fence();
  seq->submit();
  seq->wait_for_fence();
}

// ----------------------------------------------------------------------------
// Stage 2 (Morton -> Sorted Morton)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_2(VkAppData_Safe &appdata) {
  LOG_KERNEL(LogKernelType::kVK, 2, &appdata);

  // auto algo = cached_algorithms.at("merge_sort").get();

  // algo->update_descriptor_set(0,
  //                             {
  //                                 engine.get_buffer_info(appdata.u_morton_keys_s1),
  //                                 engine.get_buffer_info(appdata.u_morton_keys_sorted_s2),
  //                             });

  // algo->update_push_constant(MergeSortPushConstants{
  //     .n_logical_blocks = 1,
  //     .n = appdata.get_n_input(),
  //     .width = 1,
  //     .num_pairs = 1,
  // });

  auto algo = cached_algorithms.at("radixsort").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_morton_keys_s1),
                                  engine.get_buffer_info(appdata.u_morton_keys_sorted_s2_out),
                              });

  algo->update_push_constant(InputSizePushConstantsUnsigned{
      .n = appdata.get_n_input(),
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(), {1, 1, 1});  // Special case: single workgroup
  seq->cmd_end();

  seq->reset_fence();
  seq->submit();
  seq->wait_for_fence();
}

// ----------------------------------------------------------------------------
// Stage 3 (Sorted Morton -> Unique Sorted Morton)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_3(VkAppData_Safe &appdata) {
  LOG_KERNEL(LogKernelType::kVK, 3, &appdata);

  const auto last = std::unique_copy(appdata.u_morton_keys_sorted_s2.data(),
                                     appdata.u_morton_keys_sorted_s2.data() + appdata.get_n_input(),
                                     appdata.u_morton_keys_unique_s3_out.data());
  const auto n_unique = std::distance(appdata.u_morton_keys_unique_s3_out.data(), last);

  appdata.set_n_unique(n_unique);
  appdata.set_n_brt_nodes(n_unique - 1);
}

// ----------------------------------------------------------------------------
// Stage 4 (Unique Sorted Morton -> BRT)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_4(VkAppData_Safe &appdata) {
  LOG_KERNEL(LogKernelType::kVK, 4, &appdata);

  const int32_t n = appdata.get_n_unique();
  auto algo = cached_algorithms.at("build_radix_tree").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_morton_keys_unique_s3),
                                  engine.get_buffer_info(appdata.u_brt_prefix_n_s4_out),
                                  engine.get_buffer_info(appdata.u_brt_has_leaf_left_s4_out),
                                  engine.get_buffer_info(appdata.u_brt_has_leaf_right_s4_out),
                                  engine.get_buffer_info(appdata.u_brt_left_child_s4_out),
                                  engine.get_buffer_info(appdata.u_brt_parents_s4_out),
                              });

  algo->update_push_constant(RadixTreePushConstants{
      .n = n,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(n, 256)), 1, 1});
  seq->cmd_end();

  seq->reset_fence();
  seq->submit();
  seq->wait_for_fence();
}

// ----------------------------------------------------------------------------
// Stage 5 (BRT -> Edge Count)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_5(VkAppData_Safe &appdata) {
  LOG_KERNEL(LogKernelType::kVK, 5, &appdata);

  auto algo = cached_algorithms.at("edge_count").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_brt_prefix_n_s4),    // input
                                  engine.get_buffer_info(appdata.u_brt_parents_s4),     // input
                                  engine.get_buffer_info(appdata.u_edge_count_s5_out),  // output
                              });

  algo->update_push_constant(EdgeCountPushConstants{
      .n_brt_nodes = static_cast<int32_t>(appdata.get_n_brt_nodes()),
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(
      seq->get_handle(),
      {static_cast<uint32_t>(kiss_vk::div_ceil(appdata.get_n_brt_nodes(), 512)), 1, 1});
  seq->cmd_end();

  seq->reset_fence();
  seq->submit();
  seq->wait_for_fence();
}

// ----------------------------------------------------------------------------
// Stage 6 (Edge Count -> Edge Offset, prefix sum)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_6(VkAppData_Safe &appdata) {
  LOG_KERNEL(LogKernelType::kVK, 6, &appdata);

  const int start = 0;
  const int end = appdata.get_n_brt_nodes();

  std::partial_sum(appdata.u_edge_count_s5.data() + start,
                   appdata.u_edge_count_s5.data() + end,
                   appdata.u_edge_offset_s6_out.data() + start);

  // num_octree node is the result of the partial sum
  const auto num_octree_nodes = appdata.u_edge_offset_s6_out[end - 1];

  // No-op since SafeAppData has const n_octree_nodes
  appdata.set_n_octree_nodes(num_octree_nodes);
}

//----------------------------------------------------------------------------
// Stage 7 (Edge Offset -> Octree)
//----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_7(VkAppData_Safe &appdata) {
  LOG_KERNEL(LogKernelType::kVK, 7, &appdata);

  auto algo = cached_algorithms.at("build_octree").get();

  algo->update_descriptor_set(
      0,
      {
          engine.get_buffer_info(appdata.u_oct_children_s7_out),         // output
          engine.get_buffer_info(appdata.u_oct_corner_s7_out),           // output
          engine.get_buffer_info(appdata.u_oct_cell_size_s7_out),        // output
          engine.get_buffer_info(appdata.u_oct_child_node_mask_s7_out),  // output
          engine.get_buffer_info(appdata.u_oct_child_leaf_mask_s7_out),  // output
          engine.get_buffer_info(appdata.u_edge_offset_s6),              // input
          engine.get_buffer_info(appdata.u_edge_count_s5),               // input
          engine.get_buffer_info(appdata.u_morton_keys_unique_s3),       // input
          engine.get_buffer_info(appdata.u_brt_prefix_n_s4),             // input
          engine.get_buffer_info(appdata.u_brt_parents_s4),              // input
          engine.get_buffer_info(appdata.u_brt_left_child_s4),           // input
          engine.get_buffer_info(appdata.u_brt_has_leaf_left_s4),        // input
          engine.get_buffer_info(appdata.u_brt_has_leaf_right_s4),       // input
      });

  algo->update_push_constant(OctreePushConstants{
      .min_coord = tree::kMinCoord,
      .range = tree::kRange,
      .n_brt_nodes = static_cast<int32_t>(appdata.get_n_brt_nodes()),
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(
      seq->get_handle(),
      {static_cast<uint32_t>(kiss_vk::div_ceil(appdata.get_n_octree_nodes(), 256)), 1, 1});
  seq->cmd_end();

  seq->reset_fence();
  seq->submit();
  seq->wait_for_fence();
}

}  // namespace tree::vulkan
