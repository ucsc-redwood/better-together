#include "dispatchers.hpp"

#include <cstdint>

#include "platform/util/debug_logger.hpp"

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

// Multi-workgroup (device-wide) radix sort push constants (shared by the
// histogram and scatter passes); see multi_radixsort{,_histograms}.comp.
struct RadixSortPushConstants {
  uint32_t g_num_elements;
  uint32_t g_shift;
  uint32_t g_num_workgroups;
  uint32_t g_num_blocks_per_workgroup;
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
VulkanDispatcher::VulkanDispatcher() : engine(), default_seq(engine.make_seq()) {
  spdlog::info("VulkanDispatcher instance created.");

  auto morton_algo = engine.make_algo("tree_morton")
                         ->work_group_size(256, 1, 1)
                         ->num_buffers(2)
                         ->push_constant<MortonPushConstants>()
                         ->build();

  cached_algorithms.try_emplace("morton", std::move(morton_algo));

  // Multi-workgroup (device-wide) LSD radix sort: histogram pass (in,
  // histograms) + scatter pass (in, out, histograms), 4x8-bit, ping-pong. The 4
  // radix passes are recorded into one command buffer (one submit); record_bind
  // captures each pass's ping-pong buffers per dispatch.
  auto radix_histograms_algo = engine.make_algo("multi_radixsort_histograms")
                                   ->work_group_size(256, 1, 1)
                                   ->num_buffers(2)
                                   ->push_constant<RadixSortPushConstants>()
                                   ->build();

  cached_algorithms.try_emplace("radix_histograms", std::move(radix_histograms_algo));

  // The scatter pass's subgroup-sum array is sized by the subgroup width: one shader,
  // specialized with the device's subgroup size (pinned where supported).
  auto radix_scatter_algo = engine.make_algo("multi_radixsort")
                                ->work_group_size(256, 1, 1)
                                ->subgroup_size_dependent()
                                ->num_buffers(3)
                                ->push_constant<RadixSortPushConstants>()
                                ->build();

  cached_algorithms.try_emplace("radix_scatter", std::move(radix_scatter_algo));

  auto find_dups_algo = engine.make_algo("tree_find_dups")
                            ->work_group_size(256, 1, 1)
                            ->num_buffers(2)
                            ->push_constant<FindDupsPushConstants>()
                            ->build();

  cached_algorithms.try_emplace("find_dups", std::move(find_dups_algo));

  // Device-wide inclusive scan (3 passes), used by stage 3 (flags) and stage 6
  // (edge counts). See tree_scan_*.comp. Each pass is its own algo; all three
  // are recorded into one command buffer, each ordered after the pass it reads
  // (the Sequence's dependency tracking). Stage 3 and stage 6 can share one command buffer (a
  // whole-pipeline chunk); record_bind keeps each scan's bindings its own.
  auto scan_local_algo = engine.make_algo("tree_scan_local")
                             ->work_group_size(256, 1, 1)
                             ->num_buffers(3)
                             ->push_constant<PrefixSumPushConstants>()
                             ->build();

  cached_algorithms.try_emplace("scan_local", std::move(scan_local_algo));

  auto scan_block_sums_algo = engine.make_algo("tree_scan_block_sums")
                                  ->work_group_size(256, 1, 1)
                                  ->num_buffers(1)
                                  ->push_constant<GlobalPushConstants>()
                                  ->build();

  cached_algorithms.try_emplace("scan_block_sums", std::move(scan_block_sums_algo));

  auto scan_add_algo = engine.make_algo("tree_scan_add")
                           ->work_group_size(256, 1, 1)
                           ->num_buffers(2)
                           ->push_constant<PrefixSumPushConstants>()
                           ->build();

  cached_algorithms.try_emplace("scan_add", std::move(scan_add_algo));

  auto move_dups_algo = engine.make_algo("tree_move_dups")
                            ->work_group_size(256, 1, 1)
                            ->num_buffers(4)
                            ->push_constant<MoveDupsPushConstants>()
                            ->build();

  cached_algorithms.try_emplace("move_dups", std::move(move_dups_algo));

  auto build_radix_tree_algo = engine.make_algo("tree_build_radix_tree")
                                   ->work_group_size(256, 1, 1)
                                   ->num_buffers(6)
                                   ->push_constant<RadixTreePushConstants>()
                                   ->build();

  cached_algorithms.try_emplace("build_radix_tree", std::move(build_radix_tree_algo));

  auto edge_count_algo = engine.make_algo("tree_edge_count")
                             ->work_group_size(512, 1, 1)  // Edge count uses 512 threads
                             ->num_buffers(3)
                             ->push_constant<EdgeCountPushConstants>()
                             ->build();

  cached_algorithms.try_emplace("edge_count", std::move(edge_count_algo));

  auto build_octree_algo = engine.make_algo("tree_build_octree")
                               ->work_group_size(256, 1, 1)
                               ->num_buffers(13)
                               ->push_constant<OctreePushConstants>()
                               ->build();

  cached_algorithms.try_emplace("build_octree", std::move(build_octree_algo));
}

// ----------------------------------------------------------------------------
// run_stage_k: per-stage single-submit wrappers (bm_main per-stage device timing).
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_1(VkAppData_Safe& appdata) { dispatch_multi_stage(appdata, 1, 1); }
void VulkanDispatcher::run_stage_2(VkAppData_Safe& appdata) { dispatch_multi_stage(appdata, 2, 2); }
void VulkanDispatcher::run_stage_3(VkAppData_Safe& appdata) { dispatch_multi_stage(appdata, 3, 3); }
void VulkanDispatcher::run_stage_4(VkAppData_Safe& appdata) { dispatch_multi_stage(appdata, 4, 4); }
void VulkanDispatcher::run_stage_5(VkAppData_Safe& appdata) { dispatch_multi_stage(appdata, 5, 5); }
void VulkanDispatcher::run_stage_6(VkAppData_Safe& appdata) { dispatch_multi_stage(appdata, 6, 6); }
void VulkanDispatcher::run_stage_7(VkAppData_Safe& appdata) { dispatch_multi_stage(appdata, 7, 7); }

// ----------------------------------------------------------------------------
// Stage 1 (Input -> Morton)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_1(VkAppData_Safe& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 1, &appdata);

  auto algo = cached_algorithms.at("morton").get();

  const auto algo_pc = MortonPushConstants{
      .n = static_cast<uint32_t>(appdata.get_n_input()),
      .min_coord = tree::kMinCoord,
      .range = tree::kRange,
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_input_points_s0),
                        engine.get_buffer_info(appdata.u_morton_keys_s1_out),
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(
      seq, {static_cast<uint32_t>(kiss_vk::div_ceil(appdata.get_n_input(), 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 2 (Morton -> Sorted Morton)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_2(VkAppData_Safe& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 2, &appdata);

  // Device-wide multi-workgroup LSD radix sort (Mirco Werner / Embree). Each of
  // the 4 8-bit passes runs two dispatches: a histogram pass (per-workgroup
  // 256-bin counts) and a scatter pass (global offsets + reorder). Buffers
  // ping-pong so the result lands in u_morton_keys_sorted_s2_out after 4 passes:
  //   src/dst:  s1->tmp -> out -> tmp -> out  (final = out).
  // The 8 dispatches are recorded into the chunk's single command buffer; the
  // fence wait is once per chunk, not per pass.

  auto* hist_algo = cached_algorithms.at("radix_histograms").get();
  auto* scatter_algo = cached_algorithms.at("radix_scatter").get();

  const uint32_t n = appdata.get_n_input();
  constexpr uint32_t kWorkgroupSize = 256;
  const uint32_t num_workgroups = VkAppData_Safe::kRadixNumWorkgroups;
  // Cover all elements: each workgroup processes blocks_per_wg blocks of
  // kWorkgroupSize elements. ceil(ceil(n / WG) / num_workgroups).
  const uint32_t total_blocks = static_cast<uint32_t>(kiss_vk::div_ceil(n, kWorkgroupSize));
  const uint32_t blocks_per_workgroup =
      static_cast<uint32_t>(kiss_vk::div_ceil(total_blocks, num_workgroups));

  // Ping-pong buffer sequence so the final pass writes into _out.
  const std::pmr::vector<uint32_t>* src_seq[4] = {
      &appdata.u_morton_keys_s1,
      &appdata.u_sort_tmp,
      &appdata.u_morton_keys_sorted_s2_out,
      &appdata.u_sort_tmp,
  };
  std::pmr::vector<uint32_t>* dst_seq[4] = {
      &appdata.u_sort_tmp,
      &appdata.u_morton_keys_sorted_s2_out,
      &appdata.u_sort_tmp,
      &appdata.u_morton_keys_sorted_s2_out,
  };

  // Every dispatch depends on the one before it: histogram->scatter shares the
  // histograms buffer, and pass i's scatter output is pass i+1's histogram/scatter
  // input (ping-pong). The Sequence records a dependency (RAW + WAW) before each
  // (design A3). A fence-wait between separate submits guarantees execution order
  // but NOT memory availability/visibility across submits -- AMD/Xclipse are
  // coherent and tolerate it, but Mali-G710 is not. Doing it in one submit with
  // explicit dependencies makes them correct on all three backends.

  for (uint32_t pass = 0; pass < 4; ++pass) {
    const RadixSortPushConstants pc{
        .g_num_elements = n,
        .g_shift = 8u * pass,
        .g_num_workgroups = num_workgroups,
        .g_num_blocks_per_workgroup = blocks_per_workgroup,
    };

    // histogram pass (src -> histograms)
    hist_algo->record_bind(seq,
                           {
                               engine.get_buffer_info(*src_seq[pass]),
                               engine.get_buffer_info(appdata.u_sort_histograms),
                           });
    hist_algo->record_push_constants(seq, pc);
    hist_algo->record_dispatch(seq, {num_workgroups, 1, 1});

    // scatter pass (src + histograms -> dst)
    scatter_algo->record_bind(seq,
                              {
                                  engine.get_buffer_info(*src_seq[pass]),
                                  engine.get_buffer_info(*dst_seq[pass]),
                                  engine.get_buffer_info(appdata.u_sort_histograms),
                              });
    scatter_algo->record_push_constants(seq, pc);
    scatter_algo->record_dispatch(seq, {num_workgroups, 1, 1});
  }
}

// ----------------------------------------------------------------------------
// Stage 3 (Sorted Morton -> Unique Sorted Morton)
// ----------------------------------------------------------------------------

// Record a device-wide INCLUSIVE scan of `src` (n elements) into `dst`, using
// u_scan_block_sums as scratch, into `seq`'s open command buffer.
// Three passes (local scan -> scan block totals -> add block offsets), each ordered
// after the pass it reads by the Sequence's dependency tracking, all in one submit.
// Caller does cmd_begin/end + submit + fence-wait.
void VulkanDispatcher::record_device_scan(kiss_vk::Sequence& seq,
                                          vk::DescriptorBufferInfo src,
                                          vk::DescriptorBufferInfo dst,
                                          vk::DescriptorBufferInfo block_sums,
                                          uint32_t n) {
  auto* local_algo = cached_algorithms.at("scan_local").get();
  auto* block_algo = cached_algorithms.at("scan_block_sums").get();
  auto* add_algo = cached_algorithms.at("scan_add").get();

  const uint32_t elems_per_wg = VkAppData_Safe::kScanElementsPerWg;
  const uint32_t num_blocks = static_cast<uint32_t>(kiss_vk::div_ceil(n, elems_per_wg));

  // Pass 1: per-block inclusive scan + per-block totals.
  const auto local_algo_pc = PrefixSumPushConstants{.inputSize = n};
  local_algo->record_bind(seq, {src, dst, block_sums});
  local_algo->record_push_constants(seq, local_algo_pc);
  local_algo->record_dispatch(seq, {num_blocks, 1, 1});

  // Pass 2: exclusive scan of the per-block totals (single workgroup, in place).
  const auto block_algo_pc = GlobalPushConstants{.n_blocks = num_blocks};
  block_algo->record_bind(seq, {block_sums});
  block_algo->record_push_constants(seq, block_algo_pc);
  block_algo->record_dispatch(seq, {1, 1, 1});

  // Pass 3: add the exclusive block offset to every element of each block.
  const auto add_algo_pc = PrefixSumPushConstants{.inputSize = n};
  add_algo->record_bind(seq, {dst, block_sums});
  add_algo->record_push_constants(seq, add_algo_pc);
  add_algo->record_dispatch(seq, {num_blocks, 1, 1});
}

void VulkanDispatcher::record_stage_3(VkAppData_Safe& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 3, &appdata);

  // Stream compaction on the GPU: find_dups (run-head flags) -> device-wide
  // inclusive scan of the flags (compacted index) -> move_dups (scatter the
  // kept keys). All recorded into one command buffer / one submit; the Sequence
  // orders each dispatch after the ones it depends on.
  const uint32_t n = appdata.get_n_input();

  auto* find_algo = cached_algorithms.at("find_dups").get();
  auto* move_algo = cached_algorithms.at("move_dups").get();

  // Flags -> u_contributes; inclusive scan -> u_out_idx.
  const auto find_algo_pc = FindDupsPushConstants{.n = static_cast<int32_t>(n)};

  const auto move_algo_pc = MoveDupsPushConstants{.n = n};

  // find_dups: flags into u_contributes.
  find_algo->record_bind(seq,
                         {
                             engine.get_buffer_info(appdata.u_morton_keys_sorted_s2),
                             engine.get_buffer_info(appdata.u_contributes),
                         });
  find_algo->record_push_constants(seq, find_algo_pc);
  find_algo->record_dispatch(seq, {static_cast<uint32_t>(kiss_vk::div_ceil(n, 256)), 1, 1});

  // device-wide inclusive scan of the flags -> u_out_idx.
  record_device_scan(seq,
                     engine.get_buffer_info(appdata.u_contributes),
                     engine.get_buffer_info(appdata.u_out_idx),
                     engine.get_buffer_info(appdata.u_scan_block_sums),
                     n);

  // move_dups: scatter kept keys to their compacted positions.
  move_algo->record_bind(seq,
                         {
                             engine.get_buffer_info(appdata.u_contributes),
                             engine.get_buffer_info(appdata.u_out_idx),
                             engine.get_buffer_info(appdata.u_morton_keys_sorted_s2),
                             engine.get_buffer_info(appdata.u_morton_keys_unique_s3_out),
                         });
  move_algo->record_push_constants(seq, move_algo_pc);
  move_algo->record_dispatch(seq, {static_cast<uint32_t>(kiss_vk::div_ceil(n, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 4 (Unique Sorted Morton -> BRT)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_4(VkAppData_Safe& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 4, &appdata);

  const int32_t n = appdata.get_n_unique();
  auto algo = cached_algorithms.at("build_radix_tree").get();

  const auto algo_pc = RadixTreePushConstants{
      .n = n,
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_morton_keys_unique_s3),
                        engine.get_buffer_info(appdata.u_brt_prefix_n_s4_out),
                        engine.get_buffer_info(appdata.u_brt_has_leaf_left_s4_out),
                        engine.get_buffer_info(appdata.u_brt_has_leaf_right_s4_out),
                        engine.get_buffer_info(appdata.u_brt_left_child_s4_out),
                        engine.get_buffer_info(appdata.u_brt_parents_s4_out),
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(seq, {static_cast<uint32_t>(kiss_vk::div_ceil(n, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 5 (BRT -> Edge Count)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_5(VkAppData_Safe& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 5, &appdata);

  auto algo = cached_algorithms.at("edge_count").get();

  const auto algo_pc = EdgeCountPushConstants{
      .n_brt_nodes = static_cast<int32_t>(appdata.get_n_brt_nodes()),
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_brt_prefix_n_s4),    // input
                        engine.get_buffer_info(appdata.u_brt_parents_s4),     // input
                        engine.get_buffer_info(appdata.u_edge_count_s5_out),  // output
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(
      seq, {static_cast<uint32_t>(kiss_vk::div_ceil(appdata.get_n_brt_nodes(), 512)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 6 (Edge Count -> Edge Offset, prefix sum)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_6(VkAppData_Safe& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 6, &appdata);

  // Device-wide INCLUSIVE prefix sum of the edge counts (matches std::partial_sum
  // / cub::DeviceScan::InclusiveSum), entirely on the GPU. The buffers are int32
  // but the values are non-negative edge counts, so a uint scan is bit-identical.
  const uint32_t n = appdata.get_n_brt_nodes();

  record_device_scan(seq,
                     engine.get_buffer_info(appdata.u_edge_count_s5),
                     engine.get_buffer_info(appdata.u_edge_offset_s6_out),
                     engine.get_buffer_info(appdata.u_scan_block_sums),
                     n);
}

//----------------------------------------------------------------------------
// Stage 7 (Edge Offset -> Octree)
//----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_7(VkAppData_Safe& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 7, &appdata);

  auto algo = cached_algorithms.at("build_octree").get();

  const auto algo_pc = OctreePushConstants{
      .min_coord = tree::kMinCoord,
      .range = tree::kRange,
      .n_brt_nodes = static_cast<int32_t>(appdata.get_n_brt_nodes()),
  };

  algo->record_bind(seq,
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
  algo->record_push_constants(seq, algo_pc);
  // The shader loops over brt node indices [1, n_brt_nodes); size the grid for
  // n_brt_nodes (the grid-stride loop covers the rest).
  algo->record_dispatch(
      seq, {static_cast<uint32_t>(kiss_vk::div_ceil(appdata.get_n_brt_nodes(), 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Genuinely-chained overloads (VkAppData, no golden/_out split) -- same shaders
// and dispatch configuration as the VkAppData_Safe bodies above; the only
// change is that every OUTPUT buffer binds the plain field name (no `_out`
// suffix) instead of a separate `_out` field, since this instance's own next
// stage will read that same field as its real input. Every INPUT buffer keeps
// the same field name it already had -- for VkAppData_Safe that name was the
// (immutable, construction-time) golden; for VkAppData it's this instance's
// own previous-stage output, so no rename was needed there.
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_1(VkAppData& appdata) { dispatch_multi_stage(appdata, 1, 1); }
void VulkanDispatcher::run_stage_2(VkAppData& appdata) { dispatch_multi_stage(appdata, 2, 2); }
void VulkanDispatcher::run_stage_3(VkAppData& appdata) { dispatch_multi_stage(appdata, 3, 3); }
void VulkanDispatcher::run_stage_4(VkAppData& appdata) { dispatch_multi_stage(appdata, 4, 4); }
void VulkanDispatcher::run_stage_5(VkAppData& appdata) { dispatch_multi_stage(appdata, 5, 5); }
void VulkanDispatcher::run_stage_6(VkAppData& appdata) { dispatch_multi_stage(appdata, 6, 6); }
void VulkanDispatcher::run_stage_7(VkAppData& appdata) { dispatch_multi_stage(appdata, 7, 7); }

void VulkanDispatcher::record_stage_1(VkAppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 1, &appdata);

  auto algo = cached_algorithms.at("morton").get();

  const auto algo_pc = MortonPushConstants{
      .n = static_cast<uint32_t>(appdata.get_n_input()),
      .min_coord = tree::kMinCoord,
      .range = tree::kRange,
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_input_points_s0),
                        engine.get_buffer_info(appdata.u_morton_keys_s1),
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(
      seq, {static_cast<uint32_t>(kiss_vk::div_ceil(appdata.get_n_input(), 256)), 1, 1});
}

void VulkanDispatcher::record_stage_2(VkAppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 2, &appdata);

  auto* hist_algo = cached_algorithms.at("radix_histograms").get();
  auto* scatter_algo = cached_algorithms.at("radix_scatter").get();

  const uint32_t n = appdata.get_n_input();
  constexpr uint32_t kWorkgroupSize = 256;
  const uint32_t num_workgroups = VkAppData::kRadixNumWorkgroups;
  const uint32_t total_blocks = static_cast<uint32_t>(kiss_vk::div_ceil(n, kWorkgroupSize));
  const uint32_t blocks_per_workgroup =
      static_cast<uint32_t>(kiss_vk::div_ceil(total_blocks, num_workgroups));

  // Ping-pong buffer sequence so the final pass writes into the plain (chained)
  // stage-2 field -- this instance's real output, unlike VkAppData_Safe's _out.
  const std::pmr::vector<uint32_t>* src_seq[4] = {
      &appdata.u_morton_keys_s1,
      &appdata.u_sort_tmp,
      &appdata.u_morton_keys_sorted_s2,
      &appdata.u_sort_tmp,
  };
  std::pmr::vector<uint32_t>* dst_seq[4] = {
      &appdata.u_sort_tmp,
      &appdata.u_morton_keys_sorted_s2,
      &appdata.u_sort_tmp,
      &appdata.u_morton_keys_sorted_s2,
  };

  for (uint32_t pass = 0; pass < 4; ++pass) {
    const RadixSortPushConstants pc{
        .g_num_elements = n,
        .g_shift = 8u * pass,
        .g_num_workgroups = num_workgroups,
        .g_num_blocks_per_workgroup = blocks_per_workgroup,
    };

    hist_algo->record_bind(seq,
                           {
                               engine.get_buffer_info(*src_seq[pass]),
                               engine.get_buffer_info(appdata.u_sort_histograms),
                           });
    hist_algo->record_push_constants(seq, pc);
    hist_algo->record_dispatch(seq, {num_workgroups, 1, 1});

    scatter_algo->record_bind(seq,
                              {
                                  engine.get_buffer_info(*src_seq[pass]),
                                  engine.get_buffer_info(*dst_seq[pass]),
                                  engine.get_buffer_info(appdata.u_sort_histograms),
                              });
    scatter_algo->record_push_constants(seq, pc);
    scatter_algo->record_dispatch(seq, {num_workgroups, 1, 1});
  }
}

void VulkanDispatcher::record_stage_3(VkAppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 3, &appdata);

  const uint32_t n = appdata.get_n_input();

  auto* find_algo = cached_algorithms.at("find_dups").get();
  auto* move_algo = cached_algorithms.at("move_dups").get();

  const auto find_algo_pc = FindDupsPushConstants{.n = static_cast<int32_t>(n)};

  const auto move_algo_pc = MoveDupsPushConstants{.n = n};

  find_algo->record_bind(seq,
                         {
                             engine.get_buffer_info(appdata.u_morton_keys_sorted_s2),
                             engine.get_buffer_info(appdata.u_contributes),
                         });
  find_algo->record_push_constants(seq, find_algo_pc);
  find_algo->record_dispatch(seq, {static_cast<uint32_t>(kiss_vk::div_ceil(n, 256)), 1, 1});

  // Device-wide inclusive scan of the flags -> u_out_idx. u_out_idx[n-1] after
  // this is exactly n_unique (the running total of "is a new unique value"
  // flags) -- dispatch_multi_stage's VkAppData overload reads this back after
  // stage 3 completes (see its comment for why: stage 4's dispatch size is a
  // host-side vkCmdDispatch argument, so n_unique must be known before stage 4
  // is recorded).
  record_device_scan(seq,
                     engine.get_buffer_info(appdata.u_contributes),
                     engine.get_buffer_info(appdata.u_out_idx),
                     engine.get_buffer_info(appdata.u_scan_block_sums),
                     n);

  move_algo->record_bind(seq,
                         {
                             engine.get_buffer_info(appdata.u_contributes),
                             engine.get_buffer_info(appdata.u_out_idx),
                             engine.get_buffer_info(appdata.u_morton_keys_sorted_s2),
                             engine.get_buffer_info(appdata.u_morton_keys_unique_s3),
                         });
  move_algo->record_push_constants(seq, move_algo_pc);
  move_algo->record_dispatch(seq, {static_cast<uint32_t>(kiss_vk::div_ceil(n, 256)), 1, 1});
}

void VulkanDispatcher::record_stage_4(VkAppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 4, &appdata);

  const int32_t n = static_cast<int32_t>(appdata.get_n_unique());
  auto algo = cached_algorithms.at("build_radix_tree").get();

  const auto algo_pc = RadixTreePushConstants{
      .n = n,
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_morton_keys_unique_s3),
                        engine.get_buffer_info(appdata.u_brt_prefix_n_s4),
                        engine.get_buffer_info(appdata.u_brt_has_leaf_left_s4),
                        engine.get_buffer_info(appdata.u_brt_has_leaf_right_s4),
                        engine.get_buffer_info(appdata.u_brt_left_child_s4),
                        engine.get_buffer_info(appdata.u_brt_parents_s4),
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(seq, {static_cast<uint32_t>(kiss_vk::div_ceil(n, 256)), 1, 1});
}

void VulkanDispatcher::record_stage_5(VkAppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 5, &appdata);

  auto algo = cached_algorithms.at("edge_count").get();

  const auto algo_pc = EdgeCountPushConstants{
      .n_brt_nodes = static_cast<int32_t>(appdata.get_n_brt_nodes()),
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_brt_prefix_n_s4),  // input
                        engine.get_buffer_info(appdata.u_brt_parents_s4),   // input
                        engine.get_buffer_info(appdata.u_edge_count_s5),    // output
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(
      seq, {static_cast<uint32_t>(kiss_vk::div_ceil(appdata.get_n_brt_nodes(), 512)), 1, 1});
}

void VulkanDispatcher::record_stage_6(VkAppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 6, &appdata);

  const uint32_t n = appdata.get_n_brt_nodes();

  // u_edge_offset_s6[n_brt_nodes-1] after this is exactly n_octree_nodes --
  // dispatch_multi_stage's VkAppData overload reads this back once this call's
  // range has included stage 6 (bookkeeping value; stage 7's own dispatch size
  // does not depend on it, only on n_brt_nodes, already known since stage 4).
  record_device_scan(seq,
                     engine.get_buffer_info(appdata.u_edge_count_s5),
                     engine.get_buffer_info(appdata.u_edge_offset_s6),
                     engine.get_buffer_info(appdata.u_scan_block_sums),
                     n);
}

void VulkanDispatcher::record_stage_7(VkAppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 7, &appdata);

  auto algo = cached_algorithms.at("build_octree").get();

  const auto algo_pc = OctreePushConstants{
      .min_coord = tree::kMinCoord,
      .range = tree::kRange,
      .n_brt_nodes = static_cast<int32_t>(appdata.get_n_brt_nodes()),
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_oct_children_s7),         // output
                        engine.get_buffer_info(appdata.u_oct_corner_s7),           // output
                        engine.get_buffer_info(appdata.u_oct_cell_size_s7),        // output
                        engine.get_buffer_info(appdata.u_oct_child_node_mask_s7),  // output
                        engine.get_buffer_info(appdata.u_oct_child_leaf_mask_s7),  // output
                        engine.get_buffer_info(appdata.u_edge_offset_s6),          // input
                        engine.get_buffer_info(appdata.u_edge_count_s5),           // input
                        engine.get_buffer_info(appdata.u_morton_keys_unique_s3),   // input
                        engine.get_buffer_info(appdata.u_brt_prefix_n_s4),         // input
                        engine.get_buffer_info(appdata.u_brt_parents_s4),          // input
                        engine.get_buffer_info(appdata.u_brt_left_child_s4),       // input
                        engine.get_buffer_info(appdata.u_brt_has_leaf_left_s4),    // input
                        engine.get_buffer_info(appdata.u_brt_has_leaf_right_s4),   // input
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(
      seq, {static_cast<uint32_t>(kiss_vk::div_ceil(appdata.get_n_brt_nodes(), 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// dispatch_multi_stage(VkAppData&, ...) -- unlike the VkAppData_Safe overload,
// this cannot always batch [start_stage, end_stage] into one command
// buffer/submit: stage 4's dispatch size (n_unique/n_brt_nodes) and stage 7's
// eventual n_octree_nodes bookkeeping value are only known once stage 3 (resp.
// stage 6) has actually run on the GPU and the host has read the result back.
//
// The only HARD split is at the stage 3|4 boundary WITHIN a single call: stage
// 4's vkCmdDispatch grid size is a host-side argument fixed at command-buffer
// record time, so if this call's own range spans both stage 3 and stage 4, the
// batch must break there (submit + fence-wait + host readback of n_unique/
// n_brt_nodes, THEN record stage 4+). If the range starts after stage 3 (an
// earlier call already covered it) or ends at/before stage 3, no split is
// needed -- either the count is already set on this instance, or nothing in
// this call needs it yet.
//
// n_octree_nodes has no such hard requirement (stage 7's own dispatch size
// only needs n_brt_nodes, already known): it's simply read back after whatever
// batch this call's range causes stage 6 to finish in, since the wait
// already invalidates every buffer that batch touched.
uint64_t VulkanDispatcher::submit_multi_stage(VkAppData& data,
                                              const int start_stage,
                                              const int end_stage,
                                              kiss_vk::Sequence& seq) {
  if (start_stage < 1 || end_stage > 7 || start_stage > end_stage)
    throw std::out_of_range("Invalid stage");

  auto submit_batch = [&](int lo, int hi) {
    seq.cmd_begin();
    for (int stage = lo; stage <= hi; ++stage) {
      (this->*record_functions_chained[stage - 1])(data, seq);
    }
    seq.cmd_end();
    return seq.submit();
  };

  if (start_stage <= 3 && end_stage >= 4) {  // crosses 3|4: n_unique first
    seq.wait(submit_batch(start_stage, 3));
    const uint32_t n_input = data.get_n_input();
    const uint32_t n_unique = data.u_out_idx[n_input - 1];
    data.set_n_unique(n_unique);
    data.set_n_brt_nodes(n_unique - 1);
    return submit_batch(4, end_stage);
  }
  return submit_batch(start_stage, end_stage);
}

void VulkanDispatcher::complete_multi_stage(VkAppData& data,
                                            const int start_stage,
                                            const int end_stage,
                                            kiss_vk::Sequence& seq,
                                            const uint64_t ticket) {
  seq.wait(ticket);

  // A range that ends at stage 3 (without crossing into 4) produced n_unique just now.
  if (start_stage <= 3 && end_stage == 3) {
    const uint32_t n_input = data.get_n_input();
    const uint32_t n_unique = data.u_out_idx[n_input - 1];
    data.set_n_unique(n_unique);
    data.set_n_brt_nodes(n_unique - 1);
  }

  if (start_stage <= 6 && end_stage >= 6) {
    data.set_n_octree_nodes(data.u_edge_offset_s6[data.get_n_brt_nodes() - 1]);
  }
}

}  // namespace tree::vulkan
