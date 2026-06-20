#include "dispatchers.hpp"

#include <cstdint>

#include "platform/registry/device_registry.hpp"
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
// histogram and scatter passes); see multi_radixsort_*.comp.
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
VulkanDispatcher::VulkanDispatcher() : engine(), seq(engine.make_seq()) {
  spdlog::info("VulkanDispatcher instance created.");

  auto morton_algo = engine.make_algo("tree_morton")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(2)
                         ->push_constant<MortonPushConstants>()
                         ->build();

  cached_algorithms.try_emplace("morton", std::move(morton_algo));

  // Multi-workgroup (device-wide) LSD radix sort: histogram pass (in,
  // histograms) + scatter pass (in, out, histograms), 4x8-bit, ping-pong.
  // 4 descriptor sets each: the 4 radix passes are recorded into one command
  // buffer (one submit), so each pass needs its own descriptor set (a single
  // shared set would be overwritten and all passes would use the last binding).
  auto radix_histograms_algo = engine.make_algo("multi_radixsort_histograms")
                                   ->work_group_size(256, 1, 1)
                                   ->num_sets(4)
                                   ->num_buffers(2)
                                   ->push_constant<RadixSortPushConstants>()
                                   ->build();

  cached_algorithms.try_emplace("radix_histograms", std::move(radix_histograms_algo));

  auto radix_scatter_algo =
      engine.make_algo("multi_radixsort_warp" + std::to_string(get_vulkan_warp_size()))
          ->work_group_size(256, 1, 1)
          ->num_sets(4)
          ->num_buffers(3)
          ->push_constant<RadixSortPushConstants>()
          ->build();

  cached_algorithms.try_emplace("radix_scatter", std::move(radix_scatter_algo));

  auto find_dups_algo = engine.make_algo("tree_find_dups")
                            ->work_group_size(256, 1, 1)
                            ->num_sets(1)
                            ->num_buffers(2)
                            ->push_constant<FindDupsPushConstants>()
                            ->build();

  cached_algorithms.try_emplace("find_dups", std::move(find_dups_algo));

  // Device-wide inclusive scan (3 passes), used by stage 3 (flags) and stage 6
  // (edge counts). See tree_scan_*.comp. Each pass is its own algo; all three
  // are recorded into one command buffer with shaderWrite->shaderRead barriers
  // between them.
  auto scan_local_algo = engine.make_algo("tree_scan_local")
                             ->work_group_size(256, 1, 1)
                             ->num_sets(1)
                             ->num_buffers(3)
                             ->push_constant<PrefixSumPushConstants>()
                             ->build();

  cached_algorithms.try_emplace("scan_local", std::move(scan_local_algo));

  auto scan_block_sums_algo = engine.make_algo("tree_scan_block_sums")
                                  ->work_group_size(256, 1, 1)
                                  ->num_sets(1)
                                  ->num_buffers(1)
                                  ->push_constant<GlobalPushConstants>()
                                  ->build();

  cached_algorithms.try_emplace("scan_block_sums", std::move(scan_block_sums_algo));

  auto scan_add_algo = engine.make_algo("tree_scan_add")
                           ->work_group_size(256, 1, 1)
                           ->num_sets(1)
                           ->num_buffers(2)
                           ->push_constant<PrefixSumPushConstants>()
                           ->build();

  cached_algorithms.try_emplace("scan_add", std::move(scan_add_algo));

  auto move_dups_algo = engine.make_algo("tree_move_dups")
                            ->work_group_size(256, 1, 1)
                            ->num_sets(1)
                            ->num_buffers(4)
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

void VulkanDispatcher::run_stage_1(VkAppData_Safe& appdata) {
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

void VulkanDispatcher::run_stage_2(VkAppData_Safe& appdata) {
  LOG_KERNEL(LogKernelType::kVK, 2, &appdata);

  // Device-wide multi-workgroup LSD radix sort (Mirco Werner / Embree). Each of
  // the 4 8-bit passes runs two dispatches: a histogram pass (per-workgroup
  // 256-bin counts) and a scatter pass (global offsets + reorder). Buffers
  // ping-pong so the result lands in u_morton_keys_sorted_s2_out after 4 passes:
  //   src/dst:  s1->tmp -> out -> tmp -> out  (final = out).
  // kiss-vk exposes no inter-dispatch pipeline barrier, so each dispatch is its
  // own submit+fence-wait; the fence wait is a full memory barrier between
  // passes (histogram writes -> scatter reads, and pass i -> pass i+1).

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

  // All 4 passes (8 dispatches) are recorded into ONE command buffer / ONE
  // submit. Between every dispatch we insert a shaderWrite->shaderRead memory
  // barrier: histogram->scatter shares the histograms buffer, and pass i's
  // scatter output is pass i+1's histogram/scatter input (ping-pong). A
  // fence-wait between separate submits guarantees execution order but NOT
  // memory availability/visibility across submits -- AMD/Xclipse are coherent
  // and tolerate it, but Mali-G710 is not (the next stage saw a partially
  // written buffer, non-deterministic ~65/256 workgroups, -> garbage). Doing it
  // in one submit with explicit barriers makes the dependencies correct on all
  // three backends. Each pass uses its own descriptor set (num_sets(4)).
  const vk::MemoryBarrier mem_barrier{
      .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
      .dstAccessMask = vk::AccessFlagBits::eShaderRead,
  };
  auto record_barrier = [&] {
    seq->get_handle().pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                      vk::PipelineStageFlagBits::eComputeShader,
                                      vk::DependencyFlags{},
                                      mem_barrier,
                                      nullptr,
                                      nullptr);
  };

  // Bind buffers up-front: each pass -> its own descriptor set index.
  for (uint32_t pass = 0; pass < 4; ++pass) {
    hist_algo->update_descriptor_set(pass,
                                     {
                                         engine.get_buffer_info(*src_seq[pass]),
                                         engine.get_buffer_info(appdata.u_sort_histograms),
                                     });
    scatter_algo->update_descriptor_set(pass,
                                        {
                                            engine.get_buffer_info(*src_seq[pass]),
                                            engine.get_buffer_info(*dst_seq[pass]),
                                            engine.get_buffer_info(appdata.u_sort_histograms),
                                        });
  }

  seq->cmd_begin();
  for (uint32_t pass = 0; pass < 4; ++pass) {
    const RadixSortPushConstants pc{
        .g_num_elements = n,
        .g_shift = 8u * pass,
        .g_num_workgroups = num_workgroups,
        .g_num_blocks_per_workgroup = blocks_per_workgroup,
    };

    // histogram pass (src -> histograms)
    hist_algo->update_push_constant(pc);
    hist_algo->record_bind_core(seq->get_handle(), pass);
    hist_algo->record_bind_push(seq->get_handle());
    hist_algo->record_dispatch(seq->get_handle(), {num_workgroups, 1, 1});
    record_barrier();

    // scatter pass (src + histograms -> dst)
    scatter_algo->update_push_constant(pc);
    scatter_algo->record_bind_core(seq->get_handle(), pass);
    scatter_algo->record_bind_push(seq->get_handle());
    scatter_algo->record_dispatch(seq->get_handle(), {num_workgroups, 1, 1});
    if (pass != 3) record_barrier();
  }
  seq->cmd_end();
  seq->reset_fence();
  seq->submit();
  seq->wait_for_fence();
}

// ----------------------------------------------------------------------------
// Stage 3 (Sorted Morton -> Unique Sorted Morton)
// ----------------------------------------------------------------------------

// Record a device-wide INCLUSIVE scan of `src` (n elements) into `dst`, using
// u_scan_block_sums as scratch, into the already-open command buffer `cmd`.
// Three passes (local scan -> scan block totals -> add block offsets) separated
// by shaderWrite->shaderRead barriers (kiss-vk has no inter-dispatch pipeline
// barrier API, so everything stays in one submit). Caller does cmd_begin/end +
// submit + fence-wait. `descriptor_set` selects which set index each scan algo
// uses (so two scans in one cmd buffer don't clobber each other's bindings).
void VulkanDispatcher::record_device_scan(vk::CommandBuffer cmd,
                                          vk::DescriptorBufferInfo src,
                                          vk::DescriptorBufferInfo dst,
                                          vk::DescriptorBufferInfo block_sums,
                                          uint32_t n,
                                          uint32_t descriptor_set) {
  auto* local_algo = cached_algorithms.at("scan_local").get();
  auto* block_algo = cached_algorithms.at("scan_block_sums").get();
  auto* add_algo = cached_algorithms.at("scan_add").get();

  const uint32_t elems_per_wg = VkAppData_Safe::kScanElementsPerWg;
  const uint32_t num_blocks = static_cast<uint32_t>(kiss_vk::div_ceil(n, elems_per_wg));

  const vk::MemoryBarrier mem_barrier{
      .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
      .dstAccessMask = vk::AccessFlagBits::eShaderRead,
  };
  auto barrier = [&] {
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::DependencyFlags{},
                        mem_barrier,
                        nullptr,
                        nullptr);
  };

  // Pass 1: per-block inclusive scan + per-block totals.
  local_algo->update_descriptor_set(descriptor_set, {src, dst, block_sums});
  local_algo->update_push_constant(PrefixSumPushConstants{.inputSize = n});
  local_algo->record_bind_core(cmd, descriptor_set);
  local_algo->record_bind_push(cmd);
  local_algo->record_dispatch(cmd, {num_blocks, 1, 1});
  barrier();

  // Pass 2: exclusive scan of the per-block totals (single workgroup, in place).
  block_algo->update_descriptor_set(descriptor_set, {block_sums});
  block_algo->update_push_constant(GlobalPushConstants{.n_blocks = num_blocks});
  block_algo->record_bind_core(cmd, descriptor_set);
  block_algo->record_bind_push(cmd);
  block_algo->record_dispatch(cmd, {1, 1, 1});
  barrier();

  // Pass 3: add the exclusive block offset to every element of each block.
  add_algo->update_descriptor_set(descriptor_set, {dst, block_sums});
  add_algo->update_push_constant(PrefixSumPushConstants{.inputSize = n});
  add_algo->record_bind_core(cmd, descriptor_set);
  add_algo->record_bind_push(cmd);
  add_algo->record_dispatch(cmd, {num_blocks, 1, 1});
}

void VulkanDispatcher::run_stage_3(VkAppData_Safe& appdata) {
  LOG_KERNEL(LogKernelType::kVK, 3, &appdata);

  // Stream compaction on the GPU: find_dups (run-head flags) -> device-wide
  // inclusive scan of the flags (compacted index) -> move_dups (scatter the
  // kept keys). All recorded into one command buffer / one submit with
  // shaderWrite->shaderRead barriers between dependent dispatches.
  const uint32_t n = appdata.get_n_input();

  auto* find_algo = cached_algorithms.at("find_dups").get();
  auto* move_algo = cached_algorithms.at("move_dups").get();

  // Flags -> u_contributes; inclusive scan -> u_out_idx.
  find_algo->update_descriptor_set(0,
                                   {
                                       engine.get_buffer_info(appdata.u_morton_keys_sorted_s2),
                                       engine.get_buffer_info(appdata.u_contributes),
                                   });
  find_algo->update_push_constant(FindDupsPushConstants{.n = static_cast<int32_t>(n)});

  move_algo->update_descriptor_set(0,
                                   {
                                       engine.get_buffer_info(appdata.u_contributes),
                                       engine.get_buffer_info(appdata.u_out_idx),
                                       engine.get_buffer_info(appdata.u_morton_keys_sorted_s2),
                                       engine.get_buffer_info(appdata.u_morton_keys_unique_s3_out),
                                   });
  move_algo->update_push_constant(MoveDupsPushConstants{.n = n});

  const vk::MemoryBarrier mem_barrier{
      .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
      .dstAccessMask = vk::AccessFlagBits::eShaderRead,
  };
  auto barrier = [&] {
    seq->get_handle().pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                      vk::PipelineStageFlagBits::eComputeShader,
                                      vk::DependencyFlags{},
                                      mem_barrier,
                                      nullptr,
                                      nullptr);
  };

  seq->cmd_begin();
  // find_dups: flags into u_contributes.
  find_algo->record_bind_core(seq->get_handle(), 0);
  find_algo->record_bind_push(seq->get_handle());
  find_algo->record_dispatch(seq->get_handle(),
                             {static_cast<uint32_t>(kiss_vk::div_ceil(n, 256)), 1, 1});
  barrier();

  // device-wide inclusive scan of the flags -> u_out_idx.
  record_device_scan(seq->get_handle(),
                     engine.get_buffer_info(appdata.u_contributes),
                     engine.get_buffer_info(appdata.u_out_idx),
                     engine.get_buffer_info(appdata.u_scan_block_sums),
                     n,
                     0);
  barrier();

  // move_dups: scatter kept keys to their compacted positions.
  move_algo->record_bind_core(seq->get_handle(), 0);
  move_algo->record_bind_push(seq->get_handle());
  move_algo->record_dispatch(seq->get_handle(),
                             {static_cast<uint32_t>(kiss_vk::div_ceil(n, 256)), 1, 1});
  seq->cmd_end();

  seq->reset_fence();
  seq->submit();
  seq->wait_for_fence();
}

// ----------------------------------------------------------------------------
// Stage 4 (Unique Sorted Morton -> BRT)
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_4(VkAppData_Safe& appdata) {
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

void VulkanDispatcher::run_stage_5(VkAppData_Safe& appdata) {
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

void VulkanDispatcher::run_stage_6(VkAppData_Safe& appdata) {
  LOG_KERNEL(LogKernelType::kVK, 6, &appdata);

  // Device-wide INCLUSIVE prefix sum of the edge counts (matches std::partial_sum
  // / cub::DeviceScan::InclusiveSum), entirely on the GPU. The buffers are int32
  // but the values are non-negative edge counts, so a uint scan is bit-identical.
  const uint32_t n = appdata.get_n_brt_nodes();

  seq->cmd_begin();
  record_device_scan(seq->get_handle(),
                     engine.get_buffer_info(appdata.u_edge_count_s5),
                     engine.get_buffer_info(appdata.u_edge_offset_s6_out),
                     engine.get_buffer_info(appdata.u_scan_block_sums),
                     n,
                     0);
  seq->cmd_end();

  seq->reset_fence();
  seq->submit();
  seq->wait_for_fence();
}

//----------------------------------------------------------------------------
// Stage 7 (Edge Offset -> Octree)
//----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_7(VkAppData_Safe& appdata) {
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
  // The shader loops over brt node indices [1, n_brt_nodes); size the grid for
  // n_brt_nodes (the grid-stride loop covers the rest).
  algo->record_dispatch(
      seq->get_handle(),
      {static_cast<uint32_t>(kiss_vk::div_ceil(appdata.get_n_brt_nodes(), 256)), 1, 1});
  seq->cmd_end();

  seq->reset_fence();
  seq->submit();
  seq->wait_for_fence();
}

}  // namespace tree::vulkan
