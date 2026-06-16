#include "safe_tree_appdata.hpp"

#include <omp.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory_resource>
#include <numeric>

#include "omp/func_brt.hpp"
#include "omp/func_edge.hpp"
#include "omp/func_morton.hpp"
#include "omp/func_octree.hpp"

namespace tree {

void HostTreeManager::initialize() {
  constexpr bool kPrint = false;

  auto mr = std::pmr::new_delete_resource();
  appdata_ = std::make_unique<tree::AppData>(mr, kDefaultInputSize);

  auto& appdata = *appdata_;

  if constexpr (kPrint) {
    // print first 10 points
    for (size_t i = 0; i < 10; ++i) {
      spdlog::info("point {} = ({}, {}, {}, {})",
                   i,
                   appdata.u_input_points_s0[i].x,
                   appdata.u_input_points_s0[i].y,
                   appdata.u_input_points_s0[i].z,
                   appdata.u_input_points_s0[i].w);
    }
  }

  // stage 1
  {
    const int start = 0;
    const int end = appdata.get_n_input();

#pragma omp parallel for
    for (int i = start; i < end; ++i) {
      appdata.u_morton_keys_s1[i] =
          tree::omp::xyz_to_morton32(appdata.u_input_points_s0[i], tree::kMinCoord, tree::kRange);
    }
  }

  // print first 10 points
  if constexpr (kPrint) {
    for (size_t i = 0; i < 10; ++i) {
      spdlog::info("morton key {} = {}", i, appdata.u_morton_keys_s1[i]);
    }
  }

  // stage 2
  {
    std::ranges::sort(appdata.u_morton_keys_s1);
    std::ranges::copy(appdata.u_morton_keys_s1, appdata.u_morton_keys_sorted_s2.begin());
  }

  // print first 10 points
  if constexpr (kPrint) {
    for (size_t i = 0; i < 10; ++i) {
      spdlog::info("sorted morton key {} = {}", i, appdata.u_morton_keys_sorted_s2[i]);
    }
  }

  // stage 3
  {
    const auto last =
        std::unique_copy(appdata.u_morton_keys_sorted_s2.data(),
                         appdata.u_morton_keys_sorted_s2.data() + appdata.get_n_input(),
                         appdata.u_morton_keys_unique_s3.data());
    const auto n_unique = std::distance(appdata.u_morton_keys_unique_s3.data(), last);

    appdata.set_n_unique(n_unique);
    appdata.set_n_brt_nodes(n_unique - 1);
  }

  if constexpr (kPrint) {
    spdlog::info("n_unique = {}", appdata.get_n_unique());
    spdlog::info("n_brt_nodes = {}", appdata.get_n_brt_nodes());
  }

  // stage 4
  {
    const int start = 0;
    // brt has exactly n_brt_nodes (= n_unique - 1) internal nodes. Iterating to
    // n_unique ran one extra node i == n_brt_nodes that reads codes[i+1] (OOB:
    // codes only has n_unique valid entries) and writes a non-existent brt slot.
    // CUDA's k_BuildRadixTree loops i < n_brt_nodes. Match it.
    const int end = appdata.get_n_brt_nodes();
#pragma omp parallel for
    for (int i = start; i < end; ++i) {
      tree::omp::v1::process_radix_tree_i(i,
                                          appdata.get_n_brt_nodes(),
                                          appdata.u_morton_keys_unique_s3.data(),
                                          appdata.u_brt_prefix_n_s4.data(),
                                          appdata.u_brt_has_leaf_left_s4.data(),
                                          appdata.u_brt_has_leaf_right_s4.data(),
                                          appdata.u_brt_left_child_s4.data(),
                                          appdata.u_brt_parents_s4.data());
    }
  }

  if constexpr (kPrint) {
    for (size_t i = 0; i < 10; ++i) {
      spdlog::info("brt prefix n {} = {}", i, appdata.u_brt_prefix_n_s4[i]);
    }
  }

  // stage 5 (brt -> edge count)
  {
    const int start = 0;
    const int end = appdata.get_n_brt_nodes();

#pragma omp parallel for
    for (int i = start; i < end; ++i) {
      tree::omp::v1::process_edge_count_i(i,
                                          appdata.u_brt_prefix_n_s4.data(),
                                          appdata.u_brt_parents_s4.data(),
                                          appdata.u_edge_count_s5.data());
    }
  }

  if constexpr (kPrint) {
    for (size_t i = 0; i < 10; ++i) {
      spdlog::info("edge count {} = {}", i, appdata.u_edge_count_s5[i]);
    }
  }

  // Stage 6 (edge count -> edge offset)
  {
    const int start = 0;
    const int end = appdata.get_n_brt_nodes();

    std::partial_sum(appdata.u_edge_count_s5.data() + start,
                     appdata.u_edge_count_s5.data() + end,
                     appdata.u_edge_offset_s6.data() + start);

    // num_octree node is the result of the partial sum
    const auto num_octree_nodes = appdata.u_edge_offset_s6[end - 1];

    appdata.set_n_octree_nodes(num_octree_nodes);
  }

  if constexpr (kPrint) {
    for (size_t i = 0; i < 10; ++i) {
      spdlog::info("edge offset {} = {}", i, appdata.u_edge_offset_s6[i]);
    }
    spdlog::info("n_octree_nodes = {}", appdata.get_n_octree_nodes());
  }

  // stage 7 (everything -> octree)
  {
    // brt node 0 now contributes the full-domain ROOT octree node (edge_count[0]
    // == 1), so it must be processed too -- start at 0, not 1.
    const int start = 0;
    // i is a BRT node index, so the bound is n_brt_nodes (CUDA k_MakeOctNodes
    // loops i < n_brt_nodes). Using n_octree_nodes here (an octree-node count)
    // left more than half of the octree nodes unwritten. NOTE: a separate,
    // pre-existing octree defect remains -- some octree nodes are written by >1
    // brt node with conflicting geometry (the edge_offset prefix-sum convention);
    // tracked in docs/BUGS-FOUND.md, fixed across all backends separately.
    const int end = appdata.get_n_brt_nodes();

#pragma omp parallel for
    for (int i = start; i < end; ++i) {
      tree::omp::process_oct_node(i,
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

      // The golden previously omitted process_link_leaf, leaving the golden's
      // children / child_leaf_mask incomplete (and the oracle's claim that
      // link-leaf was "completed" untrue). Run it so the golden matches the
      // full pipeline the dispatcher (run_stage_7) executes.
      tree::omp::process_link_leaf(i,
                                   reinterpret_cast<int(*)[8]>(appdata.u_oct_children_s7.data()),
                                   appdata.u_oct_child_leaf_mask_s7.data(),
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

  if constexpr (kPrint) {
    for (size_t i = 0; i < 10; ++i) {
      spdlog::info("octree node {} = {:08b}", i, appdata.u_oct_child_node_mask_s7[i]);
    }
  }
}

SafeAppData::SafeAppData(std::pmr::memory_resource* mr)
    :  // Get data from singleton
      n_input(HostTreeManager::getInstance().getAppData()->get_n_input()),
      n_unique(HostTreeManager::getInstance().getAppData()->get_n_unique()),
      n_brt_nodes(HostTreeManager::getInstance().getAppData()->get_n_brt_nodes()),
      n_octree_nodes(HostTreeManager::getInstance().getAppData()->get_n_octree_nodes()),
      // Copy vectors from singleton
      u_input_points_s0(HostTreeManager::getInstance().getAppData()->u_input_points_s0, mr),
      u_morton_keys_s1(HostTreeManager::getInstance().getAppData()->u_morton_keys_s1, mr),
      u_morton_keys_s1_out(n_input, mr),  // Same size as input
      u_morton_keys_sorted_s2(HostTreeManager::getInstance().getAppData()->u_morton_keys_sorted_s2,
                              mr),
      u_morton_keys_sorted_s2_out(n_input, mr),  // Same size as input
      u_morton_keys_unique_s3(HostTreeManager::getInstance().getAppData()->u_morton_keys_unique_s3,
                              mr),
      u_morton_keys_unique_s3_out(n_input, mr),  // Same size as input
      u_num_selected_out(1, mr),                 // Used by CUDA for unique count
      u_brt_prefix_n_s4(HostTreeManager::getInstance().getAppData()->u_brt_prefix_n_s4, mr),
      u_brt_has_leaf_left_s4(HostTreeManager::getInstance().getAppData()->u_brt_has_leaf_left_s4,
                             mr),
      u_brt_has_leaf_right_s4(HostTreeManager::getInstance().getAppData()->u_brt_has_leaf_right_s4,
                              mr),
      u_brt_left_child_s4(HostTreeManager::getInstance().getAppData()->u_brt_left_child_s4, mr),
      u_brt_parents_s4(HostTreeManager::getInstance().getAppData()->u_brt_parents_s4, mr),
      u_brt_prefix_n_s4_out(n_input, mr),        // Same size as input
      u_brt_has_leaf_left_s4_out(n_input, mr),   // Same size as input
      u_brt_has_leaf_right_s4_out(n_input, mr),  // Same size as input
      u_brt_left_child_s4_out(n_input, mr),      // Same size as input
      u_brt_parents_s4_out(n_input, mr),         // Same size as input
      u_edge_count_s5(HostTreeManager::getInstance().getAppData()->u_edge_count_s5, mr),
      u_edge_count_s5_out(n_input, mr),  // Same size as input
      u_edge_offset_s6(HostTreeManager::getInstance().getAppData()->u_edge_offset_s6, mr),
      u_edge_offset_s6_out(n_input, mr),  // Same size as input
      u_oct_children_s7(HostTreeManager::getInstance().getAppData()->u_oct_children_s7, mr),
      u_oct_corner_s7(HostTreeManager::getInstance().getAppData()->u_oct_corner_s7, mr),
      u_oct_cell_size_s7(HostTreeManager::getInstance().getAppData()->u_oct_cell_size_s7, mr),
      u_oct_child_node_mask_s7(
          HostTreeManager::getInstance().getAppData()->u_oct_child_node_mask_s7, mr),
      u_oct_child_leaf_mask_s7(
          HostTreeManager::getInstance().getAppData()->u_oct_child_leaf_mask_s7, mr),
      // Initialize output vectors with same sizes as their input counterparts
      u_oct_children_s7_out(n_input * 8 * kMemoryRatio, mr),  // 8x for children
      u_oct_corner_s7_out(n_input * kMemoryRatio, mr),
      u_oct_cell_size_s7_out(n_input * kMemoryRatio, mr),
      u_oct_child_node_mask_s7_out(n_input * kMemoryRatio, mr),
      u_oct_child_leaf_mask_s7_out(n_input * kMemoryRatio, mr) {
  if (!HostTreeManager::getInstance().getAppData()) {
    throw std::runtime_error(
        "Tree data not initialized. Call HostTreeManager::getInstance().initialize() first.");
  }

  size_t total_memory = 0;

  // Calculate memory for each vector
  total_memory += u_input_points_s0.size() * sizeof(glm::vec4);
  total_memory += u_morton_keys_s1.size() * sizeof(uint32_t);
  total_memory += u_morton_keys_s1_out.size() * sizeof(uint32_t);
  total_memory += u_morton_keys_sorted_s2.size() * sizeof(uint32_t);
  total_memory += u_morton_keys_sorted_s2_out.size() * sizeof(uint32_t);
  total_memory += u_morton_keys_unique_s3.size() * sizeof(uint32_t);
  total_memory += u_morton_keys_unique_s3_out.size() * sizeof(uint32_t);
  total_memory += u_num_selected_out.size() * sizeof(uint32_t);
  total_memory += u_brt_prefix_n_s4.size() * sizeof(uint8_t);
  total_memory += u_brt_has_leaf_left_s4.size() * sizeof(uint8_t);
  total_memory += u_brt_has_leaf_right_s4.size() * sizeof(uint8_t);
  total_memory += u_brt_left_child_s4.size() * sizeof(int32_t);
  total_memory += u_brt_parents_s4.size() * sizeof(int32_t);
  total_memory += u_brt_prefix_n_s4_out.size() * sizeof(uint8_t);
  total_memory += u_brt_has_leaf_left_s4_out.size() * sizeof(uint8_t);
  total_memory += u_brt_has_leaf_right_s4_out.size() * sizeof(uint8_t);
  total_memory += u_brt_left_child_s4_out.size() * sizeof(int32_t);
  total_memory += u_brt_parents_s4_out.size() * sizeof(int32_t);
  total_memory += u_edge_count_s5.size() * sizeof(int32_t);
  total_memory += u_edge_count_s5_out.size() * sizeof(int32_t);
  total_memory += u_edge_offset_s6.size() * sizeof(int32_t);
  total_memory += u_edge_offset_s6_out.size() * sizeof(int32_t);
  total_memory += u_oct_children_s7.size() * sizeof(uint32_t);
  total_memory += u_oct_corner_s7.size() * sizeof(glm::vec4);
  total_memory += u_oct_cell_size_s7.size() * sizeof(float);
  total_memory += u_oct_child_node_mask_s7.size() * sizeof(uint8_t);
  total_memory += u_oct_child_leaf_mask_s7.size() * sizeof(uint8_t);
  total_memory += u_oct_children_s7_out.size() * sizeof(uint32_t);
  total_memory += u_oct_corner_s7_out.size() * sizeof(glm::vec4);
  total_memory += u_oct_cell_size_s7_out.size() * sizeof(float);
  total_memory += u_oct_child_node_mask_s7_out.size() * sizeof(uint8_t);
  total_memory += u_oct_child_leaf_mask_s7_out.size() * sizeof(uint8_t);

  spdlog::info(
      "Total memory used: {} bytes ({} MB)", total_memory, total_memory / (1024.0f * 1024.0f));
}

}  // namespace tree
