#include "safe_tree_appdata.hpp"

#include <omp.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <memory_resource>

#include "omp/dispatchers.hpp"

namespace tree {

void HostTreeManager::initialize() {
  constexpr bool kPrint = false;

  // Real-data mode (BT_TREE_DATA_DIR set): honor BT_TREE_INPUT_SIZE as the point
  // count to load, falling back to kRealDataDefaultInputSize if unset. Synthetic
  // mode (unset): unchanged, always kDefaultInputSize.
  size_t n_input = kDefaultInputSize;
  if (std::getenv("BT_TREE_DATA_DIR")) {
    n_input = kRealDataDefaultInputSize;
    if (const char* size_str = std::getenv("BT_TREE_INPUT_SIZE")) {
      n_input = std::strtoull(size_str, nullptr, 10);
    }
  }

  auto mr = std::pmr::new_delete_resource();
  appdata_ = std::make_unique<tree::AppData>(mr, n_input);

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

  // Stages 1-7: genuinely-chained AppData dispatch (single buffer per field,
  // no golden/_out split) -- see apps/tree/omp/dispatchers.{hpp,cpp} for the
  // extracted run_stage_N(AppData&) functions this used to inline directly.
  // This is the ONE place that builds the golden every SafeAppData
  // differential test compares against; a pure extraction, not new logic.
  tree::omp::run_stage_1(appdata);
  if constexpr (kPrint) {
    for (size_t i = 0; i < 10; ++i) {
      spdlog::info("morton key {} = {}", i, appdata.u_morton_keys_s1[i]);
    }
  }

  tree::omp::run_stage_2(appdata);
  if constexpr (kPrint) {
    for (size_t i = 0; i < 10; ++i) {
      spdlog::info("sorted morton key {} = {}", i, appdata.u_morton_keys_sorted_s2[i]);
    }
  }

  tree::omp::run_stage_3(appdata);
  if constexpr (kPrint) {
    spdlog::info("n_unique = {}", appdata.get_n_unique());
    spdlog::info("n_brt_nodes = {}", appdata.get_n_brt_nodes());
  }

  tree::omp::run_stage_4(appdata);
  if constexpr (kPrint) {
    for (size_t i = 0; i < 10; ++i) {
      spdlog::info("brt prefix n {} = {}", i, appdata.u_brt_prefix_n_s4[i]);
    }
  }

  tree::omp::run_stage_5(appdata);
  if constexpr (kPrint) {
    for (size_t i = 0; i < 10; ++i) {
      spdlog::info("edge count {} = {}", i, appdata.u_edge_count_s5[i]);
    }
  }

  tree::omp::run_stage_6(appdata);
  if constexpr (kPrint) {
    for (size_t i = 0; i < 10; ++i) {
      spdlog::info("edge offset {} = {}", i, appdata.u_edge_offset_s6[i]);
    }
    spdlog::info("n_octree_nodes = {}", appdata.get_n_octree_nodes());
  }

  tree::omp::run_stage_7(appdata);
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
