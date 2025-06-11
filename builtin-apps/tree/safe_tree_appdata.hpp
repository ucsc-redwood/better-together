#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <memory_resource>

#include "builtin-apps/base_appdata.hpp"
#include "tree_appdata.hpp"

namespace tree {

struct SafeAppData : public BaseAppData {
  explicit SafeAppData(std::pmr::memory_resource* mr);

  // --------------------------------------------------------------------------
  // Essential data
  // --------------------------------------------------------------------------
  const uint32_t n_input;
  const uint32_t n_unique;
  const uint32_t n_brt_nodes;
  const uint32_t n_octree_nodes;

  // --------------------------------------------------------------------------
  // Stage 1: xyz -> morton
  // --------------------------------------------------------------------------
  const std::pmr::vector<glm::vec4> u_input_points_s0;
  const std::pmr::vector<uint32_t> u_morton_keys_s1;

  std::pmr::vector<uint32_t> u_morton_keys_s1_out;

  // --------------------------------------------------------------------------
  // Stage 2: morton -> sorted morton
  // --------------------------------------------------------------------------
  const std::pmr::vector<uint32_t> u_morton_keys_sorted_s2;

  std::pmr::vector<uint32_t> u_morton_keys_sorted_s2_out;

  // --------------------------------------------------------------------------
  // Stage 3: sorted morton -> unique morton
  // --------------------------------------------------------------------------
  const std::pmr::vector<uint32_t> u_morton_keys_unique_s3;

  std::pmr::vector<uint32_t> u_morton_keys_unique_s3_out;

  // Used by CUDA implementation for unique count
  std::pmr::vector<uint32_t> u_num_selected_out;

  // --------------------------------------------------------------------------
  // Stage 4: unique morton -> Binary Radix Tree (BRT)
  // --------------------------------------------------------------------------
  const std::pmr::vector<uint8_t> u_brt_prefix_n_s4;
  const std::pmr::vector<uint8_t> u_brt_has_leaf_left_s4;
  const std::pmr::vector<uint8_t> u_brt_has_leaf_right_s4;
  const std::pmr::vector<int32_t> u_brt_left_child_s4;
  const std::pmr::vector<int32_t> u_brt_parents_s4;

  std::pmr::vector<uint8_t> u_brt_prefix_n_s4_out;
  std::pmr::vector<uint8_t> u_brt_has_leaf_left_s4_out;
  std::pmr::vector<uint8_t> u_brt_has_leaf_right_s4_out;
  std::pmr::vector<int32_t> u_brt_left_child_s4_out;
  std::pmr::vector<int32_t> u_brt_parents_s4_out;

  // --------------------------------------------------------------------------
  // Stage 5: BRT -> edge count
  // --------------------------------------------------------------------------
  const std::pmr::vector<int32_t> u_edge_count_s5;

  std::pmr::vector<int32_t> u_edge_count_s5_out;

  // --------------------------------------------------------------------------
  // Stage 6: edge count -> edge offset
  // --------------------------------------------------------------------------
  const std::pmr::vector<int32_t> u_edge_offset_s6;

  std::pmr::vector<int32_t> u_edge_offset_s6_out;

  // --------------------------------------------------------------------------
  // Stage 7: Build Octree
  // --------------------------------------------------------------------------
  const std::pmr::vector<int32_t> u_oct_children_s7;  // 8 * sizeof(int32_t) * n_octree_nodes
  const std::pmr::vector<glm::vec4> u_oct_corner_s7;
  const std::pmr::vector<float> u_oct_cell_size_s7;
  const std::pmr::vector<int32_t> u_oct_child_node_mask_s7;
  const std::pmr::vector<int32_t> u_oct_child_leaf_mask_s7;

  std::pmr::vector<int32_t> u_oct_children_s7_out;
  std::pmr::vector<glm::vec4> u_oct_corner_s7_out;
  std::pmr::vector<float> u_oct_cell_size_s7_out;
  std::pmr::vector<int32_t> u_oct_child_node_mask_s7_out;
  std::pmr::vector<int32_t> u_oct_child_leaf_mask_s7_out;

  // --------------------------------------------------------------------------
  // Getters / Setters
  // --------------------------------------------------------------------------

  [[nodiscard]] uint32_t get_n_input() const { return n_input; }
  [[nodiscard]] uint32_t get_n_unique() const { return n_unique; }
  [[nodiscard]] uint32_t get_n_brt_nodes() const { return n_brt_nodes; }
  [[nodiscard]] uint32_t get_n_octree_nodes() const { return n_octree_nodes; }

  void set_n_unique([[maybe_unused]] const uint32_t n_unique) {}
  void set_n_brt_nodes([[maybe_unused]] const uint32_t n_brt_nodes) {}
  void set_n_octree_nodes([[maybe_unused]] const uint32_t n_octree_nodes) {}
};

class HostTreeManager {
 public:
  static HostTreeManager& getInstance() {
    static HostTreeManager instance;
    return instance;
  }

  // Delete copy constructor and assignment operator
  HostTreeManager(const HostTreeManager&) = delete;
  HostTreeManager& operator=(const HostTreeManager&) = delete;

  // Get the AppData
  tree::AppData* getAppData() {
    if (!appdata_) {
      initialize();
    }
    return appdata_.get();
  }

 private:
  HostTreeManager() = default;
  void initialize();  // Made private since it's called automatically
  std::unique_ptr<tree::AppData> appdata_;
};

}  // namespace tree
