#include "tree_appdata.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <random>

namespace tree {

// clang-format off
// Buffer Allocation Summary:
// -----------------------------------------------------------------------------------------------
// | Stage | Buffer Name                  | Allocated Size               | Real Data Used          |
// |-------|------------------------------|------------------------------|-------------------------|
// | 1     | u_input_points_s0            | n_input                      | n_input                 |
// | 1     | u_morton_keys_s1             | n_input                      | n_input                 |
// | 2     | u_morton_keys_sorted_s2      | n_input                      | n_input                 |
// | 3     | u_morton_keys_unique_s3      | n_input                      | n_unique                |
// | 4     | u_brt_prefix_n_s4            | n_input                      | n_brt_nodes             |
// | 4     | u_brt_has_leaf_left_s4       | n_input                      | n_brt_nodes             |
// | 4     | u_brt_has_leaf_right_s4      | n_input                      | n_brt_nodes             |
// | 4     | u_brt_left_child_s4          | n_input                      | n_brt_nodes             |
// | 4     | u_brt_parents_s4             | n_input                      | n_brt_nodes             |
// | 5     | u_edge_count_s5              | n_input                      | n_brt_nodes             |
// | 6     | u_edge_offset_s6             | n_input                      | n_brt_nodes             |
// | 7     | u_oct_corner_s7              | n_input * 0.6f               | n_octree_nodes          |
// | 7     | u_oct_cell_size_s7           | n_input * 0.6f               | n_octree_nodes          |
// | 7     | u_oct_child_node_mask_s7     | n_input * 0.6f               | n_octree_nodes          |
// | 7     | u_oct_child_leaf_mask_s7     | n_input * 0.6f               | n_octree_nodes          |
// | 7     | u_oct_children_s7            | 8 * n_input * 0.6f           | 8 * n_octree_nodes      |
// | 7     | u_num_selected_out           | 1                            | 1                       |
// ------------------------------------------------------------------------------------------------
// clang-format on
AppData::AppData(std::pmr::memory_resource* mr, const size_t n_input)
    : n_input(n_input),
      u_input_points_s0(n_input, mr),
      u_morton_keys_s1(n_input, mr),
      u_morton_keys_sorted_s2(n_input, mr),
      u_morton_keys_unique_s3(n_input, mr),
      u_brt_prefix_n_s4(n_input, mr),
      u_brt_has_leaf_left_s4(n_input, mr),
      u_brt_has_leaf_right_s4(n_input, mr),
      u_brt_left_child_s4(n_input, mr),
      u_brt_parents_s4(n_input, mr),
      u_edge_count_s5(n_input, mr),
      u_edge_offset_s6(n_input, mr),
      u_oct_children_s7(n_input * 8 * kMemoryRatio, mr),
      u_oct_corner_s7(n_input * kMemoryRatio, mr),
      u_oct_cell_size_s7(n_input * kMemoryRatio, mr),
      u_oct_child_node_mask_s7(n_input * kMemoryRatio, mr),
      u_oct_child_leaf_mask_s7(n_input * kMemoryRatio, mr),
      u_num_selected_out(1, mr) {
  static std::mt19937 gen(114514);
  static std::uniform_real_distribution dis(kMinCoord, kMinCoord + kRange);

  // generate random points
  std::ranges::generate(u_input_points_s0,
                        [&]() { return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f); });

  // Calculate total memory allocation in bytes
  const size_t total_bytes = u_input_points_s0.size() * sizeof(glm::vec4) +       // Stage 1
                             u_morton_keys_s1.size() * sizeof(uint32_t) +         // Stage 1
                             u_morton_keys_sorted_s2.size() * sizeof(uint32_t) +  // Stage 2
                             u_morton_keys_unique_s3.size() * sizeof(uint32_t) +  // Stage 3
                             u_brt_prefix_n_s4.size() * sizeof(uint8_t) +         // Stage 4
                             u_brt_has_leaf_left_s4.size() * sizeof(uint8_t) +    // Stage 4
                             u_brt_has_leaf_right_s4.size() * sizeof(uint8_t) +   // Stage 4
                             u_brt_left_child_s4.size() * sizeof(int32_t) +       // Stage 4
                             u_brt_parents_s4.size() * sizeof(int32_t) +          // Stage 4
                             u_edge_count_s5.size() * sizeof(int32_t) +           // Stage 5
                             u_edge_offset_s6.size() * sizeof(int32_t) +          // Stage 6
                             u_oct_children_s7.size() * sizeof(int32_t) +         // Stage 7
                             u_oct_corner_s7.size() * sizeof(glm::vec4) +         // Stage 7
                             u_oct_cell_size_s7.size() * sizeof(float) +          // Stage 7
                             u_oct_child_node_mask_s7.size() * sizeof(int32_t) +  // Stage 7
                             u_oct_child_leaf_mask_s7.size() * sizeof(int32_t) +  // Stage 7
                             u_num_selected_out.size() * sizeof(int32_t);         // Stage 7

  const float total_mb = total_bytes / (1024.0f * 1024.0f);

  spdlog::trace("Tree construction appdata allocated:");
  spdlog::trace("\tInput size: {} points", n_input);
  spdlog::trace("\tTotal memory: {:.2f} MB", total_mb);
}

}  // namespace tree
