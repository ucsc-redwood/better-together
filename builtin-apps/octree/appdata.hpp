#pragma once

#include <glm/vec4.hpp>
#include <memory_resource>
#include <random>
#include <vector>

namespace octree {

template <typename T>
using UsmVec = std::pmr::vector<T>;

using MortonT = uint32_t;

constexpr auto kDefaultInputSize = 1366 * 768;

struct AppData {
  explicit AppData(std::pmr::memory_resource* mr, const size_t n_input = kDefaultInputSize)
      : n_input(n_input),
        n_unique_codes(std::numeric_limits<size_t>::max()),
        n_brt_nodes(std::numeric_limits<size_t>::max()),
        n_octree_nodes(std::numeric_limits<size_t>::max()),
        u_positions(n_input, mr),
        u_morton_codes(n_input, mr),
        u_parents(n_input, mr),
        u_left_child(n_input, mr),
        u_has_leaf_left(n_input, mr),
        u_has_leaf_right(n_input, mr),
        u_prefix_length(n_input, mr),
        u_edge_count(n_input, mr),
        u_offsets(n_input, mr),
        u_children(8 * n_input, mr) {
    // generate random positions
    static std::mt19937 gen(114514);
    static std::uniform_real_distribution dis(0.0, 9999999999.0);
    std::ranges::generate(u_positions,
                          [&]() { return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f); });
  }

  const size_t n_input;
  size_t n_unique_codes;
  size_t n_brt_nodes;
  size_t n_octree_nodes;

  UsmVec<glm::vec4> u_positions;
  UsmVec<MortonT> u_morton_codes;

  // Radix Tree
  UsmVec<int> u_parents;
  UsmVec<int> u_left_child;
  UsmVec<uint8_t> u_has_leaf_left;
  UsmVec<uint8_t> u_has_leaf_right;
  UsmVec<int> u_prefix_length;

  UsmVec<int> u_edge_count;
  UsmVec<int> u_offsets;

  UsmVec<int> u_children;  // length = 8 * n_input
};

}  // namespace octree
