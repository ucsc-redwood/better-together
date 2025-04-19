#pragma once

#include <spdlog/spdlog.h>

#include <glm/vec4.hpp>
#include <iostream>
#include <memory_resource>
#include <random>
#include <vector>

namespace octree {

template <typename T>
using UsmVec = std::pmr::vector<T>;

using MortonT = uint32_t;

constexpr auto kDefaultInputSize = 1366 * 768;
// constexpr auto kDefaultInputSize = 640 * 480;
constexpr auto kMinCoord = 0.0;
constexpr auto kMaxCoord = 1024.0;
constexpr auto kFraction = 0.5f;

struct AppData {
  explicit AppData(std::pmr::memory_resource* mr, const size_t n_input = kDefaultInputSize)
      : n(n_input),
        reserved_n(n_input * kFraction),
        m(std::numeric_limits<size_t>::max()),
        total_children(std::numeric_limits<size_t>::max()),
        u_positions(n_input, mr),
        u_morton_codes_alt(n_input, mr),
        u_morton_codes(n_input, mr),
        u_parents(reserved_n, mr),
        u_left_child(reserved_n, mr),
        u_has_leaf_left(reserved_n, mr),
        u_has_leaf_right(reserved_n, mr),
        u_prefix_length(reserved_n, mr),
        u_edge_count(reserved_n, mr),
        u_offsets(reserved_n, mr),
        u_children(8 * reserved_n, mr) {
    // generate random positions
    static std::mt19937 gen(114514);
    static std::uniform_real_distribution dis(kMinCoord, kMaxCoord);
    std::ranges::generate(u_positions,
                          [&]() { return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f); });

    // compute the memory usage
    size_t total_memory = 0;
    total_memory += u_positions.size() * sizeof(glm::vec4);
    total_memory += u_morton_codes_alt.size() * sizeof(MortonT);
    total_memory += u_morton_codes.size() * sizeof(MortonT);
    total_memory += u_parents.size() * sizeof(int);
    total_memory += u_left_child.size() * sizeof(int);
    total_memory += u_has_leaf_left.size() * sizeof(uint8_t);
    total_memory += u_has_leaf_right.size() * sizeof(uint8_t);
    total_memory += u_prefix_length.size() * sizeof(int);
    total_memory += u_edge_count.size() * sizeof(int);
    total_memory += u_offsets.size() * sizeof(int);
    total_memory += u_children.size() * sizeof(int) * 8;

    spdlog::trace("Memory usage: {} MB", total_memory / 1024.0 / 1024.0);
  }

  // ----------------------------------------------------------------------------
  // Data
  // ----------------------------------------------------------------------------

  const size_t n;  // number of input points
  const size_t reserved_n;

  size_t m;  // num nuque and brt nodes
  size_t total_children;

  UsmVec<glm::vec4> u_positions;

  // I let xyz -> morton stored in "alt" first, then copy it to "morton_codes"
  // This is because I want to use the original morton codes for sorting
  UsmVec<MortonT> u_morton_codes_alt;
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

  // ----------------------------------------------------------------------------
  // Helper functions
  // ----------------------------------------------------------------------------

  // Print the flat radix tree in AppData.
  // Assumes `n_brt_nodes` is set to the active node count.
  void print_radix_tree(const AppData& app, const size_t n_display) {
    auto num_display = std::min(n_display, app.m);

    std::cout << "=== Radix Tree (" << num_display << " nodes) ===\n";
    for (size_t i = 0; i < num_display; ++i) {
      int parent = app.u_parents[i];
      int lc = app.u_left_child[i];
      int rc = lc >= 0 ? lc + 1 : -1;
      bool leafL = app.u_has_leaf_left[i];
      bool leafR = app.u_has_leaf_right[i];
      int pfx = app.u_prefix_length[i];

      std::cout << "Node[" << i << "] " << "(parent=" << parent << ") " << "prefix_len=" << pfx
                << "  " << "L=" << lc << (leafL ? "[leaf]" : "[int]") << "  " << "R=" << rc
                << (leafR ? "[leaf]" : "[int]") << "\n";
    }
    std::cout << "=====================================\n";
  }

  inline void print_octree_nodes(const AppData& app) {
    size_t totalChildren = app.u_offsets[app.m - 1] + app.u_edge_count[app.m - 1];

    std::cout << "=== Octree (" << app.m << " internal nodes, " << totalChildren
              << " total child refs) ===\n";

    for (size_t i = 0; i < app.m; ++i) {
      int offset = app.u_offsets[i];
      int cnt = app.u_edge_count[i];

      std::cout << "Node[" << i << "] " << "children_count=" << cnt << " -> { ";

      for (int j = 0; j < cnt; ++j) {
        int childIdx = app.u_children[offset + j];
        std::cout << childIdx;
        if (j + 1 < cnt) std::cout << ", ";
      }

      std::cout << " }\n";
    }

    std::cout << "=====================================\n";
  }
};

}  // namespace octree
