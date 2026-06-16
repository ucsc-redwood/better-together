#pragma once

#include <cassert>
#include <cstdint>

namespace tree {

namespace omp {

// ----------------------------------------------------------------------------
// Old version
// ----------------------------------------------------------------------------

namespace v1 {

inline void process_edge_count_i(const int i,
                                 const uint8_t* prefix_n,
                                 const int* parents,
                                 int* edge_count) {
  // BRT node 0 is the radix-tree root (parents[0] is a self-loop, never set), so
  // the level-difference formula yields 0 -> it would contribute NO octree node
  // and there would be no full-domain ROOT octree node. The cross-brt parent walk
  // then degenerates to octree node 0 (the deepest cell), linking top-level nodes
  // backwards. Give the root brt node exactly ONE octree node (the full-domain
  // root at level prefix_n[0]/3, cell = range); process_oct_node writes its
  // geometry, and every top-level chain links up to it (edge_count[0] != 0 now
  // terminates the parent walk at the root instead of falling through to node 0).
  if (i == 0) {
    edge_count[0] = 1;
    return;
  }
  // Octree level of a radix node = number of WHOLE octants (3-bit groups) shared
  // by its key range. prefix_n is clz(xor)-1 over a 30-bit morton held in a
  // 32-bit word, so it counts the 2 always-zero high bits minus 1 -> it is the
  // shared-bit count + 1. The geometric octant depth is therefore
  // (prefix_n - 1)/3, NOT prefix_n/3: the latter rounds UP by one octant exactly
  // at an octant boundary (when (prefix_n-1) % 3 == 2), which made boundary radix
  // nodes claim a phantom octree level and mis-rooted their children one cell to
  // the side. Use the corrected depth here and in process_oct_node/link_leaf.
  const auto my_depth = (static_cast<int>(prefix_n[i]) - 1) / 3;
  const auto parent_depth = (static_cast<int>(prefix_n[parents[i]]) - 1) / 3;
  edge_count[i] = my_depth - parent_depth;
}

}  // namespace v1

//-----------------------------------------------------------------------------
// Step 5: count, for each radix‐tree node i, how many distinct octants appear
// in its [first..last] code‐range.
//
// codes           : sorted, unique Morton codes (30 bits: xyz interleaved)
// n               : number of codes / nodes
// left_child      : from build_radix_tree()
// prefix_length   : from build_radix_tree()
// edge_count_out  : length‑n raw array to fill
//
static inline void compute_edge_count_kernel(const uint32_t* codes,
                                             const int n,
                                             const int* left_child,
                                             const int* prefix_length,
                                             int* edge_count_out) {
  constexpr int MORTON_BITS = 30;
#pragma omp for
  for (int i = 0; i < n; ++i) {
    int j = left_child[i];
    int first = (i < j ? i : j);
    int last = (i < j ? j : i);

    // which bit‐triplet depth do we inspect?
    int depth = prefix_length[i];
    int shift = MORTON_BITS - depth - 3;  // next 3 bits
    assert(shift >= 0);

    // mark which of the 8 octants appear
    bool seen[8] = {false};
    for (int k = first; k <= last; ++k) {
      int oct = (codes[k] >> shift) & 0x7;
      seen[oct] = true;
    }
    int cnt = 0;
    for (int o = 0; o < 8; ++o)
      if (seen[o]) ++cnt;
    edge_count_out[i] = cnt;
  }
}

}  // namespace omp

}  // namespace tree