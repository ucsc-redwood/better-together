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
  const auto my_depth = prefix_n[i] / 3;
  const auto parent_depth = prefix_n[parents[i]] / 3;
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