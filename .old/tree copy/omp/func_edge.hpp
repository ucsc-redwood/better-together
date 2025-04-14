#pragma once

#include <cstdint>

namespace tree {

namespace omp {

inline void process_edge_count_i(const int i,
                                 const uint8_t *prefix_n,
                                 const int *parents,
                                 int *edge_count) {
  const auto my_depth = prefix_n[i] / 3;
  const auto parent_depth = prefix_n[parents[i]] / 3;
  edge_count[i] = my_depth - parent_depth;
}

}  // namespace omp

}  // namespace tree