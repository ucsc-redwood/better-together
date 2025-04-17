#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace tree {

namespace omp {

using FakeBool = uint8_t;

#if defined(__GNUC__) || defined(__clang__)
#define CLZ(x) __builtin_clz(x)
#elif defined(_MSC_VER)
#include <intrin.h>
#define CLZ(x) _lzcnt_u32(x)
#else
#error "CLZ not supported on this platform"
#endif

namespace v1 {

inline unsigned int ceil_div_u32(const unsigned int a, const unsigned int b) {
  assert(b != 0);
  return (a + b - 1) / b;
}

inline uint8_t delta_u32(const unsigned int a, const unsigned int b) {
  [[maybe_unused]] constexpr unsigned int bit1_mask = static_cast<unsigned int>(1)
                                                      << (sizeof(a) * 8 - 1);
  assert((a & bit1_mask) == 0);
  assert((b & bit1_mask) == 0);
  return static_cast<uint8_t>(CLZ(a ^ b) - 1);
}

inline int log2_ceil_u32(const unsigned int x) {
  // Counting from LSB to MSB, number of bits before last '1'
  // This is floor(log(x))
  const auto n_lower_bits = ((8 * sizeof(x)) - CLZ(x) - 1);

  // Add 1 if 2^n_lower_bits is less than x
  //     (i.e. we rounded down because x was not a power of 2)
  return static_cast<int>(n_lower_bits + ((1 << n_lower_bits) < x));
}

inline void process_radix_tree_i(const int i,
                                 const int n /*n_brt_nodes*/,
                                 const uint32_t* codes,
                                 //  const RadixTree* out_brt
                                 uint8_t* brt_prefix_n,
                                 FakeBool* brt_has_leaf_left,
                                 FakeBool* brt_has_leaf_right,
                                 int32_t* brt_left_child,
                                 int32_t* brt_parents) {
  // 'i' is the iterator within a chunk
  // 'codes' is the base address of the whole data, for each chunk, we need to
  // use the offset 'out_brt' is the base address of the whole data, for each
  // chunk, we need to use the offset

  const auto code_i = codes[i];

  const auto prefix_n = brt_prefix_n;
  const auto has_leaf_left = brt_has_leaf_left;
  const auto has_leaf_right = brt_has_leaf_right;
  const auto left_child = brt_left_child;
  const auto parent = brt_parents;

  // Determine direction of the range (+1 or -1)
  int d;
  if (i == 0) {
    d = 1;
  } else {
    const auto delta_diff_right = delta_u32(code_i, codes[i + 1]);
    const auto delta_diff_left = delta_u32(code_i, codes[i - 1]);
    const auto direction_difference = delta_diff_right - delta_diff_left;
    d = (direction_difference > 0) - (direction_difference < 0);
  }

  // Compute upper bound for the length of the range

  auto l = 0;
  if (i == 0) {
    // First node is root, covering whole tree
    l = n - 1;
  } else {
    const auto delta_min = delta_u32(code_i, codes[i - d]);
    auto l_max = 2;
    // Cast to ptrdiff_t so in case the result is negative (since d is +/- 1),
    // we can catch it and not index out of bounds
    while (i + static_cast<std::ptrdiff_t>(l_max) * d >= 0 && i + l_max * d <= n &&
           delta_u32(code_i, codes[i + l_max * d]) > delta_min) {
      l_max *= 2;
    }
    const auto l_cutoff = (d == -1) ? i : n - i;
    int t;
    int divisor;
    // Find the other end using binary search
    for (t = l_max / 2, divisor = 2; t >= 1; divisor *= 2, t = l_max / divisor) {
      if (l + t <= l_cutoff && delta_u32(code_i, codes[i + (l + t) * d]) > delta_min) {
        l += t;
      }
    }
  }

  const auto j = i + l * d;

  // Find the split position using binary search
  const auto delta_node = delta_u32(codes[i], codes[j]);
  prefix_n[i] = delta_node;
  auto s = 0;
  const auto max_divisor = 1 << log2_ceil_u32(l);
  auto divisor = 2;
  const auto s_cutoff = (d == -1) ? i - 1 : n - i - 1;
  for (auto t = ceil_div_u32(l, 2); divisor <= max_divisor;
       divisor <<= 1, t = ceil_div_u32(l, divisor)) {
    // Yanwen: 2025-2-25 Fix
    if (s + (int)t <= s_cutoff && delta_u32(code_i, codes[i + (s + t) * d]) > delta_node) {
      s += t;
    }
  }

  // Split position
  const auto gamma = i + s * d + std::min(d, 0);
  left_child[i] = gamma;
  has_leaf_left[i] = (std::min(i, j) == gamma);
  has_leaf_right[i] = (std::max(i, j) == gamma + 1);
  // Set parents of left and right children, if they aren't leaves
  // can't set this node as parent of its leaves, because the
  // leaf also represents an internal node with a differnent parent
  if (!has_leaf_left[i]) {
    parent[gamma] = i;
  }
  if (!has_leaf_right[i]) {
    parent[gamma + 1] = i;
  }
}

}  // namespace v1

// ----------------------------------------------------------------------------
// New version
// ----------------------------------------------------------------------------

namespace v2 {

// count leading zeros in a 32‑bit word (undefined if x == 0)
static inline int clz32(uint32_t x) {
  assert(x != 0);
  return __builtin_clz(x);
}

// number of common leading bits between x and y
static inline int common_prefix_bits(uint32_t x, uint32_t y) {
  if (x == y) return 32;
  return clz32(x ^ y);
}

// build a binary radix tree over sorted, unique 32‑bit Morton codes
//
// codes         : input array of length n (must be sorted, no duplicates)
// n             : number of codes
// parents       : output array length n; parents[i] = parent index of node i (root stays -1)
// left_child    : output array length n; left_child[i] = index of left subtree root for node i
// has_leaf_left : output array length n; nonzero if left child of i is a leaf
// has_leaf_right: output array length n; nonzero if right child of i is a leaf
// prefix_length : output array length n; shared‑prefix length between the two children of node i
//
static inline void build_radix_tree(const uint32_t* codes,
                                    int n,
                                    int* parents,
                                    int* left_child,
                                    uint8_t* has_leaf_left,
                                    uint8_t* has_leaf_right,
                                    int* prefix_length) {
  assert(n > 0);

  // initialize outputs
#pragma omp for
  for (int i = 0; i < n; ++i) {
    parents[i] = -1;
    left_child[i] = -1;
    has_leaf_left[i] = 0;
    has_leaf_right[i] = 0;
    prefix_length[i] = 0;
  }

#pragma omp for
  for (int i = 0; i < n; ++i) {
    // 1) choose direction d = +1 or -1 based on which neighbor shares more bits
    int d;
    if (i == 0) {
      d = +1;
    } else if (i == n - 1) {
      d = -1;
    } else {
      int lcp_left = common_prefix_bits(codes[i], codes[i - 1]);
      int lcp_right = common_prefix_bits(codes[i], codes[i + 1]);
      d = (lcp_right > lcp_left) ? +1 : -1;
    }

    // 2) find neighbor LCP to bound our search
    int neighbor_idx = i - d;
    int neighbor_lcp = common_prefix_bits(codes[i], codes[neighbor_idx]);

    // exponential search to find a range where LCP ≤ neighbor_lcp
    int step = 1;
    while (true) {
      int j = i + d * step;
      if (j < 0 || j >= n) break;
      int cur_lcp = common_prefix_bits(codes[i], codes[j]);
      if (cur_lcp <= neighbor_lcp) break;
      step <<= 1;
    }

    // binary search in [0, step) for the farthest index j with LCP > neighbor_lcp
    int lo = 0, hi = step;
    while (hi - lo > 1) {
      int mid = (lo + hi) >> 1;
      int j = i + d * mid;
      int cur_lcp = (j >= 0 && j < n) ? common_prefix_bits(codes[i], codes[j]) : 0;
      if (cur_lcp > neighbor_lcp) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    int j = i + d * lo;

    // 3) this node’s prefix length is the LCP between codes[i] and codes[j]
    int node_lcp = common_prefix_bits(codes[i], codes[j]);
    prefix_length[i] = node_lcp;

    // 4) split the range [min(i,j), max(i,j)] by LCP > node_lcp
    int first = (i < j) ? i : j;
    int last = (i < j) ? j : i;
    int lo_idx = first, hi_idx = last;
    while (lo_idx + 1 < hi_idx) {
      int mid = (lo_idx + hi_idx) >> 1;
      int mid_lcp = common_prefix_bits(codes[first], codes[mid]);
      if (mid_lcp > node_lcp) {
        lo_idx = mid;
      } else {
        hi_idx = mid;
      }
    }
    int split = lo_idx;

    // 5) record children and leaf flags
    left_child[i] = split;
    bool leaf_left = (split == first);
    bool leaf_right = (split + 1 == last);
    has_leaf_left[i] = leaf_left;
    has_leaf_right[i] = leaf_right;

    if (!leaf_left) parents[split] = i;
    if (!leaf_right) parents[split + 1] = i;
  }
}

}  // namespace v2

}  // namespace omp

}  // namespace tree