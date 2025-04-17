#pragma once

#include <libmorton/morton.h>

#include <cfloat>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace octree::omp {

// ----------------------------------------------------------------------------
// Stage 1 (xyz -> morton)
// ----------------------------------------------------------------------------
inline float xmin = FLT_MAX, ymin = FLT_MAX, zmin = FLT_MAX;
inline float xmax = -FLT_MAX, ymax = -FLT_MAX, zmax = -FLT_MAX;

// positions   : length‑n array of glm::vec4
// bounds_min  : glm::vec3 of the minimum corner
// bounds_max  : glm::vec3 of the maximum corner
// codes_out   : pre‑allocated length‑n array of uint32_t
static inline void compute_morton_codes_with_range(const glm::vec4* positions,
                                                   int n,
                                                   const glm::vec3& bounds_min,
                                                   const glm::vec3& bounds_max,
                                                   uint32_t* codes_out) {
  float dx = bounds_max.x - bounds_min.x;
  float dy = bounds_max.y - bounds_min.y;
  float dz = bounds_max.z - bounds_min.z;

#pragma omp for schedule(static)
  for (int i = 0; i < n; ++i) {
    const auto& p = positions[i];

    // normalize into [0,1]
    float nx = (dx > 0.0f) ? ((p.x - bounds_min.x) / dx) : 0.0f;
    float ny = (dy > 0.0f) ? ((p.y - bounds_min.y) / dy) : 0.0f;
    float nz = (dz > 0.0f) ? ((p.z - bounds_min.z) / dz) : 0.0f;

    // quantize to 10 bits
    uint16_t xi = uint16_t(nx * 1023.0f);
    uint16_t yi = uint16_t(ny * 1023.0f);
    uint16_t zi = uint16_t(nz * 1023.0f);

    codes_out[i] = libmorton::morton3D_32_encode(xi, yi, zi);
  }
}

inline void compute_morton_codes(const glm::vec4* positions, const size_t n, uint32_t* codes_out) {
  // 1a) find bounding box via a parallel reduction
  //   float xmin = FLT_MAX, ymin = FLT_MAX, zmin = FLT_MAX;
  //   float xmax = -FLT_MAX, ymax = -FLT_MAX, zmax = -FLT_MAX;

#pragma omp for reduction(min : xmin, ymin, zmin) reduction(max : xmax, ymax, zmax)
  for (size_t i = 0; i < n; ++i) {
    const glm::vec4& p = positions[i];
    xmin = fminf(xmin, p.x);
    ymin = fminf(ymin, p.y);
    zmin = fminf(zmin, p.z);
    xmax = fmaxf(xmax, p.x);
    ymax = fmaxf(ymax, p.y);
    zmax = fmaxf(zmax, p.z);
  }

#pragma omp barrier

  const float dx = xmax - xmin;
  const float dy = ymax - ymin;
  const float dz = zmax - zmin;

  // 1b) normalize, quantize to 10 bits, then encode
#pragma omp for
  for (size_t i = 0; i < n; ++i) {
    const glm::vec4& p = positions[i];

    // normalize into [0,1]
    float nx = (dx > 0.0f) ? ((p.x - xmin) / dx) : 0.0f;
    float ny = (dy > 0.0f) ? ((p.y - ymin) / dy) : 0.0f;
    float nz = (dz > 0.0f) ? ((p.z - zmin) / dz) : 0.0f;

    // quantize to 10 bits (0..1023)
    uint16_t xi = uint16_t(nx * 1023.0f);
    uint16_t yi = uint16_t(ny * 1023.0f);
    uint16_t zi = uint16_t(nz * 1023.0f);

    // libmorton call
    codes_out[i] = libmorton::morton3D_32_encode(xi, yi, zi);
  }
}

// ----------------------------------------------------------------------------
// Stage 2 (sort morton codes)
// ----------------------------------------------------------------------------

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

static inline void build_radix_tree(const uint32_t* codes,
                                    int n,
                                    int* parents,
                                    int* left_child,
                                    uint8_t* has_leaf_left,
                                    uint8_t* has_leaf_right,
                                    int* prefix_length) {
  assert(n > 0);

// 1) initialize arrays in parallel
#pragma omp for schedule(static)
  for (int i = 0; i < n; ++i) {
    parents[i] = -1;
    left_child[i] = -1;
    has_leaf_left[i] = 0;
    has_leaf_right[i] = 0;
    prefix_length[i] = 0;
  }

// 2) build each node in parallel
#pragma omp for schedule(static)
  for (int i = 0; i < n; ++i) {
    // pick direction based on neighbor LCP
    int d;
    if (i == 0)
      d = +1;
    else if (i == n - 1)
      d = -1;
    else {
      int lcp_l = common_prefix_bits(codes[i], codes[i - 1]);
      int lcp_r = common_prefix_bits(codes[i], codes[i + 1]);
      d = (lcp_r > lcp_l) ? +1 : -1;
    }

    int neigh = i - d;
    int bound = common_prefix_bits(codes[i], codes[neigh]);

    // exponential search
    int step = 1;
    while (true) {
      int j = i + d * step;
      if (j < 0 || j >= n || common_prefix_bits(codes[i], codes[j]) <= bound) break;
      step <<= 1;
    }

    // binary search in [0,step)
    int lo = 0, hi = step;
    while (hi - lo > 1) {
      int mid = (lo + hi) >> 1;
      int j = i + d * mid;
      int lcp = (j >= 0 && j < n) ? common_prefix_bits(codes[i], codes[j]) : 0;
      if (lcp > bound)
        lo = mid;
      else
        hi = mid;
    }
    int j = i + d * lo;

    // record this node’s prefix length
    int node_lcp = common_prefix_bits(codes[i], codes[j]);
    prefix_length[i] = node_lcp;

    // split [min(i,j),max(i,j)]
    int first = (i < j) ? i : j;
    int last = (i < j) ? j : i;
    int L = first, R = last;
    while (L + 1 < R) {
      int m = (L + R) >> 1;
      if (common_prefix_bits(codes[first], codes[m]) > node_lcp)
        L = m;
      else
        R = m;
    }
    int split = L;

    // wire up children & parents
    left_child[i] = split;
    bool leafL = (split == first);
    bool leafR = (split + 1 == last);
    has_leaf_left[i] = leafL;
    has_leaf_right[i] = leafR;

    if (!leafL) parents[split] = i;
    if (!leafR) parents[split + 1] = i;
  }
}

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
                                             int n,
                                             const int* left_child,
                                             const int* prefix_length,
                                             int* edge_count_out) {
  constexpr int MORTON_BITS = 30;
#pragma omp for schedule(static)
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

//-----------------------------------------------------------------------------
// Step 7: scatter each node’s children into a flat array via the offsets.
//
// codes, n, left_child, prefix_length  : as above
// edge_count             : computed in Step 5
// offsets                : exclusive prefix‐sum of edge_count (length n)
// children_out           : length = offsets[n-1] + edge_count[n-1]
//
static inline void build_octree_nodes_kernel(const uint32_t* codes,
                                             int n,
                                             const int* left_child,
                                             const int* prefix_length,
                                             [[maybe_unused]] const int* edge_count,
                                             const int* offsets,
                                             int* children_out) {
  constexpr int MORTON_BITS = 30;
#pragma omp for schedule(static)
  for (int i = 0; i < n; ++i) {
    int j = left_child[i];
    int first = (i < j ? i : j);
    int last = (i < j ? j : i);

    int depth = prefix_length[i];
    int shift = MORTON_BITS - depth - 3;
    assert(shift >= 0);

    // per‐octant cursor
    int local_cursor[8] = {0};
    bool used[8] = {false};
    int base = offsets[i];

    for (int k = first; k <= last; ++k) {
      int oct = (codes[k] >> shift) & 0x7;
      if (!used[oct]) {
        int out_idx = base + local_cursor[oct]++;
        children_out[out_idx] = k;
        used[oct] = true;
      }
    }
  }
}

}  // namespace octree::omp
