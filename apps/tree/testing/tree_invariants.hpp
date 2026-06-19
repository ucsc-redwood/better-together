#pragma once
// ----------------------------------------------------------------------------
// Structural invariants + tiny brute-force anchors for the tree OMP golden
// (stages 4 radix-tree and 7 octree).
//
// We deliberately do NOT clone a second hand-rolled Karras radix-tree / octree
// builder: two people reading the same paper reproduce the same misconception
// (the exact bug class this repo already hit). Instead we anchor with:
//   (a) property/invariant checks that hold for ANY correct radix tree /
//       octree, needing no second implementation; and
//   (b) a tiny brute-force ground truth on a small hand-built input, where an
//       obvious O(n^2)/exhaustive method is trivially correct.
//
// Each check returns ::testing::AssertionResult so call sites read naturally and
// the first violation is reported with the offending node/index. If a check
// fails on the CURRENT golden, that is a REAL OMP bug -- report it, do not
// weaken the check.
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "tree_ref.hpp"  // kMortonBits

namespace tree::testing {

// Number of leading bits shared by a and b, over the 30-bit morton domain (the
// kernel's delta = clz(a^b) - 1 convention drops the unused MSB; we replicate
// only the well-defined "shared prefix length" notion here for the invariant).
[[nodiscard]] inline int CommonPrefixLen30(uint32_t a, uint32_t b) {
  if (a == b) return kMortonBits;
  // clz over 32 bits; the morton values occupy the low 30 bits, and the kernel
  // subtracts 1 for the dropped sign-style top bit -> prefix length = clz-1.
  const int clz = __builtin_clz(a ^ b);
  return clz - 1;
}

// ---------------------------------------------------------------------------
// Stage 1/3 precondition shared by stages 4 & 7: codes are sorted & unique.
// ---------------------------------------------------------------------------
[[nodiscard]] inline ::testing::AssertionResult SortedUnique(const uint32_t* codes, int n) {
  for (int i = 1; i < n; ++i) {
    if (!(codes[i - 1] < codes[i])) {
      return ::testing::AssertionFailure()
             << "morton codes not strictly increasing at i=" << i << " (codes[" << (i - 1)
             << "]=" << codes[i - 1] << ", codes[" << i << "]=" << codes[i] << ")";
    }
  }
  return ::testing::AssertionSuccess();
}

// ---------------------------------------------------------------------------
// Stage 4 (radix tree) invariants. For each internal node i the kernel emits:
//   left_child[i] (= split index gamma), has_leaf_left/right, prefix_n[i],
//   parents[i]. We recover the node's covered range [first,last] from the split
//   and the leaf flags and assert the Karras structural invariants:
//     - split gamma and gamma+1 are valid indices,
//     - prefix_n[i] equals the actual common-prefix length of the boundary
//       codes of the node's two children (the defining property of an internal
//       radix-tree node),
//     - a non-leaf child's parent pointer points back to i (tree consistency),
//     - children split levels are strictly deeper than the node (prefix grows
//       downward).
// These hold for ANY correct binary radix tree without a second builder.
// ---------------------------------------------------------------------------
[[nodiscard]] inline ::testing::AssertionResult RadixTreeInvariants(const uint32_t* codes,
                                                                    int n_brt,
                                                                    int n_unique,
                                                                    const uint8_t* prefix_n,
                                                                    const uint8_t* has_leaf_left,
                                                                    const uint8_t* has_leaf_right,
                                                                    const int* left_child,
                                                                    const int* parents) {
  for (int i = 0; i < n_brt; ++i) {
    const int gamma = left_child[i];  // split position
    if (gamma < 0 || gamma + 1 >= n_unique) {
      return ::testing::AssertionFailure() << "node " << i << ": split index gamma=" << gamma
                                           << " out of range [0," << (n_unique - 1) << ")";
    }

    // The two children straddle the split: the left subtree's deepest boundary
    // key is codes[gamma], the right subtree's is codes[gamma+1]. prefix_n[i]
    // is, by definition, the length of the common prefix of those boundary keys
    // (the bit position where the node splits its range into two octsubtrees).
    const int actual = CommonPrefixLen30(codes[gamma], codes[gamma + 1]);
    if (static_cast<int>(prefix_n[i]) != actual) {
      return ::testing::AssertionFailure()
             << "node " << i << ": prefix_n=" << +prefix_n[i] << " != common-prefix-len(codes["
             << gamma << "], codes[" << (gamma + 1) << "])=" << actual;
    }

    // A non-leaf child is itself an internal node and must point back to i.
    if (!has_leaf_left[i]) {
      const int c = gamma;
      if (c < 0 || c >= n_brt || parents[c] != i) {
        return ::testing::AssertionFailure()
               << "node " << i << ": left internal child " << c << " has parent "
               << (c >= 0 && c < n_brt ? parents[c] : -999) << " (expected " << i << ")";
      }
      // child is strictly deeper (its prefix is longer than its parent's).
      if (static_cast<int>(prefix_n[c]) <= static_cast<int>(prefix_n[i])) {
        return ::testing::AssertionFailure()
               << "node " << i << ": left internal child " << c << " prefix_n=" << +prefix_n[c]
               << " not deeper than parent prefix_n=" << +prefix_n[i];
      }
    }
    if (!has_leaf_right[i]) {
      const int c = gamma + 1;
      if (c < 0 || c >= n_brt || parents[c] != i) {
        return ::testing::AssertionFailure()
               << "node " << i << ": right internal child " << c << " has parent "
               << (c >= 0 && c < n_brt ? parents[c] : -999) << " (expected " << i << ")";
      }
      if (static_cast<int>(prefix_n[c]) <= static_cast<int>(prefix_n[i])) {
        return ::testing::AssertionFailure()
               << "node " << i << ": right internal child " << c << " prefix_n=" << +prefix_n[c]
               << " not deeper than parent prefix_n=" << +prefix_n[i];
      }
    }
  }

  // Exactly one node (the root) has no parent assigned. The kernel never writes
  // parents[0] (root), so it keeps whatever it was initialized to; rather than
  // asserting on the uninitialized root slot, assert every OTHER node is the
  // non-leaf child of exactly one parent (a forest would be a structural bug).
  std::vector<int> indeg(n_brt, 0);
  for (int i = 0; i < n_brt; ++i) {
    const int gamma = left_child[i];
    if (!has_leaf_left[i] && gamma >= 0 && gamma < n_brt) ++indeg[gamma];
    if (!has_leaf_right[i] && gamma + 1 >= 0 && gamma + 1 < n_brt) ++indeg[gamma + 1];
  }
  int roots = 0;
  for (int i = 0; i < n_brt; ++i) {
    if (indeg[i] == 0) ++roots;
    if (indeg[i] > 1) {
      return ::testing::AssertionFailure()
             << "node " << i << " has in-degree " << indeg[i] << " (>1: not a tree)";
    }
  }
  if (roots != 1) {
    return ::testing::AssertionFailure()
           << "radix tree has " << roots << " roots (expected exactly 1)";
  }
  return ::testing::AssertionSuccess();
}

// ---------------------------------------------------------------------------
// Stage 4 tiny brute-force: build the correct radix tree for a small sorted,
// unique code array by the OBVIOUS O(n^2) definition and compare prefix_n and
// the split index. For node i, Karras defines:
//   - direction d, range [i, j] = maximal run where the LCP with codes[i]
//     exceeds the LCP with the i-d neighbor; we find j by linear scan (O(n)).
//   - prefix_n[i] = LCP(codes[i], codes[j]).
//   - split gamma = last index in [min,max) whose LCP(codes[first], .) still
//     exceeds prefix_n[i]; found by linear scan.
// O(n^2) total, trivially correct for tiny n.
// ---------------------------------------------------------------------------
struct BrtNodeRef {
  int prefix_n;
  int gamma;
};

[[nodiscard]] inline BrtNodeRef BrtNodeBruteForce(int i, const uint32_t* codes, int n_unique) {
  auto lcp = [&](int a, int b) -> int {
    if (a < 0 || a >= n_unique || b < 0 || b >= n_unique) return -1;
    return CommonPrefixLen30(codes[a], codes[b]);
  };
  // direction
  int d;
  if (i == 0) {
    d = 1;
  } else {
    const int lr = lcp(i, i + 1);
    const int ll = lcp(i, i - 1);
    d = (lr - ll > 0) - (lr - ll < 0);
  }
  // delta_min = LCP with the i-d neighbor
  const int delta_min = lcp(i, i - d);
  // find farthest j (= i + l*d) with LCP > delta_min by linear scan
  int l = 0;
  while (true) {
    const int next = i + (l + 1) * d;
    if (next < 0 || next >= n_unique) break;
    if (lcp(i, next) <= delta_min) break;
    ++l;
  }
  const int j = i + l * d;
  const int prefix_n = lcp(i, j);
  // split: largest s in [0, l) with LCP(codes[i], codes[i+(s+1)*d]) > prefix_n
  // (kernel takes gamma = i + s*d + min(d,0)). Find s by linear scan.
  int s = 0;
  for (int t = 1; t <= l; ++t) {
    const int idx = i + t * d;
    if (lcp(i, idx) > prefix_n) {
      s = t;
    } else {
      break;
    }
  }
  const int gamma = i + s * d + (d < 0 ? d : 0);
  return BrtNodeRef{prefix_n, gamma};
}

// ---------------------------------------------------------------------------
// Stage 7 (octree) invariants. The octree geometry is derived per node from the
// morton corner + cell size. For ANY correct octree:
//   - cell_size is a positive power-of-two fraction of `range`,
//   - each node's corner lies on the lattice for its level (corner is an
//     integer multiple of its own cell_size within [min,min+range]),
//   - a child cell is geometrically contained in (and exactly 1/2 the size of)
//     its parent cell, for every populated child slot in child_node_mask.
// These need no second octree builder; they falsify a wrong cell geometry or a
// mis-linked parent/child relationship.
// ---------------------------------------------------------------------------
[[nodiscard]] inline ::testing::AssertionResult OctreeGeometryInvariants(const glm::vec4* corner,
                                                                         const float* cell_size,
                                                                         const int* child_node_mask,
                                                                         const int (*children)[8],
                                                                         int n_oct,
                                                                         float min_coord,
                                                                         float range) {
  for (int v = 0; v < n_oct; ++v) {
    const float cs = cell_size[v];
    if (!(cs > 0.0f) || cs > range) {
      return ::testing::AssertionFailure()
             << "octnode " << v << ": cell_size=" << cs << " not in (0," << range << "]";
    }
    // cell_size must be range / 2^L for some integer L >= 0.
    const float ratio = range / cs;
    const float rounded = std::round(ratio);
    if (std::abs(ratio - rounded) > 1e-3f * ratio) {
      return ::testing::AssertionFailure()
             << "octnode " << v << ": range/cell_size=" << ratio << " is not an integer";
    }
    // power of two?
    int L = static_cast<int>(rounded);
    if (L < 1 || (L & (L - 1)) != 0) {
      return ::testing::AssertionFailure()
             << "octnode " << v << ": range/cell_size=" << L << " is not a power of two";
    }
    // corner must lie on this node's lattice: (corner - min) / cs is an integer.
    for (int axis = 0; axis < 3; ++axis) {
      const float q = (corner[v][axis] - min_coord) / cs;
      if (std::abs(q - std::round(q)) > 1e-2f) {
        return ::testing::AssertionFailure()
               << "octnode " << v << " axis " << axis << ": corner " << corner[v][axis]
               << " not on lattice of cell_size " << cs << " (q=" << q << ")";
      }
    }
  }

  // parent/child containment for every populated child slot.
  for (int v = 0; v < n_oct; ++v) {
    const int mask = child_node_mask[v];
    for (int c = 0; c < 8; ++c) {
      if (!(mask & (1 << c))) continue;
      const int cv = children[v][c];
      if (cv < 0 || cv >= n_oct) continue;  // may reference a leaf, not an octnode
      // child cell is half the parent's
      if (std::abs(cell_size[cv] * 2.0f - cell_size[v]) > 1e-3f * cell_size[v]) {
        return ::testing::AssertionFailure() << "octnode " << v << " child slot " << c << " (node "
                                             << cv << "): child cell_size=" << cell_size[cv]
                                             << " != parent/2=" << (cell_size[v] / 2.0f);
      }
      // child corner is contained in parent cell.
      for (int axis = 0; axis < 3; ++axis) {
        const float lo = corner[v][axis];
        const float hi = corner[v][axis] + cell_size[v];
        const float cc = corner[cv][axis];
        if (cc < lo - 1e-2f || cc > hi + 1e-2f) {
          return ::testing::AssertionFailure()
                 << "octnode " << v << " child " << cv << " axis " << axis << ": child corner "
                 << cc << " outside parent cell [" << lo << "," << hi << "]";
        }
      }
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace tree::testing
