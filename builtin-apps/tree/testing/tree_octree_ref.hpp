#pragma once
// ----------------------------------------------------------------------------
// Tiny independent BRUTE-FORCE octree ground truth (stage 7 anchor).
//
// Purpose: a doubt-free, second-source octree built from a small set of morton
// codes by the OBVIOUS definition -- with NO reuse of the repo's radix-tree /
// edge-count / octree code. Used to decide whether the production octree's
// children[] / child_node_mask linking is correct or buggy, by matching nodes
// across the two builders by GEOMETRY (corner + cell_size), since node indexing
// differs between the two.
//
// Definition used (the textbook pointer-octree over sorted Morton codes):
//   * Each octree node is identified by a morton PREFIX of length 3*level bits
//     (level = number of resolved octants from the top). The set of octree
//     internal nodes = every distinct prefix at which at least two of the input
//     codes first diverge, i.e. every prefix that has >= 2 distinct child
//     octants among the codes that share it. (We also always include the root
//     prefix at the coarsest level where the whole set still shares bits.)
//   * For an internal node at `level`, its children are the distinct values of
//     the NEXT 3-bit octant (bits [morton_bits-3*(level+1) .. -3*level)) among
//     the codes sharing the node's prefix.
//   * A child octant whose sub-range contains >= 2 distinct deeper octants (or
//     more precisely, whose codes do not all share a longer common prefix that
//     skips straight to a leaf) is itself an internal node; otherwise it is a
//     leaf. For the children-linking question we only care about node->node
//     edges, so we record, for each (parent prefix, child octant), the prefix
//     of the child node IF that child is itself an internal node.
//   * Geometry: corner = decode(prefix << (morton_bits-3*level)); cell_size =
//     range / 2^level. (range/2^level, NOT range/2^(level-root_level): the
//     geometric truth is anchored to the absolute morton lattice, independent of
//     where the repo chooses to put its octree "root".)
//
// O(n^2)-ish over a tiny n: trivially correct.
// ----------------------------------------------------------------------------

#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "../omp/func_morton.hpp"  // morton32_to_xyz, morton_bits (decode only)

namespace tree::testing {

inline constexpr int kOctMortonBits = 30;  // = tree::omp::morton_bits

// One internal octree node in the brute-force ground truth, keyed by morton
// prefix + level. Geometry is derived; children records, per occupied octant,
// whether that octant leads to another INTERNAL node (and its prefix/level).
struct OctRefNode {
  uint32_t prefix;  // the node's morton prefix, left-aligned (top 3*level bits)
  int level;        // number of resolved octants (0 = whole domain)
  float cell_size;  // range / 2^level
  float corner[3];
  // occupied octant -> child internal node prefix (left-aligned) if the child
  // is itself internal; absent if that octant is a pure leaf.
  std::map<int, uint32_t> child_internal;  // octant(0..7) -> child prefix
  // occupied octant -> the leaf's full morton code, when that octant resolves to
  // a SINGLE input code (a pure leaf rather than a deeper internal node). This is
  // the independent ground truth for leaf LINKING (which point hangs off which
  // octnode+octant), used to validate process_link_leaf.
  std::map<int, uint32_t> child_leaf;  // octant(0..7) -> leaf morton code
  int n_occupied_octants = 0;              // distinct next-octants present
};

// Decode a left-aligned morton prefix to a corner using the repo's own decoder
// (decode is not the thing under test; the LINKING is).
inline void DecodePrefixCorner(uint32_t left_aligned_prefix,
                               float min_coord,
                               float range,
                               float out_corner[3]) {
  glm::vec4 c;
  tree::omp::morton32_to_xyz(&c, left_aligned_prefix, min_coord, range);
  out_corner[0] = c[0];
  out_corner[1] = c[1];
  out_corner[2] = c[2];
}

// Build the brute-force octree from a sorted, unique set of morton codes.
//
// Returns one OctRefNode per distinct internal-node prefix. An internal node at
// `level` exists iff the codes sharing its (3*level)-bit prefix span >= 2
// distinct next-octants. The root we expose is the deepest prefix shared by ALL
// codes (level = common prefix length / 3) -- matching how a Karras/edge octree
// roots itself at the coarsest BRANCHING level rather than the absolute domain.
inline std::vector<OctRefNode> BuildBruteForceOctree(const std::vector<uint32_t>& codes,
                                                     float min_coord,
                                                     float range) {
  // octant of `code` at a given level (the 3 bits resolved at depth level+1).
  auto octant_at = [](uint32_t code, int level) -> int {
    // bits [morton_bits - 3*(level+1), morton_bits - 3*level)
    const int shift = kOctMortonBits - 3 * (level + 1);
    return (code >> shift) & 0b111;
  };
  // left-aligned prefix of `code` to `level` octants.
  auto prefix_at = [](uint32_t code, int level) -> uint32_t {
    if (level <= 0) return 0u;
    const int keep = 3 * level;                       // top bits to keep
    const uint32_t mask = ~((1u << (kOctMortonBits - keep)) - 1u);
    return code & mask & ((1u << kOctMortonBits) - 1u);
  };

  // Recursively (here: iteratively via a worklist) discover internal nodes.
  // An internal node is a (prefix, level) under which the covered codes span
  // >= 2 distinct next-octants. We start at the coarsest branching level.
  struct Span {
    int lo, hi;  // [lo, hi) index range into codes (all share `level`-prefix)
    int level;
  };

  // coarsest common level for the whole set.
  auto common_level = [&](int lo, int hi) -> int {
    // largest L such that all codes[lo..hi) share their first 3*L bits.
    int L = 0;
    while (L < kOctMortonBits / 3) {
      const uint32_t p0 = prefix_at(codes[lo], L + 1);
      bool same = true;
      for (int k = lo + 1; k < hi; ++k) {
        if (prefix_at(codes[k], L + 1) != p0) {
          same = false;
          break;
        }
      }
      if (!same) break;
      ++L;
    }
    return L;
  };

  std::vector<OctRefNode> nodes;

  if (codes.size() < 2) return nodes;

  // Canonical recursive octree over the codes: an internal node spans [lo,hi)
  // and a level. We collapse straight chains: at each step we descend to the
  // node's own common_level (>= the level its parent assigned). A code range is
  // an internal node iff it spans >= 2 distinct codes that diverge below the
  // collapsed level (i.e. partition into >= 2 octants at that level). Single
  // codes / fully-identical ranges are leaves.
  std::vector<Span> work;
  const int root_level = common_level(0, static_cast<int>(codes.size()));
  work.push_back({0, static_cast<int>(codes.size()), root_level});

  while (!work.empty()) {
    Span s = work.back();
    work.pop_back();

    // collapse chain: descend to the deepest level still shared by all codes in
    // [lo,hi) -- that is where this node actually branches.
    const int lvl = common_level(s.lo, s.hi);

    OctRefNode node;
    node.prefix = prefix_at(codes[s.lo], lvl);
    node.level = lvl;
    node.cell_size = range / static_cast<float>(1 << lvl);
    DecodePrefixCorner(node.prefix, min_coord, range, node.corner);

    // partition [lo,hi) by the octant at THIS node's branching level.
    std::map<int, std::pair<int, int>> sub;  // octant -> [lo,hi)
    for (int k = s.lo; k < s.hi; ++k) {
      const int oct = octant_at(codes[k], lvl);
      if (sub.find(oct) == sub.end())
        sub[oct] = {k, k + 1};
      else
        sub[oct].second = k + 1;
    }
    node.n_occupied_octants = static_cast<int>(sub.size());

    for (auto& [oct, rng] : sub) {
      const int clo = rng.first, chi = rng.second;
      // This octant leads to another internal node iff its codes still branch
      // below: >= 2 distinct codes that diverge at some deeper level.
      bool child_is_internal = false;
      if (chi - clo >= 2) {
        const int cl = common_level(clo, chi);
        // they branch below `cl`, so there is a real internal node there.
        child_is_internal = true;
        node.child_internal[oct] = prefix_at(codes[clo], cl);
        work.push_back({clo, chi, cl});
      } else {
        // single code -> pure leaf octant. Record the leaf's code so the leaf
        // LINKING (process_link_leaf) can be validated against this ground truth.
        node.child_leaf[oct] = codes[clo];
      }
      (void)child_is_internal;
    }

    nodes.push_back(std::move(node));
  }

  return nodes;
}

}  // namespace tree::testing
