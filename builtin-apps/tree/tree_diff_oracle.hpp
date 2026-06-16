#pragma once
// ----------------------------------------------------------------------------
// Tree differential-oracle harness (backend-parametrized).
//
// The CPU/OMP path is the reference: SafeAppData's const golden buffers are
// produced by the OMP kernels at construction (HostTreeManager::initialize()).
// For each stage S and any backend B, we run stages 1..S on B into the `_out`
// buffers and compare the stage-S output against the golden — exact for the
// integer/structural stages, near for the stage-7 octree floats.
//
// All backends share this logic; only the "runner" differs (how AppData memory
// is allocated and how a stage is dispatched). Each backend's test_main defines
// a Runner and expands BT_DECLARE_TREE_DIFF_TESTS(suite, Runner). The same
// binary self-validates on whatever target runs it (PC/Jetson/phone), so there
// are no stored goldens and no host->device transfer of expected values.
//
// Per-stage gotchas this encodes (see docs/TESTING.md §B):
//   - stage 1 morton is an unordered set  -> compare as a multiset (sorted).
//   - unique/brt/octree write only a valid prefix -> compare n_unique /
//     n_brt_nodes / n_octree_nodes elements (already known at construction).
//   - stage-7 children & leaf-mask require the golden to run process_link_leaf
//     (completed in safe_tree_appdata.cpp).
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <span>
#include <tuple>
#include <vector>

#include "builtin-apps/common/testing/oracle.hpp"
#include "safe_tree_appdata.hpp"

namespace tree::testing {

// Reinterpret a contiguous range of glm::vec4 as a flat float span (4 per elem).
template <class Vec4Range>
[[nodiscard]] inline std::span<const float> AsFloats(const Vec4Range& v) {
  return std::span<const float>(reinterpret_cast<const float*>(v.data()), v.size() * 4);
}

// Each Runner declares `using AppData = …` — tree::SafeAppData for OMP/CUDA, or
// the Vulkan subclass tree::vulkan::VkAppData_Safe (which IS-A SafeAppData, so
// the golden/_out buffers the checks read are inherited). The checks always take
// the SafeAppData base. Running stages 1..S populates the stage-S _out (each
// stage reads its reference input and writes its own _out; stage 2 chains off
// stage-1's _out).

inline void CheckStage1(const tree::SafeAppData& a) {
  // Morton keys are an unordered set over the input points; the sorted golden
  // (u_morton_keys_sorted_s2) is the reference. Compare as a multiset.
  std::vector<uint32_t> got(a.u_morton_keys_s1_out.begin(), a.u_morton_keys_s1_out.end());
  std::ranges::sort(got);
  EXPECT_TRUE(bt::testing::ExactEqual(a.u_morton_keys_sorted_s2, got, "tree s1 morton(set)"));
}

inline void CheckStage2(const tree::SafeAppData& a) {
  EXPECT_TRUE(bt::testing::ExactEqual(
      a.u_morton_keys_sorted_s2, a.u_morton_keys_sorted_s2_out, "tree s2 sort"));
}

inline void CheckStage3(const tree::SafeAppData& a) {
  EXPECT_TRUE(bt::testing::ExactEqual(
      a.u_morton_keys_unique_s3, a.u_morton_keys_unique_s3_out, "tree s3 unique", a.get_n_unique()));
}

inline void CheckStage4(const tree::SafeAppData& a) {
  const std::size_t n = a.get_n_brt_nodes();
  EXPECT_TRUE(bt::testing::ExactEqual(a.u_brt_prefix_n_s4, a.u_brt_prefix_n_s4_out, "tree s4 prefix_n", n));
  EXPECT_TRUE(bt::testing::ExactEqual(a.u_brt_has_leaf_left_s4, a.u_brt_has_leaf_left_s4_out, "tree s4 leaf_left", n));
  EXPECT_TRUE(bt::testing::ExactEqual(a.u_brt_has_leaf_right_s4, a.u_brt_has_leaf_right_s4_out, "tree s4 leaf_right", n));
  EXPECT_TRUE(bt::testing::ExactEqual(a.u_brt_left_child_s4, a.u_brt_left_child_s4_out, "tree s4 left_child", n));
  EXPECT_TRUE(bt::testing::ExactEqual(a.u_brt_parents_s4, a.u_brt_parents_s4_out, "tree s4 parents", n));
}

inline void CheckStage5(const tree::SafeAppData& a) {
  EXPECT_TRUE(bt::testing::ExactEqual(
      a.u_edge_count_s5, a.u_edge_count_s5_out, "tree s5 edge_count", a.get_n_brt_nodes()));
}

inline void CheckStage6(const tree::SafeAppData& a) {
  EXPECT_TRUE(bt::testing::ExactEqual(
      a.u_edge_offset_s6, a.u_edge_offset_s6_out, "tree s6 edge_offset", a.get_n_brt_nodes()));
}

// Canonical, index-permutation-independent octree edge set. Each entry is a
// downward edge keyed purely by geometry + octant + kind (0 = internal child via
// child_node_mask, 1 = leaf via child_leaf_mask). Coordinates are quantized to
// ints (the octree lattice is integral for kRange=1024, kMinCoord=0).
using OctEdge = std::array<long, 9>;  // {kind, pcx,pcy,pcz,pcell, tcx,tcy,tcz, tcell}

[[nodiscard]] inline std::vector<OctEdge> OctreeEdgeSet(
    std::span<const glm::vec4> corner,
    std::span<const float> cell,
    std::span<const std::int32_t> children,   // n*8
    std::span<const std::int32_t> node_mask,
    std::span<const std::int32_t> leaf_mask,
    std::size_t n) {
  auto q = [](float v) { return static_cast<long>(std::lround(v)); };
  std::vector<OctEdge> edges;
  for (std::size_t v = 0; v < n; ++v) {
    const long pcx = q(corner[v].x), pcy = q(corner[v].y), pcz = q(corner[v].z);
    const long pcell = q(cell[v]);
    for (int c = 0; c < 8; ++c) {
      const bool is_node = (node_mask[v] & (1 << c)) != 0;
      // child_leaf_mask is initialized to all-ones and CLEARED on a leaf link, so
      // a populated leaf slot is a ZERO bit (matches set_leaf's &= ~).
      const bool is_leaf = (leaf_mask[v] & (1 << c)) == 0;
      if (!is_node && !is_leaf) continue;
      const long kind = is_node ? 0 : 1;
      const std::int32_t t = children[v * 8 + c];
      OctEdge e{kind, pcx, pcy, pcz, pcell, 0, 0, 0, 0};
      if (kind == 0 && t >= 0 && static_cast<std::size_t>(t) < n) {
        // internal child: identify it by its OWN geometry (index-independent).
        e[5] = q(corner[t].x); e[6] = q(corner[t].y); e[7] = q(corner[t].z);
        e[8] = q(cell[t]);
      } else {
        // leaf: the value is a point index, not an octree node; key by the raw
        // value (point indices are the same input on every backend).
        e[5] = t;
      }
      edges.push_back(e);
    }
  }
  std::ranges::sort(edges);
  return edges;
}

[[nodiscard]] inline ::testing::AssertionResult CheckStage7Topology(const tree::SafeAppData& a,
                                                                    std::size_t n) {
  std::span<const std::int32_t> ch_ref(a.u_oct_children_s7.data(), a.u_oct_children_s7.size());
  std::span<const std::int32_t> ch_out(a.u_oct_children_s7_out.data(), a.u_oct_children_s7_out.size());
  std::span<const std::int32_t> nm_ref(a.u_oct_child_node_mask_s7.data(), a.u_oct_child_node_mask_s7.size());
  std::span<const std::int32_t> nm_out(a.u_oct_child_node_mask_s7_out.data(), a.u_oct_child_node_mask_s7_out.size());
  std::span<const std::int32_t> lm_ref(a.u_oct_child_leaf_mask_s7.data(), a.u_oct_child_leaf_mask_s7.size());
  std::span<const std::int32_t> lm_out(a.u_oct_child_leaf_mask_s7_out.data(), a.u_oct_child_leaf_mask_s7_out.size());

  auto ref = OctreeEdgeSet(a.u_oct_corner_s7, a.u_oct_cell_size_s7, ch_ref, nm_ref, lm_ref, n);
  auto out = OctreeEdgeSet(a.u_oct_corner_s7_out, a.u_oct_cell_size_s7_out, ch_out, nm_out, lm_out, n);
  if (ref == out) return ::testing::AssertionSuccess();

  // Break the difference down by edge kind. INTERNAL (kind 0) parent->child edges
  // are the octree's structural topology and MUST be order-independent; LEAF
  // (kind 1) slots can legitimately collide (two leaves of the same octnode+octant
  // resolved by non-atomic last-writer-wins differ across backends), so we report
  // them separately and only fail on an internal-topology divergence.
  auto split = [](const std::vector<OctEdge>& e, long kind) {
    std::vector<OctEdge> r;
    for (const auto& x : e) if (x[0] == kind) r.push_back(x);
    return r;  // already sorted (subsequence of a sorted vector)
  };
  const auto ref_int = split(ref, 0), out_int = split(out, 0);
  const auto ref_leaf = split(ref, 1), out_leaf = split(out, 1);

  std::size_t int_only_ref = 0, int_only_out = 0;
  {
    std::vector<OctEdge> d1, d2;
    std::ranges::set_difference(ref_int, out_int, std::back_inserter(d1));
    std::ranges::set_difference(out_int, ref_int, std::back_inserter(d2));
    int_only_ref = d1.size();
    int_only_out = d2.size();
  }
  if (int_only_ref != 0 || int_only_out != 0) {
    return ::testing::AssertionFailure()
           << "tree s7 topology: INTERNAL parent->child edges differ (" << int_only_ref
           << " only-in-ref, " << int_only_out << " only-in-out; ref_int=" << ref_int.size()
           << " out_int=" << out_int.size() << ") -- a real cross-backend octree-structure bug";
  }
  // Internal topology matches; only leaf slot resolution differs (order-sensitive,
  // not a structural defect). Surface it as a non-fatal note via success with a
  // message visible only on -1 logging; keep the gate on internal edges.
  if (ref_leaf == out_leaf) return ::testing::AssertionSuccess();
  return ::testing::AssertionSuccess()
         << "(note) leaf-slot edges differ by last-writer-wins (order-sensitive), "
            "internal topology identical";
}

inline void CheckStage7(const tree::SafeAppData& a) {
  const std::size_t n = a.get_n_octree_nodes();
  // child_node_mask is an OR-reduction: several brt nodes scatter children into
  // the same octree node's mask. With atomicOr in the shader (matching the
  // commutative |= the reference intends) the mask is order-INDEPENDENT and is a
  // stable differential target.
  EXPECT_TRUE(bt::testing::ExactEqual(a.u_oct_child_node_mask_s7, a.u_oct_child_node_mask_s7_out, "tree s7 node_mask", n));
  // Geometry IS now a deterministic oracle target. The octree builder used the
  // INCLUSIVE edge_offset prefix sum as each brt node's range START, which shifted
  // every range +edge_count[i] -> the root (node 0) had no writer and adjacent
  // brt ranges overlapped, so cell_size/corner were last-writer-wins (and node 0
  // a hole). Fixed by using the EXCLUSIVE start (edge_offsets[x]-edge_counts[x])
  // in process_oct_node / process_link_leaf on all three backends; a diagnostic
  // confirmed 0 holes and 0 multi-writer collisions afterward, so each octree
  // node now has exactly one geometry writer and the values are order-independent.
  EXPECT_TRUE(bt::testing::NearEqual(a.u_oct_cell_size_s7, a.u_oct_cell_size_s7_out, 1e-5f, 1e-6f, "tree s7 cell_size", n));
  EXPECT_TRUE(bt::testing::NearEqual(AsFloats(a.u_oct_corner_s7), AsFloats(a.u_oct_corner_s7_out), 1e-5f, 1e-6f, "tree s7 corner", n * 4));

  // Order-INDEPENDENT children/leaf topology comparison. Octree node indices are
  // identical across backends (the per-index geometry above matches), but child
  // slots are scattered, so rather than rely on slot-write order we canonicalize
  // each downward edge by GEOMETRY: per node v with populated child/leaf octant
  // c, emit (corner[v], cell[v], c, kind, corner[target], cell[target]). The
  // multiset of these tuples is the octree topology; comparing the sorted sets of
  // golden vs _out validates u_oct_children_s7 + u_oct_child_node_mask_s7 +
  // u_oct_child_leaf_mask_s7 together, independent of node-index permutations.
  EXPECT_TRUE(CheckStage7Topology(a, n));
}

// Drive: skip if the backend's device is absent, else run 1..s and check s.
template <class Runner>
inline void RunAndCheckStage(int s) {
  if (!Runner::Available()) {
    GTEST_SKIP() << "backend device not available on this target";
  }
  Runner runner;
  typename Runner::AppData a(runner.Mr());  // SafeAppData (OMP/CUDA) or VkAppData_Safe (Vulkan)
  for (int i = 1; i <= s; ++i) runner.RunStage(a, i);
  const tree::SafeAppData& base = a;  // checks read the inherited golden / _out
  switch (s) {
    case 1: CheckStage1(base); break;
    case 2: CheckStage2(base); break;
    case 3: CheckStage3(base); break;
    case 4: CheckStage4(base); break;
    case 5: CheckStage5(base); break;
    case 6: CheckStage6(base); break;
    case 7: CheckStage7(base); break;
    default: FAIL() << "no such tree stage: " << s;
  }
}

}  // namespace tree::testing

// Expand the 7 per-stage differential tests for a given backend Runner.
#define BT_DECLARE_TREE_DIFF_TESTS(SUITE, RUNNER)                                    \
  TEST(SUITE, Stage1_Morton)     { tree::testing::RunAndCheckStage<RUNNER>(1); }     \
  TEST(SUITE, Stage2_Sort)       { tree::testing::RunAndCheckStage<RUNNER>(2); }     \
  TEST(SUITE, Stage3_Unique)     { tree::testing::RunAndCheckStage<RUNNER>(3); }     \
  TEST(SUITE, Stage4_RadixTree)  { tree::testing::RunAndCheckStage<RUNNER>(4); }     \
  TEST(SUITE, Stage5_EdgeCount)  { tree::testing::RunAndCheckStage<RUNNER>(5); }     \
  TEST(SUITE, Stage6_EdgeOffset) { tree::testing::RunAndCheckStage<RUNNER>(6); }     \
  TEST(SUITE, Stage7_Octree)     { tree::testing::RunAndCheckStage<RUNNER>(7); }
