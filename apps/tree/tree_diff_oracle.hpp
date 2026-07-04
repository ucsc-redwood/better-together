#pragma once
// ----------------------------------------------------------------------------
// Tree differential-oracle harness (backend-parametrized). Two mechanisms live
// here side by side, sharing the same per-stage comparison logic:
//
// 1. SafeAppData family (Vulkan only, BT_DECLARE_TREE_DIFF_TESTS /
//    RunAndCheckStage / single-arg CheckStageN): SafeAppData's const golden
//    buffers are produced by the OMP kernels at construction
//    (HostTreeManager::initialize()). For each stage S we run stages 1..S on
//    the backend into the `_out` buffers and compare the stage-S output
//    against the golden.
// 2. tree::AppData (OMP/CUDA, BT_DECLARE_TREE_DIFF_TESTS_APPDATA /
//    RunAndCheckStageAppData / two-arg CheckStageN): tree::AppData has no
//    golden/_out split (single buffer per stage, genuinely chained), so
//    instead we build TWO instances from the SAME input points -- `ref` runs
//    the OMP oracle chain, `out` is the backend under test -- and compare
//    corresponding fields directly. SafeAppData is NOT a subclass of
//    tree::AppData (independent struct, see safe_tree_appdata.hpp), so this is
//    a parallel set of overloads, not a signature change to #1.
//
// Either way: only the "runner" differs (how AppData memory is allocated and
// how a stage is dispatched). Each backend's test_main defines a Runner and
// expands the matching BT_DECLARE_TREE_DIFF_TESTS(_APPDATA) macro. The same
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

#include "apps/tree/omp/dispatchers.hpp"
#include "platform/util/testing/oracle.hpp"
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
  // Guard the count: a 0-length ExactEqual passes vacuously, masking a stage that
  // produced nothing -- make that a failure, not a silent green (review #25).
  ASSERT_GT(a.get_n_unique(), 0u) << "tree s3: zero unique keys (compare would be vacuous)";
  EXPECT_TRUE(bt::testing::ExactEqual(a.u_morton_keys_unique_s3,
                                      a.u_morton_keys_unique_s3_out,
                                      "tree s3 unique",
                                      a.get_n_unique()));
}

inline void CheckStage4(const tree::SafeAppData& a) {
  const std::size_t n = a.get_n_brt_nodes();
  ASSERT_GT(n, 0u) << "tree s4: zero BRT nodes (compare would be vacuous)";
  EXPECT_TRUE(
      bt::testing::ExactEqual(a.u_brt_prefix_n_s4, a.u_brt_prefix_n_s4_out, "tree s4 prefix_n", n));
  EXPECT_TRUE(bt::testing::ExactEqual(
      a.u_brt_has_leaf_left_s4, a.u_brt_has_leaf_left_s4_out, "tree s4 leaf_left", n));
  EXPECT_TRUE(bt::testing::ExactEqual(
      a.u_brt_has_leaf_right_s4, a.u_brt_has_leaf_right_s4_out, "tree s4 leaf_right", n));
  EXPECT_TRUE(bt::testing::ExactEqual(
      a.u_brt_left_child_s4, a.u_brt_left_child_s4_out, "tree s4 left_child", n));
  EXPECT_TRUE(
      bt::testing::ExactEqual(a.u_brt_parents_s4, a.u_brt_parents_s4_out, "tree s4 parents", n));
}

inline void CheckStage5(const tree::SafeAppData& a) {
  ASSERT_GT(a.get_n_brt_nodes(), 0u) << "tree s5: zero BRT nodes (compare would be vacuous)";
  EXPECT_TRUE(bt::testing::ExactEqual(
      a.u_edge_count_s5, a.u_edge_count_s5_out, "tree s5 edge_count", a.get_n_brt_nodes()));
}

inline void CheckStage6(const tree::SafeAppData& a) {
  ASSERT_GT(a.get_n_brt_nodes(), 0u) << "tree s6: zero BRT nodes (compare would be vacuous)";
  EXPECT_TRUE(bt::testing::ExactEqual(
      a.u_edge_offset_s6, a.u_edge_offset_s6_out, "tree s6 edge_offset", a.get_n_brt_nodes()));
}

// Canonical, index-permutation-independent octree edge set. Each entry is a
// downward edge keyed purely by geometry + octant + kind (0 = internal child via
// child_node_mask, 1 = leaf via child_leaf_mask). Coordinates are quantized to
// ints (the octree lattice is integral for kRange=1024, kMinCoord=0).
//
// Field 9 (`oct`, the parent's child-slot index) is part of the key for BOTH
// kinds; it is deterministic regardless of node-index permutation. For a LEAF
// edge the target field tcx (e[5]) carries the slot's resolved point index as a
// VALUE, not slot identity: two leaves of the same (octnode, octant) collide and
// the surviving point index is non-atomic last-writer-wins, so it can differ
// across backends. The leaf check therefore compares the slot KEY set (parent
// geometry + octant) with the value dropped -- see LeafSlotKey / CheckStage7Topology.
using OctEdge = std::array<long, 10>;  // {kind, pcx,pcy,pcz,pcell, tcx,tcy,tcz,tcell, oct}

// Strip the value-bearing target fields from a leaf edge, keeping only its slot
// identity (kind + parent geometry + octant). Two backends that disagree only on
// which point index won a contended (octnode, octant) leaf slot map to the same
// key, so a difference between the ref and out key sets is a real
// missing/spurious/misplaced leaf link, independent of last-writer-wins.
[[nodiscard]] inline OctEdge LeafSlotKey(const OctEdge& e) {
  return OctEdge{e[0], e[1], e[2], e[3], e[4], 0, 0, 0, 0, e[9]};
}

[[nodiscard]] inline std::vector<OctEdge> OctreeEdgeSet(
    std::span<const glm::vec4> corner,
    std::span<const float> cell,
    std::span<const std::int32_t> children,  // n*8
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
      OctEdge e{kind, pcx, pcy, pcz, pcell, 0, 0, 0, 0, c};
      if (kind == 0 && t >= 0 && static_cast<std::size_t>(t) < n) {
        // internal child: identify it by its OWN geometry (index-independent).
        e[5] = q(corner[t].x);
        e[6] = q(corner[t].y);
        e[7] = q(corner[t].z);
        e[8] = q(cell[t]);
      } else {
        // leaf: the value is a point index, not an octree node; carry the raw
        // value (point indices are the same input on every backend), but it is a
        // value, not slot identity -- see LeafSlotKey.
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
  std::span<const std::int32_t> ch_out(a.u_oct_children_s7_out.data(),
                                       a.u_oct_children_s7_out.size());
  std::span<const std::int32_t> nm_ref(a.u_oct_child_node_mask_s7.data(),
                                       a.u_oct_child_node_mask_s7.size());
  std::span<const std::int32_t> nm_out(a.u_oct_child_node_mask_s7_out.data(),
                                       a.u_oct_child_node_mask_s7_out.size());
  std::span<const std::int32_t> lm_ref(a.u_oct_child_leaf_mask_s7.data(),
                                       a.u_oct_child_leaf_mask_s7.size());
  std::span<const std::int32_t> lm_out(a.u_oct_child_leaf_mask_s7_out.data(),
                                       a.u_oct_child_leaf_mask_s7_out.size());

  auto ref = OctreeEdgeSet(a.u_oct_corner_s7, a.u_oct_cell_size_s7, ch_ref, nm_ref, lm_ref, n);
  auto out =
      OctreeEdgeSet(a.u_oct_corner_s7_out, a.u_oct_cell_size_s7_out, ch_out, nm_out, lm_out, n);
  if (ref == out) return ::testing::AssertionSuccess();

  // Break the difference down by edge kind. INTERNAL (kind 0) parent->child edges
  // are the octree's structural topology and MUST be order-independent; LEAF
  // (kind 1) slots can legitimately collide (two leaves of the same octnode+octant
  // resolved by non-atomic last-writer-wins differ across backends), so we report
  // them separately and only fail on an internal-topology divergence.
  auto split = [](const std::vector<OctEdge>& e, long kind) {
    std::vector<OctEdge> r;
    for (const auto& x : e)
      if (x[0] == kind) r.push_back(x);
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

  // Internal topology matches. Now validate the LEAF links (the entire output of
  // process_link_leaf -- which child[] slots resolve to which point indices), the
  // surface the previous oracle threw away. The slot VALUE (point index) is
  // legitimately last-writer-wins for a contended (octnode, octant) slot, so we
  // cannot compare raw leaf edges; but WHICH slots are leaf-populated -- keyed by
  // parent geometry + octant -- is deterministic. A residual difference in that
  // key set (after dropping the multiply-written value) is a real missing,
  // spurious, or misplaced leaf link (a wrong child[] index, leaf-mask bit, or
  // off-by-one leaf code), so we fail on it.
  auto slot_keys = [](const std::vector<OctEdge>& leaf) {
    std::vector<OctEdge> keys;
    keys.reserve(leaf.size());
    for (const auto& e : leaf) keys.push_back(LeafSlotKey(e));
    std::ranges::sort(keys);
    keys.erase(std::ranges::unique(keys).begin(), keys.end());  // collapse contended slots
    return keys;
  };
  const auto ref_keys = slot_keys(ref_leaf), out_keys = slot_keys(out_leaf);
  std::size_t leaf_only_ref = 0, leaf_only_out = 0;
  {
    std::vector<OctEdge> d1, d2;
    std::ranges::set_difference(ref_keys, out_keys, std::back_inserter(d1));
    std::ranges::set_difference(out_keys, ref_keys, std::back_inserter(d2));
    leaf_only_ref = d1.size();
    leaf_only_out = d2.size();
  }
  if (leaf_only_ref != 0 || leaf_only_out != 0) {
    return ::testing::AssertionFailure()
           << "tree s7 topology: LEAF child slots differ (" << leaf_only_ref << " only-in-ref, "
           << leaf_only_out << " only-in-out; ref_leaf_slots=" << ref_keys.size()
           << " out_leaf_slots=" << out_keys.size()
           << ") -- a real missing/spurious/misplaced leaf link (the slot key set, parent "
              "geometry + octant, is deterministic; only the resolved point index is "
              "last-writer-wins and is excluded)";
  }
  // Slot key sets match; the only residual difference is the resolved point index
  // of a contended (octnode, octant) slot (last-writer-wins, order-sensitive).
  if (ref_leaf == out_leaf) return ::testing::AssertionSuccess();
  return ::testing::AssertionSuccess()
         << "(note) leaf-slot VALUES differ by last-writer-wins on contended slots; "
            "leaf slot key set + internal topology identical";
}

inline void CheckStage7(const tree::SafeAppData& a) {
  const std::size_t n = a.get_n_octree_nodes();
  // child_node_mask is an OR-reduction: several brt nodes scatter children into
  // the same octree node's mask. With atomicOr in the shader (matching the
  // commutative |= the reference intends) the mask is order-INDEPENDENT and is a
  // stable differential target.
  EXPECT_TRUE(bt::testing::ExactEqual(
      a.u_oct_child_node_mask_s7, a.u_oct_child_node_mask_s7_out, "tree s7 node_mask", n));
  // Geometry IS now a deterministic oracle target. The octree builder used the
  // INCLUSIVE edge_offset prefix sum as each brt node's range START, which shifted
  // every range +edge_count[i] -> the root (node 0) had no writer and adjacent
  // brt ranges overlapped, so cell_size/corner were last-writer-wins (and node 0
  // a hole). Fixed by using the EXCLUSIVE start (edge_offsets[x]-edge_counts[x])
  // in process_oct_node / process_link_leaf on all three backends; a diagnostic
  // confirmed 0 holes and 0 multi-writer collisions afterward, so each octree
  // node now has exactly one geometry writer and the values are order-independent.
  EXPECT_TRUE(bt::testing::NearEqual(
      a.u_oct_cell_size_s7, a.u_oct_cell_size_s7_out, 1e-5f, 1e-6f, "tree s7 cell_size", n));
  EXPECT_TRUE(bt::testing::NearEqual(AsFloats(a.u_oct_corner_s7),
                                     AsFloats(a.u_oct_corner_s7_out),
                                     1e-5f,
                                     1e-6f,
                                     "tree s7 corner",
                                     n * 4));

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
    case 1:
      CheckStage1(base);
      break;
    case 2:
      CheckStage2(base);
      break;
    case 3:
      CheckStage3(base);
      break;
    case 4:
      CheckStage4(base);
      break;
    case 5:
      CheckStage5(base);
      break;
    case 6:
      CheckStage6(base);
      break;
    case 7:
      CheckStage7(base);
      break;
    default:
      FAIL() << "no such tree stage: " << s;
  }
}

// ----------------------------------------------------------------------------
// tree::AppData variant: genuinely-chained (no golden/_out split, see
// tree_appdata.hpp), used by the OMP/CUDA differential + pipeline-e2e suites so
// they exercise the SAME AppData path production profiling uses (Vulkan stays
// on the SafeAppData family above -- no chained path exists for it yet).
//
// There is no single instance to read golden-vs-_out from anymore, so instead
// we build TWO tree::AppData: `ref` runs the OMP oracle chain, `out` is
// whatever the Runner/backend under test produced -- both seeded with the SAME
// input points (copied onto `ref` post-construction, exactly as
// test_pipeline_chained.cpp's CheckItemChained already does). Field names match
// SafeAppData's golden fields 1:1 (just without the _out suffix), so the
// ExactEqual/NearEqual calls below are otherwise identical to the checks above.
// ----------------------------------------------------------------------------

inline void CheckStage1(const tree::AppData& ref, const tree::AppData& out) {
  std::vector<uint32_t> got(out.u_morton_keys_s1.begin(), out.u_morton_keys_s1.end());
  std::ranges::sort(got);
  EXPECT_TRUE(bt::testing::ExactEqual(ref.u_morton_keys_sorted_s2, got, "tree s1 morton(set)"));
}

inline void CheckStage2(const tree::AppData& ref, const tree::AppData& out) {
  EXPECT_TRUE(bt::testing::ExactEqual(
      ref.u_morton_keys_sorted_s2, out.u_morton_keys_sorted_s2, "tree s2 sort"));
}

inline void CheckStage3(const tree::AppData& ref, const tree::AppData& out) {
  ASSERT_EQ(ref.get_n_unique(), out.get_n_unique()) << "tree s3: ref/out disagree on unique count";
  ASSERT_GT(ref.get_n_unique(), 0u) << "tree s3: zero unique keys (compare would be vacuous)";
  EXPECT_TRUE(bt::testing::ExactEqual(ref.u_morton_keys_unique_s3,
                                      out.u_morton_keys_unique_s3,
                                      "tree s3 unique",
                                      ref.get_n_unique()));
}

inline void CheckStage4(const tree::AppData& ref, const tree::AppData& out) {
  ASSERT_EQ(ref.get_n_brt_nodes(), out.get_n_brt_nodes())
      << "tree s4: ref/out disagree on BRT node count";
  const std::size_t n = ref.get_n_brt_nodes();
  ASSERT_GT(n, 0u) << "tree s4: zero BRT nodes (compare would be vacuous)";
  EXPECT_TRUE(
      bt::testing::ExactEqual(ref.u_brt_prefix_n_s4, out.u_brt_prefix_n_s4, "tree s4 prefix_n", n));
  EXPECT_TRUE(bt::testing::ExactEqual(
      ref.u_brt_has_leaf_left_s4, out.u_brt_has_leaf_left_s4, "tree s4 leaf_left", n));
  EXPECT_TRUE(bt::testing::ExactEqual(
      ref.u_brt_has_leaf_right_s4, out.u_brt_has_leaf_right_s4, "tree s4 leaf_right", n));
  EXPECT_TRUE(bt::testing::ExactEqual(
      ref.u_brt_left_child_s4, out.u_brt_left_child_s4, "tree s4 left_child", n));
  EXPECT_TRUE(
      bt::testing::ExactEqual(ref.u_brt_parents_s4, out.u_brt_parents_s4, "tree s4 parents", n));
}

inline void CheckStage5(const tree::AppData& ref, const tree::AppData& out) {
  ASSERT_EQ(ref.get_n_brt_nodes(), out.get_n_brt_nodes())
      << "tree s5: ref/out disagree on BRT node count";
  ASSERT_GT(ref.get_n_brt_nodes(), 0u) << "tree s5: zero BRT nodes (compare would be vacuous)";
  EXPECT_TRUE(bt::testing::ExactEqual(
      ref.u_edge_count_s5, out.u_edge_count_s5, "tree s5 edge_count", ref.get_n_brt_nodes()));
}

inline void CheckStage6(const tree::AppData& ref, const tree::AppData& out) {
  ASSERT_EQ(ref.get_n_brt_nodes(), out.get_n_brt_nodes())
      << "tree s6: ref/out disagree on BRT node count";
  ASSERT_GT(ref.get_n_brt_nodes(), 0u) << "tree s6: zero BRT nodes (compare would be vacuous)";
  EXPECT_TRUE(bt::testing::ExactEqual(
      ref.u_edge_offset_s6, out.u_edge_offset_s6, "tree s6 edge_offset", ref.get_n_brt_nodes()));
}

[[nodiscard]] inline ::testing::AssertionResult CheckStage7Topology(const tree::AppData& ref_app,
                                                                    const tree::AppData& out_app,
                                                                    std::size_t n) {
  std::span<const std::int32_t> ch_ref(ref_app.u_oct_children_s7.data(),
                                       ref_app.u_oct_children_s7.size());
  std::span<const std::int32_t> ch_out(out_app.u_oct_children_s7.data(),
                                       out_app.u_oct_children_s7.size());
  std::span<const std::int32_t> nm_ref(ref_app.u_oct_child_node_mask_s7.data(),
                                       ref_app.u_oct_child_node_mask_s7.size());
  std::span<const std::int32_t> nm_out(out_app.u_oct_child_node_mask_s7.data(),
                                       out_app.u_oct_child_node_mask_s7.size());
  std::span<const std::int32_t> lm_ref(ref_app.u_oct_child_leaf_mask_s7.data(),
                                       ref_app.u_oct_child_leaf_mask_s7.size());
  std::span<const std::int32_t> lm_out(out_app.u_oct_child_leaf_mask_s7.data(),
                                       out_app.u_oct_child_leaf_mask_s7.size());

  auto ref =
      OctreeEdgeSet(ref_app.u_oct_corner_s7, ref_app.u_oct_cell_size_s7, ch_ref, nm_ref, lm_ref, n);
  auto out =
      OctreeEdgeSet(out_app.u_oct_corner_s7, out_app.u_oct_cell_size_s7, ch_out, nm_out, lm_out, n);
  if (ref == out) return ::testing::AssertionSuccess();

  // Break the difference down by edge kind. INTERNAL (kind 0) parent->child edges
  // are the octree's structural topology and MUST be order-independent; LEAF
  // (kind 1) slots can legitimately collide (two leaves of the same octnode+octant
  // resolved by non-atomic last-writer-wins differ across backends), so we report
  // them separately and only fail on an internal-topology divergence.
  auto split = [](const std::vector<OctEdge>& e, long kind) {
    std::vector<OctEdge> r;
    for (const auto& x : e)
      if (x[0] == kind) r.push_back(x);
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

  auto slot_keys = [](const std::vector<OctEdge>& leaf) {
    std::vector<OctEdge> keys;
    keys.reserve(leaf.size());
    for (const auto& e : leaf) keys.push_back(LeafSlotKey(e));
    std::ranges::sort(keys);
    keys.erase(std::ranges::unique(keys).begin(), keys.end());  // collapse contended slots
    return keys;
  };
  const auto ref_keys = slot_keys(ref_leaf), out_keys = slot_keys(out_leaf);
  std::size_t leaf_only_ref = 0, leaf_only_out = 0;
  {
    std::vector<OctEdge> d1, d2;
    std::ranges::set_difference(ref_keys, out_keys, std::back_inserter(d1));
    std::ranges::set_difference(out_keys, ref_keys, std::back_inserter(d2));
    leaf_only_ref = d1.size();
    leaf_only_out = d2.size();
  }
  if (leaf_only_ref != 0 || leaf_only_out != 0) {
    return ::testing::AssertionFailure()
           << "tree s7 topology: LEAF child slots differ (" << leaf_only_ref << " only-in-ref, "
           << leaf_only_out << " only-in-out; ref_leaf_slots=" << ref_keys.size()
           << " out_leaf_slots=" << out_keys.size()
           << ") -- a real missing/spurious/misplaced leaf link (the slot key set, parent "
              "geometry + octant, is deterministic; only the resolved point index is "
              "last-writer-wins and is excluded)";
  }
  if (ref_leaf == out_leaf) return ::testing::AssertionSuccess();
  return ::testing::AssertionSuccess()
         << "(note) leaf-slot VALUES differ by last-writer-wins on contended slots; "
            "leaf slot key set + internal topology identical";
}

inline void CheckStage7(const tree::AppData& ref, const tree::AppData& out) {
  ASSERT_EQ(ref.get_n_octree_nodes(), out.get_n_octree_nodes())
      << "tree s7: ref/out disagree on octree node count";
  const std::size_t n = ref.get_n_octree_nodes();
  EXPECT_TRUE(bt::testing::ExactEqual(
      ref.u_oct_child_node_mask_s7, out.u_oct_child_node_mask_s7, "tree s7 node_mask", n));
  EXPECT_TRUE(bt::testing::NearEqual(
      ref.u_oct_cell_size_s7, out.u_oct_cell_size_s7, 1e-5f, 1e-6f, "tree s7 cell_size", n));
  EXPECT_TRUE(bt::testing::NearEqual(AsFloats(ref.u_oct_corner_s7),
                                     AsFloats(out.u_oct_corner_s7),
                                     1e-5f,
                                     1e-6f,
                                     "tree s7 corner",
                                     n * 4));
  EXPECT_TRUE(CheckStage7Topology(ref, out, n));
}

// Drive: skip if the backend's device is absent, else run 1..s on `out`, then run
// the FULL 1..7 OMP chain on a fresh `ref` seeded with `out`'s own input, and diff
// stage s. `ref` always runs to completion (not just 1..s) because CheckStage1's
// reference target is a stage-2 field (the sorted golden) and CheckStage3+ read
// counts set by later stages -- mirroring how SafeAppData's golden above is
// always the fully-built pipeline result regardless of how far `_out` got
// dispatched. Unlike RunAndCheckStage above, both instances are ordinary
// tree::AppData -- there's no golden field to read off `out` itself.
template <class Runner>
inline void RunAndCheckStageAppData(int s) {
  if (!Runner::Available()) {
    GTEST_SKIP() << "backend device not available on this target";
  }
  Runner runner;
  typename Runner::AppData out(runner.Mr());
  for (int i = 1; i <= s; ++i) runner.RunStage(out, i);

  tree::AppData ref(std::pmr::new_delete_resource(), out.get_n_input());
  ref.u_input_points_s0 = out.u_input_points_s0;
  tree::omp::dispatch_multi_stage(ref, 1, 7);

  switch (s) {
    case 1:
      CheckStage1(ref, out);
      break;
    case 2:
      CheckStage2(ref, out);
      break;
    case 3:
      CheckStage3(ref, out);
      break;
    case 4:
      CheckStage4(ref, out);
      break;
    case 5:
      CheckStage5(ref, out);
      break;
    case 6:
      CheckStage6(ref, out);
      break;
    case 7:
      CheckStage7(ref, out);
      break;
    default:
      FAIL() << "no such tree stage: " << s;
  }
}

}  // namespace tree::testing

// Expand the 7 per-stage differential tests for a given backend Runner.
#define BT_DECLARE_TREE_DIFF_TESTS(SUITE, RUNNER)                                \
  TEST(SUITE, Stage1_Morton) { tree::testing::RunAndCheckStage<RUNNER>(1); }     \
  TEST(SUITE, Stage2_Sort) { tree::testing::RunAndCheckStage<RUNNER>(2); }       \
  TEST(SUITE, Stage3_Unique) { tree::testing::RunAndCheckStage<RUNNER>(3); }     \
  TEST(SUITE, Stage4_RadixTree) { tree::testing::RunAndCheckStage<RUNNER>(4); }  \
  TEST(SUITE, Stage5_EdgeCount) { tree::testing::RunAndCheckStage<RUNNER>(5); }  \
  TEST(SUITE, Stage6_EdgeOffset) { tree::testing::RunAndCheckStage<RUNNER>(6); } \
  TEST(SUITE, Stage7_Octree) { tree::testing::RunAndCheckStage<RUNNER>(7); }

// Same 7-test expansion, for a Runner whose AppData is tree::AppData (genuinely
// chained) rather than a SafeAppData family member -- see RunAndCheckStageAppData.
#define BT_DECLARE_TREE_DIFF_TESTS_APPDATA(SUITE, RUNNER)                               \
  TEST(SUITE, Stage1_Morton) { tree::testing::RunAndCheckStageAppData<RUNNER>(1); }     \
  TEST(SUITE, Stage2_Sort) { tree::testing::RunAndCheckStageAppData<RUNNER>(2); }       \
  TEST(SUITE, Stage3_Unique) { tree::testing::RunAndCheckStageAppData<RUNNER>(3); }     \
  TEST(SUITE, Stage4_RadixTree) { tree::testing::RunAndCheckStageAppData<RUNNER>(4); }  \
  TEST(SUITE, Stage5_EdgeCount) { tree::testing::RunAndCheckStageAppData<RUNNER>(5); }  \
  TEST(SUITE, Stage6_EdgeOffset) { tree::testing::RunAndCheckStageAppData<RUNNER>(6); } \
  TEST(SUITE, Stage7_Octree) { tree::testing::RunAndCheckStageAppData<RUNNER>(7); }
