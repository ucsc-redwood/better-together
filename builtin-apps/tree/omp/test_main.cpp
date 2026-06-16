#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory_resource>

#include "../../app.hpp"
#include "../testing/tree_invariants.hpp"
#include "../testing/tree_octree_ref.hpp"
#include "../testing/tree_ref.hpp"
#include "../tree_diff_oracle.hpp"
#include "dispatchers.hpp"
#include "func_brt.hpp"
#include "func_edge.hpp"
#include "func_octree.hpp"

// ----------------------------------------------------------------------------
// Tree × OMP differential oracle. OMP is the reference, so this run is a
// self-consistency check: dispatching each stage must reproduce the golden
// computed at SafeAppData construction (same kernels), proving the harness and
// the per-stage comparisons before CUDA/Vulkan adopt the identical BT_DECLARE
// expansion. Always available (every target has a CPU).
// ----------------------------------------------------------------------------

namespace {
struct OmpTreeRunner {
  using AppData = tree::SafeAppData;
  static constexpr bool Available() { return true; }
  static std::pmr::memory_resource* Mr() { return std::pmr::new_delete_resource(); }
  void RunStage(tree::SafeAppData& a, int stage) { tree::omp::dispatch_stage(a, stage); }
};
}  // namespace

BT_DECLARE_TREE_DIFF_TESTS(TreeDiffOmp, OmpTreeRunner)

// ----------------------------------------------------------------------------
// INDEPENDENT correctness anchors for the OMP golden (stages 1/4/5/7).
//
// The BT_DECLARE tests above compare _out vs the golden, but for stages 1/4/5/7
// the golden and the dispatcher call the SAME pure function, so that comparison
// is a tautology. These anchor tests validate the GOLDEN itself against a
// reference / invariants / tiny brute-force that are INDEPENDENT of the kernel
// under test, giving the (shared) golden a real ground truth -- which the GPU
// differential tests then inherit. A failure here is a real OMP bug.
//
// The golden lives in SafeAppData's const buffers (built once at construction
// from HostTreeManager::initialize()); we only read it here.
// ----------------------------------------------------------------------------
namespace {

const tree::SafeAppData& Golden() {
  static tree::SafeAppData a(std::pmr::new_delete_resource());
  return a;
}

// Stage 1: the multiset of golden morton keys must equal the multiset produced
// by an independent (libmorton naive-loop) encoder over the input points.
// Compared as a multiset because morton keys are an unordered set over the
// points, and the golden u_morton_keys_s1 is sorted in place at stage 2; the
// sorted golden lives in u_morton_keys_sorted_s2 (the same reference the diff
// oracle's CheckStage1 uses).
TEST(TreeAnchorOmp, Stage1_MortonRef) {
  const auto& a = Golden();
  const int n = static_cast<int>(a.get_n_input());
  std::vector<uint32_t> ref(n);
  for (int i = 0; i < n; ++i) {
    ref[i] = tree::testing::MortonRef(a.u_input_points_s0[i], tree::kMinCoord, tree::kRange);
  }
  std::ranges::sort(ref);
  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(ref[i], a.u_morton_keys_sorted_s2[i])
        << "morton golden(sorted) != libmorton reference(sorted) at rank " << i;
  }
}

// Stage 4: golden radix tree must satisfy the Karras structural invariants, and
// a tiny brute-force (run on the first prefix of the real, sorted-unique codes)
// must match prefix_n and the split index exactly.
TEST(TreeAnchorOmp, Stage4_RadixTreeInvariants) {
  const auto& a = Golden();
  const int n_unique = static_cast<int>(a.get_n_unique());
  const int n_brt = static_cast<int>(a.get_n_brt_nodes());
  const uint32_t* codes = a.u_morton_keys_unique_s3.data();

  EXPECT_TRUE(tree::testing::SortedUnique(codes, n_unique));
  EXPECT_TRUE(tree::testing::RadixTreeInvariants(codes,
                                                 n_brt,
                                                 n_unique,
                                                 a.u_brt_prefix_n_s4.data(),
                                                 a.u_brt_has_leaf_left_s4.data(),
                                                 a.u_brt_has_leaf_right_s4.data(),
                                                 a.u_brt_left_child_s4.data(),
                                                 a.u_brt_parents_s4.data()));
}

TEST(TreeAnchorOmp, Stage4_RadixTreeBruteForce) {
  const auto& a = Golden();
  const int n_unique = static_cast<int>(a.get_n_unique());
  const int n_brt = static_cast<int>(a.get_n_brt_nodes());
  const uint32_t* codes = a.u_morton_keys_unique_s3.data();

  // O(n^2) brute force over a small prefix of the REAL codes (trivially correct
  // at this size; exhaustive over all nodes would be O(n^2) on ~300k nodes).
  const int m = std::min(n_brt, 256);
  for (int i = 0; i < m; ++i) {
    const auto bf = tree::testing::BrtNodeBruteForce(i, codes, n_unique);
    EXPECT_EQ(bf.prefix_n, static_cast<int>(a.u_brt_prefix_n_s4[i]))
        << "brute-force prefix_n mismatch at brt node " << i;
    EXPECT_EQ(bf.gamma, a.u_brt_left_child_s4[i])
        << "brute-force split index mismatch at brt node " << i;
  }
}

// Stage 5: golden edge_count must equal the octree level-difference recomputed
// from the stage-4-ANCHORED prefix_n/parents, with the explicit root=0 rule.
TEST(TreeAnchorOmp, Stage5_EdgeCountRef) {
  const auto& a = Golden();
  const int n_brt = static_cast<int>(a.get_n_brt_nodes());
  for (int i = 0; i < n_brt; ++i) {
    const int ref =
        tree::testing::EdgeCountRef(i, a.u_brt_prefix_n_s4.data(), a.u_brt_parents_s4.data());
    ASSERT_EQ(ref, a.u_edge_count_s5[i])
        << "edge_count golden != level-difference reference at brt node " << i;
  }
}

// Stage 7: golden octree geometry must satisfy structural invariants (cell size
// a power-of-two fraction of range, corners on the lattice, child cells half &
// contained in parent cells).
TEST(TreeAnchorOmp, Stage7_OctreeInvariants) {
  const auto& a = Golden();
  const int n_oct = static_cast<int>(a.get_n_octree_nodes());
  // DIAG: dump root + its declared children geometry
  {
    const auto* cs = a.u_oct_cell_size_s7.data();
    const auto* corner = a.u_oct_corner_s7.data();
    const auto* mask = a.u_oct_child_node_mask_s7.data();
    const auto* ch = reinterpret_cast<const int(*)[8]>(a.u_oct_children_s7.data());
    fprintf(stderr, "STAGE7 DIAG: n_oct=%d  root cell_size=%g corner=(%g,%g,%g) mask=%d\n",
            n_oct, cs[0], corner[0].x, corner[0].y, corner[0].z, mask[0]);
    for (int c = 0; c < 8; ++c) if (mask[0] & (1<<c)) {
      int cv = ch[0][c];
      fprintf(stderr, "   slot %d -> oct %d  cell_size=%g corner=(%g,%g,%g)\n",
              c, cv, (cv>=0&&cv<n_oct)?cs[cv]:-1, (cv>=0&&cv<n_oct)?corner[cv].x:-1,
              (cv>=0&&cv<n_oct)?corner[cv].y:-1, (cv>=0&&cv<n_oct)?corner[cv].z:-1);
    }
    // also: how many octnodes have cell_size larger than the root's?
    int bigger=0, maxidx=0; float maxcs=0;
    for(int v=0;v<n_oct;++v){ if(cs[v]>cs[0]) ++bigger; if(cs[v]>maxcs){maxcs=cs[v];maxidx=v;} }
    fprintf(stderr, "   octnodes with cell_size>root: %d ; max cell_size=%g at %d\n", bigger, maxcs, maxidx);
  }
  EXPECT_TRUE(tree::testing::OctreeGeometryInvariants(
      a.u_oct_corner_s7.data(),
      a.u_oct_cell_size_s7.data(),
      a.u_oct_child_node_mask_s7.data(),
      reinterpret_cast<const int(*)[8]>(a.u_oct_children_s7.data()),
      n_oct,
      tree::kMinCoord,
      tree::kRange));
}

// ----------------------------------------------------------------------------
// DECISIVE tiny brute-force octree test.
//
// Runs the REPO's actual stage-4..7 functions (process_radix_tree_i,
// process_edge_count_i, partial_sum, process_oct_node) on a TINY hand-built set
// of sorted-unique morton codes, then compares the resulting children/
// child_node_mask against an INDEPENDENT brute-force octree (tree_octree_ref)
// built from the same codes by the obvious prefix-branching definition.
//
// Matching is by GEOMETRY (corner), since node indexing differs. The brute-force
// gives the CORRECT parent->child edges (a child cell is contained in and half
// the size of its parent). If the repo's edges disagree, it is a real linking
// bug; if they agree, the production octree is correct and the invariant is
// over-strict.
// ----------------------------------------------------------------------------

// Run the repo's stage 4..7 on a tiny sorted-unique morton-code array and return
// the produced octree arrays.
struct TinyOct {
  int n_oct = 0;
  std::vector<int> children;          // n_oct*8
  std::vector<glm::vec4> corner;      // n_oct
  std::vector<float> cell_size;       // n_oct
  std::vector<int> child_node_mask;   // n_oct
};

TinyOct RunRepoOctreeTiny(const std::vector<uint32_t>& codes) {
  const int n_unique = static_cast<int>(codes.size());
  const int n_brt = n_unique - 1;

  std::vector<uint8_t> prefix_n(n_unique, 0), hll(n_unique, 0), hlr(n_unique, 0);
  std::vector<int32_t> left_child(n_unique, 0), parents(n_unique, 0);

  // Stage 4 (single-threaded, deterministic, exactly as the golden).
  for (int i = 0; i < n_brt; ++i) {
    tree::omp::v1::process_radix_tree_i(
        i, n_brt, codes.data(), prefix_n.data(), hll.data(), hlr.data(),
        left_child.data(), parents.data());
  }

  // Stage 5.
  std::vector<int32_t> edge_count(n_brt, 0);
  for (int i = 0; i < n_brt; ++i) {
    tree::omp::v1::process_edge_count_i(i, prefix_n.data(), parents.data(), edge_count.data());
  }

  // Stage 6 (inclusive prefix sum), as in HostTreeManager::initialize().
  std::vector<int32_t> edge_offset(n_brt, 0);
  std::partial_sum(edge_count.data(), edge_count.data() + n_brt, edge_offset.data());
  const int n_oct = n_brt > 0 ? edge_offset[n_brt - 1] : 0;

  TinyOct out;
  out.n_oct = n_oct;
  out.children.assign(n_oct * 8, 0);
  out.corner.assign(n_oct, glm::vec4(0));
  out.cell_size.assign(n_oct, 0.0f);
  out.child_node_mask.assign(n_oct, 0);

  // Stage 7 (skip root brt node 0, exactly as the golden), single-threaded.
  for (int i = 1; i < n_brt; ++i) {
    tree::omp::process_oct_node(
        i,
        reinterpret_cast<int(*)[8]>(out.children.data()),
        out.corner.data(),
        out.cell_size.data(),
        out.child_node_mask.data(),
        edge_offset.data(),
        edge_count.data(),
        codes.data(),
        prefix_n.data(),
        parents.data(),
        tree::kMinCoord,
        tree::kRange);
  }
  return out;
}

// Find the repo octnode whose corner AND cell_size match. Returns -1 if none.
// (corner alone is ambiguous: a chain of nested cells can share corner (0,0,0)).
int FindRepoNode(const TinyOct& o, float cx, float cy, float cz, float cell, float eps = 1e-2f) {
  for (int v = 0; v < o.n_oct; ++v) {
    if (std::abs(o.corner[v].x - cx) < eps && std::abs(o.corner[v].y - cy) < eps &&
        std::abs(o.corner[v].z - cz) < eps &&
        std::abs(o.cell_size[v] - cell) < 1e-3f * cell) {
      return v;
    }
  }
  return -1;
}

TEST(TreeAnchorOmp, Stage7_TinyBruteForceOctree) {
  // A tiny, hand-chosen sorted-unique morton-code set. Mix of shallow and deep
  // branching so the parent->child geometry is non-trivial. (Values < 2^30.)
  // Two codes share a deep prefix (force a deep internal chain); others branch
  // near the top (force shallow internal nodes) -- exactly the situation the
  // failing invariant flags (node 0 deep, children shallow).
  std::vector<uint32_t> codes = {
      0x00000001u,  // octant-0 chain, deep
      0x00000005u,
      0x00000007u,
      0x08000000u,  // octant 1 at top level (bit 27)
      0x10000000u,  // octant 2 at top level
      0x18000000u,  // octant 3
      0x20000000u,  // octant 4
      0x28000000u,  // octant 5
  };
  std::ranges::sort(codes);
  codes.erase(std::unique(codes.begin(), codes.end()), codes.end());
  ASSERT_TRUE(tree::testing::SortedUnique(codes.data(), static_cast<int>(codes.size())));

  const TinyOct o = RunRepoOctreeTiny(codes);
  const auto ref = tree::testing::BuildBruteForceOctree(codes, tree::kMinCoord, tree::kRange);
  ASSERT_GT(o.n_oct, 0);
  ASSERT_FALSE(ref.empty());

  fprintf(stderr, "\n[TINY] repo octree: n_oct=%d\n", o.n_oct);
  for (int v = 0; v < o.n_oct; ++v) {
    fprintf(stderr, "  repo oct %d: corner=(%g,%g,%g) cell=%g mask=%d children=[",
            v, o.corner[v].x, o.corner[v].y, o.corner[v].z, o.cell_size[v],
            o.child_node_mask[v]);
    for (int c = 0; c < 8; ++c) fprintf(stderr, "%d%s", o.children[v * 8 + c], c < 7 ? "," : "");
    fprintf(stderr, "]\n");
  }
  fprintf(stderr, "[TINY] brute-force octree (ground truth): %zu internal nodes\n", ref.size());
  for (const auto& rn : ref) {
    fprintf(stderr, "  ref node lvl=%d corner=(%g,%g,%g) cell=%g  edges:", rn.level,
            rn.corner[0], rn.corner[1], rn.corner[2], rn.cell_size);
    for (auto& [oct, cp] : rn.child_internal) fprintf(stderr, " oct%d", oct);
    fprintf(stderr, "\n");
  }

  // ---- Decisive comparison 1: every repo child link must be geometrically
  // valid (child cell = half parent cell). This is the textbook octree property
  // and is INDEPENDENT of the repo's node indexing. Any failure is a wrong link.
  int repo_links = 0, linking_violations = 0;
  for (int v = 0; v < o.n_oct; ++v) {
    const float pcell = o.cell_size[v];
    for (int c = 0; c < 8; ++c) {
      if (!(o.child_node_mask[v] & (1 << c))) continue;
      const int cv = o.children[v * 8 + c];
      if (cv < 0 || cv >= o.n_oct) continue;
      ++repo_links;
      const float ccell = o.cell_size[cv];
      if (std::abs(ccell * 2.0f - pcell) > 1e-3f * pcell) {
        ++linking_violations;
        fprintf(stderr,
                "[TINY] VIOLATION: repo node %d (corner=(%g,%g,%g) cell=%g) child slot %d -> node %d"
                " has cell=%g (NOT parent/2=%g) -- backwards/wrong link\n",
                v, o.corner[v].x, o.corner[v].y, o.corner[v].z, pcell, c, cv, ccell, pcell / 2);
      }
    }
  }

  // ---- Decisive comparison 2: for each CORRECT parent->internal-child edge in
  // the ground truth, the repo must contain that exact edge (parent corner+cell
  // -> a child slot pointing to a node with the child's corner+cell).
  int gt_edges = 0, gt_edges_present = 0;
  for (const auto& rn : ref) {
    const int pv = FindRepoNode(o, rn.corner[0], rn.corner[1], rn.corner[2], rn.cell_size);
    for (auto& [oct, cprefix] : rn.child_internal) {
      ++gt_edges;
      float ccorner[3];
      tree::testing::DecodePrefixCorner(cprefix, tree::kMinCoord, tree::kRange, ccorner);
      // the child's level = first deeper branching level; its cell is strictly
      // < parent cell. We don't know its exact level a priori from prefix alone,
      // so match the repo child by corner with cell == parent/2 OR any smaller
      // power-of-two cell consistent with a descendant. The textbook child cell
      // is parent/2.
      bool present = false;
      if (pv >= 0) {
        for (int c = 0; c < 8; ++c) {
          if (!(o.child_node_mask[pv] & (1 << c))) continue;
          const int cv = o.children[pv * 8 + c];
          if (cv < 0 || cv >= o.n_oct) continue;
          if (std::abs(o.corner[cv].x - ccorner[0]) < 1e-2f &&
              std::abs(o.corner[cv].y - ccorner[1]) < 1e-2f &&
              std::abs(o.corner[cv].z - ccorner[2]) < 1e-2f &&
              o.cell_size[cv] < rn.cell_size) {
            present = true;
            break;
          }
        }
      }
      if (present) ++gt_edges_present;
      else
        fprintf(stderr,
                "[TINY] MISSING EDGE: ground-truth parent (corner=(%g,%g,%g) cell=%g) -> child "
                "octant %d (corner=(%g,%g,%g)) is NOT present as a valid downward link in repo "
                "(repo parent node idx=%d)\n",
                rn.corner[0], rn.corner[1], rn.corner[2], rn.cell_size, oct, ccorner[0], ccorner[1],
                ccorner[2], pv);
    }
  }

  fprintf(stderr,
          "[TINY] repo_links=%d linking_violations=%d | ground_truth_edges=%d present=%d\n",
          repo_links, linking_violations, gt_edges, gt_edges_present);

  EXPECT_EQ(linking_violations, 0)
      << linking_violations
      << " repo child links have child_cell != parent/2 (geometrically backwards links).";
  EXPECT_EQ(gt_edges_present, gt_edges)
      << "Repo octree is missing " << (gt_edges - gt_edges_present)
      << " of the brute-force ground truth's correct parent->child edges.";
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  // Run the oracle deterministically: the radix-tree (stage 4) and octree
  // (stage 7) builds have cross-node writes whose result is order-sensitive
  // under parallel `omp for`, so a multi-threaded reference is not bitwise
  // stable run-to-run. A correctness oracle must be deterministic; speed is not
  // under test here. (Cross-backend GPU comparison of these structural stages
  // will still need invariant/canonical checks — see docs/TESTING.md.)
  omp_set_num_threads(1);
  return RUN_ALL_TESTS();
}
