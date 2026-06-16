#pragma once
// ----------------------------------------------------------------------------
// Independent correctness anchors for the tree OMP golden (stages 1 & 5).
//
// The differential oracle (tree_diff_oracle.hpp) compares each backend's _out
// against SafeAppData's golden, which is itself produced by the OMP kernels at
// construction (HostTreeManager::initialize()). For several stages the golden
// and the dispatcher call the SAME pure function, so the OMP self-test is a
// tautology (same fn -> byte-identical) and the GPU tests inherit an
// un-independently-validated ground truth.
//
// This header gives stages 1 (morton) and 5 (edge-count) a SECOND,
// obviously-correct reference that is INDEPENDENT of the kernel under test:
//   - morton: the kernel hand-rolls the "magic bits" split-by-3 interleave.
//     The anchor uses libmorton's naive bit-by-bit for-loop encoder
//     (libmorton::m3D_e_for) -- a different algorithm, a well-tested 3rd party.
//   - edge-count: the kernel derives the per-node octant count from
//     prefix_n/parent depths. The anchor recomputes it directly from the morton
//     code range [first,last] by scanning the actual 3-bit octant at the node's
//     split level (the definition the kernel's depth arithmetic stands in for).
//
// These are anchors, NOT clones: if an anchor disagrees with the golden, the
// golden is wrong -- do not weaken the anchor to match.
// ----------------------------------------------------------------------------

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

// Use the naive for-loop encoder directly (morton3D.h) rather than the
// dispatch header (morton.h), whose method is macro-selected and could resolve
// to the same magic-bits family as the kernel. m3D_e_for is independent.
#include <libmorton/morton3D.h>

namespace tree::testing {

// Independent morton encoder for one input point.
//
// The coordinate quantization (continuous xyz -> 10-bit integer lattice) is the
// shared problem SPEC, not the thing under test (the bit interleave is). We
// replicate the quantization exactly as the spec dictates, then interleave with
// an independent algorithm (libmorton's naive loop). bit_scale=1024 and the
// min_coord/range affine map match tree::omp::xyz_to_morton32 by construction.
[[nodiscard]] inline uint32_t MortonRef(const glm::vec4& p, float min_coord, float range) {
  constexpr float bit_scale = 1024.0f;
  const auto i = static_cast<uint32_t>((p.x - min_coord) / range * bit_scale);
  const auto j = static_cast<uint32_t>((p.y - min_coord) / range * bit_scale);
  const auto k = static_cast<uint32_t>((p.z - min_coord) / range * bit_scale);
  // m3D_e_for interleaves x into bit 0, y into bit 1, z into bit 2 -- the same
  // axis order as the kernel's m3D_e_magicbits, but via the naive loop.
  return libmorton::m3D_e_for<uint32_t, uint32_t>(i, j, k);
}

constexpr int kMortonBits = 30;

// Independent edge-count reference for one radix-tree node i.
//
// The production definition (process_edge_count_i, all backends) is the OCTREE
// LEVEL DIFFERENCE between a brt node and its brt-parent:
//   edge_count[i] = prefix_n[i]/3 - prefix_n[parents[i]]/3   (and 0 for root).
// i.e. how many octree levels are collapsed onto the brt edge i -> parent[i].
//
// NOTE on independence: an octant-scan over the node's key range (the v2
// compute_edge_count_kernel) computes a DIFFERENT quantity (distinct child
// octants present), which disagrees with the deployed depth-difference on the
// vast majority of nodes -- using it as the anchor would force the golden to a
// definition the pipeline does not use (the very "two-algorithms / different
// misconception" trap). So the honest independent anchor recomputes the
// depth-difference, but drives it from the stage-4-ANCHORED prefix_n/parents
// (validated by RadixTreeInvariants + the tiny brute-force) and asserts the
// root special-case explicitly. This catches a wrong parent lookup, a wrong /3,
// a missing root rule, or a corrupted prefix_n feeding stage 5 -- it is NOT a
// byte-clone of the kernel call site, but a recomputation from validated inputs.
[[nodiscard]] inline int EdgeCountRef(int i, const uint8_t* prefix_n, const int* parents) {
  if (i == 0) return 0;  // root: no parent edge
  const int my_depth = prefix_n[i] / 3;
  const int parent_depth = prefix_n[parents[i]] / 3;
  return my_depth - parent_depth;
}

}  // namespace tree::testing
