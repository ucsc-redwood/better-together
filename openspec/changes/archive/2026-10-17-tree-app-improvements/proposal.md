# Proposal

## Why

A review of the tree app's GPU kernels found two kinds of problems. The correctness bugs come first: they're invisible to today's tests, and the performance work changes the same kernels.

### A. Correctness (hidden from today's differential tests)
1. **Morton encoding wraps at the upper bound.** All three backends quantize with `(x - min) / range * 1024` cast to `uint`, then mask with `& 0x3ff` while spreading bits:
   - Vulkan `tree_morton.comp`;
   - OMP `apps/tree/omp/func_morton.hpp:34-37`;
   - CUDA `apps/tree/cuda/func_morton.cuh:28-31`.

   A coordinate at the top of the range gives 1024, which masks to 0, so the point lands at the opposite edge. The inputs are drawn from `[kMinCoord, kMinCoord + kRange)`, so this only fires at the boundary or on float rounding. It's latent, and the oracle shares it.
2. **Leaf-mask updates race.** `set_leaf` clears a bit with a non-atomic `&=` inside parallel loops:
   - CUDA `apps/tree/cuda/07_octree.cu:33`, in the grid-stride `k_LinkLeafNodes`;
   - OMP `apps/tree/omp/func_octree.hpp:35`, under `#pragma omp for` in `run_stage_7`.

   Several BRT nodes can link leaves into the same octree node, so bits can be lost. The neighbouring `set_child` was already made atomic for exactly this reason (`func_octree.hpp:21-24`).
3. **Vulkan stage 7 never links leaves.** `tree_build_octree.comp`'s `main` only calls `k_MakeOctNodes`; `k_LinkLeafNodes` is dead code. Yet `TreeDiffVulkan.Stage7_Octree`, whose `CheckStage7Topology` includes a leaf-slot comparison, passes (confirmed locally 2026-10-15, not skipped). Why is not yet explained (to verify). The oracle should catch missing leaf links and, apparently, doesn't here.

### B. Performance (tree stages 2–3; the numbers below were measured on rocky-ryzen / Radeon 780M before this change and motivate it, while its gates now run on the phones)
- **Stage 2, the radix sort:** 0.785 ms for n = 307,200 keys, half of tree's ≈1.57 ms GPU total. The floor is ≈0.18 ms (4 passes × 12 B/key over DRAM). Two causes:
  - In every scatter pass, each of the 256 workgroups re-reads the whole 256 × 256 histogram table (≈64 MB of L2 per pass, plus a 256-iteration serial loop per thread), while each workgroup sorts only ≈1,200 keys.
  - There are 8 dispatches with 7 global barriers.
- **Stage 3, compaction:** 5 dispatches (find_dups, a 3-kernel device scan, move_dups), 0.179 ms in total, each kernel 20–40 µs. It's dispatch- and barrier-bound.

## What Changes

### A. Correctness (first)
- **Reproducer tests first,** each failing before its fix:
  - a point at the coordinate upper bound;
  - several leaf links into one node in parallel;
  - an oracle check that detects missing leaf links. That check also explains, or exposes, the Vulkan stage-7 pass.
- **Morton:** clamp the quantized value to `[0, 1023]` on OMP, CUDA and Vulkan together. **BREAKING (numerical):** codes change only for points that previously wrapped.
- **Atomic leaf-mask clear on every backend:** `#pragma omp atomic`, CUDA `atomicAnd`, Vulkan `atomicAnd`.
- **Vulkan stage 7 links leaves** (design A3 decides how: second dispatch vs in-kernel second phase).

### B. Performance (after A)
- **Quick knob first:** `kRadixNumWorkgroups` 256 → 32/64/128 (`apps/tree/vulkan/vk_appdata.hpp`, both appdata classes).
- **Global prefix computed once per pass,** or one upfront all-digit histogram (reduce-then-scan).
- **LDS local sort before scatter,** for coalesced writes.
- **Subgroup-aggregated histogram atomics.**
- **`REQUIRE_FULL_SUBGROUPS`** for the subgroup-dependent sort kernels (needs the optional `computeFullSubgroups` feature).
- **Stage-3 compaction in fewer than 5 dispatches.**

## Capabilities

### New Capabilities
- `tree-app/morton-codes`: how points are quantized and encoded into 30-bit Morton codes, identically on every backend, including at the range boundary.
- `tree-app/octree-build`: the stage-7 contract: atomic mask updates, leaf links on every backend, and an oracle that detects missing leaf links.
- `gpu-kernels/tree-sort`: the device-wide sort (stage 2) and compaction (stage 3) kernels: exact oracle match at subgroup 16, 32 and 64, full subgroups where guaranteed, and performance bounds.

### Modified Capabilities
<!-- none. The engine side of `computeFullSubgroups` (enabling the optional feature, the
     Algorithm flag) belongs to `vk-engine-runtime`, which runs first; this change only relies on it. -->

## Impact

- **Code:**
  - `apps/tree/omp/{func_morton.hpp,func_octree.hpp}`;
  - `apps/tree/cuda/{func_morton.cuh,07_octree.cu}`;
  - `platform/engine/vulkan/shaders/comp/{tree_morton,tree_build_octree,multi_radixsort,multi_radixsort_histograms,tree_find_dups,tree_scan_*,tree_move_dups}.comp` (compiled at build time);
  - `apps/tree/vulkan/{dispatchers.cpp,vk_appdata.hpp}`;
  - `apps/tree/tree_diff_oracle.hpp` and the tree tests;
  - `platform/engine/vulkan/{base_engine,algorithm}.{hpp,cpp}` (full-subgroups flag).
- **Gates:**
  - local `ctest -L omp`;
  - rocky-ryzen `ctest -L vulkan` plus the tree suites at subgroup 64 and forced 32 (correctness only, nothing is profiled there);
  - performance: per-stage A/B on the Pixel 7a and the Samsung against a recorded baseline;
  - Pixel 7a (16) and Samsung (64 and forced 32);
  - `ctest -L cuda` on duck-stable for the CUDA fixes.
- **Profiling:** stage 7 gets slower on VK (it does more work), and stages 2–3 get faster. Refresh the tree × VK profile on the phones after landing.
