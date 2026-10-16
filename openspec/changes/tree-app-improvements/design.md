# Design

## Context

See proposal.md (Why). The facts below were read from the code on 2026-10-15.

- **The golden is OMP, computed at runtime.** `HostTreeManager::initialize` (`apps/tree/safe_tree_appdata.cpp:46-91`) runs `tree::omp::run_stage_1..7`. `run_stage_7` (`apps/tree/omp/dispatchers.cpp`) loops BRT nodes under `#pragma omp for`, calling `process_oct_node` and then `process_link_leaf`. There are no stored goldens.
- **`CheckStage7Topology` (`apps/tree/tree_diff_oracle.hpp`)** canonicalises edges by geometry:
  - internal edges must match;
  - leaf slots are compared as a set of (parent geometry, octant) keys, with the point index excluded as last-writer-wins.

  A leaf slot is defined as a ZERO bit in `child_leaf_mask` (the mask is initialised all-ones and cleared on link).
- **Vulkan `_out` leaf mask allocation.** `u_oct_child_leaf_mask_s7_out` is allocated in `SafeAppData` (`safe_tree_appdata.cpp:144`). Its initial value, and whether Vulkan writes it at all, still has to be checked: that's how the Vulkan pass must be explained (to verify).
- **Sort structure.** 4 LSD passes × (`multi_radixsort_histograms` → `multi_radixsort`), with `kRadixNumWorkgroups = 256` and `blocks_per_workgroup = ceil(ceil(n/256)/256)`. The scatter's threads with `x < 256` loop over all workgroups' histograms; the per-block ranking uses `bin_flags` + `bitCount`; the writes are random.
- **Stage 3.** `record_stage_3` runs `find_dups` → `record_device_scan` (3 dispatches) → `move_dups`. `record_device_scan` is shared with stage 6.
- **Subgroups (Phase B).** Pixel 7a fixed at 16; rocky-ryzen and Samsung default 64, range 32..64. Pinned via `requiredSubgroupSize`; `BT_VK_SUBGROUP_SIZE` forces another supported size.

## Goals / Non-Goals

**Goals:**
- A: boundary-safe Morton codes, race-free mask updates, Vulkan leaf links, and an oracle that catches missing leaf links.
- B: stage 2 at ≤ 0.40 ms on rocky-ryzen; stage 3 in fewer than 5 dispatches and no slower. Both exact at subgroup 16, 32 and 64.

**Non-Goals:**
- Changing the key width (30-bit Morton in `uint`), the digit order seen by stage 4, or the stage boundaries.
- Making the leaf-slot *value* (which point a contended slot resolves to) deterministic. That stays last-writer-wins and excluded from the oracle.
- Stage 4–6 kernels, apart from keeping stage 6 exact if `record_device_scan` changes.

## Decisions — A. Correctness

### A1. Reproducers before fixes
Each fix lands with a test that fails without it:
- **Morton:** encode synthetic points at `kMinCoord + kRange` and at a value whose scaled product rounds to 1024.0, on every backend.
- **Leaf race:** a constructed input where many BRT nodes link into the same octree node. Run the CUDA/OMP stage 7 repeatedly and compare masks against a serial reference.
- **Oracle:** feed `CheckStage7Topology` an output whose leaf masks were reset to "no leaves". The check must fail. If it doesn't, the check is vacuous and is fixed first.

### A2. Clamp, not widen
Clamp the quantized integer to `[0, 1023]` (`min(uint(v), 1023u)`) after the cast, identically on all three backends, so the oracle and the GPU codes stay equal. Using 1023.999 as the scale was considered and rejected: it changes every code, not just the wrapping ones.

### A3. Atomic leaf-mask clear; Vulkan links leaves in a second dispatch
- **Atomic clear:** `set_leaf`'s clear becomes atomic: `#pragma omp atomic` on OMP, `atomicAnd` on CUDA and in the shader's `SetLeaf`. The child slot store stays plain; it's last-writer-wins by contract.
- **Vulkan leaf linking:** a separate dispatch after a compute→compute barrier, reusing the same shader through a push-constant phase selector, or a second entry. The leaf pass reads masks and offsets that the `MakeOctNodes` pass of *other* workgroups wrote, so a barrier inside the dispatch isn't enough.
  - *Alternative:* both passes in one dispatch with a grid-wide barrier. Rejected: Vulkan has no grid barrier.

## Decisions — B. Performance

### B1. Measure the cheap knob first
Sweep `kRadixNumWorkgroups` ∈ {32, 64, 128, 256} and keep the best. The O(WG²) histogram re-read falls quadratically. The sweep shows how much of the 0.785 ms is re-read versus scatter or dispatch, which decides whether B2–B4 are needed.

### B2. Global prefix once per pass (reduce-then-scan)
Add a small per-pass scan of the 256 × WG table into per-workgroup offsets, or compute all 4 digits' histograms in one upfront read (FidelityFX ParallelSort structure). The scatter then reads only its own offsets.

### B3. Not Onesweep
Onesweep's decoupled look-back needs forward progress between workgroups, which Vulkan doesn't guarantee. A waiting workgroup can deadlock on a never-scheduled predecessor, a particular risk on Mali (to verify). Reduce-then-scan is portable at the cost of one extra key read per pass.

### B4. LDS local sort and aggregated atomics
- **Local sort:** rank keys by digit within each workgroup's block (reusing the `bin_flags` ranking), then write in digit order, so writes within a bin run are contiguous.
- **Aggregated atomics:** in the histogram kernel, combine equal bins per subgroup via `GL_KHR_shader_subgroup_ballot` before the shared `atomicAdd`.

### B5. Full subgroups
- **Engine:** owned by `vk-engine-runtime`, which runs before this change. It enables `computeFullSubgroups` as an optional feature and gives every subgroup-size-dependent pipeline `REQUIRE_FULL_SUBGROUPS` (its `vulkan-engine/pipeline-config` delta). This change does not touch the engine side.
- **Pipelines:** the sort kernels opt in by being subgroup-size-dependent; group 7 only verifies this.
- **Validity:** VUID-VkPipelineShaderStageCreateInfo-pNext-02757 needs local size X to be a multiple of `requiredSubgroupSize`. 256 is a multiple of 16, 32 and 64.
- **Order:** campaign run order is `vk-engine-runtime` → `tree-app-improvements` → `cifar-gpu-kernels`.

### B6. Stage-3 compaction
Per-workgroup flags plus a subgroup/LDS scan, one scan of workgroup totals, and a fused scatter: 2–3 dispatches instead of 5. Leave `record_device_scan` unchanged for stage 6 unless both stages re-verify exactly.

## Risks / Trade-offs

- **[Risk]** Clamping changes codes, and so the downstream tree, for boundary points. **→ Mitigation:** it's applied to all backends and the oracle at once; existing inputs never hit the boundary (half-open range), so the current test data is unchanged (to verify by diffing stage-1 output before/after).
- **[Risk]** Atomic leaf clears cost a little on CUDA/OMP stage 7. **→ Mitigation:** measure stage 7 before/after on duck-stable and the CPU tiers; the contention is low.
- **[Risk]** New LDS footprints exceed the smallest `maxComputeSharedMemorySize`. **→ Mitigation:** log it on all three GPUs and size for the minimum.
- **[Risk]** Faster stages 2–3 and a slower Vulkan stage 7 shift the hybrid schedules. **→ Mitigation:** refresh the tree × VK profile (a task).

## Migration Plan

Topic A lands first, each fix behind its reproducer, with the full gate including `ctest -L cuda` on duck-stable. Topic B then lands in this order: B1, B2, B4, B5, B6, each as a separate, revertible commit behind the VK gate. Baked `*_spv.h` are committed with each shader change, because Android and rocky use the committed headers.

## Open Questions

- Whether B1 alone reaches the 0.40 ms target. If it does, B2/B4 become optional and should be re-scoped with the owner.
