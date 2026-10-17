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
- B: stage 2 faster than its recorded baseline beyond noise on the Pixel 7a and the Samsung; stage 3 in fewer than 5 dispatches and not slower on either phone. Both exact at subgroup 16, 32 and 64. Performance is measured on the phones only; rocky-ryzen runs correctness tests (the owner does not allow profiling there).

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
*As implemented (task 2.1):* the clamp is applied to the scaled float before the cast, `uint(clamp(v, 0, 1023))`. For
every v >= 0 that equals `min(uint(v), 1023u)`; below the bottom it gives 0 as the morton-codes spec requires, where
casting a negative float to an unsigned integer is undefined (x86 gave 1023, the Radeon 780M 0).

### A3. Atomic leaf-mask clear; Vulkan links leaves in a second dispatch
- **Atomic clear:** `set_leaf`'s clear becomes atomic: `#pragma omp atomic` on OMP, `atomicAnd` on CUDA and in the shader's `SetLeaf`. The child slot store stays plain; it's last-writer-wins by contract.
- **Vulkan leaf linking:** a separate dispatch after a compute→compute barrier, reusing the same shader through a push-constant phase selector, or a second entry. The leaf pass reads masks and offsets that the `MakeOctNodes` pass of *other* workgroups wrote, so a barrier inside the dispatch isn't enough.
  - *Alternative:* both passes in one dispatch with a grid-wide barrier. Rejected: Vulkan has no grid barrier.
  - *As implemented (task 3.3):* a push-constant `phase` (0: nodes, 1: leaves), the pipeline recorded twice. Phase 1
    writes the children buffer phase 0 wrote, so the `Sequence` records the barrier itself. Correction to the rationale
    above: once the masks are initialized at allocation (task 3.2), the leaf pass reads nothing the node pass writes
    (both only write disjoint child slots), so running both per brt node in one dispatch, as OMP's `run_stage_7` does,
    would also be correct. The second dispatch costs about a barrier; the measured +1.7 ms on the Pixel 7a is the leaf
    pass itself (task 3.4).
- **Why the Vulkan stage-7 test passed without leaf links (task 1.4).** Nothing ever set the leaf masks to all-ones:
  every mask buffer is a value-initialized `std::pmr::vector<int32_t>` (the golden's `tree::AppData` buffer and
  `SafeAppData`'s `_out`), so it starts at 0, and `set_leaf` only clears bits. The OMP golden's masks stayed all zeros
  (0 of 146,708 non-zero on the default input), and so did Vulkan's, which never writes binding 4. `OctreeEdgeSet` reads
  a zero bit as a leaf, so both sides listed every octant without a child node as a leaf slot and agreed. The oracle
  fix (task 3.2) therefore starts at the data: the masks are initialized to 0xFF where they are allocated, which is
  the contract this design already assumed. With that alone, both Vulkan stage-7 tests fail with every leaf slot
  missing on the Vulkan side.

## Decisions — B. Performance

### B1. Measure the cheap knob first
Sweep `kRadixNumWorkgroups` ∈ {32, 64, 128, 256} and keep the best. The O(WG²) histogram re-read falls quadratically. The sweep shows how much of the 0.785 ms is re-read versus scatter or dispatch, which decides whether B2–B4 are needed.

*Done before this change was applied (PR #26):* 256 → 64 in both appdata classes. On the 780M the 32..128 plateau
sorted 307k keys ~1.9x faster than 256 (`docs/reports-for-human/2026-10-15-vk-shader-profile.md` 4.4); measured on
the phones, stage 2 was -20 % on the Pixel 7a and -42 % on the Samsung (same report, section 9). The phone baseline
of task 4.1 is taken with 64.

### B2. Global prefix once per pass (reduce-then-scan)
Add a small per-pass scan of the 256 × WG table into per-workgroup offsets, or compute all 4 digits' histograms in one upfront read (FidelityFX ParallelSort structure). The scatter then reads only its own offsets.

*Chosen (task 5.1): the per-pass scan.* The upfront all-digit histogram gives each digit's global bin totals, but a
workgroup's offset inside a bin also needs the counts of the workgroups before it, and those change every pass, so it
does not remove the per-pass table read. `multi_radixsort_scan.comp` (one workgroup) writes every workgroup's 256
offsets in place; it carries the subgroup scan over the bins, so it is the subgroup-size-dependent kernel and the
scatter no longer uses subgroups. *Measured (task 5.3):* no gain on the phones (Pixel 7a level for level 0.2-1.3 %
slower, Samsung -1.9 %, all within the spread); at 64 workgroups the re-read was already small.

### B3. Not Onesweep
Onesweep's decoupled look-back needs forward progress between workgroups, which Vulkan doesn't guarantee. A waiting workgroup can deadlock on a never-scheduled predecessor, a particular risk on Mali (to verify). Reduce-then-scan is portable at the cost of one extra key read per pass.

### B4. LDS local sort and aggregated atomics
- **Local sort:** rank keys by digit within each workgroup's block (reusing the `bin_flags` ranking), then write in digit order, so writes within a bin run are contiguous.
- **Aggregated atomics:** in the histogram kernel, combine equal bins per subgroup via `GL_KHR_shader_subgroup_ballot` before the shared `atomicAdd`.
- *Measured (task 6.3) and reverted:* on top of B2, the local sort made stage 2 +24 % (Pixel 7a) / +2 % (Samsung), the
  aggregated atomics +77 % / +11 %. With 256-key blocks and 256 bins a bin run is about one key long, and the Morton
  digits are nearly uniform, so neither has anything to win and both add work.

### B5. Full subgroups
- **Engine:** owned by `vk-engine-runtime`, which runs before this change. It enables `computeFullSubgroups` as an optional feature and gives every subgroup-size-dependent pipeline `REQUIRE_FULL_SUBGROUPS` (its `vulkan-engine/pipeline-config` delta). This change does not touch the engine side.
- **Pipelines:** the sort kernels opt in by being subgroup-size-dependent; group 7 only verifies this.
- **Validity:** VUID-VkPipelineShaderStageCreateInfo-pNext-02757 needs local size X to be a multiple of `requiredSubgroupSize`. 256 is a multiple of 16, 32 and 64.
- **Order:** campaign run order is `vk-engine-runtime` → `tree-app-improvements` → `cifar-gpu-kernels`.

### B6. Stage-3 compaction
Per-workgroup flags plus a subgroup/LDS scan, one scan of workgroup totals, and a fused scatter: 2–3 dispatches instead of 5. Leave `record_device_scan` unchanged for stage 6 unless both stages re-verify exactly.

*As implemented (task 8.1):* 2 dispatches. Pass 1 flags and scans each 2048-key tile in shared memory (no subgroup
ops); pass 2 lets every tile sum the totals of the tiles before it (150 for the default input), so the separate scan
of the tile totals disappears, then scatters. `record_device_scan` is unchanged for stage 6. *Measured (task 8.2):*
stage 3 -16 % on the Pixel 7a, -42 % on the Samsung.

## Risks / Trade-offs

- **[Risk]** Clamping changes codes, and so the downstream tree, for boundary points. **→ Mitigation:** it's applied to all backends and the oracle at once; existing inputs never hit the boundary (half-open range), so the current test data is unchanged (to verify by diffing stage-1 output before/after).
- **[Risk]** Atomic leaf clears cost a little on CUDA/OMP stage 7. **→ Mitigation:** measure stage 7 before/after on duck-stable and the CPU tiers; the contention is low.
  *Measured (task 3.4):* not a little on CUDA. Stage 7 on duck-stable went 4.43 → 6.80 ms (+53 %), all of it the
  `atomicAnd` on the zero-copy pinned buffers (a non-atomic control is 4.44 ms). OMP on the Jetson's cores +6 %, within
  the spread. The spec requires the atomic, so it stays. Combining a warp's clears to one node (`__match_any_sync`
  plus one `atomicAnd` per node) could recover part of it; not done here.
- **[Risk]** New LDS footprints exceed the smallest `maxComputeSharedMemorySize`. **→ Mitigation:** log it on all three GPUs and size for the minimum.
- **[Risk]** Faster stages 2–3 and a slower Vulkan stage 7 shift the hybrid schedules. **→ Mitigation:** refresh the tree × VK profile (a task).

## Migration Plan

Topic A lands first, each fix behind its reproducer, with the full gate including `ctest -L cuda` on duck-stable. Topic B then lands in this order: B1, B2, B4, B5, B6, each as a separate, revertible commit behind the VK gate. Baked `*_spv.h` are committed with each shader change, because Android and rocky use the committed headers.

## Open Questions

- Whether B2/B4 still give a gain beyond noise on the phones on top of B1 (already 64). If neither does, record it; the spec's phone gate decides, not a fixed target.
  - *Answer (tasks 5.3 and 6.3):* neither does. B2 is neutral (Pixel 7a level for level 0.2-1.3 % slower, Samsung
    -2..-4 %, within the spread), B4 is slower on both phones and was reverted, and re-sweeping the workgroup count on
    top of B2 (32 / 128 / 256) found nothing faster on both phones. So the spec requirement "The sort is faster on the
    phones" (`gpu-kernels/tree-sort`) is not met by this change. Before archiving, the owner has to decide whether to
    drop or relax that requirement, or to fund a different sort (e.g. bigger tiles per workgroup, a different digit
    width). Both phones run these stages mostly at their lowest GPU clock, and their runs land on discrete levels
    10-50 % apart (4.1), so a gain also has to be large to be seen.

## Follow-ups

- **Owner decision before archiving:** the `gpu-kernels/tree-sort` requirement "The sort is faster on the phones" is
  not met (Open Questions; tasks 5.3 and 6.3). Every other requirement of the change is met.
- CUDA stage 7 is 53 % slower on duck-stable because of the required `atomicAnd` on zero-copy memory (Risks); a
  warp-aggregated clear could recover part of it.
- The tree × VK profiles on the phones lose most cells to the loader's `max_cv=0.1` filter at the default sampling
  knobs, before and after this change (task 9.2).
