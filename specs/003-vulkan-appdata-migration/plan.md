# Implementation Plan: Vulkan Genuinely-Chained AppData Migration

**Branch**: `003-vulkan-appdata-migration` | **Date**: 2026-07-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-vulkan-appdata-migration/spec.md`

## Summary

Build a genuinely-chained (single-buffer-per-stage, no golden/`_out` split) `AppData`
path for the Vulkan tree backend, mirroring the CUDA path already proven this session —
then wire production Vulkan profilers to it and migrate the routine Vulkan correctness
tests onto it, verifying on a Jetson devkit and both Android phones (Pixel/Samsung).

Research (below) found this is **not** a mechanical copy of the CUDA work: Vulkan's
current dispatcher reads every stage's input from `SafeAppData`'s CONST golden field
(never its own `_out` output) and needs extra scratch buffers (radix-sort histograms,
scan block sums) that don't exist on plain `tree::AppData`. Genuine chaining also forces
two new host-readback synchronization points (after stage 3, after stage 6) that don't
exist in the current single-command-buffer-per-chunk batching, because those two counts
must be known on the host before sizing the next GPU dispatch.

## Technical Context

**Language/Version**: C++20 (repo-wide), GLSL compute shaders (`.comp`, pre-baked via
`bt_bake_shaders`)

**Primary Dependencies**: `kiss_vk` engine wrapper (`platform/engine/vulkan/`) — VMA-based
pmr allocator, `Sequence`/command-buffer abstraction, cache-maintenance
(`flush_touched`/`invalidate_touched`) for `HOST_CACHED` memory (Mali coherency fix);
Vulkan Headers/Loader; googletest.

**Storage**: N/A (in-memory GPU buffers via `VulkanMemoryResource`, no persistence)

**Testing**: googletest via ctest, `BT_DECLARE_TREE_DIFF_TESTS_APPDATA`-style macro
expansion (the exact pattern CUDA/OMP just migrated onto in this session,
`apps/tree/tree_diff_oracle.hpp`), OMP-as-oracle differential comparison (Principle III)

**Target Platform**: integrated-GPU Vulkan targets only (`kiss_vk` hard-selects
`eIntegratedGpu`) — Jetson Orin (`duck-stable`/`duck-naughty`, required verification
target), Android phones (Pixel 7a subgroup-16, Samsung Galaxy subgroup-32, both required
verification targets); rocky-ryzen (x86 iGPU) available for dev iteration but not a
required verification target per the spec's Assumptions

**Project Type**: single C++/CMake systems project (existing repo structure, no new
top-level component)

**Performance Goals**: N/A — this feature is a correctness/architecture migration, not a
performance target; existing per-stage profiling numbers are expected to change (they'll
now reflect genuine chaining cost) but no specific throughput/latency goal is set

**Constraints**: must preserve the existing golden-decoupled `VkAppData_Safe` path
unmodified (additive, not replacing — mirrors how `SafeAppData` itself was left in place
for CUDA/OMP); the two new host-readback sync points must use the existing
`flush_touched`/`invalidate_touched` + fence-wait cache-maintenance machinery already in
`platform/engine/vulkan/sequence.cpp` (no new cache-coherency mechanism to invent, given
the prior Mali coherency defect history)

**Scale/Scope**: tree app only, Vulkan backend only; 7 pipeline stages; 3 required
hardware verification targets (Jetson, Pixel, Samsung)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Simplicity First)**: the scope below is proportionate — every file
  touched is touched because Vulkan's dispatcher hardcodes `VkAppData_Safe` by type (a
  class with member-function-pointer dispatch tables, not free functions), so a parallel
  chained overload set requires real per-stage edits, not a one-line alias swap like
  CUDA's `const.hpp` switch. No speculative abstraction is introduced beyond what CUDA's
  already-accepted pattern (`tree::AppData` + `dispatch_stage`/`dispatch_multi_stage`
  overloads) established.
- **Principle II (Surgical, Traceable Changes)**: the existing `VkAppData_Safe` path and
  its `record_stage_N`/`run_stage_N` bodies are left untouched; new chained-path
  overloads are added alongside (same pattern used for OMP/CUDA — see
  `apps/tree/omp/dispatchers.hpp`'s side-by-side `SafeAppData` and `tree::AppData`
  overload sets, and `tree_diff_oracle.hpp`'s parallel `BT_DECLARE_TREE_DIFF_TESTS` /
  `BT_DECLARE_TREE_DIFF_TESTS_APPDATA` macros).
- **Principle III (OMP-as-Oracle Differential Testing, NON-NEGOTIABLE)**: the new chained
  Vulkan path MUST be verified against the OMP reference at the fixed seed before it
  touches production profilers or the routine gate, exactly as CUDA's Phase 1 did with
  `test-pipeline-chained-cu` before Phase 2/3 wired it further in.
- **Principle IV (Goal-Driven Verification)**: each user story's "Independent Test" in
  the spec is a concrete `ctest`/binary run on real hardware, not just "it compiles."
- **Principle VI (Isolated Measurement Environment)**: production-profiler verification
  (User Story 2) on Jetson/Android must confirm no other process is competing for the
  GPU/CPU before treating any timing output as meaningful (the `llama-server`-on-Jetson
  incident this session is the concrete precedent).

No violations requiring Complexity Tracking — the extra scratch-buffer struct and the
two forced sync points are inherent to Vulkan's existing architecture (discovered in
research, not a design choice being made here), not added complexity.

## Project Structure

### Documentation (this feature)

```text
specs/003-vulkan-appdata-migration/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks, not this command)
```

### Source Code (repository root)

Existing single-project C++/CMake layout; no new top-level directories. Files this
feature touches or adds (all under the existing `apps/tree/` and `profiler/` trees):

```text
apps/tree/
├── vulkan/
│   ├── vk_appdata.hpp            # ADD: tree::vulkan::VkAppData (chained), alongside
│   │                              #      existing VkAppData_Safe (untouched)
│   ├── dispatchers.hpp           # ADD: run_stage_N/dispatch_stage/dispatch_multi_stage
│   │                              #      overloads for VkAppData (untouched SafeAppData
│   │                              #      overloads stay)
│   ├── dispatchers.cpp           # ADD: record_stage_N bodies for VkAppData -- rebind
│   │                              #      each stage's _out-suffixed target to the plain
│   │                              #      chained field name; dispatch_multi_stage splits
│   │                              #      into sub-batches at the stage3|4 and stage6|7
│   │                              #      boundaries (host readback of n_unique /
│   │                              #      n_octree_nodes), single batch otherwise
│   ├── test_main.cpp             # MODIFY: add BT_DECLARE_TREE_DIFF_TESTS_APPDATA suite
│   │                              #      (existing SafeAppData suite untouched -- same
│   │                              #      side-by-side pattern as CUDA/OMP)
│   └── test_pipeline_main_vk.cpp # MODIFY: switch AppTraits<VulkanDispatcher> + CheckItem
│                                  #      to the ref/out chained-diff pattern (mirrors
│                                  #      apps/tree/cuda/test_pipeline_main_cu.cu)
└── tree_diff_oracle.hpp           # no change expected -- already generic over
                                    # tree::AppData via BT_DECLARE_TREE_DIFF_TESTS_APPDATA

profiler/tree-vk/
└── const.hpp                      # MODIFY: AppDataT alias switch (mirrors
                                    #      profiler/tree-cu/const.hpp's Phase 2 change)
```

**Structure Decision**: single existing project, additive changes only (no files
deleted, no new top-level structure) — same shape as the CUDA/OMP migration this
session, just with the extra `VkAppData` scratch-buffer struct and the dispatcher's
per-stage record bodies needing real edits rather than a type-alias switch.

## Complexity Tracking

*No violations — table intentionally omitted (see Constitution Check above).*
