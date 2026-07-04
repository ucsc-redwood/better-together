---

description: "Task list for Vulkan Genuinely-Chained AppData Migration"
---

# Tasks: Vulkan Genuinely-Chained AppData Migration

**Input**: Design documents from `/specs/003-vulkan-appdata-migration/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/vulkan-dispatch-chained.md](./contracts/vulkan-dispatch-chained.md), [quickstart.md](./quickstart.md)

**Tests**: this feature's purpose IS test/verification infrastructure, so test tasks are
core to every phase, not optional extras.

**Organization**: tasks are grouped by user story (US1/US2/US3, priority order from
spec.md) so each story is independently verifiable on real hardware.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, or independent hardware targets with no
  file conflicts)
- **[Story]**: which user story this task belongs to (US1/US2/US3)
- Every task names its exact file path(s)

---

## Phase 1: Setup

- [X] T001 Confirm build/hardware access for this feature: `cmake --list-presets` shows
      `vulkan`/`jetson`/`android`; confirm `ssh doremy@duck-stable` works and `adb
      devices` (run from `rocky-ryzen`) lists both `3A021JEHN02756` (Pixel 7a) and
      `R5CY21Y3VEV` (Samsung Galaxy) per `docs/instruction-for-ai/01-hardware.md`. No
      code changes.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: the genuinely-chained Vulkan dispatch path itself — every user story below
depends on this existing and compiling first.

**⚠️ CRITICAL**: no user story work can begin until this phase is complete.

- [X] T002 Define `tree::vulkan::VkAppData` (chained) struct in
      `apps/tree/vulkan/vk_appdata.hpp`: inherits `tree::AppData`, adds the scratch
      fields `u_contributes`, `u_out_idx`, `u_sums`, `u_prefix_sums`, `u_sort_tmp`,
      `u_sort_histograms` (`kRadixBins * kRadixNumWorkgroups`), `u_scan_block_sums`
      (mirrors `VkAppData_Safe`'s scratch shape exactly, per data-model.md). Leave
      `VkAppData_Safe` untouched.
- [X] T003 Declare the chained overloads in `apps/tree/vulkan/dispatchers.hpp`:
      `run_stage_1..7(VkAppData&)`, `dispatch_stage(VkAppData&, int)`,
      `dispatch_multi_stage(VkAppData&, int, int)` on `VulkanDispatcher`, alongside the
      existing `VkAppData_Safe` overloads (depends on T002).
- [X] T004 Implement the chained `record_stage_1` body in
      `apps/tree/vulkan/dispatchers.cpp`: same shader/dispatch config as the existing
      `VkAppData_Safe` version, rebind the output buffer to `VkAppData::u_morton_keys_s1`
      (no `_out` suffix) (depends on T003).
- [X] T005 Implement the chained `record_stage_2` body in
      `apps/tree/vulkan/dispatchers.cpp`: radix sort now reads
      `VkAppData::u_morton_keys_s1` — this instance's OWN stage-1 output, not a
      construction-time golden — and writes `u_morton_keys_sorted_s2` (depends on T004).
- [X] T006 Implement the chained `record_stage_3` body in
      `apps/tree/vulkan/dispatchers.cpp`: dedup writes `u_morton_keys_unique_s3` +
      `u_num_selected_out` (the host-readback counter) (depends on T005).
- [X] T007 Implement the chained `record_stage_4` body in
      `apps/tree/vulkan/dispatchers.cpp`: radix-tree build reads this instance's own
      `u_morton_keys_unique_s3` (depends on T006).
- [X] T008 Implement the chained `record_stage_5` body in
      `apps/tree/vulkan/dispatchers.cpp` (depends on T007).
- [X] T009 Implement the chained `record_stage_6` body in
      `apps/tree/vulkan/dispatchers.cpp`: writes `u_edge_offset_s6` (the value the
      n_octree_nodes host readback reads back afterward) (depends on T008).
- [X] T010 Implement the chained `record_stage_7` body in
      `apps/tree/vulkan/dispatchers.cpp` (depends on T009).
- [X] T011 Implement `dispatch_multi_stage`'s sub-batch split for `VkAppData` in
      `apps/tree/vulkan/dispatchers.cpp`: per the pre/post table in
      contracts/vulkan-dispatch-chained.md, insert `submit()`/`wait_for_fence()` +
      host readback (`set_n_unique`/`set_n_brt_nodes`) whenever `[start_stage,
      end_stage]` spans the stage 3→4 boundary, and another readback
      (`set_n_octree_nodes`) whenever it spans stage 6→7; otherwise keep the existing
      single-command-buffer batching (depends on T004-T010).
- [X] T012 Confirm `apps/tree/CMakeLists.txt`'s existing `bt_bake_shaders` shader
      coverage is sufficient for the chained path (no new `.comp` shaders are expected —
      T004-T010 reuse the existing compute shaders, only C++ buffer bindings change); add
      any missing shader registration if the build surfaces a gap (depends on T004-T011).

**Checkpoint**: the chained Vulkan dispatch path builds under the `vulkan` preset. User
story work can now begin.

---

## Phase 3: User Story 1 - The Vulkan tree pipeline computes correct results without the golden-decoupled scaffold (Priority: P1) 🎯 MVP

**Goal**: prove the chained Vulkan path is correct (per-stage vs. OMP) and that a hybrid
CPU+GPU schedule genuinely overlaps, on all three required hardware targets.

**Independent Test**: run the new differential + concurrency-proof suites standalone;
they don't touch production profilers or the routine gate.

- [X] T013 [P] [US1] Add `VulkanChainedTreeRunner` (`AppData = tree::vulkan::VkAppData`,
      `RunStage` calls `disp.dispatch_stage`) and
      `BT_DECLARE_TREE_DIFF_TESTS_APPDATA(TreeDiffVulkanChained,
      VulkanChainedTreeRunner)` in `apps/tree/vulkan/test_main.cpp`, alongside the
      existing `VulkanTreeRunner`/`TreeDiffVulkan` suite (unchanged) — per
      contracts/vulkan-dispatch-chained.md.
- [X] T014 [P] [US1] Create `apps/tree/vulkan/test_pipeline_chained_vk.cpp`: hybrid
      CPU+Vulkan-GPU concurrency+correctness proof over the real SPSC ring, mirroring
      `apps/tree/cuda/test_pipeline_chained_cu.cu`'s `CheckItemChained` pattern (build a
      fresh OMP `ref` from the item's own input, diff the final octree). Register a new
      `test-pipeline-chained-vk` ctest target with the `experimental` label (not part of
      `ctest -L omp`/routine gates) in `apps/tree/CMakeLists.txt`.
- [X] T015 [US1] Build (`vulkan` preset) and run `TreeDiffVulkanChained` +
      `test-pipeline-chained-vk` on `rocky-ryzen` (fastest dev-iteration loop, not a
      required verification target); fix any issues surfaced before moving to hardware
      that costs more round-trip time (depends on T013, T014).
- [X] T016 [P] [US1] Cross-build (`jetson` preset via `bt-cross:7.2`) and run both suites
      on the Jetson devkit (`duck-stable`) via `scripts/run-on-jetson.sh`; confirm pass
      (depends on T015).
- [X] T017 [P] [US1] Build (`android` preset) and run both suites on the Pixel 7a via
      `scripts/run-on-android.sh 3A021JEHN02756`; confirm pass (depends on T015).
- [X] T018 [P] [US1] Run both suites on the Samsung Galaxy via
      `scripts/run-on-android.sh R5CY21Y3VEV`; confirm pass (depends on T015).

**Checkpoint**: US1 done — the chained Vulkan path is proven correct and genuinely
concurrent, independently on Jetson + both phones (SC-001, SC-002).

---

## Phase 4: User Story 2 - Production Vulkan profiling tools measure the real production path (Priority: P2)

**Goal**: switch the production Vulkan profiling tools to the chained path.

**Independent Test**: run each production Vulkan profiling tool and confirm it
completes without error using the chained path.

- [X] T019 [US2] Switch `profiler/tree-vk/const.hpp`'s `AppDataT` alias from
      `tree::vulkan::VkAppData_Safe` to `tree::vulkan::VkAppData` (mirrors
      `profiler/tree-cu/const.hpp`'s Phase 2 change this session) (depends on
      Foundational phase; independent of US1).
- [X] T020 [US2] Rebuild all Vulkan profiler binaries (`bm-baseline-tree-vk`,
      `bm-fully-tree-vk`, `bm-gen-logs-tree-vk`, `bm-prof-tree-vk`) and fix any compile
      errors surfaced (depends on T019). CORRECTION found during implementation: Vulkan
      has no `run-pipe-tree-vk` equivalent (that's CUDA-only, see
      `profiler/CMakeLists.txt`) — 4 tools, not 5.
- [X] T021 [US2] Before any profiling run below: confirm no competing process is running
      on the target device (Constitution Principle VI — the `llama-server`-on-Jetson
      precedent this session).
- [X] T022 [P] [US2] Run all 4 profiler tools on the Jetson devkit (`duck-stable`);
      confirm each completes without error (depends on T020, T021).
- [X] T023 [P] [US2] Run all 4 profiler tools on the Pixel 7a; confirm each completes
      without error (depends on T020, T021).
- [X] T024 [P] [US2] Run all 4 profiler tools on the Samsung Galaxy; confirm each
      completes without error (depends on T020, T021).

**Checkpoint**: US2 done — production Vulkan profilers measure the real chained path,
verified on Jetson + both phones (SC-003).

---

## Phase 5: User Story 3 - The project's routine Vulkan correctness checks exercise the same path production code uses (Priority: P3)

**Goal**: switch the routine `test-pipeline-e2e-vk` gate to the chained path (the
`AppTraits<VulkanDispatcher>` specialization can only exist once per binary, so this is
an in-place switch, not additive — the same constraint CUDA's Phase 3 had).

**Independent Test**: run the migrated suites and confirm they pass.

- [X] T025 [US3] Switch `apps/tree/vulkan/test_pipeline_main_vk.cpp`'s
      `AppTraits<tree::vulkan::VulkanDispatcher>` specialization and `CheckItem` to the
      chained ref/out diff pattern (build a fresh OMP `ref`, diff against `a`) —
      mirrors `apps/tree/cuda/test_pipeline_main_cu.cu`'s Phase 3 change (depends on
      Foundational phase; independent of US1/US2).
- [X] T026 [US3] Rebuild `test-pipeline-e2e-vk` and fix any compile errors surfaced
      (depends on T025).
- [X] T027 [P] [US3] Run `test-tree-vk` (both the existing `TreeDiffVulkan` suite and
      the `TreeDiffVulkanChained` suite from US1) + `test-pipeline-e2e-vk` on the Jetson
      devkit; confirm 100% pass — zero regression on the untouched `VkAppData_Safe`
      suite, and the chained suite/gate passes too (depends on T026, T013).
- [X] T028 [P] [US3] Run the same on the Pixel 7a; confirm 100% pass (depends on T026,
      T013).
- [X] T029 [P] [US3] Run the same on the Samsung Galaxy; confirm 100% pass (depends on
      T026, T013).

**Checkpoint**: US3 done — routine Vulkan correctness checks exercise the chained path
in production-equivalent form, on all three required targets, with no regression to the
existing `VkAppData_Safe` suites (SC-004).

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T030 [P] Run `just fmt` on all touched files; revert any `.specify/` collateral
      via `git checkout -- .specify/` (established discipline this session).
- [X] T031 Run `ctest --test-dir build/pc -L omp --output-on-failure` locally; confirm
      100% green as the zero-regression check for everything OMP-side (unaffected by
      this Vulkan-only feature, but must stay green).
- [X] T032 [P] Update `docs/instruction-for-ai/03-unit-testing.md` and/or
      `05-profiling.md` only if they name `VkAppData_Safe`/the Vulkan suites specifically
      enough that leaving them unchanged would mislead a reader about which path is now
      exercised.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: depends on Setup — BLOCKS all user stories (T002-T012 are
  a strict sequential chain within `dispatchers.cpp`/`vk_appdata.hpp`/`dispatchers.hpp`,
  since they build one coherent dispatch path).
- **US1 (Phase 3)**: depends on Foundational only.
- **US2 (Phase 4)**: depends on Foundational only — independent of US1 (different
  files: `profiler/tree-vk/const.hpp` vs. `apps/tree/vulkan/test_main.cpp`/new file).
- **US3 (Phase 5)**: depends on Foundational only for its own edits (T025/T026); its
  verification tasks (T027-T029) also want US1's `TreeDiffVulkanChained` suite (T013) to
  exist so the "both suites pass" checkpoint has something to check.
- **Polish (Phase 6)**: depends on all three user stories being complete.

### Parallel Opportunities

- T013 and T014 (different files) can run in parallel.
- Once each story's build/fix task is done, its three device-verification tasks
  (T016-T018, T022-T024, T027-T029) are independent hardware targets with no file
  conflicts and can run in parallel (three engineers, or three sequential sessions in
  any order).
- US2 (Phase 4) and US3 (Phase 5) touch entirely different files
  (`profiler/tree-vk/const.hpp` vs. `test_pipeline_main_vk.cpp`) and can proceed in
  parallel once Foundational is done, independent of each other and of US1.

---

## Parallel Example: User Story 1

```bash
# T013 and T014 touch different files -- launch together:
Task: "Add VulkanChainedTreeRunner + TreeDiffVulkanChained suite in apps/tree/vulkan/test_main.cpp"
Task: "Create apps/tree/vulkan/test_pipeline_chained_vk.cpp"

# After T015 (rocky dev-iteration fixes land), the three hardware verifications are independent:
Task: "Cross-build + run on Jetson devkit (duck-stable)"
Task: "Build + run on Pixel 7a"
Task: "Run on Samsung Galaxy"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup) + Phase 2 (Foundational — the chained dispatch path itself).
2. Complete Phase 3 (US1): prove correctness + concurrency on Jetson + both phones.
3. **STOP and VALIDATE**: this alone answers "does the chained Vulkan path even work,"
   the foundational question everything else depends on.

### Incremental Delivery

1. Setup + Foundational → the chained path compiles.
2. US1 → proven correct/concurrent on all three required targets (MVP).
3. US2 → production profilers measure the real path.
4. US3 → the routine gate exercises the real path, zero regression to the existing
   golden-decoupled suites.

### Parallel Team Strategy

Once Foundational is done, US2 and US3 can proceed in parallel with each other (disjoint
files); US1 should land first since US3's verification checkpoint wants US1's new
`TreeDiffVulkanChained` suite to already exist.

---

## Notes

- [P] tasks = different files, or independent hardware targets with no file conflicts.
- Foundational tasks (T004-T010) are intentionally NOT marked [P] — they all edit
  `apps/tree/vulkan/dispatchers.cpp` and are easiest to reason about as one sequential
  per-stage pass, even though no stage's correctness logically depends on another's.
- Every device-verification task's failure must name which device, which stage, and
  what diverged (FR-006) — don't just report pass/fail.
- `VkAppData_Safe` and its consuming suites/tools are never modified or deleted by this
  feature (per spec Assumptions) — every task above is additive except T025 (forced by
  the single-specialization-per-dispatcher-type constraint on `AppTraits`).
