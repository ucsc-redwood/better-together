---

description: "Task list for Remove Demo Runner Apps"
---

# Tasks: Remove Demo Runner Apps

**Input**: Design documents from `/specs/004-remove-demo-apps/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md)

**Tests**: no new tests — this is a pure-deletion change; the existing `ctest -L omp`
gate is the regression check (see research.md Finding 5).

**Organization**: tasks are grouped by user story (US1/US2/US3, priority order from
spec.md).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: which user story this task belongs to (US1/US2/US3)
- Every task names its exact file path(s)

---

## Phase 1: Setup

- [X] T001 Capture the pre-removal baseline: run
      `ctest --test-dir build/pc -L omp --output-on-failure` and record the pass count
      (currently 9/9) so Phase 4's regression check has an exact number to match, not
      just "looks green."

---

## Phase 2: Foundational

*(none — this feature has no shared blocking prerequisite beyond the Phase 1 baseline;
proceed directly to User Story 1.)*

---

## Phase 3: User Story 1 - The build no longer ships ad-hoc, assertion-free demo binaries (Priority: P1) 🎯 MVP

**Goal**: delete the five demo/runner targets, their source files, their
`CMakeLists.txt` registrations, and the two CMake helper macros left orphaned by their
removal.

**Independent Test**: after this phase, list build targets and confirm none of the five
remain; confirm the build still succeeds.

- [X] T002 [P] [US1] Delete `apps/tree/cuda/main.cu` (`run-tree-cu` source)
- [X] T003 [P] [US1] Delete `apps/tree/vulkan/run_main.cpp` (`run-tree-vk` source)
- [X] T004 [P] [US1] Delete `apps/cifar-dense/omp/main.cpp` (`run-cifar-dense-omp` source)
- [X] T005 [P] [US1] Delete `apps/cifar-dense/cuda/main.cu` (`run-cifar-dense-cu` source)
- [X] T006 [P] [US1] Delete `apps/cifar-sparse/cuda/main.cu` (`run-cifar-sparse-cu` source)
- [X] T007 [P] [US1] Remove the `# ---- runners ----` block (lines 27-33) from
      `apps/tree/CMakeLists.txt` — the `bt_add_cuda_app(run-tree-cu ...)` and
      `bt_add_vk_run(run-tree-vk ...)` registrations, including their `if(BT_ENABLE_*)`
      guards (depends on T002, T003)
- [X] T008 [P] [US1] Remove the `# ---- runners ----` block (lines 26-30) from
      `apps/cifar-dense/CMakeLists.txt` — the `bt_add_omp_run(run-cifar-dense-omp ...)`
      and `bt_add_cuda_app(run-cifar-dense-cu ...)` registrations (depends on T004, T005)
- [X] T009 [P] [US1] Remove the `# ---- runners ----` block (lines 23-26) from
      `apps/cifar-sparse/CMakeLists.txt` — the `bt_add_cuda_app(run-cifar-sparse-cu ...)`
      registration and its `if(BT_ENABLE_CUDA)` guard (depends on T006)
- [X] T010 [US1] Remove the `bt_add_omp_run` (lines 9-12) and `bt_add_vk_run` (lines
      34-37) function definitions from `cmake/bt_targets.cmake` — per research.md
      Finding 2, each has exactly one caller and both are now gone (depends on T007, T008)
- [X] T011 [US1] Build the `pc` preset (`cmake --preset pc && cmake --build --preset
      pc`) and confirm `cmake --build --preset pc --target help` lists none of
      `run-tree-cu`, `run-tree-vk`, `run-cifar-dense-omp`, `run-cifar-dense-cu`,
      `run-cifar-sparse-cu` (depends on T002-T010)

**Checkpoint**: US1 done — the five demo targets and their orphaned helper macros no
longer exist; the `pc` preset builds clean (SC-001).

---

## Phase 4: User Story 2 - Removing the demo apps doesn't silently break anything (Priority: P1)

**Goal**: confirm, after removal, that nothing else in the project depended on the
deleted targets.

**Independent Test**: routine correctness gate still passes; no CI/script/orchestration
file references a removed target name.

- [X] T012 [P] [US2] Re-run the reference search from research.md Finding 3 post-removal:
      `grep -rn "run-tree-cu\|run-tree-vk\|run-cifar-dense-omp\|run-cifar-dense-cu\|run-cifar-sparse-cu" .github/ scripts/ optimizer/` —
      confirm zero matches (depends on T002-T010)
- [X] T013 [P] [US2] Confirm zero remaining references to the removed helper macros:
      `grep -rn "bt_add_omp_run\|bt_add_vk_run" cmake/ apps/ profiler/ runtime/ platform/ tools/` —
      confirm zero matches (depends on T010)
- [X] T014 [US2] Run `ctest --test-dir build/pc -L omp --output-on-failure` and confirm
      it matches the Phase 1 baseline exactly (same pass count, no new failures)
      (depends on T011)
- [X] T015 [P] [US2] Cross-build under the `jetson` preset (CUDA+Vulkan enabled) and the
      `vulkan` preset to confirm the backend-guarded deletions (the `if(BT_ENABLE_CUDA)`/
      `if(BT_ENABLE_VULKAN)` blocks removed in T007-T009) don't break those
      configurations (depends on T002-T010)

**Checkpoint**: US2 done — zero live references anywhere, routine gate unchanged, both
backend-enabled presets still build (SC-002, SC-003).

---

## Phase 5: User Story 3 - Existing documentation doesn't mislead readers (Priority: P2)

**Goal**: update the one current how-to doc that instructs building/running a removed
target; leave historical records alone.

**Independent Test**: the updated doc no longer references a nonexistent target;
historical docs are unchanged.

- [X] T016 [US3] Edit `specs/001-octomap-real-workload/quickstart.md` lines 89-91: per
      research.md Finding 4, remove the "or `run-tree-cu` for a smoke run" clause and the
      "the smoke runner `run-tree-cu`" description, keeping the still-accurate "no CUDA
      benchmark for tree... use `test-tree-cu` for the differential correctness check"
      guidance
- [X] T017 [P] [US3] Confirm (no edit) that `specs/001-octomap-real-workload/tasks.md:210`,
      `docs/reports-for-human/cmake-migration-rfc.md:201`, and
      `docs/reports-for-human/project-evaluation-2026-06-19.md:99` remain unchanged —
      these are historical record per research.md Finding 4 and spec User Story 3

**Checkpoint**: US3 done — no current documentation misleads a reader about a deleted
target (SC-004).

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T018 [P] Run `just fmt` on touched files; revert any `.specify/` collateral via
      `git checkout -- .specify/` (established discipline this session).
- [X] T019 Final full-repo sanity sweep:
      `grep -rn "run-tree-cu\|run-tree-vk\|run-cifar-dense-omp\|run-cifar-dense-cu\|run-cifar-sparse-cu\|bt_add_omp_run\|bt_add_vk_run" .` —
      confirm the only remaining hits are inside `specs/004-remove-demo-apps/` itself and
      the historical docs from T017 (belt-and-suspenders check covering any file not
      explicitly enumerated in T012/T013). FOUND during execution: one additional
      historical reference not caught during planning —
      `docs/reports-for-human/code-review-2026-06-18.md:231` (a dated code-review finding
      about `bt_add_omp_run`/`bt_add_vk_run`'s ARGN semantics) — left unchanged, same
      historical-record treatment as the other dated reports.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: empty for this feature.
- **US1 (Phase 3)**: depends on Setup (T001, so there's a baseline to compare against
  later) — the actual deletions have no dependency on T001 itself, only the *comparison*
  in Phase 4 does.
- **US2 (Phase 4)**: depends on US1 being complete (there's nothing to verify the
  absence of until the targets are actually removed).
- **US3 (Phase 5)**: independent of US1/US2 — could technically run in parallel with
  them, but is sequenced last here since it's the lowest priority and smallest change.
- **Polish (Phase 6)**: depends on all three user stories being complete.

### Parallel Opportunities

- T002-T006 (all deletions, different files) run in parallel.
- T007-T009 (all CMakeLists.txt edits, different files) run in parallel, each depending
  only on its own app's deletions.
- T012, T013, T015 (independent verification checks) run in parallel once US1 is done.
- T017 (doc no-op verification) can run any time independent of everything else.

---

## Parallel Example: User Story 1

```bash
# All five deletions, different files -- launch together:
Task: "Delete apps/tree/cuda/main.cu"
Task: "Delete apps/tree/vulkan/run_main.cpp"
Task: "Delete apps/cifar-dense/omp/main.cpp"
Task: "Delete apps/cifar-dense/cuda/main.cu"
Task: "Delete apps/cifar-sparse/cuda/main.cu"

# Then the three CMakeLists.txt edits, different files -- launch together:
Task: "Remove runners block from apps/tree/CMakeLists.txt"
Task: "Remove runners block from apps/cifar-dense/CMakeLists.txt"
Task: "Remove runners block from apps/cifar-sparse/CMakeLists.txt"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup — baseline).
2. Complete Phase 3 (US1 — the actual removal).
3. **STOP and VALIDATE**: target list no longer shows the five demo binaries, build
   succeeds. This alone delivers the core ask.

### Incremental Delivery

1. Setup → baseline captured.
2. US1 → demo targets gone, build clean (MVP).
3. US2 → confirmed nothing else depended on them, routine gate unchanged.
4. US3 → documentation cleaned up.

---

## Notes

- [P] tasks = different files, no dependencies.
- This is a small, mechanical, low-risk refactor — most of the "work" was already done
  during `/speckit-plan`'s research phase (exact line numbers, exact reference
  inventory); implementation is largely executing what research.md already found.
- Do not touch any `test-*` target, any `bm-*` benchmark, or anything under `profiler/`
  — explicitly out of scope (FR-003).
