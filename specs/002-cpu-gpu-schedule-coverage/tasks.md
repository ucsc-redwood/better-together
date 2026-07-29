---
description: "Task list for: CPU/GPU Schedule Permutation & Overlap Coverage for Tree"
---

# Tasks: CPU/GPU Schedule Permutation & Overlap Coverage for Tree

**Input**: Design documents from `/specs/002-cpu-gpu-schedule-coverage/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [quickstart.md](./quickstart.md)

**Tests**: Not requested as a separate TDD pass — this feature *is* a test suite. Per
FR-008 and the prior session's precedent, it's a new, deliberately `experimental`-labeled
gtest binary excluded from `ctest -L cuda`, not a change gated by existing tests.

**Execution notes (2026-07-04)**: implemented and verified end-to-end on real Jetson
hardware (`duck-stable`). Both gtest cases (`AllScheduleCorrectness`,
`OverlapAcrossRepeatedRuns`) passed on the first real run: all 29 schedules correct,
every CPU+GPU schedule showed genuine overlap in >= 3/5 runs. US3's diagnostics were
verified by deliberately injecting a correctness fault and an overlap fault (each
temporarily, then fully reverted and re-verified clean) — both failure messages name
the exact schedule and are clearly distinguishable from each other. One real bug was
caught and fixed along the way: `run_schedule_once` initially used `ASSERT_TRUE` inside
a function returning `RunResult` (non-`void`), which doesn't compile — gtest's `ASSERT_*`
macros require a `void`-returning function; switched to `EXPECT_TRUE`.

**Organization**: Tasks are grouped by user story (from spec.md) to enable independent
implementation and verification of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on an incomplete task)
- **[Story]**: US1 / US2 / US3, per spec.md's prioritized user stories
- Nearly every task below touches the same single new file
  (`apps/tree/cuda/test_schedule_permutation_cu.cu`), per plan.md's Structure Decision —
  so almost nothing here is `[P]` against anything else in that file; parallelism is
  limited to the two Polish tasks touching unrelated files.

## Path Conventions

Extension of the existing `apps/tree` test suite (no new project) — one new file plus
one new CMake registration, exactly mirroring the prior session's
`test_pipeline_chained_cu.cu` / `test-pipeline-chained-cu` precedent.

---

## Phase 1: Setup

**Purpose**: Scaffold the new test file and CMake target so later tasks have something
to build into.

- [X] T001 Create `apps/tree/cuda/test_schedule_permutation_cu.cu` with the standard
  includes (`gtest/gtest.h`, `cuda_runtime.h`, `apps/tree/omp/dispatchers.hpp`,
  `dispatchers.cuh`, `platform/registry/device_registry.hpp`,
  `runtime/pipeline_runner.hpp` or `runtime/pipeline.hpp` directly, `runtime/record.hpp`,
  `runtime/schedule.hpp`, `runtime/spsc_queue.hpp`) plus an empty `main()`
  (`::testing::InitGoogleTest` + `parse_args_test` + `spdlog::set_level(off)` +
  `RUN_ALL_TESTS()`, matching `test_pipeline_chained_cu.cu`'s shape). Register it in
  `apps/tree/CMakeLists.txt` as `bt_add_cuda_test(test-schedule-permutation-cu
  cuda/test_schedule_permutation_cu.cu)` with
  `set_tests_properties(test-schedule-permutation-cu PROPERTIES LABELS
  "experimental")`. Cross-compile (`bt-cross:7.2`, `cmake --build --preset jetson
  --target test-schedule-permutation-cu`) to confirm the empty scaffold builds clean.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The shared schedule-generation and single-run harness every user story
depends on.

**⚠️ CRITICAL**: No user story task can be implemented until this phase is complete.

- [X] T002 In `apps/tree/cuda/test_schedule_permutation_cu.cu`: implement
  `generate_all_schedules()` returning `std::vector<Schedule>` — the all-CPU schedule
  (one `ChunkConfig{kOMP, 1, 7, first_present_cpu_type()}`) plus, for every `gpu_start`
  in `1..7` and `gpu_end` in `gpu_start..7` (28 combinations), a `Schedule` with an
  optional CPU-before chunk `[1, gpu_start-1]`, the GPU chunk
  `{kCuda, gpu_start, gpu_end, std::nullopt}`, and an optional CPU-after chunk
  `[gpu_end+1, 7]` (both CPU chunks using `first_present_cpu_type()`). Give each
  schedule a descriptive `uid` (e.g. `"gpu-3-5"`, `"all-cpu"`, `"all-gpu"`). Validate
  every generated schedule with `validate_schedule_coverage(sched, 7)` and
  `first_concurrent_gpu_chunk(sched)` (must return `std::nullopt`) — fail the test
  immediately (`FAIL()`) if generation ever produces an invalid schedule. Depends on
  T001 (file must exist).
- [X] T003 In `apps/tree/cuda/test_schedule_permutation_cu.cu`: implement
  `run_schedule_once(const Schedule& sched, size_t pool_size, size_t n_items) ->
  std::pair<std::vector<std::unique_ptr<tree::AppData>>, Logger<N>>` (or an equivalent
  struct return) — construct a `tree::cuda::CudaDispatcher`, pool `tree::AppData` via
  `make_dataset<tree::AppData>(disp, pool_size)`, build one `SPSCQueue<tree::AppData*,
  64>` per chunk, pre-fill queue 0, spawn one `worker_with_record` thread per chunk
  (GPU chunks call `disp.dispatch_multi_stage(*app, start, end)`; OMP chunks call
  `tree::omp::dispatch_multi_stage(*app, start, end)`, ignoring core-pinning per the
  prior session's established simplification), join all threads, and return the
  pooled dataset (for correctness) plus the populated `Logger` (for overlap). Mirrors
  `profiler/bm_gen_log_common.hpp`'s `run_schedule()` and the prior session's
  `test_pipeline_chained_cu.cu`, but pools `tree::AppData` (not `SafeAppData`) and
  returns data instead of dumping to stdout. Depends on T001.

**Checkpoint**: the file builds, can generate all 29 schedules, and can run any one of
them once through the real concurrent runtime, capturing timing. Every user story below
builds on this.

---

## Phase 3: User Story 1 - Every CPU/GPU stage split still computes the correct result (Priority: P1) 🎯 MVP

**Goal**: Every one of the 29 schedule permutations produces output matching a
sequential OMP reference, exactly.

**Independent Test**: run the suite; every schedule's pooled items pass the correctness
check, and every failure (if any) names its exact schedule and mismatching
stage/buffer (FR-005 — folded in here so this story ships actionable failures from the
start, not as a later retrofit).

### Implementation for User Story 1

- [X] T004 [US1] In `apps/tree/cuda/test_schedule_permutation_cu.cu`: implement
  `CheckItemChained(tree::AppData& item)` — build a fresh reference `tree::AppData`
  seeded with `item.u_input_points_s0`, run `tree::omp::run_stage_1..7` sequentially
  (the OMP oracle), then `EXPECT_EQ` every element of `u_oct_child_node_mask_s7` and
  `u_oct_child_leaf_mask_s7` against the reference, with a `<<` failure message naming
  the node index. Reuse verbatim from the prior session's
  `apps/tree/cuda/test_pipeline_chained_cu.cu`'s `CheckItemChained` — this is the same
  check, not a new one.
- [X] T005 [US1] In `apps/tree/cuda/test_schedule_permutation_cu.cu`: add
  `TEST(SchedulePermutation, AllScheduleCorrectness)` — for each schedule from
  `generate_all_schedules()` (T002), call `run_schedule_once()` (T003) once, then run
  `CheckItemChained` (T004) on every pooled item; on any failure, do **not** abort the
  whole test (per the spec's edge case) — use `ADD_FAILURE() << "schedule [" <<
  sched.uid << "] stage/item mismatch: ..."` so gtest keeps evaluating every remaining
  schedule and reports all failures in one run, each tagged with its schedule's `uid`.

**Checkpoint**: User Story 1 is fully functional and independently verifiable — every
schedule's correctness is checked, and any failure names its exact schedule.

---

## Phase 4: User Story 2 - CPU and GPU genuinely overlap instead of taking turns (Priority: P2)

**Goal**: For every schedule with both a CPU and a GPU chunk, measured evidence across
5 repeated runs confirms genuine concurrent execution, not serialization.

**Independent Test**: run the suite; every CPU+GPU schedule reports an overlap verdict
based on >= 3 of 5 repeated runs showing measured concurrent time, and every failure
names its exact schedule and which runs lacked overlap (FR-006 — folded in here for the
same reason as T005).

### Implementation for User Story 2

- [X] T006 [US2] In `apps/tree/cuda/test_schedule_permutation_cu.cu`: implement
  `double MeasureConcurrentMs(const Logger<N>& logger, int cpu_chunk_id, int
  gpu_chunk_id, uint64_t freq)` — port `dashboard/generate.py`'s `_coverage_time`
  sweep-line algorithm (research.md §3): collect `(start, end)` cycle intervals from
  `logger.records_[*][cpu_chunk_id]` and `logger.records_[*][gpu_chunk_id]` (skip the
  first `PIPE_WARMUP`-equivalent few tasks per chunk as cold-start), convert to
  milliseconds via `Logger::cycles_to_milliseconds(..., freq)`, build `+1/-1` sweep
  events, and return the total time where >= 2 intervals are concurrently active.
  Return 0.0 for a chunk pair that never both appear (the all-CPU/all-GPU boundary
  schedules — overlap is not applicable there, per the spec's edge case).
- [X] T007 [US2] In `apps/tree/cuda/test_schedule_permutation_cu.cu`: add
  `TEST(SchedulePermutation, OverlapAcrossRepeatedRuns)` — for each schedule from
  `generate_all_schedules()` (T002) that has both a CPU chunk and a GPU chunk (skip the
  two all-one-PU boundary schedules), call `run_schedule_once()` (T003) **5 times**,
  compute `MeasureConcurrentMs()` (T006) per run, and count runs where the result is
  `> 0`. `EXPECT_GE(overlapping_runs, 3) << "schedule [" << sched.uid << "] only
  overlapped in " << overlapping_runs << "/5 runs"` — using `EXPECT_GE` (not
  `ASSERT_GE`) so, like T005, one schedule's failure doesn't stop the rest from being
  checked and reported.

**Checkpoint**: User Stories 1 AND 2 both work independently — correctness is proven
across every schedule (US1), and genuine CPU/GPU overlap is proven, with repeated-run
evidence, across every schedule that has both a CPU and a GPU chunk (US2).

---

## Phase 5: User Story 3 - A failing permutation is actionable, not just a red mark (Priority: P3)

**Goal**: Confirm the diagnostics T004-T007 already produce are genuinely actionable —
naming the exact schedule and distinguishing a correctness failure from a
lack-of-overlap failure — by deliberately observing both failure paths once.

**Independent Test**: temporarily force one of each failure type and confirm the
report is specific and distinguishable, then revert the deliberate change.

### Verification for User Story 3

- [X] T008 [US3] Deliberately verify the correctness-failure report path: temporarily
  corrupt one value in `CheckItemChained`'s reference computation (T004) — e.g. flip
  one bit of `ref.u_oct_child_node_mask_s7[0]` right before the comparison — rebuild
  and run `test-schedule-permutation-cu` on `duck-stable`, confirm the failure output
  names the specific schedule `uid` and node index, then revert the temporary change
  and rebuild to confirm the suite passes clean again.
- [X] T009 [US3] Deliberately verify the overlap-failure report path: temporarily force
  `run_schedule_once()` (T003) to run chunks sequentially instead of concurrently for
  one schedule (e.g. `t.join()` immediately after each `emplace_back` instead of after
  the loop) — rebuild and run, confirm `OverlapAcrossRepeatedRuns` (T007) fails for
  exactly that schedule with a message distinguishable from T008's correctness-failure
  message, then revert the temporary change and rebuild to confirm the suite passes
  clean again.

**Checkpoint**: All three user stories are independently verified — correctness,
overlap, and the actionability of both stories' failure reporting.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Full-hardware validation and documentation, spanning all three stories.

- [X] T010 Cross-build `test-schedule-permutation-cu` for Jetson (`bt-cross:7.2`,
  `cmake --build --preset jetson --target test-schedule-permutation-cu`), deploy via
  `scripts/run-on-jetson.sh test-schedule-permutation-cu`, and confirm the full sweep
  (29 schedules, `AllScheduleCorrectness` + `OverlapAcrossRepeatedRuns`) completes and
  reports results, per `quickstart.md` steps 1-3.
- [X] T011 [P] Run `ctest --test-dir build/jetson -L cuda --output-on-failure` on
  `duck-stable` and confirm `test-schedule-permutation-cu` does **not** appear in the
  output (its `LABELS` is `experimental`, not `cuda`) — per `quickstart.md` step 4 and
  FR-008.
- [X] T012 [P] Add a short pointer to `test-schedule-permutation-cu` (what it covers,
  that it's on-demand/not in `ctest -L cuda`) in
  `docs/instruction-for-ai/03-unit-testing.md`, near the existing
  `test-pipeline-e2e-cu` description.
- [X] T013 Run `just fmt` across `apps/tree/cuda/test_schedule_permutation_cu.cu` and
  `apps/tree/CMakeLists.txt`, then confirm `just fmt-check` is clean for both files.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup (T001, the file must exist) — BLOCKS all
  user stories.
- **User Stories (Phase 3-5)**: All depend on Foundational (Phase 2).
  - US1 (T004-T005) has no dependency on US2.
  - US2 (T006-T007) has no dependency on US1's tasks, only on Foundational — but since
    all tasks share one file, implement sequentially regardless of logical
    independence.
  - US3 (T008-T009) depends on US1 (T004-T005) and US2 (T006-T007) existing, since it
    verifies their diagnostic output.
- **Polish (Phase 6)**: Depends on all three user stories being complete.

### Within Each Phase

- T002 before T003 is not required (independent functions), but both must precede
  every user story task.
- T004 before T005 (T005 calls T004).
- T006 before T007 (T007 calls T006).
- T008 and T009 depend on T005 and T007 respectively already existing to verify.

### Parallel Opportunities

- None within Phases 1-5 — every task touches the same single file
  (`apps/tree/cuda/test_schedule_permutation_cu.cu`), per the plan's Structure
  Decision, so edits must be sequential regardless of logical story independence.
- T011 and T012 (Polish) can run in parallel — different files, no shared dependency.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001).
2. Complete Phase 2: Foundational (T002-T003) — schedule generation + single-run
   harness.
3. Complete Phase 3: User Story 1 (T004-T005) — correctness across all 29 schedules.
4. **STOP and VALIDATE**: cross-build, deploy to `duck-stable`, confirm
   `AllScheduleCorrectness` passes (or reports specific, actionable failures).
5. This is a demonstrable MVP: every CPU/GPU schedule permutation is proven correct on
   real hardware.

### Incremental Delivery

1. Setup + Foundational → schedule generation and run harness ready.
2. Add User Story 1 → validate independently on `duck-stable` → MVP demo-able.
3. Add User Story 2 → validate independently (overlap proven across repeated runs).
4. Add User Story 3 → validate independently (both failure paths confirmed
   actionable, then reverted).
5. Polish → full sweep run, gate-exclusion confirmed, docs pointer, formatting.

### Parallel Team Strategy

Limited by the single-file structure — realistically one contributor at a time on
Phases 1-5 (merge conflicts on the same file otherwise). Once Phase 5 is done, a second
contributor could pick up T012 (docs) while the first runs T010-T011/T013 on hardware.

---

## Notes

- [P] tasks touch different files and have no dependency on an incomplete task — rare
  in this feature given its intentionally small, single-file scope (plan.md's
  Structure Decision).
- [Story] labels (US1/US2/US3) map every task to its spec.md user story for
  traceability.
- No new automated tests beyond the feature's own gtest binary — this feature *is* the
  test; per FR-008 it stays `experimental`-labeled, out of `ctest -L cuda`.
- Commit after each task or logical group, per this repo's Surgical, Traceable Changes
  principle.
- Stop at any checkpoint (end of Phase 2, 3, 4, or 5) to validate that increment
  independently before continuing.
