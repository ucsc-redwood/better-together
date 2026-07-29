---

description: "Task list for Tree Scheduler Re-Validation Post-AppData Migration"
---

# Tasks: Tree Scheduler Re-Validation Post-AppData Migration

**Input**: Design documents from `/specs/005-tree-scheduler-revalidation/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md)

**Tests**: no new automated tests; per FR-007, an in-scope contingency fix (if needed)
must pass the existing `ctest -L <backend>` differential gate before being trusted —
that gate itself is not new.

**Organization**: tasks are grouped by user story (US1/US2/US3, priority order from
spec.md). US1 and US2 map to `00_run_fleet.py`'s `--phases` flag: US1 =
`build,profile`, US2 = `schedule,run,summary` — the tool's own phase split matches the
spec's story split exactly.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files/devices, no dependencies on incomplete
  tasks)
- **[Story]**: which user story this task belongs to (US1/US2/US3)
- Every task names its exact command or file path

---

## Phase 1: Setup

- [X] T001 Constitution Principle VI precondition: confirm no competing process on
      `duck-stable` (`ssh doremy@duck-stable 'ps aux --sort=-%cpu | head -10'`) and on
      `rocky-ryzen` (`ssh rocky-ryzen 'bash -lc "ps aux --sort=-%cpu | head -10"'`) — the
      adb host both phones' profiling/schedule-run traffic transits. If either shows a
      competing load, stop and resolve before continuing (do not silently proceed).
- [X] T002 Confirm (no action) that local `data/profiling/`, `data/schedules_{btpm,isolated}/`,
      `data/sched_logs/` contain no `duck-stable`/`duck-naughty`-keyed tree data, and that
      any `jetson`-keyed tree data present is dated before 2026-07-01 (per research.md
      Finding 5) — this justifies using `--fresh` in Phase 3 rather than skipping it.

---

## Phase 2: Foundational

*(none — `00_run_fleet.py` is pre-existing tooling; there is no shared setup beyond
Phase 1's preconditions.)*

---

## Phase 3: User Story 1 - Fresh, current profiling data replaces stale pre-migration data (Priority: P1) 🎯 MVP

**Goal**: recollect tree's profiling data on `duck-stable` (CUDA+Vulkan), `pixel`
(Vulkan), and `samsung` (Vulkan) from scratch.

**Independent Test**: the resulting `data/profiling/` entries for these three devices
are dated today and contain no carried-over pre-migration/pre-reflash data.

- [X] T003 [US1] Run `uv run optimizer/orchestrate/00_run_fleet.py --only
      duck-stable,samsung,pixel --fresh --phases build,profile` — wipes any existing
      `data/profiling|schedules_{btpm,isolated}|sched_logs` entries for these three
      devices, rebuilds the benchmark binaries, then recollects isolated + interference
      profiling data for tree (and incidentally cifar-dense/cifar-sparse, per
      research.md Finding 2 — not this feature's concern) on all three devices.
- [X] T004 [US1] **CONTINGENCY, only if T003 fails with a crash/defect in the migrated
      dispatch path** (per the spec's Clarification / FR-007): diagnose the minimum fix
      needed to unblock collection, scoped strictly to that defect (no broader
      refactor); verify the fix via the relevant `ctest -L <cu|vk>` differential gate on
      the affected target before re-running T003. Skip this task entirely if T003
      succeeds cleanly.
- [X] T005 [US1] Confirm `data/profiling/duck-stable/tree/{cuda,vulkan}/{isolated,interference}/`
      and `data/profiling/{samsung,pixel}/tree/vulkan/{isolated,interference}/` all
      exist, are dated today, and are the only tree profiling data present for these
      three devices (depends on T003, and T004 if it ran).

**Checkpoint**: US1 done — fresh tree profiling data exists for all four target
combinations, nothing stale remains (SC-001).

---

## Phase 4: User Story 2 - An honest answer to "does pipelining still pay off for tree" (Priority: P1)

**Goal**: generate schedule candidates from the fresh data, run the top candidates on
real hardware, and compare against each combination's best-single-processor baseline.

**Independent Test**: `data/sched_logs/speedup-summary.md` contains a real, measured
Speedup value for each of the four tree combinations.

- [X] T006 [US2] Run `uv run optimizer/orchestrate/00_run_fleet.py --only
      duck-stable,samsung,pixel --phases schedule,run,summary` (no `--fresh` here — build
      on T003's fresh profiling data, don't wipe it) — for each device × backend × table
      (`btpm`/`isolated`): z3 generates 10 candidate schedules
      (`02_gen_schedule_merged.py`), the top 4 (default cap, per `fleet.json`) are
      actually run on the device (`03_run_schedule.py`), then
      `optimizer/analysis/speedup_summary.py` regenerates
      `data/sched_logs/speedup-summary.md` (depends on T005).
- [X] T007 [US2] Confirm `data/sched_logs/speedup-summary.md` contains exactly four
      `tree` rows — `duck-stable`/CUDA, `duck-stable`/VK, `pixel`/VK, `samsung`/VK — each
      with a non-empty Baseline, Best, and Speedup value (depends on T006).
- [X] T008 [US2] For each of the four tree rows, note whether Speedup is above, at, or
      below `1.00x` (a value below `1.00x` is an accepted, honestly-reported outcome per
      the spec's Edge Cases — not a task failure) (depends on T007).

**Checkpoint**: US2 done — every in-scope combination has an independently measured,
real (not estimated) scheduler-vs-baseline comparison (SC-002).

---

## Phase 5: User Story 3 - The result is a discoverable, dated record (Priority: P2)

**Goal**: archive the measured comparison as a new dated report in this project's
existing format.

**Independent Test**: the new dated report file exists, states all four tree outcomes
plainly, and no historical report was altered.

- [X] T009 [US3] Copy `data/sched_logs/speedup-summary.md` verbatim to
      `docs/reports-for-human/perf-results/speedup-summary-<DATE>-appdata-migration.md`
      (`<DATE>` = today, `YYYY-MM-DD`) — matching the existing
      `speedup-summary-2026-07-02-<label>.md` naming convention (depends on T007).
- [X] T010 [US3] Confirm the archived report's four tree rows (from T008) are stated
      plainly, including any row with Speedup below `1.00x` — do not edit the copied
      content to soften or omit a loss (depends on T009).
- [X] T011 [P] [US3] Confirm (no edit) that no existing dated report under
      `docs/reports-for-human/` (including the three prior `speedup-summary-2026-07-02-*.md`
      files and `2026-07-03-kernel-wave-and-definitive-baseline.md`) was modified —
      `git status`/`git diff` on `docs/reports-for-human/` shows only the new file from
      T009 as an addition, per FR-006.

**Checkpoint**: US3 done — the result is a discoverable, dated, honest record (SC-003).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies — T001 and T002 can run in parallel.
- **Foundational (Phase 2)**: empty for this feature.
- **US1 (Phase 3)**: depends on Setup (T001's no-competing-load confirmation matters
  before any measurement is trusted).
- **US2 (Phase 4)**: depends on US1 (T005) — cannot schedule/run without fresh
  profiling data to schedule from.
- **US3 (Phase 5)**: depends on US2 (T007) — cannot archive a report that doesn't exist
  yet.

### Parallel Opportunities

- T001 and T002 (Setup) are independent checks and can run in parallel.
- T011 (confirming historical reports untouched) has no dependency on T009/T010's
  content and can run any time after T009 lands, in parallel with T010.
- Within T003/T006, `00_run_fleet.py` itself already parallelizes across the three
  devices internally (one worker thread per device) — no task-level action needed to
  get that concurrency.

---

## Parallel Example: Setup

```bash
# T001 and T002 are independent checks -- launch together:
Task: "Confirm no competing process on duck-stable and rocky-ryzen"
Task: "Confirm local data/ has no duck-stable/duck-naughty tree data, only stale jetson-keyed data"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup — competing-load check).
2. Complete Phase 3 (US1 — fresh profiling data collected and confirmed).
3. **STOP and VALIDATE**: this alone answers "is the post-migration profiling data even
   collectible," the prerequisite everything else needs.

### Incremental Delivery

1. Setup → no competing load, confirmed stale baseline understood.
2. US1 → fresh profiling data for all four combinations (MVP).
3. US2 → real measured scheduler-vs-baseline answer for each combination.
4. US3 → archived as a discoverable, dated report.

---

## Notes

- [P] tasks = different files/devices, no dependencies.
- T004 is explicitly conditional — most runs won't need it. Do not treat its absence
  from a run as an incomplete task; treat it as N/A.
- Do not add an `--app` filter to `00_run_fleet.py` or otherwise try to suppress the
  incidental cifar-dense/cifar-sparse cells — out of scope per research.md Finding 2.
- A Speedup value below `1.00x` for tree on any combination is a valid, expected-possible
  outcome (per the spec's Edge Cases and the existing report format's own "Tree losses"
  section) — report it, don't chase it as a bug unless it's actually T004's kind of
  defect (a crash/blocked collection), not merely a disappointing number.
