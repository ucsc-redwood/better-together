# Implementation Plan: CPU/GPU Schedule Permutation & Overlap Coverage for Tree

**Branch**: `002-cpu-gpu-schedule-coverage` | **Date**: 2026-07-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-cpu-gpu-schedule-coverage/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Exhaustively test all 29 valid ways of splitting the tree pipeline's 7 stages between a
contiguous CPU range and a contiguous GPU range (including the two all-one-PU boundary
cases), on real Jetson (CUDA) hardware, using the genuinely-chained `tree::AppData`
dispatch path from the prior session — verifying both (a) correctness (every schedule's
output matches a sequential OMP reference) and (b) genuine CPU/GPU concurrency (the two
PUs' work windows measurably overlap, corroborated across 5 repeated runs per schedule,
not serialized).

Technical approach: almost entirely reuse, not build. `runtime/record.hpp`'s `Logger` +
`worker_with_record` already timestamp every (task, chunk) work window — the exact data
needed for overlap detection — because `profiler/bm_gen_log_common.hpp`'s production
Gantt-log tool already uses them. This feature ports that tool's existing steady-state
`concurrency` sweep-line algorithm (from `dashboard/generate.py`) directly into one new,
self-contained gtest file, alongside the prior session's already-proven
`CheckItemChained` correctness check. No production code changes; no new abstractions.

## Technical Context

**Language/Version**: C++20 (existing tree/CUDA backend + gtest), matching the prior
session's `test_pipeline_chained_cu.cu`

**Primary Dependencies**: `runtime/record.hpp` (`Logger`, cycle timestamps — existing,
unmodified), `runtime/pipeline.hpp` (`make_dataset`, `worker_with_record` — existing),
`runtime/schedule.hpp` (`Schedule`, `validate_schedule_coverage`,
`first_concurrent_gpu_chunk` — existing), the prior session's genuinely-chained
`tree::AppData` `run_stage_N`/`dispatch_multi_stage` overloads (OMP + CUDA, already on
this branch)

**Storage**: None — all measurement is in-process (`Logger::records_`, already public);
no new files, logs, or database

**Testing**: One new gtest binary (`test-schedule-permutation-cu`), explicitly excluded
from `ctest -L cuda` (per FR-008) via an `experimental` label, matching the prior
session's `test-pipeline-chained-cu` precedent

**Target Platform**: Real Jetson hardware (`duck-stable`, cross-compiled via
`bt-cross:7.2`) — per the clarification session, CUDA only, no simulated timing

**Project Type**: Extension of the existing `apps/tree` test suite — no new project,
no new production code path

**Performance Goals**: N/A as an SLA — the sweep (29 schedules × 5 repeated runs = 145
pipeline executions) is bounded but not gated on a time budget, since it's explicitly
on-demand (FR-008), not part of routine CI

**Constraints**: Must not modify shared production profiler code
(`profiler/tree-cu/const.hpp` and friends, which bind `AppDataT = tree::SafeAppData` for
every production `bm_*` binary); must exercise the real concurrent runtime, never a
mock (FR-002); CUDA + real Jetson hardware only (clarification)

**Scale/Scope**: 29 schedule permutations (every contiguous GPU-range placement across
7 stages, plus the all-CPU case) × 5 repeated runs each, for a bounded, single-sweep
workload sized for on-demand use, not a CI gate

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Assessment |
|---|---|
| I. Simplicity First | **Pass.** Reuses `Logger`/`worker_with_record`/`make_dataset` (existing runtime primitives) and ports an already-written, already-validated overlap algorithm (`dashboard/generate.py`'s `_coverage_time`) rather than inventing new instrumentation or analysis. |
| II. Surgical, Traceable Changes | **Pass.** One new test file (+ CMake registration). No production code touched — deliberately avoids modifying the shared `profiler/tree-cu/const.hpp` that every production profiler binary depends on. |
| III. OMP-as-Oracle Differential Testing (NON-NEGOTIABLE) | **Pass / not triggered.** No kernel changes are made — this feature validates *scheduling/orchestration* correctness across permutations of already-oracle-verified stages, reusing the exact `CheckItemChained` (OMP-as-oracle) check from the prior session. It's an additional, deliberately on-demand assurance layer (per the clarification session), not a substitute for or exemption from the everyday `ctest -L omp`/`cuda` gate, which is unaffected. |
| IV. Goal-Driven Verification | **Pass.** The feature *is* a verification tool; `quickstart.md` defines how to run it and confirm its own reports are accurate end-to-end on real hardware. |
| V. Data & Docs as Source of Truth | **Pass.** No device topology or hardware-access facts change; this is pure test code. A short pointer will be added to `docs/instruction-for-ai/05-profiling.md` or `03-unit-testing.md` during implementation (task, not a plan-level concern). |

No violations. **Complexity Tracking is empty — no entries required.**

*Post-design re-check (after Phase 1): unchanged — data-model.md and quickstart.md
introduce nothing beyond what was assessed here; all five principles still pass.*

## Project Structure

### Documentation (this feature)

```text
specs/002-cpu-gpu-schedule-coverage/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md         # Phase 1 output
└── tasks.md              # Phase 2 output (/speckit-tasks — not created here)
```

No `contracts/` — this is an internal verification tool with no external interface
(no API, no CLI contract beyond "run the gtest binary"); skipped per the plan
template's own allowance for purely internal projects.

### Source Code (repository root)

Extension of the existing `apps/tree` test suite — one new file, one new CMake target:

```text
apps/tree/cuda/
└── test_schedule_permutation_cu.cu   # new: generates all 29 schedules, runs each
                                       # 5x through make_dataset+worker_with_record,
                                       # checks correctness (reused CheckItemChained
                                       # pattern) + overlap (ported _coverage_time
                                       # sweep over Logger::records_)

apps/tree/CMakeLists.txt              # +1 bt_add_cuda_test registration,
                                       # LABELS "experimental" (excluded from
                                       # ctest -L cuda), mirroring
                                       # test-pipeline-chained-cu's precedent
```

No changes to: `runtime/record.hpp`, `runtime/pipeline.hpp`, `runtime/schedule.hpp`
(all reused as-is), `profiler/tree-cu/*` (production profiler binaries, untouched),
`apps/tree/omp/dispatchers.*` / `apps/tree/cuda/dispatchers.*` (the genuinely-chained
overloads already exist from the prior session).

**Structure Decision**: extend the existing `apps/tree/cuda` test directory with one
new file, exactly following the prior session's `test_pipeline_chained_cu.cu`
structural precedent (own `main()`, own pool/queue setup, `experimental` CMake label).
No new top-level directory or build target category.

## Complexity Tracking

*No entries — Constitution Check reported no violations.*
