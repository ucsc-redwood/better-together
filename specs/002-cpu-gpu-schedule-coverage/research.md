# Phase 0 Research: CPU/GPU Schedule Permutation & Overlap Coverage for Tree

All Technical Context items resolved directly from existing repo infrastructure — this
feature reuses far more than it builds new.

## 1. How to generate "every valid CPU/GPU schedule permutation"

**Decision**: The GPU stage-range is a contiguous `[i, j]` with `1 <= i <= j <= 7`
(28 non-empty placements), plus the all-CPU case (GPU range empty) = **29 schedules
total**. For `i > 1`, a CPU chunk covers `[1, i-1]`; for `j < 7`, a CPU chunk covers
`[j+1, 7]`; both use `first_present_cpu_type()` (matching the existing convention in
`bm_gen_log_common.hpp`'s `make_warmup_schedule`). This is generated in-process by a
small loop, not read from a schedule JSON file.

**Rationale**: `runtime/schedule.hpp`'s `first_concurrent_gpu_chunk()` guard (already
enforced everywhere schedules run) rejects more than one GPU chunk per schedule — so a
contiguous single GPU range is the *entire* valid space, not a simplification. 29 is
small enough to run exhaustively on real hardware in one sweep.

**Alternatives considered**:
- Reading schedules from `schemas/schedule.schema.json` / z3-generated JSON — rejected:
  this feature needs the *exhaustive* space, not whatever z3 happened to propose;
  generating it directly is simpler and guarantees completeness (Simplicity First).

## 2. How to run each schedule through the real concurrent runtime

**Decision**: Reuse `runtime/pipeline.hpp`'s `make_dataset`/`worker_with_record` and
`runtime/record.hpp`'s `Logger<kNumToProcess>` directly — the same primitives
`profiler/bm_gen_log_common.hpp`'s `run_schedule()` already uses for production
Gantt-log generation, just pooling the genuinely-chained `tree::AppData` (from the
prior session) instead of `tree::SafeAppData`.

**Rationale**: `worker_with_record` already timestamps every (task, chunk) work window
via `Logger::start_tick`/`end_tick` — exactly the data FR-004's overlap measurement
needs. No new instrumentation required, only a new call site.

**Alternatives considered**:
- Modifying `profiler/tree-cu/const.hpp` (which binds `AppDataT = tree::SafeAppData` for
  every `profiler/tree-cu/*.cu` production binary) to add a build-time switch — rejected:
  would entangle this deliberately-experimental feature with production profiling tools
  (`bm_prof`, `bm_baseline`, `bm_fully_vs_normal`, `bm_gen_log`), violating the prior
  session's explicit precedent of keeping the new AppData path isolated in dedicated
  test files.

## 3. How to detect genuine overlap from the recorded timestamps

**Decision**: Port `dashboard/generate.py`'s existing `_coverage_time(intervals,
min_cover)` sweep-line algorithm (already used to compute the dashboard's "Pipeline
timeline" steady-state `concurrency_pct` from these exact `Logger` records) directly
into the new C++ test: sort each chunk-interval's start/end as +1/-1 events, sweep, and
sum the wall-time where >= 2 chunks (a CPU one and the GPU one, for two *different*
tasks) are simultaneously active. "Steady-state" = discard the first few processed
items (cold-start/ramp-up), matching the dashboard's `PIPE_WARMUP` convention.

**Rationale**: this exact overlap math is already written, already validated (it's what
produces the dashboard's existing measured Gantt/concurrency numbers), and small enough
to re-implement in ~20 lines of C++ against `Logger::records_` (public) directly —
avoids a round-trip through text logs and Python for a self-contained gtest.

**Alternatives considered**:
- Shelling out to the existing Python `dashboard/generate.py` code from the C++ test —
  rejected: adds a runtime dependency (Python + the dashboard's data layout) to what
  should be a self-contained, on-demand C++ verification tool; the core algorithm is
  tiny enough to port directly.

## 4. Correctness check per schedule

**Decision**: Reuse the exact `CheckItemChained` pattern from the prior session's
`test_pipeline_chained_cu.cu`: for each pooled item, build a fresh reference
`tree::AppData` seeded with the same input points, run `tree::omp::run_stage_1..7`
sequentially (the OMP oracle), and diff the final octree buffers exactly.

**Rationale**: already written, already proven correct on real Jetson hardware in the
prior session — this feature is that same check, run across the full schedule space
instead of the 1-2 schedules already covered.

**Alternatives considered**: none — this is a direct reuse, not a new design point.

## 5. Repeated-run evidentiary bar for the overlap verdict

**Decision**: Run each schedule **5 times** (odd number avoids ties); a schedule's
overlap verdict is "genuinely overlapping" only if the measured `concurrency_pct`-style
signal is non-zero in **at least 3 of 5** runs.

**Rationale**: matches the clarification session's decision (majority-of-repeated-runs,
not a single observation) with a concrete, boundable number — 29 schedules × 5 runs is a
bounded, single-sweep-sized workload appropriate for an on-demand, non-gating tool
(FR-008), while still ruling out a one-off timing fluke on real (noisy) hardware.

**Alternatives considered**:
- A numeric `concurrency_pct` threshold (e.g. "must exceed 10%") — rejected per the
  spec's own resolved assumption: overlap is a qualitative/binary determination, not a
  percentage target; any non-zero measured concurrent-execution time already
  distinguishes "overlapping" from "fully serialized."

## 6. Where this lives / how it's gated

**Decision**: One new gtest file, `apps/tree/cuda/test_schedule_permutation_cu.cu`,
built as CMake target `test-schedule-permutation-cu`, labeled `experimental` (excluded
from `ctest -L cuda`) — following the exact precedent set by
`test-pipeline-chained-cu` in the prior session.

**Rationale**: consistent, already-reviewed pattern for "new verification of the
genuinely-chained AppData path that isn't ready to be a maintained gate."

## Technical Context resolution summary

| Item | Resolution |
|---|---|
| Language/Version | C++20 (existing tree/CUDA backend + gtest), reusing `runtime/` and `apps/tree/` infra as-is |
| Primary Dependencies | `runtime/record.hpp` (`Logger`, cycle timestamps), `runtime/pipeline.hpp` (`make_dataset`, `worker_with_record`), `runtime/schedule.hpp` (`Schedule`, `validate_schedule_coverage`, `first_concurrent_gpu_chunk`), the prior session's `tree::AppData` genuinely-chained `run_stage_N`/`dispatch_multi_stage` overloads (OMP + CUDA) |
| Storage | None — all measurement is in-process (`Logger::records_`, public, read directly); no new files or logs |
| Testing | One new `experimental`-labeled gtest binary; explicitly excluded from `ctest -L omp`/`cuda` per FR-008 |
| Target Platform | Real Jetson hardware (`duck-stable`, cross-compiled via `bt-cross:7.2`) |
| Project Type | Extension of the existing `apps/tree` test suite — no new project |
| Performance Goals | N/A as a target — the sweep's own runtime (29 schedules × 5 runs) is bounded but not SLA'd, since it's on-demand (FR-008) |
| Constraints | Must not modify shared production profiler code (`profiler/tree-cu/const.hpp` et al.); must reuse the real concurrent runtime, not a mock (FR-002); CUDA + real Jetson hardware only (per clarification) |
| Scale/Scope | 29 schedule permutations × 5 repeated runs = 145 pipeline executions per full sweep, each with `kPoolSize`/`kNumToProcess` sized like the prior session's experimental tests |
