# Design

## Context

- `tree::omp::dispatch_multi_stage(cores, n, tree::AppData&, a, b)` ignores `cores` and `n` and calls each
  `run_stage_N(AppData&)` in turn. Stages 1, 4, 5 and 7 open their own `#pragma omp parallel for` (default team size,
  no affinity); stages 2, 3 and 6 are single-threaded on the calling thread, which is the pipeline's chunk thread
  and is not pinned either.
- The `SafeAppData` overloads (the oracle path) and the cifar apps pin: one outer `#pragma omp parallel
  num_threads(n)` whose threads each call `bind_thread_to_cores(cores)`.
- libgomp keeps a thread pool per master thread; a new pool thread is created by `pthread_create` and inherits the
  creating thread's affinity mask.

## Goals / Non-Goals

**Goals:**
- Tree CPU chunks and tree tier profiling run on exactly their tier, with their thread count.
- A test that would have caught this, for every app.

**Non-Goals:**
- Changing tree's per-stage parallelism (which stages are parallel, how).
- The profiling loader CLI's `max_cv` default (solver gate is 1.0).
- GPU-side noise (DVFS levels).

## Decisions

### D1. Pin the calling thread and size the inner teams
The AppData overloads that take `cores, n` bind the calling thread to `cores`, run the stages with the OpenMP thread
count set to `n` for that thread (`omp_set_num_threads(n)` or `num_threads(n)` on each inner region), and restore
the previous mask and count afterwards. Inner team threads then inherit the mask.

*Check needed in task 1.1:* libgomp reuses pool threads across regions, so a pool created before pinning keeps its
old mask. If the test shows that, each inner `parallel for` binds its threads itself (`bind_thread_to_cores` at the
top of the region), as the SafeAppData path does.

*Alternative:* one outer team around the stage loop, as SafeAppData does. Rejected for this path: stages 2, 3 and 6
have no worksharing construct and would race inside a shared team (the comment in `dispatchers.hpp`), and wrapping
them in `omp single` changes the parallel structure.

### D2. Test from inside the stage
The test runs each app's CPU dispatch with a one-tier core list (on the build box: a subset of its cores) and records,
from inside every thread that executes stage work, `sched_getaffinity` and the thread id. A test hook in the
dispatch path is acceptable if a stage body cannot be observed otherwise; it must compile out of release builds.

### D3. Re-profile and re-execute tree only
Only tree's CPU records change. Re-profile `bm-prof-tree-vk` on both phones (isolated and interference, 2 runs
each), then solve top-10 tree schedules per (phone, table) with the per-stage model and execute them as in the
2026-10-18 gate (3 rounds, 20 s apart). The spec scenario "Profiling a tree stage on one tier" is a one-off check
against a `taskset`-pinned process.

## Risks / Trade-offs

- **[Pinned tree chunks get slower]** (fewer cores than the unpinned default) → Expected and correct: the old
  numbers borrowed other tiers' cores. The gate checks prediction error and the best period, not chunk speed.
- **[Restoring the mask is missed on an exception path]** → Use a scope guard.

## Migration Plan

Code fix plus a refreshed tree store. Rollback: revert; the superseded store stays beside the new one.

## Measurements

### D1's inheritance (task 1.2)

Binding only the calling thread (and setting its thread count) does not pin the inner teams. With that variant,
`CpuPinningOmp.TreeAppData` saw 6 threads run stage work for cores {0-5}: the caller with mask {0-5}, and 5 pool
threads with {0-11}. libgomp created them during the test's unpinned run, from the unbound caller, and reused
them for every later region. The profiler's main thread is in that position, moving from one tier's cells to the
next. So every tree parallel region binds its own threads (D1's fallback), as the SafeAppData path does.

### A tree stage on the little tier (task 1.3)

Pixel 7a, the binary of task 1.2, `BT_PROF_SCENARIO=isolated`, `--benchmark_filter='^little/stage[12]/'`, 3 rounds
alternating `bm-prof-tree-vk` as is and under `taskset f` (little cores 0-3), 20 s before every run. 125-129
samples per stage-1 run (cv 0.05-0.07), 11 per stage-2 run (cv 0.002-0.007). p50 in ms:

| | run 1 | run 2 | run 3 | median | spread |
|---|---|---|---|---|---|
| stage 1, `bm-prof-tree-vk` | 2.472 | 2.424 | 2.432 | 2.432 | 2.424-2.472 |
| stage 1, `taskset f` | 2.468 | 2.446 | 2.391 | 2.446 | 2.391-2.468 |
| stage 2, `bm-prof-tree-vk` | 26.220 | 26.253 | 26.194 | 26.220 | 26.194-26.253 |
| stage 2, `taskset f` | 26.238 | 26.270 | 26.242 | 26.242 | 26.238-26.270 |

Each median lies inside the other's spread. The store's unpinned `little` stage 2 was 7.020 / 7.148 ms
(2026-10-18 runs 1 / 2): the sort ran on faster cores than its label.

## Open Questions

- ~~Whether libgomp pool reuse defeats D1's inheritance~~: it does (Measurements, "D1's inheritance"); tree binds
  per region.
