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

## Open Questions

- Whether libgomp pool reuse defeats D1's inheritance (task 1.1 answers it).
