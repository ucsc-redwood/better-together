# Proposal

## Why

Tree's CPU chunks are not pinned to their tier. The `tree::AppData` OMP path that the pipeline (`bm-gen-logs-tree-vk`)
and the profiler (`bm-prof-tree-vk`) call takes a core list and thread count and ignores both
(`apps/tree/omp/dispatchers.hpp`, "Core-pinning intentionally unsupported for the AppData path"). Each
`run_stage_N(AppData&)` opens its own `#pragma omp parallel for` with the default team: all cores, unpinned.
The cifar apps pin (`bind_thread_to_cores` inside one outer team).

Consequences, all measured in earlier changes:
- **Tree's `big` / `medium` / `little` profiling cells are not tier measurements** (`docs/instruction-for-ai/05-profiling.md`,
  "Tree's CPU tiers are nominal"). The solver assigns tiers from them anyway.
- **Pipelined tree CPU chunks compete for every core**, with each other and with the GPU chunk's host thread. The
  Pixel 7a's best tree btpm schedule ran 9.84 / 21.17 / 28.78 ms in three runs (cpu-chunk-costs task 3.3): bimodal,
  because a "Little" chunk holding the sort runs wherever the scheduler puts it.
- Tree is the app where CPU chunks are the largest remaining error (cpu-chunk-costs gate: 0.22-1.17 median CPU-chunk
  error, against 0.04-0.41 for cifar).

## What Changes

- **Tree's AppData path pins.** `dispatch_multi_stage(cores, n, AppData&, …)` and `dispatch_stage` run the stages
  on exactly the given cores with the given thread count, for every stage (including the single-threaded 2, 3 and 6).
  The results are unchanged.
- **A test checks pinning for every app.** Every OMP thread a CPU chunk uses runs on a core of its tier, for tree,
  cifar-dense and cifar-sparse.
- **Re-profile tree on both phones**, then re-solve and execute tree schedules, and compare with the 2026-10-18
  cpu-chunk-costs gate.

## Capabilities

### New Capabilities
- `runtime/cpu-chunks`: a CPU chunk runs only on its tier's cores, with its tier's thread count.

### Modified Capabilities
- `optimizer/cost-model`: adds the tree accuracy check after pinning.

## Impact

- **Code:** `apps/tree/omp/{dispatchers.hpp,dispatchers.cpp}`; a pinning test in the OMP test suites.
- **Data:** `data/profiling/<serial>/tree/vulkan/{isolated,interference}/run-*.jsonl` refreshed (the previous runs move
  to a `superseded-<date>/` directory), new tree executions in `data/sched_logs`.
- **Docs:** remove "Tree's CPU tiers are nominal" from `docs/instruction-for-ai/05-profiling.md`.
- **Measurement:** phones only. Nothing is profiled on rocky-ryzen.
- **Out of scope:** the profiling loader's CLI `max_cv=0.1` default. The solver's loader already gates at 1.0
  (`smt/data_loader.py::SCENARIO_GATES`), so no solver input is dropped by it.
