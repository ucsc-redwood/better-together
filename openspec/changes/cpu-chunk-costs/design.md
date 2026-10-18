# Design

## Context

- `bm-prof` (`profiler/bm_prof_common.hpp`) times a cell by calling `time_once` repeatedly: one stage, one tier,
  one AppData, calibrated sample count, no input reset between samples. That is correct only for stages that
  leave their input untouched.
- Tree's OMP stage 2 (`apps/tree/omp/dispatchers.cpp::run_stage_2`, `func_sort.hpp::parallel_sort`) sorts
  each thread's segment of `u_morton_keys_s1_out` in place, then merges into `u_morton_keys_sorted_s2_out`.
  From the second sample on, `std::sort` sees sorted segments.
- The per-stage model prices a CPU chunk as its stage sum plus `per_chunk + n * per_stage` from
  `data/profiling/<serial>/overhead.json` when that file exists (`smt/overhead.py`). Neither phone has one.
- Executed schedules and the accuracy summary exist (`data/sched_logs`, `optimizer/analysis/accuracy.py`,
  `speedup_summary.py`) from gpu-chunk-cost-model.

## Goals / Non-Goals

**Goals:**
- CPU stage times that match what a pipeline item costs.
- A fitted, validated CPU overhead on both phones.
- A measured before/after of the CPU-chunk error and of the best measured period.

**Non-Goals:**
- The interference table's over-pricing of CPU chunks (Pixel cifar-dense btpm: measured 0.62x predicted).
  It is reported by the gate, not changed here.
- GPU chunk prices, the range model, solver encodings.
- rocky-ryzen and the Jetsons: not re-profiled.

## Decisions

### D1. Fix the kernels, not the harness
Stages keep their inputs intact. Tree stage 2 copies its input segment into the output buffer and sorts and
merges there (using a scratch buffer for the merge ping-pong if needed); its result is unchanged.

*Alternative:* the profiler snapshots and restores a stage's inputs before every sample. Rejected: it needs
per-app knowledge of which buffers are inputs, adds copy time near the timed region, and leaves a kernel
whose behaviour depends on how often it ran.

### D2. One test guards every app
A shared OMP test helper runs each stage twice on one AppData (after producing its inputs with the stages before
it), compares the input buffers with a copy taken before, and compares the two outputs. It runs in `ctest -L omp`
for tree, cifar-dense and cifar-sparse, so a future in-place kernel fails CI.

### D3. Re-profile only what changed
Only tree's CPU records change. Re-run `bm-prof-tree-vk` on both phones, isolated and interference, 2 runs each,
replacing the tree/vulkan store (its GPU records are re-measured in the same runs, which is harmless). The stage-2
validation of the spec ("fresh input") is a one-off measurement with a small harness variant, recorded in this
design, not a permanent tool.

### D4. Fit on old executions, judge on new ones
- Fit `overhead.json` per phone from the existing executions (`data/sched_logs`, re-priced on the refreshed
  table, as `fit_overhead.py` already does).
- Solve the top-10 per (phone, app, table) with the refreshed store and that overhead (cifar-sparse top-5), and
  execute them as in the 2026-10-17 gate (3 rounds, 20 s apart).
- The new executions are the held-out set: the spec's held-out scenario compares their CPU-chunk error with and
  without the overhead, and the gate compares them with the 2026-10-17 numbers.

## Risks / Trade-offs

- **[Copying the input costs time]** → One 1.2 MB copy per item; it is part of the real stage cost and is
  measured with it.
- **[Overhead fits are pooled over CPU tiers]** (`cpu` class) → If the held-out check fails because tiers differ,
  record it; fitting per tier is a follow-up, not part of this change.
- **[New schedules move stage 2 off the CPU]** → Expected: with the true price the solver avoids it. The CPU-chunk
  error is then judged on whatever CPU chunks remain.

## Migration Plan

Kernel fix plus refreshed data. Rollback: revert the commits; old stores stay in git history.

## Measurements

### Stage 2 on fresh input (task 1.3)

- **Harness variant (one-off, not committed).** `profiler/bm_prof_common.hpp` was patched for one build only: with
  `BT_PROF_FRESH_INPUT` set, a CPU cell's `time_once` re-runs stages 1..s-1 (`prep`) before every sample, outside the
  timed region. The binary was copied aside as `bm-prof-tree-vk-fresh` and the source restored; the rebuilt
  `bm-prof-tree-vk` is byte-identical to the one built before the patch.
- **Run.** Pixel 7a, both binaries from the fixed code, `BT_PROF_SCENARIO=isolated`, `--benchmark_filter='^medium/stage2/'`,
  the whole process pinned to the Medium cores 4-5 with `taskset 30` (see "Tree CPU stages are not pinned" below),
  3 rounds alternating the two binaries, 20 s apart. 32-33 samples per run, cv 0.3-0.9 %.

| | run 1 | run 2 | run 3 | median | spread |
|---|---|---|---|---|---|
| `bm-prof-tree-vk` (repeated samples on one AppData) | 9.110 | 9.156 | 9.116 | 9.116 | 9.110-9.156 |
| fresh input before every sample | 9.108 | 9.126 | 9.114 | 9.114 | 9.108-9.126 |
| price before this change (`Medium [2,2]`, proposal) | | | | 0.64 | |

p50 in ms. The medians differ by 0.002 ms and each lies inside the other's spread: with the input left intact,
repeated samples on one AppData cost what a fresh item does, 14x the old price.

### Tree CPU stages are not pinned

`tree::omp::dispatch_multi_stage(cores, n, tree::AppData&, ...)` ignores `cores` (`apps/tree/omp/dispatchers.hpp`:
"Core-pinning intentionally unsupported for the AppData path"), and that is the overload both `bm-prof-tree-vk` and
`bm-gen-logs-tree-vk` call for an OMP cell or chunk. Stages 2, 3 and 6 are single-threaded on that path and run
on whatever core the scheduler gives the calling thread; stages 1, 4, 5 and 7 open an unpinned OMP team. So the
tree table's `big` / `medium` / `little` columns are the same unpinned measurement taken three times (e.g. the
Pixel's isolated stage 3 before this change: big 1.066 / 0.243, medium 0.193 / 0.194, little 0.194 / 0.194 ms
over the two runs), and a tree CPU chunk runs wherever the scheduler puts it. The cifar apps pin. This change
does not alter it (kernel input purity is its scope); it bounds how well any tree CPU-chunk price can match.
Task 1.3 pins the process with `taskset` so that its numbers are from the Medium tier.

## Open Questions

- ~~Whether other stages beyond tree stage 2 mutate their input~~: answered by task 1.1, only tree stage 2 does
  (on both tree AppData paths).
