# CPU chunk costs: the acceptance gate (2026-10-18)

**Question.** Do the cpu-chunk-costs changes bring the per-stage model's CPU-chunk prices closer to executed
pipelines on the Pixel 7a and the Samsung? There are two changes:

- tree stage 2 is profiled on intact input;
- each phone gets a fitted per-chunk CPU overhead.

The spec (`openspec/changes/cpu-chunk-costs/specs/optimizer/cost-model/spec.md`) sets four bounds:

1. The median CPU-chunk `|measured / predicted - 1|` falls below the 2026-10-17 gate's value for tree on both tables.
2. It is not higher for cifar-dense and cifar-sparse.
3. The best measured period per (phone, app, table) is not worse beyond run-to-run noise.
4. Each phone's fitted overhead lowers the CPU-chunk error of executions it was not fitted on (held-out check).

**Decision.**

- **The stage-2 fix stays.** It meets bounds 1 and 3 in every cell. Tree's CPU-chunk error falls in all four
  (phone, table) cells (Samsung isolated 3.03 to 1.17, Pixel btpm 0.37 to 0.22). No best period is worse. Two are
  better beyond the noise: the Samsung's isolated tree (7.11 ms, was 22.30) and the Pixel's isolated cifar-dense
  (an unchanged schedule, so that one is session-to-session variation). The tree "before" also predates
  tree-app-improvements (see section 1), so not all of tree's gain is this change's.
- **No `overhead.json` ships for either phone.** Both fail the held-out check. On the Pixel the fitted constants
  are zero, so the error is unchanged (0.315 with and without). On the Samsung the 0.553 ms per chunk makes the
  error worse (0.307 with, 0.244 without). The fitted files are kept in
  `data/archive/2026-10-18-cpu-chunk-costs-rejected-overhead/`.
- **Bound 2 fails as written in 5 of the 8 cifar cells.** The 3 Pixel cells are repeat measurements of the same
  schedules at the same prices: the Pixel's cifar tables are unchanged and its overhead is zero, so the error moved
  only with session-to-session variation, which the bound makes no allowance for. The 2 Samsung cells
  (cifar-dense btpm, cifar-sparse isolated) were solved under the rejected overhead and picked other schedules.

So the change's spec is **not met**: the "Each phone ships a fitted CPU chunk overhead" requirement fails on both
phones, and the cifar part of "CPU chunk predictions are checked against executed schedules" fails in 5 cells.

## 1. What was run

- **Before.** The 2026-10-17 gate ran the code of commit 965e294 on the store of 2026-10-16. That was before
  tree-app-improvements (merged 4765310), which changed tree stages 1, 2 (GPU), 3 and 7 and re-profiled the tree
  store. Tree's before/after therefore includes that change's kernels as well as this one; the cifar code is
  the same on both sides.
- **Code.** Tree stage 2 copies its keys and sorts the copy (`run_stage_2(tree::AppData&)`; `omp::parallel_sort`
  takes a const input). `InputPurityOmp.*` checks every OMP stage of the three apps, and only tree stage 2 failed it
  before the fix. Commit ef8026a.
- **Store.** Tree re-profiled on both phones, isolated and interference, 2 runs each, 2026-10-18 07:45 UTC
  (`data/profiling/<serial>/tree/vulkan/{isolated,interference}/run-00{1,2}.jsonl`, commit b042d19).
  - Stage 2 on a CPU tier is now 5.4-16 ms; it was 0.3-1.2 ms.
  - The cifar stores and every range record are unchanged.
  - The store the 2026-10-17 gate solved on is kept beside it in `superseded-2026-10-17/`.
- **Overhead fit.** `optimizer/analysis/fit_overhead.py` was fitted on the 2026-10-17 executions, re-priced on the
  refreshed tree table (367 chunk samples):
  - Pixel: cpu 0, gpu_vulkan 0 (model `none`).
  - Samsung: cpu 0.5528 ms per chunk (model `median`), gpu_vulkan 0.
  - Those executions and their schedule files moved to `data/archive/2026-10-17-gpu-chunk-cost-model/`, so they do not
    pool with the new runs. `fit_overhead.py --dry-run --sched-logs data/archive/2026-10-17-gpu-chunk-cost-model/sched_logs --schedules-root 'data/archive/2026-10-17-gpu-chunk-cost-model/schedules_{table}'`
    reproduces the fit.
- **Schedules.** For each phone, app and table:

  ```
  02_gen_schedule_merged.py --device <serial> --app <app> --backend vk --table_type <btpm|isolated>
    --minimize_mode tmax --num_solutions 10 --output_folder data/schedules_<table>
  ```

  - The per-stage model (the default), with the fitted `overhead.json` in the store.
  - With an overhead file present, `02` also runs the union sweep, which adds 0-2 plain-model candidates.
- **Executions** (2026-10-18 00:52-01:59 local time, as in the 2026-10-17 gate):
  - Per (phone, app, table), the top 10 by predicted period (cifar-sparse top 5), each in its own `bm-gen-logs-<app>-vk`
    process with a one-schedule JSON, from `/data/local/tmp/cpucost` with the real weights.
  - 3 rounds in the same order, 20 s idle before every run, the two phones at the same time.
  - 50 runs per round per phone, 300 runs, none failed. Logs: `data/sched_logs/<serial>_<app>_vk_<table>/`.
- **Measures.** The same method as the 2026-10-17 report, sections 1-3.
  - Period per run: each chunk's mean duration summed per PU, maxed over PUs. A schedule's period is the median
    over the runs of its assignment, pooled over both tables' folders; its spread is their min-max.
  - Per chunk: the median over those runs of the chunk's mean duration, against the chunk's price.
  - CPU (GPU) error of a cell: the median `|ratio - 1|` over the CPU (GPU) chunks of its executed schedules.
  - Noise for bound 3: the larger min-max spread of the two best schedules.
  - The script that computes this is kept outside the repository. Run on the 2026-10-17 logs, schedules and store,
    it reproduces that report's per-stage rows exactly (sections 2 and 3, every cell). `speedup_summary.py`
    writes the period summary (`data/sched_logs/speedup-summary.md`).

## 2. The gate, every cell

"Before" is the 2026-10-17 gate's per-stage row. "After, as solved" prices each CPU chunk as it was solved, so the
Samsung's include the fitted 0.553 ms. "After, no overhead" prices it as the plain stage sum, which is what the
store now ships.

| Phone | App | Table | CPU chunks before / after | CPU error before | after, as solved | after, no overhead | Bound | Holds | Best before ms [min-max] | Best after ms [min-max] | Change vs noise | Holds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pixel 7a | tree | btpm | 21 / 16 | 0.372 | 0.218 | 0.218 | lower | yes | 11.70 [10.39-23.37] | 11.54 [10.44-21.44] | -0.16 vs 12.98 | yes |
| Pixel 7a | tree | isolated | 28 / 30 | 0.376 | 0.297 | 0.297 | lower | yes | 28.12 [27.91-28.74] | 26.55 [24.96-26.70] | -1.57 vs 1.73 | yes |
| Pixel 7a | cifar-dense | btpm | 28 / 28 | 0.401 | 0.409 | 0.409 | not higher | **no** | 43.26 [42.99-43.40] | 43.19 [42.90-43.43] | -0.08 vs 0.53 | yes |
| Pixel 7a | cifar-dense | isolated | 30 / 30 | 0.245 | 0.260 | 0.260 | not higher | **no** | 39.74 [39.72-39.79] | 39.59 [39.55-39.63] | -0.14 vs 0.07 | yes |
| Pixel 7a | cifar-sparse | btpm | 17 / 17 | 0.134 | 0.215 | 0.215 | not higher | **no** | 444.91 [438.17-469.96] | 431.24 [419.81-451.62] | -13.67 vs 31.81 | yes |
| Pixel 7a | cifar-sparse | isolated | 17 / 17 | 0.348 | 0.329 | 0.329 | not higher | yes | 444.91 [438.17-469.96] | 431.24 [419.81-451.62] | -13.67 vs 31.81 | yes |
| Samsung | tree | btpm | 19 / 16 | 1.531 | 0.366 | 0.229 | lower | yes | 5.10 [3.57-5.25] | 3.88 [3.83-4.84] | -1.23 vs 1.67 | yes |
| Samsung | tree | isolated | 30 / 30 | 3.028 | 1.165 | 1.783 | lower | yes | 22.30 [22.30-29.64] | 7.11 [6.84-9.55] | -15.19 vs 7.35 | yes |
| Samsung | cifar-dense | btpm | 27 / 27 | 0.072 | 0.186 | 0.093 | not higher | **no** | 11.46 [11.44-11.49] | 11.26 [11.22-11.45] | -0.21 vs 0.22 | yes |
| Samsung | cifar-dense | isolated | 30 / 30 | 0.074 | 0.044 | 0.061 | not higher | yes | 10.93 [10.89-11.08] | 10.99 [10.92-11.02] | +0.06 vs 0.19 | yes |
| Samsung | cifar-sparse | btpm | 14 / 14 | 0.425 | 0.296 | 0.346 | not higher | yes | 237.54 [212.42-267.32] | 213.92 [211.10-215.12] | -23.62 vs 54.90 | yes |
| Samsung | cifar-sparse | isolated | 20 / 13 | 0.113 | 0.157 | 0.155 | not higher | **no** | 192.13 [191.86-225.03] | 183.20 [182.14-194.23] | -8.93 vs 33.17 | yes |

The verdict on bound 2 is the same under either pricing in every cell. The best schedules after:

| Phone | App | Table | Best schedule | Runs |
|---|---|---|---|---|
| Pixel 7a | tree | btpm | SCH-L2G5-G160-4fd8 | 3 |
| Pixel 7a | tree | isolated | SCH-B1M1L4G1-G590-c1af | 3 |
| Pixel 7a | cifar-dense | btpm | SCH-M4B1G5L1-G4821-09d0 | 6 |
| Pixel 7a | cifar-dense | isolated | SCH-L2B3G4M2-G4187-d33c | 3 |
| Pixel 7a | cifar-sparse | btpm | SCH-B4G3L1M3-G44949-fd1f | 6 |
| Pixel 7a | cifar-sparse | isolated | SCH-B4G3L1M3-G41421-a525 | 6 |
| Samsung | tree | btpm | SCH-L1G6-G175-d87b | 3 |
| Samsung | tree | isolated | SCH-G2L1B2M2-G334-d4eb | 3 |
| Samsung | cifar-dense | btpm | SCH-L2M1B1G7-G1060-853e | 3 |
| Samsung | cifar-dense | isolated | SCH-L2G6B1M2-G514-c9ab | 3 |
| Samsung | cifar-sparse | btpm | SCH-L2B2G3M4-G25650-260f | 3 |
| Samsung | cifar-sparse | isolated | SCH-M4G4B3-G1607-12f5 | 3 |

Period error and GPU error, for context (not gated). No GPU overhead was fitted. Tree's GPU cells were
re-measured in the same profiling runs, on tree-app-improvements' kernels; the cifar ones are unchanged.

| Phone | App | Table | GPU error before | after | Period error before | after |
|---|---|---|---|---|---|---|
| Pixel 7a | tree | btpm | 0.438 | 0.438 | 1.115 | 0.615 |
| Pixel 7a | tree | isolated | 0.617 | 0.348 | 4.726 | 2.818 |
| Pixel 7a | cifar-dense | btpm | 0.114 | 0.116 | 0.114 | 0.116 |
| Pixel 7a | cifar-dense | isolated | 0.119 | 0.119 | 0.121 | 0.120 |
| Pixel 7a | cifar-sparse | btpm | 0.013 | 0.009 | 0.015 | 0.009 |
| Pixel 7a | cifar-sparse | isolated | 0.085 | 0.080 | 0.090 | 0.080 |
| Samsung | tree | btpm | 0.950 | 0.687 | 1.467 | 0.085 |
| Samsung | tree | isolated | 0.518 | 0.284 | 9.695 | 1.403 |
| Samsung | cifar-dense | btpm | 0.065 | 0.066 | 0.056 | 0.060 |
| Samsung | cifar-dense | isolated | 0.122 | 0.124 | 0.035 | 0.049 |
| Samsung | cifar-sparse | btpm | 0.495 | 0.497 | 0.249 | 0.344 |
| Samsung | cifar-sparse | isolated | 0.196 | 0.199 | 0.090 | 0.162 |

## 3. The held-out check

The overhead was fitted on the 2026-10-17 executions. The new executions are an independently solved and executed
set (spec scenario "Held-out check"). Median CPU-chunk `|measured / predicted - 1|` over all of a phone's new
executions:

| Phone | CPU chunks | With the fitted overhead | Without | Lower with? | `overhead.json` |
|---|---|---|---|---|---|
| Pixel 7a | 138 | 0.315 | 0.315 | no (the fit is zero) | not shipped |
| Samsung | 130 | 0.307 | 0.244 | no (higher) | not shipped |

Per cell on the Samsung, with / without: tree btpm 0.366 / 0.229, tree isolated 1.165 / 1.783, cifar-dense btpm
0.186 / 0.093, cifar-dense isolated 0.044 / 0.061, cifar-sparse btpm 0.296 / 0.346, cifar-sparse isolated 0.157 /
0.155. The constant helps where CPU chunks run over their stage sum (tree isolated, cifar-dense isolated) and hurts
where they run at or under it. On cifar-dense btpm, for example, the CPU chunks run close to their stage sum (0.093
plain), and the 0.553 ms doubles the error (0.186).

The fit is a single constant pooled over apps, tables and tiers, and it does not fit the pooled residuals:

- **Pixel.** The median CPU residual is -0.575 ms, because the cifar tables over-price CPU chunks (cifar-dense btpm
  median measured/priced 0.62, isolated 0.77) while tree's run over (isolated 1.64). The fit clamps to zero.
- **Samsung.** The fit selects the median residual, 0.553 ms. A pooled median is not a fixed per-chunk cost. In
  the fitting set the chunks priced under 1 ms that the proposal pointed at are a handful (Samsung: 14 of 131, 8 of
  them in tree isolated).

## 4. Reading it

- **Tree: stage 2 was the error the proposal said it was.** On 2026-10-17 a chunk holding stage 2 was priced
  0.3-1.2 ms and ran 14-27 ms. Now it is priced 6-8 ms:
  - Where the GPU is cheaper the solver now keeps stage 2 there: the Samsung's GPU stage 2 costs 1.3 ms, and its
    four best-priced btpm candidates run stage 2 on the GPU.
  - On the Pixel stage 2 costs about what its GPU stage 2 does (7 ms), so most candidates still put it on a CPU tier.
- **What remains of the tree CPU error is co-running chunks.** A Pixel Medium `[2,2]` chunk priced 7.0 ms runs 27 ms
  (3.8x) in the isolated-table schedules, and 15-21 ms against a 7.9 ms price under btpm. The same sort runs 9.1 ms
  alone on the Medium cores (design.md, "Measurements"). The tree OMP path that `bm-prof` and the pipeline call
  ignores the core list (design.md, "Tree CPU stages are not pinned"), so:
  - the sort thread is unpinned;
  - every other CPU chunk's `#pragma omp parallel for` opens an unpinned team over all cores;
  - all of them share the cores with the GPU chunk's host thread.

  A per-stage table cannot price that contention, and a constant cannot either. The Samsung's isolated tree chunks
  run 1.8-8.6x their price for the same reason: their stages are short, and three CPU chunks co-run with the GPU
  chunk.
- **cifar: the CPU prices did not change on the Pixel.** Its cifar stores are untouched and the fitted overhead is
  zero, so its executed cifar schedules are the same assignments at the same prices as on 2026-10-17 (10/10, 10/10,
  6/6, 6/6). Their CPU error moved between 0.401 and 0.409, between 0.245 and 0.260, and from 0.134 to 0.215 (the
  last a median over 17 chunks of 6 schedules): what two sessions measure for the same chunks. For comparison, the
  per-round values within the new session spread over 0.398-0.405, 0.248-0.268 and 0.223-0.258. Bound 2 as written
  has no noise term, so these cells fail it. On the Samsung the fitted overhead changed the candidates (0 of each
  cell's uids are shared with 2026-10-17), and two cells are higher; cifar-dense btpm is higher under either pricing
  (0.186 as solved, 0.093 plain, 0.072 before).

## 5. What follows (not done here)

- **Pin tree's OMP chunks** (`tree::omp::dispatch_multi_stage(cores, ...)` on the `tree::AppData` path) or give the
  single-threaded stages a pinned thread. Until then a tree CPU tier is a label, and the remaining tree CPU-chunk
  error comes from co-running unpinned work rather than from the table.
- **Fit per app or per tier, or not at all.** One constant pooled over apps cannot serve the Pixel, where the cifar
  tables over-price and tree under-prices. The design already lists per-tier fits as a follow-up.
- **Give bound 2 a noise allowance.** For example: not higher beyond the per-round spread of the same session. As
  written, an unchanged cell can fail it.

## 6. How to reproduce

```bash
# store: docs/instruction-for-ai/06-end-to-end-scheduling.md Step 1 (01_collect_profiling.py, --tools prof --runs 2
#   --scenarios isolated,interference --cooldown-s 20 --dest /data/local/tmp/<dir>)
# overhead fit (the rejected constants):
uv run optimizer/analysis/fit_overhead.py --dry-run \
  --sched-logs data/archive/2026-10-17-gpu-chunk-cost-model/sched_logs \
  --schedules-root 'data/archive/2026-10-17-gpu-chunk-cost-model/schedules_{table}'
# schedules as executed: copy data/archive/2026-10-18-cpu-chunk-costs-rejected-overhead/<serial>/overhead.json
# into data/profiling/<serial>/ first (the store ships none now), then per phone, app, table:
uv run optimizer/orchestrate/02_gen_schedule_merged.py --profiling_root data/profiling --device <serial> \
  --app <app> --backend vk --table_type <btpm|isolated> --minimize_mode tmax --num_solutions 10 \
  --output_folder data/schedules_<table>
# executions: the top 10 (cifar-sparse top 5) of each file by metrics.max_time, each written to a one-schedule
# JSON and run 3 rounds x in order, 20 s apart, from /data/local/tmp/<dir> holding libc++_shared.so and weights/:
adb -s <serial> shell 'cd /data/local/tmp/<dir> && BT_WEIGHTS_DIR=/data/local/tmp/<dir>/weights \
  LD_LIBRARY_PATH=. ./bm-gen-logs-<app>-vk --device <serial> --schedule-file one-NN.json \
  --n-schedules-to-run 1' </dev/null > data/sched_logs/<serial>_<app>_vk_<table>/schedule_run_<k>.log
# period summary:
uv run optimizer/analysis/speedup_summary.py --sched-logs data/sched_logs --schedules-root data
```

Reproduction check (task 3.3), with the store as shipped (no `overhead.json`; the Pixel's fit was zero, so its
prices are those of 3.1):

- **Pixel tree btpm.** The command above returns the 10 executed assignments of 3.1 with the same uids and prices.
  Three runs of its best schedule, SCH-L2G5-G160-4fd8, measured 9.84 / 21.17 / 28.78 ms (3.1: 11.54 [10.44-21.44]).
  The cell is bimodal: its Little `[1,2]` chunk holds the unpinned sort.
- **Pixel cifar-dense btpm.** 10 of 10 assignments with the same uids and prices. SCH-M4B1G5L1-G4821-09d0 measured
  43.25 / 42.98 / 43.26 ms (3.1: 43.19 [42.90-43.43]).
