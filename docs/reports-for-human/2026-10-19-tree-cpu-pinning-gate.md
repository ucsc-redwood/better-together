# Tree CPU pinning: the acceptance gate (2026-10-19)

**Question.** Once tree's CPU chunks run on their tier (tree-cpu-pinning) and tree is re-profiled, are tree's
per-stage CPU-chunk prices at least as good as at the 2026-10-18 cpu-chunk-costs gate, on the Pixel 7a and the
Samsung? The spec (`openspec/changes/tree-cpu-pinning/specs/optimizer/cost-model/spec.md`) sets two bounds per
(phone, table):

1. The median CPU-chunk `|measured / predicted - 1|` of the executed per-stage tree schedules is not higher than
   the 2026-10-18 value by more than 0.1.
2. The best measured tree period is not worse beyond run-to-run noise.

**Decision: the gate passes in all four cells.**

- **CPU-chunk error falls in every cell**, to 0.06-0.27: Pixel btpm 0.218 to 0.084, isolated 0.297 to 0.062;
  Samsung btpm 0.229 to 0.207, isolated 1.783 to 0.268 (before priced as the store ships; see section 2).
- **No best period is worse.** Two are better beyond the noise: Pixel isolated 26.55 to 6.87 ms, Samsung isolated
  7.11 to 3.94 ms. Pixel btpm (11.54 to 8.60) is better within the noise of the old bimodal schedule, and Samsung
  btpm is unchanged (3.88 to 3.95, noise 1.01).
- **The run-to-run spread of the best schedules narrowed.** The Pixel's btpm best ran 8.46-8.98 ms over 3 runs
  (2026-10-18: 10.44-21.44, bimodal because its Little chunk held the unpinned sort).
- Not gated: the GPU-chunk error rose under the btpm table (Pixel 0.438 to 0.955, Samsung 0.687 to 1.521).
  Section 4 explains why.

## 1. What was run

- **Before.** The 2026-10-18 gate's tree cells: the code of ef8026a (unpinned tree CPU path), its store (now in
  `data/profiling/<serial>/tree/vulkan/<scenario>/superseded-2026-10-18/`), and its executions and schedule files
  (moved to `data/archive/2026-10-18-cpu-chunk-costs/{sched_logs,schedules_btpm,schedules_isolated}/`, so that they
  do not pool with the new runs by assignment).
- **Code.** Commit 3d2dc35: `dispatch_multi_stage(cores, n, tree::AppData&, …)` runs every stage on the tier's
  cores with its thread count. `CpuPinningOmp.*` checks this for all three apps (tasks 1.1-1.2).
- **Store.** Tree re-profiled on both phones, isolated and interference, 2 runs each, 2026-10-19 20:54 UTC
  (`data/profiling/<serial>/tree/vulkan/{isolated,interference}/run-00{1,2}.jsonl`, binary of 3d2dc35). The range
  records and every cifar store are unchanged. The old vs new CPU p50 per tier and stage are in tasks.md, 2.1.
- **Schedules.** Per phone and table, `02_gen_schedule_merged.py --profiling_root data/profiling --device <serial>
  --app tree --backend vk --table_type <btpm|isolated> --minimize_mode tmax --num_solutions 10 --output_folder
  data/schedules_<table>`: the per-stage model, no `overhead.json` (the store ships none), so no union sweep and
  plain stage sums. 10 candidates per file.
- **Executions** (2026-10-19 20:59-21:21 UTC):
  - Per (phone, table), the 10 candidates by `metrics.max_time`, each as a one-schedule JSON in its own
    `bm-gen-logs-tree-vk` process from `/data/local/tmp/pin`.
  - 3 rounds in the same order (btpm 1-10, then isolated 1-10), 20 s idle before every run, the two phones at the
    same time. 120 runs, none failed. Logs: `data/sched_logs/<serial>_tree_vk_<table>/schedule_run_<k>.log`, 30 per
    cell.
- **Measures.** The method of the 2026-10-17 and 2026-10-18 reports (section 1 of each), now committed as
  `optimizer/analysis/chunk_accuracy.py`:
  - Period per run: each chunk's mean duration summed per PU, maxed over PUs. A schedule's period is the median
    over the runs of its assignment, pooled over both tables' folders; its spread is their min-max.
  - Per chunk: the median over those runs of the chunk's mean duration, against the chunk's price.
  - CPU (GPU) error of a cell: the median `|ratio - 1|` over the CPU (GPU) chunks of its executed schedules.
  - Noise for bound 2: the larger min-max spread of the two best schedules (before and after).
  - Checked before use: on the 2026-10-18 logs, schedules and rejected overheads it reproduces that report's
    section 2 exactly, every cell (the four tree rows and the eight cifar rows, both pricings, GPU and period
    error); on the 2026-10-17 archive it reproduces that report's per-stage tree rows (sections 2 and 3).
  - `speedup_summary.py` rewrote the period summary (`data/sched_logs/speedup-summary.md`); only its tree rows changed.

## 2. The gate, every cell

"Before, as solved" prices each 2026-10-18 CPU chunk as it was solved (the Samsung's under the rejected 0.553 ms
per chunk); "before, no overhead" as the plain stage sum, which is how the store ships and how this change's
schedules are priced. The bound is checked against both; the verdict is the same.

| Phone | Table | CPU chunks before / after | CPU error before, as solved | before, no overhead | after | Bound (before + 0.1) | Holds | Best before ms [min-max] | Best after ms [min-max] | Change vs noise | Holds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Pixel 7a | btpm | 16 / 29 | 0.218 | 0.218 | 0.084 | 0.318 | yes | 11.54 [10.44-21.44] | 8.60 [8.46-8.98] | -2.94 vs 11.00 | yes |
| Pixel 7a | isolated | 30 / 30 | 0.297 | 0.297 | 0.062 | 0.397 | yes | 26.55 [24.96-26.70] | 6.87 [6.85-6.87] | -19.68 vs 1.74 | yes |
| Samsung | btpm | 16 / 17 | 0.366 | 0.229 | 0.207 | 0.329 (0.466 as solved) | yes | 3.88 [3.83-4.84] | 3.95 [3.83-4.23] | +0.07 vs 1.01 | yes |
| Samsung | isolated | 30 / 28 | 1.165 | 1.783 | 0.268 | 1.265 (1.883 no overhead) | yes | 7.11 [6.84-9.55] | 3.94 [3.78-4.51] | -3.17 vs 2.71 | yes |

The best schedules and their run-to-run spread:

| Phone | Table | Best before | Runs | Best after | Runs | Spread after (max - min) ms |
|---|---|---|---|---|---|---|
| Pixel 7a | btpm | SCH-L2G5-G160-4fd8 | 3 | SCH-G2M2L2B1-G386-46ae | 3 | 0.52 |
| Pixel 7a | isolated | SCH-B1M1L4G1-G590-c1af | 3 | SCH-L1B1G3M2-G428-cfdd | 3 | 0.02 |
| Samsung | btpm | SCH-L1G6-G175-d87b | 3 | SCH-L1G3B2M1-G432-3b83 | 6 | 0.40 |
| Samsung | isolated | SCH-G2L1B2M2-G334-d4eb | 3 | SCH-B1G3L2M1-G291-d80b | 3 | 0.73 |

The Samsung btpm best has 6 runs: its assignment was also proposed under the isolated table (as
SCH-L1G3B2M1-G335-370b) and ran 3 times in each folder.

GPU and period error, for context (not gated). GPU chunks are priced from the same re-measured runs.

| Phone | Table | GPU error before | after | Period error before | after |
|---|---|---|---|---|---|
| Pixel 7a | btpm | 0.438 | 0.955 | 0.615 | 0.701 |
| Pixel 7a | isolated | 0.348 | 0.157 | 2.818 | 0.027 |
| Samsung | btpm | 0.687 | 1.521 | 0.085 | 0.575 |
| Samsung | isolated | 0.284 | 0.176 | 1.403 | 0.161 |

## 3. What the pinned tiers changed

- **The tiers are distinct now.** Before, a tree `big`, `medium` and `little` cell were one unpinned measurement
  (Pixel isolated stage 1: 1.14-1.16 ms on each). Now the Pixel's stage 2 (the single-threaded sort) costs 26.1 ms
  on little, 9.1 on medium and 6.7 on big. A `taskset`-pinned process times the little cells the same (task 1.3:
  stage 2 26.22 vs 26.24 ms).
- **The sort is priced at what it costs on its tier.** On 2026-10-18, 9 of the Pixel's 10 btpm candidates ran
  stage 2 on a CPU tier. Its `[2,2]` chunks, priced 7.9-8.2 ms, ran 8.6-24.6 ms (Little `[2,2]` priced 8.12 ran
  24.6). The best one, SCH-L2G5-G160-4fd8, held it in a Little `[1,2]` chunk (priced 9.86 ms); its period ranged
  10.44-21.44 ms over the gate's runs and 9.84-28.78 in that change's reproduction check. Now the Pixel's btpm
  candidates run stage 2 on the GPU and its isolated candidates on the big tier (priced 6.74 ms); the best isolated
  schedule runs that chunk in 6.87 ms. On the Samsung every candidate runs stage 2 on the GPU.
- **CPU chunks run close to their price on the Pixel.** Measured / predicted per CPU chunk, median and range:
  btpm 0.94 (0.73-1.13), isolated 1.06 (1.01-1.15). The isolated table's period error fell from 2.82 to 0.03.

## 4. Reading it

- **The remaining tree error is on the GPU, under the btpm table.** Every executed Pixel btpm candidate is
  GPU-bound: its GPU chunk (stages 1-2, 1-3, 2-2 or 2-3) is priced 4.4-5.7 ms from the interference column and runs
  8.6-11.3 ms (median ratio 1.95). On the Samsung the btpm GPU chunks run 1.9-3.5x their price (median 2.52; `G[2,7]`
  priced 3.44 runs 6.4-8.7). This is the known artifact of the btpm GPU column (`06-end-to-end-scheduling.md`,
  "distrust the BTPM GPU column"): the background load keeps the GPU clocked up while its cell is measured. The
  interference GPU cells are below the isolated ones (Pixel stages 1+2: 4.90 vs 8.39 ms; Samsung stage 4: 0.36-0.59
  vs 1.37) and are about what they were on 2026-10-18 (this change does not touch the GPU path). `02`'s DVFS floor
  (`--dvfs-floor`) is opt-in and was not used, as on 2026-10-18.
- **Under the isolated table the period is predicted within 0.03 (Pixel) and 0.16 (Samsung)** (median period
  error), with CPU error 0.06 / 0.27 and GPU error 0.16 / 0.18.
- **The Samsung's CPU chunks still run over their price under the isolated table**: median measured / predicted
  1.27, range 1.00-1.62 for chunks priced 0.9 ms or more and up to 2.15 for the shorter ones. The little `[1,1]`
  chunk is priced 2.59 ms and runs 3.33-4.00. The pinning test passes on that phone (task 1.2), so the chunks run on
  their tier; what makes them slower in the pipeline than in the profiler is not answered here.

## 5. How to reproduce

```bash
# store: 06-end-to-end-scheduling.md Step 1 (01_collect_profiling.py --tools prof --runs 2
#   --scenarios isolated,interference --cooldown-s 20 --dest /data/local/tmp/<dir>), the binary of 3d2dc35
# schedules, per phone and table:
uv run optimizer/orchestrate/02_gen_schedule_merged.py --profiling_root data/profiling --device <serial> \
  --app tree --backend vk --table_type <btpm|isolated> --minimize_mode tmax --num_solutions 10 \
  --output_folder data/schedules_<table>
# executions: the 10 of each file by metrics.max_time, each written to a one-schedule JSON and run 3 rounds in
# order, 20 s apart, from /data/local/tmp/<dir> holding libc++_shared.so:
adb -s <serial> shell 'cd /data/local/tmp/<dir> && LD_LIBRARY_PATH=. ./bm-gen-logs-tree-vk --device <serial> \
  --schedule-file one-tree-<table>-NN.json --n-schedules-to-run 1' </dev/null > \
  data/sched_logs/<serial>_tree_vk_<table>/schedule_run_<k>.log
# the gate (after), and the before:
uv run optimizer/analysis/chunk_accuracy.py --app tree
uv run optimizer/analysis/chunk_accuracy.py --app tree --schedules-root data/archive/2026-10-18-cpu-chunk-costs \
  --sched-logs data/archive/2026-10-18-cpu-chunk-costs/sched_logs \
  --overhead-root data/archive/2026-10-18-cpu-chunk-costs-rejected-overhead
uv run optimizer/analysis/speedup_summary.py --sched-logs data/sched_logs --schedules-root data
```
