# End-to-end refresh: the October campaign, before and now (2026-10-20)

**Question.** After the October campaign's changes (gpu-chunk-cost-model through cifar-gpu-kernels), what do
scheduled pipelines measure on the Pixel 7a and the Samsung, for tree, cifar-dense and cifar-sparse, compared with
the 2026-10-17 gate? This covers best period, speedup over the best single PU, and prediction errors. The whole
procedure of `06-end-to-end-scheduling.md` (Steps 0-5) was run again from this commit: re-profile, solve, execute,
compare.

**Findings.**

- **No best period is worse than on 2026-10-17. Seven of the 12 (phone, app, table) cells are better beyond
  run-to-run noise.**
  - tree isolated: Pixel 28.12 to 8.92 ms, Samsung 22.30 to 4.05.
  - Samsung cifar-sparse: btpm 237.54 to 153.27, isolated 192.13 to 153.27.
  - Samsung cifar-dense btpm: 11.46 to 9.74.
  - Pixel cifar-dense: btpm 43.26 to 40.68, isolated 39.74 to 37.80.
  - The other five are lower but within the noise. The Pixel's tree btpm spread narrowed from 10.39-23.37 to
    8.15-9.64 ms.
- **Against a measured single-PU baseline, the pipelines win on every cell but one:**
  - tree: 2.1x (Pixel), 2.4-2.5x (Samsung);
  - Pixel cifar-dense 1.55-1.67x, Pixel cifar-sparse 1.29x, Samsung cifar-sparse 1.18x;
  - Samsung cifar-dense: 1.06x (btpm) and **0.95x (isolated)**. Its GPU alone runs the whole app in 10.31 ms.
- **The profile-summed baseline of Step 5 (`speedup_summary.py`) is not a reliable denominator any more.**
  - It is off by up to 1.6x in either direction. The Samsung's cifar-sparse GPU sums to 118.79 ms but runs the app in
    181.43 ms, which makes a 1.18x pipeline read as 0.78x. The Samsung's cifar-dense GPU sums to 16.96 ms but runs
    the app in 10.31 ms.
  - On the Pixel's cifar-dense it drops the GPU column (noisy GPU cells) and divides by the big tier.
  - Section 3 gives both baselines.
- **What got worse** (section 5):
  - Pixel tree isolated is slower than at the 2026-10-19 gate: 6.87 to 8.92 ms, beyond noise.
  - The GPU-chunk error rose for tree on both phones, and on the Samsung's cifar-sparse under the isolated table.
  - The Samsung's cifar-dense gains nothing from pipelining now.
- **The largest remaining error is the GPU-chunk price** (section 6). The likely cause is that a GPU stage is
  profiled at a GPU clock the pipeline does not hold:
  - Pixel tree isolated: period error 0.85, GPU error 1.01;
  - Samsung tree btpm: GPU error 1.50;
  - Samsung cifar-sparse isolated: 0.66.

## 1. What was run

- **Before** is the 2026-10-17 gate (`2026-10-17-gpu-chunk-cost-model-gate.md`), per-stage rows only:
  - code of 965e294, profiling store of 2026-10-16 (12e895f);
  - executions and schedule files in `data/archive/2026-10-17-gpu-chunk-cost-model/` (local, not committed, as the
    2026-10-18 and 2026-10-19 reports also use it);
  - its store: tree in `data/profiling/<serial>/tree/vulkan/<scenario>/superseded-2026-10-17/`, the cifar apps in
    `.../superseded-2026-10-18/`. The cifar store did not change between 2026-10-16 and this refresh.
  - `optimizer/analysis/e2e_refresh.py` on that archive reproduces the gate's per-stage rows exactly (sections 2 and
    3, every cell), and the 2026-10-17 `speedup-summary.md` baselines.
- **Code.** 3324b53 (`data/e2e-refresh`, = `dev` after #45). The Android preset with NDK r30 built `bm-prof`,
  `bm-gen-logs` and `bm-baseline` for the three apps.
- **Store** (Step 1). Both phones, all three apps, isolated and interference, 2 runs each, 2026-10-20 15:07-15:36 UTC.
  The command was `01_collect_profiling.py --tools prof --runs 2 --scenarios isolated,interference --cooldown-s 20
  --dest /data/local/tmp/e2e` with the real weights. 24 runs, none failed. The replaced runs moved to
  `superseded-2026-10-19/` (tree, the 2026-10-19 gate's store) and `superseded-2026-10-18/` (cifar). The range
  records were not re-collected.
- **Schedules** (Step 2). Per phone, app and table:
  - `02_gen_schedule_merged.py --minimize_mode tmax --num_solutions 10`, the per-stage model (the default);
  - no `overhead.json` (the store ships none), so no union sweep and plain stage sums;
  - the earlier schedules and executions (the 2026-10-18 cifar and 2026-10-19 tree gates) moved to
    `data/archive/2026-10-20-pre-e2e-refresh/` first, so that runs do not pool by assignment.
  - The quality gate dropped some CPU cells of stage 11 from the interference tables: all three on the Pixel's
    cifar-dense (cv up to 2.6 on a 0.05 ms stage), little on the Samsung's cifar-sparse. On the Pixel's cifar-dense
    btpm table, stage 11 must therefore run on the GPU.
- **Executions** (Step 3), run as the 2026-10-19 gate ran them:
  - per (phone, app, table), the top 10 by `metrics.max_time` (cifar-sparse top 5), each as a one-schedule JSON in
    its own `bm-gen-logs-<app>-vk` process with the real weights;
  - 3 rounds in the same order, 20 s idle before every run, the two phones at the same time;
  - Samsung 15:33-16:30 UTC, Pixel 15:43-16:48 UTC; 300 runs, none failed.
  - Logs: `data/sched_logs/<serial>_<app>_vk_<table>/`.
  - The Samsung's cifar-sparse files count 6 executed schedules, not 5, because its btpm and isolated top 5 share
    assignments that pool.
- **Baselines** (Step 5).
  - `speedup_summary.py`: the fastest single PU summed from the isolated store (`data/sched_logs/speedup-summary.md`).
  - As a measured cross-check, after each phone's executions: `bm-baseline-<app>-vk --benchmark_min_time=1s
    --benchmark_repetitions=3`, the whole app as one GPU chunk and on each CPU tier.
  - Output in `data/baselines-2026-10-20/<serial>/<app>.json`.
- **Measures**: `optimizer/analysis/chunk_accuracy.py`, the method of the three earlier gates.
  - The period of a run: each chunk's mean duration, summed per PU, maxed over PUs. A schedule's period is the median
    over the runs of its assignment, pooled over both tables' folders, and its spread is their min-max.
  - Errors: the median `|measured / predicted - 1|` of the period, and of every CPU (GPU) chunk of the executed
    schedules.
  - Noise for a best period: the larger min-max spread of the two best schedules (before and now).
  - `optimizer/analysis/e2e_refresh.py` (new) joins both sides with their baselines.

## 2. Before vs now, every cell

Best measured period and its spread, change against noise, speedup over the best single PU, and median prediction
errors. Speedup "Step 5" divides the profile-summed baseline (the before column used the same method on its store),
and "measured" divides the bm-baseline of this refresh (there is no measured baseline for before).

| Phone | App | Table | Best ms before [min-max] | Best ms now [min-max] | Change vs noise | Speedup Step 5 before | now | Speedup measured now | Period error before | now | CPU error before | now | GPU error before | now |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pixel 7a | tree | btpm | 11.70 [10.39-23.37] | 8.96 [8.15-9.64] | -2.74 vs 12.98, within | 0.98x | 1.46x | 2.13x | 1.115 | 0.487 | 0.372 | 0.198 | 0.438 | 0.886 |
| Pixel 7a | tree | isolated | 28.12 [27.91-28.74] | 8.92 [8.62-8.95] | -19.20 vs 0.83, **better** | 0.41x | 1.47x | 2.14x | 4.726 | 0.853 | 0.376 | 0.065 | 0.617 | 1.014 |
| Pixel 7a | cifar-dense | btpm | 43.26 [42.99-43.40] | 40.68 [40.65-40.72] | -2.58 vs 0.41, **better** | 1.89x | 2.41x (OMP base) | 1.55x | 0.114 | 0.122 | 0.401 | 0.378 | 0.114 | 0.122 |
| Pixel 7a | cifar-dense | isolated | 39.74 [39.72-39.79] | 37.80 [37.42-38.10] | -1.94 vs 0.68, **better** | 2.06x | 2.59x (OMP base) | 1.67x | 0.121 | 0.052 | 0.245 | 0.098 | 0.119 | 0.010 |
| Pixel 7a | cifar-sparse | btpm | 444.91 [438.17-469.96] | 419.52 [419.24-457.35] | -25.39 vs 38.11, within | 1.30x | 1.44x | 1.29x | 0.015 | 0.108 | 0.134 | 0.267 | 0.013 | 0.108 |
| Pixel 7a | cifar-sparse | isolated | 444.91 [438.17-469.96] | 419.52 [419.24-457.35] | -25.39 vs 38.11, within | 1.30x | 1.44x | 1.29x | 0.090 | 0.092 | 0.348 | 0.263 | 0.085 | 0.092 |
| Samsung | tree | btpm | 5.10 [3.57-5.25] | 4.32 [4.15-4.32] | -0.78 vs 1.68, within | 1.44x | 2.33x | 2.37x | 1.467 | 0.445 | 1.531 | 0.265 | 0.950 | 1.499 |
| Samsung | tree | isolated | 22.30 [22.30-29.64] | 4.05 [4.02-4.07] | -18.25 vs 7.34, **better** | 0.33x | 2.49x | 2.53x | 9.695 | 0.171 | 3.028 | 0.125 | 0.518 | 0.186 |
| Samsung | cifar-dense | btpm | 11.46 [11.44-11.49] | 9.74 [9.72-9.85] | -1.72 vs 0.13, **better** | 1.78x | 1.74x | 1.06x | 0.056 | 0.051 | 0.072 | 0.544 | 0.065 | 0.051 |
| Samsung | cifar-dense | isolated | 10.93 [10.89-11.08] | 10.84 [10.69-11.06] | -0.09 vs 0.37, within | 1.87x | 1.56x | **0.95x** | 0.035 | 0.067 | 0.074 | 0.120 | 0.122 | 0.169 |
| Samsung | cifar-sparse | btpm | 237.54 [212.42-267.32] | 153.27 [153.10-154.57] | -84.27 vs 54.90, **better** | 1.20x | 0.78x | 1.18x | 0.249 | 0.049 | 0.425 | 0.188 | 0.495 | 0.049 |
| Samsung | cifar-sparse | isolated | 192.13 [191.86-225.03] | 153.27 [153.10-154.57] | -38.86 vs 33.17, **better** | 1.48x | 0.78x | 1.18x | 0.090 | 0.662 | 0.113 | 0.393 | 0.196 | 0.662 |

The best schedules now. The cifar-sparse cells share one assignment across both tables, so each has a single best
schedule; Runs is 6 where that assignment ran in both tables' folders.

| Phone | App | Table | Best schedule now | Runs |
|---|---|---|---|---|
| Pixel 7a | tree | btpm | SCH-L1G1B4M1-G432-dae3 | 6 |
| Pixel 7a | tree | isolated | SCH-G2M2L2B1-G392-7c0f | 3 |
| Pixel 7a | cifar-dense | btpm | SCH-L1M2B2G6-G3896-8780 | 3 |
| Pixel 7a | cifar-dense | isolated | SCH-M3L1G2B5-G3569-cf1d | 3 |
| Pixel 7a | cifar-sparse | btpm / isolated | SCH-L1B3G3M4-G41505-5d3f / -G42693-b03e | 6 |
| Samsung | tree | btpm | SCH-B1G3L2M1-G362-310b | 3 |
| Samsung | tree | isolated | SCH-G2L1M3B1-G294-eec3 | 3 |
| Samsung | cifar-dense | btpm | SCH-M1L1G8B1-G1033-e93c | 3 |
| Samsung | cifar-dense | isolated | SCH-L1G7B1M2-G854-1782 | 3 |
| Samsung | cifar-sparse | btpm / isolated | SCH-L1G6M2B2-G8876-1877 / -G6255-5e00 | 3 |

Against the latest gates rather than 2026-10-17 (2026-10-18 for cifar, 2026-10-19 for tree; same script with
`--before-*` on `data/archive/2026-10-20-pre-e2e-refresh/`), the best periods are:

- **better beyond noise in five cells:** Pixel cifar-dense (43.19 to 40.68 btpm, 39.59 to 37.80 isolated), Samsung
  cifar-dense btpm (11.26 to 9.74), Samsung cifar-sparse (213.92 to 153.27 btpm, 183.20 to 153.27 isolated);
- **worse beyond noise in one:** Pixel tree isolated (6.87 to 8.92, +2.05 vs 0.33);
- **within noise in the other six.**

## 3. Baselines

The fastest single PU, whole app, ms:

| Phone | App | Step 5 before (store of 2026-10-16) | Step 5 now | Measured now: VK | big | medium | little | Fastest measured |
|---|---|---|---|---|---|---|---|---|
| Pixel 7a | tree | VK 11.48 | VK 13.10 | 20.29 [20.14-20.33] | 19.10 [18.61-19.30] | 22.13 [22.13-22.21] | 59.83 [59.66-60.00] | big 19.10 |
| Pixel 7a | cifar-dense | VK 81.75 | OMP 98.07 | 63.16 [63.11-63.34] | 131.42 [130.12-131.88] | 159.07 [158.88-159.25] | 850.92 [807.37-862.27] | VK 63.16 |
| Pixel 7a | cifar-sparse | VK 580.43 | VK 604.44 | 542.99 [542.92-543.38] | 2148.18 [2115.59-2155.61] | 2339.83 [2334.16-2347.47] | 7765.47 [7656.38-8251.19] | VK 542.99 |
| Samsung | tree | OMP 7.34 | VK 10.09 | 10.24 [10.13-10.33] | 16.70 [16.70-16.72] | 15.31 [15.26-15.32] | 39.18 [39.15-39.65] | VK 10.24 |
| Samsung | cifar-dense | VK 20.39 | VK 16.96 | 10.31 [10.13-10.46] | 103.38 [103.37-103.40] | 77.24 [77.20-77.34] | 264.12 [262.80-265.42] | VK 10.31 |
| Samsung | cifar-sparse | VK 284.62 | VK 118.79 | 181.43 [167.18-187.76] | 2032.07 [2031.57-2046.60] | 1514.01 [1513.05-1514.09] | 6848.06 [6813.97-7439.17] | VK 181.43 |

The measured columns give the median [min-max] over 3 repetitions. bm-baseline's unpinned "OMP all cores" row is
not one PU; it is in the JSON and the script output.

- **The GPU's whole-app time is not the sum of its stages, in either direction.**
  - Higher than the sum: Pixel tree 20.29 vs 12.52 ms (as the gpu-chunk-cost-model proposal measured), Samsung
    cifar-sparse 181.43 vs 118.79.
  - Lower than the sum: Samsung cifar-dense 10.31 vs 16.96, Pixel cifar-dense 63.16 vs 68.30. The solver's view
    keeps the noisy GPU cells; see the next point.
  - Step 5's speedups move with this error. Samsung cifar-sparse reads 0.78x on the summed baseline and 1.18x on
    the measured one.
- **The Step 5 baseline of the Pixel's cifar-dense has no GPU column.** `smt.baselines` drops every cell with cv > 1,
  and three GPU stages (2, 4, 8) have cv > 1 in both isolated runs. The baseline therefore falls to the big tier
  (98.07 ms), and the Step 5 speedups of 2.41x and 2.59x are not comparable with the 1.89x and 2.06x before. The
  solver keeps those cells (GPU exempt from the CV gate) and sums the GPU to 68.30 ms. Nothing was changed here (no
  solver code changes in this run).
- **The tree "before" baselines** sum CPU cells measured before tree-cpu-pinning: tree's `big` / `medium` / `little`
  were one unpinned measurement then. So the Samsung's OMP 7.34 is an all-cores number.

Isolated store sums per PU (the solver's view, GPU cells kept), before / now, ms. Printed by `e2e_refresh.py`:

| Phone | App | vulkan | big | medium | little |
|---|---|---|---|---|---|
| Pixel 7a | tree | 11.48 / 12.52 | 14.89 / 17.95 | 15.30 / 21.63 | 15.42 / 58.59 |
| Pixel 7a | cifar-dense | 81.75 / 68.30 | 145.60 / 98.07 | 254.28 / 159.99 | 730.37 / 741.10 |
| Pixel 7a | cifar-sparse | 580.43 / 604.44 | 3293.76 / 2309.97 | 2751.60 / 2367.09 | 8320.95 / 8562.14 |
| Samsung | tree | 12.29 / 10.09 | 8.95 / 16.64 | 7.34 / 15.16 | - |
| Samsung | cifar-dense | 20.39 / 16.96 | 110.24 / 107.69 | 81.65 / 80.46 | 272.87 / 256.37 |
| Samsung | cifar-sparse | 284.62 / 118.79 | 2961.00 / 2047.56 | 2807.00 / 1549.37 | 8000.96 / 6824.34 |

The Samsung's tree little tier has no complete column on one side ("-").

## 4. Which changes contributed

Merged between the before code (965e294) and now. The ones that change what runs on the phones:

| Change (PR) | What it changed | Cells it moved |
|---|---|---|
| cpu-chunk-costs (#34) | Tree's OMP stage 2 sorts a copy, so it is profiled on fresh input. The Pixel big tier's stage 2 went from 0.48 to 6.76 ms, the Samsung's from 1.19 to 7.09. | tree isolated on both phones. The 2026-10-17 schedules held the sort in CPU chunks priced 0.3-1.2 ms that ran 14-27 ms (the 28.12 and 22.30 ms periods). Now the solver keeps it on the GPU or the big tier. |
| tree-cpu-pinning (#40) | Tree's CPU chunks run on their tier. | tree, both phones: the bimodal runs are gone (Pixel btpm spread 10.39-23.37 to 8.15-9.64). Together with #34, tree's CPU error (btpm / isolated) falls from 0.37 / 0.38 to 0.20 / 0.07 on the Pixel and from 1.53 / 3.03 to 0.27 / 0.13 on the Samsung. |
| tree-app-improvements (#33) | Radix-sort and compaction kernels, Morton clamp, atomic leaf clear. Vulkan stage 7 now links leaves, which costs time: +133 % on the Pixel in that change's A/B. | tree GPU column: Pixel 11.48 to 12.52 ms, Samsung 12.29 to 10.09. |
| occasional-host-reads (#37) | Tree's final octree and count buffers leave the host-cached boundary class. | Pixel tree GPU chunk: the whole chunk took 29.54 to 17.79 ms in that change's gate. |
| cifar-sparse-conv (#41, cifar-gpu-kernels group 1) | One output channel per workgroup row (subgroup >= 32 path). | Samsung cifar-sparse: GPU sum 284.62 to 118.79 ms, best -35 % (btpm) and -20 % (isolated). The Pixel runs the unchanged Mali path. |
| cifar-dense-conv (#44, group 2) | Channel blocks and per-conv register tiles (subgroup >= 32 path). | Samsung cifar-dense: GPU sum 20.39 to 16.96 ms (together with #45), btpm best -15 %. |
| cifar-fc-weights (#45, group 3) | Clustered subgroup reduction in `linear_bt` (FP32 by default; INT8 is opt-in and was not used). | Both cifar apps' fc stages. The Pixel's cifar-dense fc1 / fc2 went from 7.18 / 7.21 to 2.84 / 3.25 ms in the store, which drives its cifar-dense gain (-6 %, -5 %). |
| gpu-chunk-cost-model (#30) | Range cost models, opt-in; GPU clock in provenance. | None directly: the per-stage model stays the default. The clock it records is what section 6 reads. |

One VkInstance per process (#39, a Jetson fix) and the CUDA leaf-clear fix (#42) are not expected to change the
phone timings.

Not a change: the Pixel's cifar-dense **CPU** columns are 33-37 % lower in this store (big 145.60 to 98.07 ms),
uniformly over every stage, with the same OMP code. This is session variation: the governor (`sched_pixel`) is not
locked. It shows in the Pixel's cifar-dense CPU chunks, which run below their btpm price (B[4,5] priced 38.89 runs
21.17 ms).

## 5. What got worse

- **Pixel tree isolated vs the 2026-10-19 gate: 6.87 to 8.92 ms (beyond noise; vs 2026-10-17 it is still -68 %).**
  - The new isolated store prices GPU stage 2 (the radix sort) at 4.73 ms. On 2026-10-19 it was 7.05.
  - The two runs measured 6.24 ms at a 434 MHz Mali clock (41 samples) and 4.03 ms at 701 MHz (88 samples). The loader
    weights by sample count, so the faster run dominates.
  - So the solver now puts stage 2 on the GPU (`G[1,2]` priced 5.46, or `G[2,2]` 4.73) instead of on the big tier.
    In the pipeline those chunks run 8.92-11.71 ms.
  - The same sort on the big tier, as the 2026-10-19 best schedule held it, is priced 6.76.
- **GPU-chunk error on tree rose** from 2026-10-17: Pixel 0.438 to 0.886 (btpm) and 0.617 to 1.014 (isolated),
  Samsung btpm 0.950 to 1.499. The CPU error fell in the same cells, so the GPU chunk is now the bottleneck that is
  mispriced (section 6).
- **The Samsung's cifar-dense gains nothing from pipelining.** The GPU runs the whole app in 10.31 ms, and the best
  pipelines take 9.74 (btpm, 1.06x) and 10.84 ms (isolated, 0.95x).
  - The btpm candidates give the GPU stages 3-10, and that chunk is the bottleneck (9.7-11.3 ms).
  - The isolated candidates keep the GPU at 5-7 stages. A big or medium chunk holding fc1 / fc2 is the bottleneck
    (10.8-11.8 ms).
  - Over the whole app the CPU tiers are 7.5x (medium) to 26x (little) slower than the GPU (measured), so any split
    leaves little to gain.
  - The Step 5 speedups fell from 1.78x / 1.87x to 1.74x / 1.56x, although the period is lower: the baseline fell more.
  - The Samsung's cifar-sparse Step 5 speedup fell from 1.20x / 1.48x to 0.78x, an artifact of the summed baseline
    (section 3): measured, it is 1.18x.
- **Prediction errors that rose in the cifar cells:**
  - **Samsung cifar-sparse isolated, period error 0.090 to 0.662.** The GPU chunk `G[2,7]` is priced 94.91 ms and runs
    153.27.
  - **Samsung cifar-dense btpm, CPU error 0.072 to 0.544.** These are sub-millisecond chunks: stage 11 priced 0.02
    ms runs 0.07-0.45, `M[1,1]` priced 0.84 runs 1.17-1.31. No fixed per-chunk cost is in the store. The period
    error is unaffected (0.051), because the GPU chunk is the bottleneck.
  - **Pixel cifar-sparse btpm, period error 0.015 to 0.108, CPU error 0.134 to 0.267.**

## 6. The largest remaining error and its likely cause

**The price of a GPU chunk.** Three cells, each with the largest error of its kind:

| Phone | App | Table | Error | Priced | Measured in the pipeline |
|---|---|---|---|---|---|
| Pixel 7a | tree | isolated | period 0.853, GPU 1.014 | `G[1,2]` 5.46 ms | 8.92-11.71 ms |
| Samsung | tree | btpm | GPU 1.499 | `G[2,7]` 3.55, `G[1,7]` 3.74 ms | 6.27-9.05, 11.19 ms |
| Samsung | cifar-sparse | isolated | period and GPU 0.662 | `G[2,7]` 94.91 ms | 153.27 ms |

**Likely cause: the GPU stage cells are measured at a GPU clock that the pipeline's GPU chunk does not run at.** The
pipeline logs record no clock, so this is inferred from the profiler's `provenance.gpu_clock_mhz` and from
bm-baseline; it is not measured in the pipeline itself.

- **Pixel tree.** Stage 2's isolated price comes mostly from a run at 701 MHz; the other run, at 434 MHz, is 55 %
  slower. A pipeline's GPU chunk, idle for part of every period, runs it at 8.9-11.7 ms. That is slower than either
  run, as the 2026-10-17 gate found for the whole tree chunk.
- **Samsung tree btpm.** The interference column is the known DVFS artifact (`06-end-to-end-scheduling.md`,
  "distrust the BTPM GPU column"). The background load keeps the GPU clocked up, so stage 7 is priced 0.98 ms
  against 2.25 isolated. The whole app measured alone on the GPU (10.24 ms) matches the isolated sum (10.09). `02`'s
  `--dvfs-floor` was not used, as at the earlier gates.
- **Samsung cifar-sparse isolated.** The heavy conv stages (3, 5, 6, 7) were profiled at the 1095 MHz peak. The
  whole app as one GPU chunk measures 181.43 ms against a 118.79 ms sum (1.53x), with no CPU work running. So the
  gap is in the GPU chunk itself, not interference from the co-running CPU chunks. A sustained 150-180 ms chunk most
  likely does not hold the peak clock. The btpm table prices the same chunk at 147-148 ms and is within 0.05.
  - The range records (`range-hot`) would measure a whole chunk, but they predate the new sparse kernel and were not
    re-collected (out of scope here).

Two smaller points belong with this:

- The loader's count-weighted aggregation favours the faster run, because a fixed time budget takes more samples
  when a stage runs faster.
- The interference quality gate can remove every CPU cell of a tiny stage (Pixel cifar-dense stage 11).

## 7. How to reproduce

```bash
# Step 0: binaries (NDK r30, $ANDROID_NDK_HOME)
cmake --preset android && cmake --build --preset android --target \
  bm-{prof,gen-logs,baseline}-{tree,cifar-dense,cifar-sparse}-vk
# Step 1, per phone and app (the phone's <dest> holds libc++_shared.so and weights/):
uv run optimizer/orchestrate/01_collect_profiling.py --device <serial> --app <app> --backend vk \
  --adb-serial <serial> --build-dir build/android --dest /data/local/tmp/<dir> \
  --tools prof --runs 2 --scenarios isolated,interference --cooldown-s 20
# Step 2, per phone, app and table:
uv run optimizer/orchestrate/02_gen_schedule_merged.py --profiling_root data/profiling --device <serial> \
  --app <app> --backend vk --table_type <btpm|isolated> --minimize_mode tmax --num_solutions 10 \
  --output_folder data/schedules_<table>
# Step 3: the top 10 by metrics.max_time (cifar-sparse top 5) of each file, each as a one-schedule JSON,
# 3 rounds in order, 20 s idle before every run; k = (round - 1) * N + i:
adb -s <serial> shell 'cd /data/local/tmp/<dir> && BT_WEIGHTS_DIR=/data/local/tmp/<dir>/weights \
  LD_LIBRARY_PATH=. ./bm-gen-logs-<app>-vk --device <serial> --schedule-file one-<app>-<table>-NN.json \
  --n-schedules-to-run 1' </dev/null > data/sched_logs/<serial>_<app>_vk_<table>/schedule_run_<k>.log
# Step 5: measured baseline, after the executions (4 log lines precede the JSON; drop them):
adb -s <serial> shell 'cd /data/local/tmp/<dir> && BT_WEIGHTS_DIR=/data/local/tmp/<dir>/weights \
  LD_LIBRARY_PATH=. ./bm-baseline-<app>-vk --device <serial> --benchmark_min_time=1s \
  --benchmark_repetitions=3 --benchmark_format=json 2>/dev/null' </dev/null > data/baselines-2026-10-20/<serial>/<app>.json
uv run optimizer/analysis/speedup_summary.py --sched-logs data/sched_logs --schedules-root data
# sections 2-3 (before = the 2026-10-17 archive):
uv run optimizer/analysis/e2e_refresh.py \
  --before-schedules-root data/archive/2026-10-17-gpu-chunk-cost-model \
  --before-sched-logs data/archive/2026-10-17-gpu-chunk-cost-model/sched_logs \
  --before-isolated tree=superseded-2026-10-17 --before-isolated cifar-dense=superseded-2026-10-18 \
  --before-isolated cifar-sparse=superseded-2026-10-18 --baselines data/baselines-2026-10-20
# vs the latest gates: --before-schedules-root / --before-sched-logs on data/archive/2026-10-20-pre-e2e-refresh,
#   --before-isolated tree=superseded-2026-10-19 (cifar as above)
uv run optimizer/analysis/chunk_accuracy.py      # per-cell executed counts, CPU chunk counts
```

The change-vs-noise column was computed from `e2e_refresh.py`'s best and [min-max] columns. The per-chunk prices
and medians quoted in sections 4-6 came from the same join (`analysis.accuracy.cell_rows` and
`analysis.chunk_accuracy.chunk_durations`) by a scratch script kept outside the repository.
