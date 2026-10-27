# GPU chunk cost model: the acceptance gate (2026-10-17)

**Question.** Does pricing a GPU chunk by its measured stage range (gpu-chunk-cost-model,
`range-hot` / `range-idle`) predict executed pipelines better than the per-stage sum (`per-stage`),
on the Pixel 7a and the Samsung, for tree, cifar-dense and cifar-sparse? The spec makes a range model
the default only if, on both phones and for all three apps, its median absolute error is lower than
`per-stage`'s on the same table and its best measured schedule is not slower beyond run-to-run noise.

**Decision: `per-stage` stays the default.** Neither range variant passes. Each beats `per-stage`
in 2 of the 6 (phone, app) cells per table; both lose cifar-dense on both phones and cifar-sparse
under the btpm table on both phones. `02_gen_schedule_merged.py --cost-model` keeps `per-stage` as its
default; `range-hot` and `range-idle` stay available for comparison.

## 1. What was run

- **Profiling store** (task 2.3): `data/profiling/{3A021JEHN02756,R5CY21Y3VEV}/{tree,cifar-dense,cifar-sparse}/vulkan/`,
  per-stage `run-001/002.jsonl` and range `range-001/002.jsonl`, isolated and interference, collected
  2026-10-16 22:42 to 2026-10-17 00:18 (sweep durations in the change's design.md).
- **Schedules** (from the code of commit 965e294): for each phone, app, table (btpm = the interference
  scenario, isolated) and cost model, `02_gen_schedule_merged.py --minimize_mode tmax --num_solutions 10
  --cost-model <model>` into `data/schedules_<table>/`. No `overhead.json` exists for the phones, so there
  is no union sweep and the GPU per-stage price has no fitted overhead.
- **Executions** (2026-10-17 00:11 to 02:25): per (phone, app, table), the top 10 candidates of each
  model's file by predicted period (cifar-sparse: top 5), one run per distinct chunk assignment
  (assignments two models propose run once), each in its own `bm-gen-logs-<app>-vk` process with one
  schedule, 3 rounds in the same order (configurations alternate), 20 s idle before every run. 170
  distinct (phone, app, table, assignment) runs per round, 510 runs, none failed. Logs: `data/sched_logs/<serial>_<app>_vk_<table>/`.
  The two phones ran at the same time from the build box, each phone's runs one after another.
- **Measured period** of a run (`analysis/accuracy.py`): each chunk's mean duration over its tasks
  (100 for tree, 32 for the cifar apps), summed per PU, maxed over PUs; a schedule's measured period is
  the median over every run of its assignment in the cell, and its spread is their min-max. Ratio =
  measured / predicted (`metrics.max_time`). A run counts for every schedule file of the cell that
  proposed the same assignment, so a file's "Executed" count can exceed its top-N (cifar-sparse: 6 or 7).
- **Pooled runs.** The runs of an assignment are pooled over both tables' log folders. Most
  assignments were proposed under one table only and have 3 runs. Six assignments, all on the Pixel,
  were proposed under both tables, ran 3 times in each table's folder, and have **6 runs**: 2 in
  cifar-dense (`M4B1G5L1` and `M3B2G5L1`, each in 4 files: both tables' `per-stage` and the isolated
  `range-hot` / `range-idle`) and 4 in cifar-sparse (`B4G3L1M3`, `B3L1G3M4`, `L2B2G3M4`, `L1B3G3M4`,
  each in both tables' `per-stage` files). Their 16 rows (2 x 4 + 4 x 2) of the summary's 310 have
  6 runs.
  Three best schedules in section 2 are among them, so their median and spread are over 6 runs:
  Pixel cifar-dense btpm `per-stage` (43.26 [42.99-43.40]) and Pixel cifar-sparse `per-stage` on both
  tables (444.91 [438.17-469.96]). Every other median and spread in sections 2 and 3 is over 3 runs.
- **Noise** for "not worse": the range model's best measured period may exceed `per-stage`'s best by at
  most the larger min-max spread over the runs (3, or 6 for the pooled ones) of the two best schedules.

## 2. The gate, every cell

`Lower error`: the range model's median |ratio - 1| is below `per-stage`'s on the same table.
`Best not worse`: its best measured period minus `per-stage`'s, against the noise.

| Phone | App | Table | Model | Executed | Median abs error | Best measured ms [min-max] | Best schedule | Lower error | Best not worse | Wins |
|---|---|---|---|---|---|---|---|---|---|---|
| Pixel 7a | tree | btpm | per-stage | 10 | 1.115 | 11.70 [10.39-23.37] | SCH-B1M1G5-G661-4695 |  |  |  |
| Pixel 7a | tree | btpm | range-hot | 10 | 0.663 | 11.70 [10.39-23.37] | SCH-B1M1G5-G1077-6c73 | yes | yes (+0.00 vs noise 12.98) | **yes** |
| Pixel 7a | tree | btpm | range-idle | 10 | 0.663 | 11.70 [10.39-23.37] | SCH-B1M1G5-G1077-c906 | yes | yes (+0.00 vs noise 12.98) | **yes** |
| Pixel 7a | tree | isolated | per-stage | 10 | 4.726 | 28.12 [27.91-28.74] | SCH-L2B1M1G3-G506-9a1d |  |  |  |
| Pixel 7a | tree | isolated | range-hot | 10 | 6.036 | 28.24 [28.18-29.13] | SCH-L1B2M1G3-G478-166b | no | yes (+0.12 vs noise 0.96) | no |
| Pixel 7a | tree | isolated | range-idle | 10 | 6.036 | 29.39 [29.24-30.16] | SCH-L3M1G3-G370-cf16 | no | no (+1.27 vs noise 0.92) | no |
| Pixel 7a | cifar-dense | btpm | per-stage | 10 | 0.114 | 43.26 [42.99-43.40] | SCH-M4B1G5L1-G4821-09d0 |  |  |  |
| Pixel 7a | cifar-dense | btpm | range-hot | 10 | 0.094 | 48.81 [48.58-48.95] | SCH-M3G4L1B3-G5140-2f28 | yes | no (+5.54 vs noise 0.40) | no |
| Pixel 7a | cifar-dense | btpm | range-idle | 10 | 0.094 | 48.81 [48.58-48.95] | SCH-M3G4L1B3-G5140-2f28 | yes | no (+5.54 vs noise 0.40) | no |
| Pixel 7a | cifar-dense | isolated | per-stage | 10 | 0.121 | 39.74 [39.72-39.79] | SCH-L2B3G4M2-G4187-d33c |  |  |  |
| Pixel 7a | cifar-dense | isolated | range-hot | 10 | 0.004 | 43.12 [43.07-43.19] | SCH-L2G4B3M2-G3530-e443 | yes | no (+3.38 vs noise 0.12) | no |
| Pixel 7a | cifar-dense | isolated | range-idle | 10 | 0.004 | 43.12 [43.07-43.19] | SCH-L2G4B3M2-G3530-7789 | yes | no (+3.38 vs noise 0.12) | no |
| Pixel 7a | cifar-sparse | btpm | per-stage | 6 | 0.015 | 444.91 [438.17-469.96] | SCH-L2B2G3M4-G37961-d258 |  |  |  |
| Pixel 7a | cifar-sparse | btpm | range-hot | 7 | 0.031 | 645.33 [616.63-733.82] | SCH-L2G4M1B4-G56432-74a1 | no | no (+200.42 vs noise 117.18) | no |
| Pixel 7a | cifar-sparse | btpm | range-idle | 7 | 0.031 | 645.33 [616.63-733.82] | SCH-L2G4M1B4-G56432-469c | no | no (+200.42 vs noise 117.18) | no |
| Pixel 7a | cifar-sparse | isolated | per-stage | 6 | 0.090 | 444.91 [438.17-469.96] | SCH-L2B2G3M4-G36089-38bf |  |  |  |
| Pixel 7a | cifar-sparse | isolated | range-hot | 6 | 0.069 | 433.24 [432.85-457.70] | SCH-L1M3G4B3-G38023-c21e | yes | yes (-11.67 vs noise 31.79) | **yes** |
| Pixel 7a | cifar-sparse | isolated | range-idle | 6 | 0.069 | 433.24 [432.85-457.70] | SCH-L1M3G4B3-G38023-c21e | yes | yes (-11.67 vs noise 31.79) | **yes** |
| Samsung | tree | btpm | per-stage | 10 | 1.467 | 5.10 [3.57-5.25] | SCH-L1G6-G042-972d |  |  |  |
| Samsung | tree | btpm | range-hot | 10 | 1.416 | 3.67 [3.55-4.91] | SCH-M1G6-G002-8403 | yes | yes (-1.43 vs noise 1.67) | **yes** |
| Samsung | tree | btpm | range-idle | 10 | 1.416 | 3.67 [3.55-4.91] | SCH-M1G6-G002-8403 | yes | yes (-1.43 vs noise 1.67) | **yes** |
| Samsung | tree | isolated | per-stage | 10 | 9.695 | 22.30 [22.30-29.64] | SCH-M3G2L1B1-G264-cad1 |  |  |  |
| Samsung | tree | isolated | range-hot | 10 | 7.193 | 15.00 [14.96-21.95] | SCH-M3B1G3-G155-415d | yes | yes (-7.31 vs noise 7.35) | **yes** |
| Samsung | tree | isolated | range-idle | 10 | 7.193 | 15.00 [14.96-21.95] | SCH-M3B1G3-G155-415d | yes | yes (-7.31 vs noise 7.35) | **yes** |
| Samsung | cifar-dense | btpm | per-stage | 10 | 0.056 | 11.46 [11.44-11.49] | SCH-L2M1G8-G393-e485 |  |  |  |
| Samsung | cifar-dense | btpm | range-hot | 10 | 0.111 | 11.46 [11.44-11.49] | SCH-L2M1G8-G531-fba9 | no | yes (+0.00 vs noise 0.04) | no |
| Samsung | cifar-dense | btpm | range-idle | 10 | 0.111 | 11.46 [11.44-11.49] | SCH-L2M1G8-G531-fba9 | no | yes (+0.00 vs noise 0.04) | no |
| Samsung | cifar-dense | isolated | per-stage | 10 | 0.035 | 10.93 [10.89-11.08] | SCH-L2G6B1M2-G570-9a50 |  |  |  |
| Samsung | cifar-dense | isolated | range-hot | 10 | 0.140 | 11.09 [10.98-11.28] | SCH-L1B1G7M2-G869-2c4d | no | yes (+0.16 vs noise 0.30) | no |
| Samsung | cifar-dense | isolated | range-idle | 10 | 0.140 | 11.09 [10.98-11.28] | SCH-L1B1G7M2-G869-2c4d | no | yes (+0.16 vs noise 0.30) | no |
| Samsung | cifar-sparse | btpm | per-stage | 5 | 0.249 | 237.54 [212.42-267.32] | SCH-L2B2G3M4-G25705-41f1 |  |  |  |
| Samsung | cifar-sparse | btpm | range-hot | 5 | 0.246 | 413.87 [402.02-438.24] | SCH-L2G4M2B3-G47404-376d | yes | no (+176.33 vs noise 54.90) | no |
| Samsung | cifar-sparse | btpm | range-idle | 5 | 0.246 | 413.87 [402.02-438.24] | SCH-L2G4M2B3-G47404-376d | yes | no (+176.33 vs noise 54.90) | no |
| Samsung | cifar-sparse | isolated | per-stage | 7 | 0.090 | 192.13 [191.86-225.03] | SCH-M4G4B3-G1607-e6c3 |  |  |  |
| Samsung | cifar-sparse | isolated | range-hot | 5 | 0.229 | 192.13 [191.86-225.03] | SCH-M4G4B3-G5745-6dbb | no | yes (+0.00 vs noise 33.17) | no |
| Samsung | cifar-sparse | isolated | range-idle | 5 | 0.229 | 192.13 [191.86-225.03] | SCH-M4G4B3-G5745-6dbb | no | yes (+0.00 vs noise 33.17) | no |

Wins per range variant (out of 6 (phone, app) cells): btpm 2 (Pixel tree, Samsung tree), isolated 2
(Samsung tree, Pixel cifar-sparse). The same for `range-hot` and `range-idle`.

## 3. Where the error is

The period error mixes GPU and CPU chunks. Per chunk, measured mean duration / predicted chunk time
(median over its runs, pooled as in section 1), then the median of |ratio - 1| over the executed
schedules' chunks:

| Phone | App | Table | Model | GPU chunks: median abs error | CPU chunks: median abs error | GPU measured / predicted, median |
|---|---|---|---|---|---|---|
| Pixel 7a | tree | btpm | per-stage | 0.438 (n=10) | 0.372 (n=21) | 1.44 |
| Pixel 7a | tree | btpm | range-hot | 0.088 (n=10) | 0.366 (n=18) | 0.91 |
| Pixel 7a | tree | btpm | range-idle | 0.090 (n=10) | 0.366 (n=18) | 0.91 |
| Pixel 7a | tree | isolated | per-stage | 0.617 (n=10) | 0.376 (n=28) | 1.62 |
| Pixel 7a | tree | isolated | range-hot | 0.120 (n=10) | 0.313 (n=29) | 1.12 |
| Pixel 7a | tree | isolated | range-idle | 0.120 (n=10) | 0.333 (n=29) | 1.12 |
| Pixel 7a | cifar-dense | btpm | per-stage | 0.114 (n=10) | 0.401 (n=28) | 0.89 |
| Pixel 7a | cifar-dense | btpm | range-hot | 0.094 (n=10) | 0.396 (n=28) | 0.91 |
| Pixel 7a | cifar-dense | btpm | range-idle | 0.094 (n=10) | 0.396 (n=28) | 0.91 |
| Pixel 7a | cifar-dense | isolated | per-stage | 0.119 (n=10) | 0.245 (n=30) | 0.88 |
| Pixel 7a | cifar-dense | isolated | range-hot | 0.004 (n=10) | 0.323 (n=29) | 1.00 |
| Pixel 7a | cifar-dense | isolated | range-idle | 0.004 (n=10) | 0.323 (n=29) | 1.00 |
| Pixel 7a | cifar-sparse | btpm | per-stage | 0.013 (n=6) | 0.134 (n=17) | 1.00 |
| Pixel 7a | cifar-sparse | btpm | range-hot | 0.388 (n=7) | 0.155 (n=19) | 0.61 |
| Pixel 7a | cifar-sparse | btpm | range-idle | 0.388 (n=7) | 0.155 (n=19) | 0.61 |
| Pixel 7a | cifar-sparse | isolated | per-stage | 0.085 (n=6) | 0.348 (n=17) | 1.09 |
| Pixel 7a | cifar-sparse | isolated | range-hot | 0.062 (n=6) | 0.294 (n=16) | 1.06 |
| Pixel 7a | cifar-sparse | isolated | range-idle | 0.062 (n=6) | 0.294 (n=16) | 1.06 |
| Samsung | tree | btpm | per-stage | 0.950 (n=10) | 1.531 (n=19) | 1.95 |
| Samsung | tree | btpm | range-hot | 0.459 (n=10) | 0.630 (n=18) | 1.46 |
| Samsung | tree | btpm | range-idle | 0.334 (n=10) | 0.630 (n=18) | 1.33 |
| Samsung | tree | isolated | per-stage | 0.518 (n=10) | 3.028 (n=30) | 0.57 |
| Samsung | tree | isolated | range-hot | 0.400 (n=10) | 3.227 (n=28) | 0.60 |
| Samsung | tree | isolated | range-idle | 0.400 (n=10) | 3.227 (n=28) | 0.60 |
| Samsung | cifar-dense | btpm | per-stage | 0.065 (n=10) | 0.072 (n=27) | 0.94 |
| Samsung | cifar-dense | btpm | range-hot | 0.138 (n=10) | 0.082 (n=22) | 0.86 |
| Samsung | cifar-dense | btpm | range-idle | 0.142 (n=10) | 0.085 (n=22) | 0.86 |
| Samsung | cifar-dense | isolated | per-stage | 0.122 (n=10) | 0.074 (n=30) | 0.88 |
| Samsung | cifar-dense | isolated | range-hot | 0.140 (n=10) | 0.141 (n=26) | 1.14 |
| Samsung | cifar-dense | isolated | range-idle | 0.140 (n=10) | 0.141 (n=26) | 1.14 |
| Samsung | cifar-sparse | btpm | per-stage | 0.495 (n=5) | 0.425 (n=14) | 0.50 |
| Samsung | cifar-sparse | btpm | range-hot | 0.765 (n=5) | 0.212 (n=14) | 0.24 |
| Samsung | cifar-sparse | btpm | range-idle | 0.765 (n=5) | 0.212 (n=14) | 0.24 |
| Samsung | cifar-sparse | isolated | per-stage | 0.196 (n=7) | 0.113 (n=20) | 0.80 |
| Samsung | cifar-sparse | isolated | range-hot | 0.350 (n=5) | 0.099 (n=14) | 0.65 |
| Samsung | cifar-sparse | isolated | range-idle | 0.350 (n=5) | 0.099 (n=14) | 0.65 |

Reading it:

- **tree: the range price is much closer for the GPU chunk.** On the Pixel the GPU chunk's error falls
  from 0.44 to 0.09 (btpm) and from 0.62 to 0.12 (isolated): the per-stage sum misses about a third of
  the chunk (measured / predicted 1.44 and 1.62), the range record does not. On the Samsung it falls
  from 0.95 to 0.46 (btpm, 0.33 under `range-idle`) and 0.52 to 0.40 (isolated). The period error of
  tree stays large under every model because of the **CPU chunks**: 0.31-0.38 on the Pixel, 0.63-3.2 on
  the Samsung (a Medium chunk of stages 2-3 is priced at 0.58 ms and runs in about 7 ms). CPU prices are
  outside this change (non-goal).
- **cifar-dense: no gain.** The Pixel's GPU chunk was already priced within 11-12 % per stage; the
  isolated range record is exact (0.004), but the schedules it leads to are 8-13 % slower (the best
  `range-hot` schedule 43.12 vs 39.74 ms isolated, 48.81 vs 43.26 ms btpm), beyond the 0.1-0.4 ms
  noise. On the Samsung the range price is worse than the per-stage one (0.14 vs 0.07-0.12).
- **cifar-sparse under btpm: the interference range records overprice long GPU chunks.** The GPU chunks
  run in 0.24x (Samsung) and 0.61x (Pixel) of their range price, so the solver avoids long GPU chunks
  and its best schedule is 176-200 ms (74 %, 45 %) slower than the per-stage one. The cifar-sparse
  interference range sweep also took 1.8x (Pixel) and 2.8x (Samsung) as long as the isolated one
  (design.md, "Sweep durations"). Not analysed further; the likely cause is that bm-ranges' background load (every
  CPU tier looping over the range's own stages for the whole sweep) slows the host side of a chunk more
  than a pipeline's CPU chunks do.
- **`range-idle` = `range-hot` here.** With `--minimize_mode tmax` the idle-aware price never exceeds
  the period when t0 does not (design D3 "As built"), so both models find the same schedules and
  predict the same periods: identical errors in every cell. The best schedules differ only where the
  top-10 cut falls inside a tie (Pixel tree isolated). On the GPU chunk itself `range-idle` is closer
  only on Samsung tree btpm (0.33 vs 0.46).

## 4. What follows (not done here)

- A range model could still pay off for **tree** if the CPU chunks were priced better: the GPU chunk is
  already well predicted there. The CPU tiers' per-stage sums are now the larger error on both phones.
- The **interference range sweep** needs a background load closer to a pipeline's before its numbers
  are used for long GPU chunks (cifar-sparse btpm).
- `range-idle` adds nothing in `tmax` mode; it only changes `min_time` and the gap objective.

## 5. How to reproduce

```bash
# store (per phone and app): docs/instruction-for-ai/06-end-to-end-scheduling.md Step 1
# schedules (per phone, app, table, model):
uv run optimizer/orchestrate/02_gen_schedule_merged.py --profiling_root data/profiling --device <serial> \
  --app <app> --backend vk --table_type <btpm|isolated> --minimize_mode tmax --num_solutions 10 \
  --output_folder data/schedules_<table> --cost-model <per-stage|range-hot|range-idle>
# executions: the distinct assignments of the three files' top 10 (cifar-sparse top 5), each written
# to a one-schedule JSON and run 3 rounds x in order, 20 s apart, from /data/local/tmp/<dir>:
adb -s <serial> shell 'cd /data/local/tmp/<dir> && BT_WEIGHTS_DIR=/data/local/tmp/<dir>/weights \
  LD_LIBRARY_PATH=. ./bm-gen-logs-<app>-vk --device <serial> --schedule-file one-NN.json \
  --n-schedules-to-run 1' </dev/null > data/sched_logs/<serial>_<app>_vk_<table>/schedule_run_<k>.log
# accuracy (section 2 per cost model and table) and per-schedule ratios:
uv run optimizer/analysis/speedup_summary.py --sched-logs data/sched_logs --schedules-root data
uv run optimizer/analysis/validate_btpm.py
```

`analysis.accuracy` reads both minimize modes' schedule files and reports each schedule under the
cost model it records (`metrics.cost_model`, so a range model's per-stage fallback counts as
`per-stage`), one row per (cost model, table, assignment). This run has neither gapness files nor
fallbacks: regenerated with that join, the summary has the same 310 rows and medians, and the
tables above are unchanged.

The gate columns (`Lower error`, `Best not worse`) and section 3 were computed from
`analysis.accuracy`'s rows and the same logs by scratch scripts kept outside the repository.
