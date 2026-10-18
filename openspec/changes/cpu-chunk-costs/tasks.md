# Tasks

Gates used below:
- **"local":** `ctest --test-dir build/pc -L omp` and `uv run pytest -q optimizer`.
- **"phone gate":** `scripts/run-on-android.sh <serial> test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk` on
  3A021JEHN02756 (Pixel 7a) and R5CY21Y3VEV (Samsung).
- **"phone profiling":** as in `docs/instruction-for-ai/06-end-to-end-scheduling.md` Step 1, 20 s between runs; on
  the Samsung `taskset` takes a hex mask. **Nothing is profiled on rocky-ryzen.**

## 1. Input-pure stages (profiling/cost-table)

- [x] 1.1 Add the D2 test for tree, cifar-dense and cifar-sparse (OMP). Verify: it fails on the current code for tree stage 2 (record which buffers change), and the list of every failing (app, stage) is written under this task.
  - *Test:* `bt::testing::ExpectStageInputPure` (`platform/util/testing/input_purity.hpp`) runs a stage twice on
    one AppData whose earlier stages ran. Every listed region (each buffer and scalar field) the stage does not write
    must equal its contents before the first run, after each run, and the regions it writes must match between the
    two runs. Tests `InputPurityOmp.{TreeAppData,TreeSafeAppData}` (`test-tree-omp`), `InputPurityOmp.CifarDense`
    (`test-cifar-dense-omp`), `InputPurityOmp.CifarSparse` (`test-cifar-sparse-omp`). `TreeAppData` is the chained
    `tree::AppData` that `bm-prof-tree-vk` and the pipeline run; `TreeSafeAppData` is the oracle's path and repeats
    stage 2 with 4 OMP threads on re-run stage-1 keys, since `test-tree-omp` runs single-threaded and the segment
    merge only happens with several threads.
  - *Fails on the current code (4765310 plus the test, build box):* tree stage 2 only.
    - `tree::AppData` stage 2 (`run_stage_2(tree::AppData&)`, `std::ranges::sort` in place): `u_morton_keys_s1`,
      1220389 of 1228800 bytes differ after the first run and after the second.
    - `SafeAppData` stage 2 (`omp::parallel_sort`): `u_morton_keys_s1_out`, 1220258 of 1228800 bytes differ after each
      run, with 1 thread and with 4.
    - Every other stage passes: tree 1 and 3-7 on both paths, cifar-dense 1-11, cifar-sparse 1-11 (inputs, weights,
      CSR arrays and `nnz` unchanged, outputs identical across the two runs).
- [x] 1.2 Make every failing stage keep its input intact (D1). Verify: the test passes; the OMP oracle suites and the VK/CUDA differential suites that compare with OMP still pass under local and the phone gate.
  - *Fix:* `run_stage_2(tree::AppData&)` copies `u_morton_keys_s1` into `u_morton_keys_sorted_s2` and sorts there.
    `omp::parallel_sort` takes its input `const`: each thread copies its segment into the output and sorts it there,
    and the segments are merged pairwise with `std::inplace_merge` in the output (the old ping-pong merged back into
    the input); `merge_segments` became unused and was removed.
  - *local:* `ctest --test-dir build/pc -L omp` 11/11 passed (the four `InputPurityOmp` tests included);
    `uv run pytest -q optimizer` 41 passed.
  - *phone gate* (`BT_ANDROID_DEST=/data/local/tmp/cpucost scripts/run-on-android.sh <serial> test-tree-vk
    test-cifar-dense-vk test-cifar-sparse-vk test-tree-omp`): Pixel 7a and Samsung each 20 / 16 / 17 / 23 tests
    passed, exit 0. CUDA is not in either gate (no Jetson run here).
- [x] 1.3 Validate the timing on one phone (D3): time OMP stage 2 on the Pixel's Medium tier with the fixed code, once by `bm-prof-tree-vk` and once with fresh input before every sample. Verify: the two medians agree within the spread, and both numbers plus the old 0.64 ms are recorded in design.md.
  - Pixel 7a, isolated, `medium/stage2`, process pinned to the Medium cores (`taskset 30`), 3 alternating rounds 20 s
    apart: `bm-prof-tree-vk` p50 9.110 / 9.156 / 9.116 ms (median 9.116, spread 9.110-9.156), fresh input
    9.108 / 9.126 / 9.114 ms (median 9.114, spread 9.108-9.126). Each median lies inside the other's spread. Old
    price 0.64 ms. Details and the harness variant: design.md, "Measurements".

## 2. Refresh the store and fit the overhead (optimizer/cost-model)

- [x] 2.1 Re-profile tree on both phones (isolated and interference, 2 runs each) into `data/profiling/<serial>/tree/vulkan/`. Verify: the loaders accept the store, and the old vs new stage-2 CPU times per tier are recorded here.
  - *Run (2026-10-18 07:45 UTC, both phones at once, each phone's runs in sequence):* `uv run
    optimizer/orchestrate/01_collect_profiling.py --device <serial> --app tree --backend vk --adb-serial <serial>
    --build-dir build/android --dest /data/local/tmp/cpucost --tools prof --runs 2 --scenarios isolated,interference
    --cooldown-s 20`, the binary of ef8026a. 8 runs, 28 records each, none failed. The range records
    (`range-*.jsonl`) were not re-collected (GPU only, unchanged).
  - *Loaders accept it:* `profiling_loader.py --device <serial> --app tree --backend vulkan` loads 56 records from 2
    runs (and the 112 range records) on both phones, exit 0; `smt.data_loader.load_stage_timings` builds both tables.
    Under the interference gate (`min_runs` 2) some sub-ms CPU cells of stages 3 and 6 are UNAVAILABLE as before
    (cv > 1; the 2026-10-17 store lost the same cells and also stage 2 little/big on the Pixel).
  - *Old vs new stage-2 p50 per tier, run 1 / run 2 (ms).* "Old" is the store the 2026-10-17 gate solved on
    (`superseded-2026-10-17/`, collected 2026-10-16 22:43). `data/` is git-ignored, and step 01 replaced the
    tree-app-improvements refresh of 2026-10-17 20:53 in place; its Pixel values (read before the re-run) were
    isolated little 0.394 / 0.397, medium 1.926 / 0.397, big 2.015 / 2.029, interference little 0.478 / 0.528,
    medium 0.447 / 0.498, big 0.492 / 0.494. The Samsung's are lost.

    | phone, scenario | little old | little new | medium old | medium new | big old | big new |
    |---|---|---|---|---|---|---|
    | Pixel 7a, isolated | 0.467 / 0.491 | 7.020 / 7.148 | 0.461 / 0.493 | 7.006 / 7.074 | 0.456 / 0.505 | 15.904 / 16.318 |
    | Pixel 7a, interference | 0.601 / 0.623 | 7.933 / 8.405 | 0.670 / 0.615 | 7.888 / 7.881 | 0.572 / 0.623 | 9.038 / 7.876 |
    | Samsung, isolated | 0.317 / 0.318 | 5.654 / 5.616 | 0.304 / 0.317 | 5.440 / 5.459 | 1.188 / 1.188 | 6.465 / 7.925 |
    | Samsung, interference | 0.372 / 0.376 | 6.294 / 6.488 | 0.376 / 0.379 | 6.053 / 6.448 | 0.389 / 0.351 | 7.091 / 8.127 |

    Stage 2 on a CPU now costs 5.4-16 ms instead of 0.3-1.2 ms (12-35x): on the Pixel about what its GPU takes
    (6.7-7.6 ms isolated, 4.0-4.9 interference), on the Samsung 4-6x its GPU (1.1-1.6 ms). The tiers are nominal on tree (design.md, "Tree CPU stages are not
    pinned"): the Pixel's `big` cells (cv 0.52) are the unpinned single-threaded sort landing on slower cores part of
    the time, and the pinned Medium measurement of task 1.3 (9.1 ms) is slower than the unpinned `medium` cell (7.0).
- [x] 2.2 Fit `overhead.json` for each phone with `fit_overhead.py` from the existing executions (D4). Verify: the fitted constants are recorded here, and pytest still passes.
  - `uv run optimizer/analysis/fit_overhead.py` (the 2026-10-17 executions in `data/sched_logs`, re-priced on the
    refreshed tree table and the unchanged cifar tables; 367 chunk samples) wrote
    `data/profiling/<serial>/overhead.json`:

    | phone | class | n | model | per_chunk_ms | per_stage_ms | mean abs residual before / after (ms) |
    |---|---|---|---|---|---|---|
    | Pixel 7a | cpu | 136 | none | 0 | 0 | 20.755 / 20.755 |
    | Pixel 7a | gpu_vulkan | 50 | none | 0 | 0 | 7.286 / 7.286 |
    | Samsung | cpu | 131 | median | 0.5528 | 0 | 9.910 / 9.812 |
    | Samsung | gpu_vulkan | 50 | none | 0 | 0 | 21.499 / 21.499 |

  - The Pixel's CPU fit is zero: its median residual over all CPU chunks is -0.575 ms (clamped to 0), because the
    cifar tables over-price CPU chunks (cifar-dense btpm median measured/priced 0.62, isolated 0.77) while tree's
    chunks run over their price (tree isolated 1.64); a single constant pooled over apps and tiers cannot serve both.
    The fit is shipped or not by the held-out check of task 3.2. No GPU overhead is fitted on either phone, so GPU
    chunk prices are unchanged.
  - `uv run pytest -q optimizer`: 41 passed.

## 3. Acceptance gate (optimizer/cost-model)

- [x] 3.1 Solve top-10 (cifar-sparse top-5) per phone, app and table with the per-stage model, the refreshed store and the overhead, and execute them as in the 2026-10-17 gate. Verify: every cell has executions in `data/sched_logs`.
  - The 2026-10-17 executions and schedule files were moved to `data/archive/2026-10-17-gpu-chunk-cost-model/`
    first, so that their logs (old stage-2 kernel) do not pool with the new ones by assignment.
  - *Solve:* `02_gen_schedule_merged.py --profiling_root data/profiling --device <serial> --app <app> --backend vk
    --table_type <btpm|isolated> --minimize_mode tmax --num_solutions 10 --output_folder data/schedules_<table>`
    (per-stage, the 2.2 `overhead.json` in the store; the union sweep added 0-2 candidates per file), 12 files.
  - *Execute* (2026-10-18 00:52-01:59 PDT): per cell the top 10 by `metrics.max_time` (cifar-sparse top 5), each as a
    one-schedule JSON in its own `bm-gen-logs-<app>-vk` process from `/data/local/tmp/cpucost` with the real
    weights, 3 rounds in the same order, 20 s idle before every run, both phones at once. 300 runs (50 per round per
    phone), none failed.
  - Every cell has executions: `data/sched_logs/<serial>_<app>_vk_<table>/`, 30 logs for each tree and cifar-dense
    cell and 15 for each cifar-sparse cell, on both phones.
- [x] 3.2 Compute per-chunk CPU errors on the new executions with and without the overhead, and compare with the 2026-10-17 values (gate report section 3, per-stage rows). Verify: a decision record in `docs/reports-for-human/` lists every (phone, app, table) with before / after CPU error and best measured period, and states whether each spec bound holds. If the overhead fails the held-out check, `overhead.json` is not shipped for that phone and the reason is recorded.
  - Decision record: `docs/reports-for-human/2026-10-18-cpu-chunk-costs-gate.md` (all 12 cells, before / after CPU
    error with and without the overhead, best measured period against the noise, each bound's verdict). The
    computing script reproduces the 2026-10-17 report's per-stage rows exactly on that report's data.
  - *Tree CPU error lower on both tables, both phones:* holds (Pixel 0.372 to 0.218 btpm, 0.376 to 0.297
    isolated; Samsung 1.531 to 0.366, 3.028 to 1.165).
  - *cifar CPU error not higher:* **fails** in 5 of 8 cells (Pixel cifar-dense btpm 0.401 to 0.409, isolated 0.245
    to 0.260, cifar-sparse btpm 0.134 to 0.215; Samsung cifar-dense btpm 0.072 to 0.186, cifar-sparse isolated
    0.113 to 0.157). The Pixel's three are the same schedules at the same prices as on 2026-10-17 (session variation).
  - *Best measured period not worse beyond noise:* holds in all 12 cells.
  - *Held-out check:* **fails** on both phones (median CPU-chunk error over all new executions, with / without:
    Pixel 0.315 / 0.315, the fit being zero; Samsung 0.307 / 0.244). Neither `overhead.json` is shipped: both moved
    to `data/archive/2026-10-18-cpu-chunk-costs-rejected-overhead/<serial>/`, and `data/profiling/<serial>/` holds
    none. The reasons are in the record, section 3.
- [x] 3.3 Update `docs/instruction-for-ai/05-profiling.md` (input-pure stages) and `06-end-to-end-scheduling.md` (the shipped overhead). Verify: the documented commands reproduce one cell of 3.1.
  - 05: new section "A stage is profiled on unmodified input (input-pure stages)": the rule, the
    `InputPurityOmp` guard, and that tree's CPU tiers are nominal, with how to pin with `taskset`. 06: the phones
    ship no `overhead.json`, why, where the rejected fits are, and the held-out check a refit must pass. The report
    lists the 3.1 commands (section 6).
  - *Reproduction*, with the store as shipped (the Pixel's fit was zero, so its prices are 3.1's):
    - Pixel tree btpm: the 06 Step 2 command returns 3.1's 10 executed assignments with the same uids and prices.
      Its best schedule SCH-L2G5-G160-4fd8, run 3 times by the report's command 20 s apart, measured 9.84 / 21.17 /
      28.78 ms (3.1: 11.54 [10.44-21.44]). The median is inside 3.1's range, the third run above it: the cell is
      bimodal (the Little `[1,2]` chunk holds the unpinned sort).
    - Pixel cifar-dense btpm: 10 of 10 assignments with the same uids and prices. SCH-M4B1G5L1-G4821-09d0
      measured 43.25 / 42.98 / 43.26 ms (3.1: 43.19 [42.90-43.43]).

## 4. Final gate

- [x] 4.1 local and phone gate green at the last commit. Verify: results listed in the PR.
  - At 5161488 (the commit after it changes only this file): `ctest --test-dir build/pc -L omp` 11/11 passed
    (`InputPurityOmp.*` included); `uv run pytest -q optimizer` 41 passed; phone gate from a fresh `android` build,
    `BT_ANDROID_DEST=/data/local/tmp/cpucost scripts/run-on-android.sh <serial> test-tree-vk test-cifar-dense-vk
    test-cifar-sparse-vk`: Pixel 7a and Samsung each 20 / 16 / 17 passed, 0 skipped, exit 0.
  - No PR is opened in this run (its instructions forbid it); these results are for the PR description.
