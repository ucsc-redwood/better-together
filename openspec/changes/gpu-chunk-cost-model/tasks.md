# Tasks

Gates used below:
- **"local":** `uv run pytest -q optimizer` and `ctest --test-dir build/pc -L omp`.
- **"phone gate":** `cmake --preset android -DBT_BUILD_BENCHMARKS=ON`, the four VK test targets via
  `scripts/run-on-android.sh <serial> ...` on 3A021JEHN02756 (Pixel 7a) and R5CY21Y3VEV (Samsung).
- **"phone profiling":** `adb push` the tool to `/data/local/tmp/<dir>`. Run with
  `BT_PROF_SCENARIO=<isolated|interference> BT_PROF_RUN=<n> ... --device <serial>`, capturing stdout to
  the store path, with 20 s between runs. On the Samsung, `taskset` takes a hex mask (no `-c`).
  **Nothing is profiled on rocky-ryzen.**

## 1. Clock provenance (profiling/cost-table: frequency)

- [x] 1.1 Add Mali (`/sys/class/misc/mali0/device/cur_freq`, kHz) and Exynos (`/sys/kernel/gpu/gpu_clock`, devfreq `*sgpu*/cur_freq`) readers to `read_gpu_clock_mhz()`, and a unit test of the parsing on sample file contents. Verify: the test passes under local.
  - *Evidence:* the reader moved to `profiler/gpu_clock.hpp` (no benchmark/JSON dependencies, so a plain gtest can
    include it) and takes a root prefix for tests. Units as read on the phones (2026-10-16, `adb shell` as `shell`):
    Pixel `mali0/device/cur_freq` 202000 (kHz, idle); Samsung `gpu_clock`, `22200000.sgpu/cur_freq`,
    `gpu_freq_table` and `available_frequencies` all in kHz (252000..1095000), `gpu_clock` and `cur_freq` 0 while
    idle. A 0, or anything below one MHz, counts as unknown and the next source is tried; the existing Hz sources keep
    their units.
  - `test-gpu-clock-omp` (`profiler/tests/test_gpu_clock.cpp`, labels `omp;unit`): fake sysfs trees for AMD
    `pp_dpm_sclk`, Tegra devfreq (Hz), Mali (kHz), Exynos `gpu_clock` (kHz), the `*sgpu*` devfreq beside a non-GPU
    devfreq node, the idle 0 on both Exynos nodes (-1), garbage (-1), and no source (-1).
  - Local: `ctest --test-dir build/pc -L omp` → 100 % of 11 passed (`test-gpu-clock-omp` 9/9, of which
    `GpuClockDevice.ReadCost` measures, not checks); `uv run pytest -q optimizer` → 21 passed.
- [x] 1.2 Sample the clock during each GPU cell's measured loop and record the median of the non-zero samples, -1 if there are none (D5); measure the reader's cost per call on the Pixel. Verify: a `bm-prof-tree-vk` run on each phone writes positive `gpu_clock_mhz` on every GPU record, and design.md records the read cost and the chosen stride.
  - *Evidence:* `GpuClockSampler` (`profiler/gpu_clock.hpp`) reads on its own thread every `BT_PROF_CLOCK_PERIOD_MS`
    (5) while `measure_calibrated` runs a GPU cell; `emit_jsonl` writes its median of the positive samples as the GPU
    record's `provenance.gpu_clock_mhz` (-1 if none) and their count as `gpu_clock_samples`. CPU records keep the
    run-level snapshot as before.
  - Read cost on the Pixel: 30.3 us per call (`./test-gpu-clock-omp --device 3A021JEHN02756`,
    `GpuClockDevice.ReadCost`); design.md "Open Questions" records it and the 5 ms stride.
  - Android build at f820476 + this group, pushed to `/data/local/tmp/bt-costmodel`;
    `BT_PROF_SCENARIO=isolated BT_PROF_RUN=1 ./bm-prof-tree-vk --device <serial>`, 28 records each. Every GPU record
    has a positive clock (stage: p50 ms, MHz, samples):

    | stage | Pixel 7a | Samsung |
    |---|---|---|
    | 1 | 1.364, 202, 164 | 1.585, 252, 432 |
    | 2 | 6.773, 400, 84 | 1.374, 800, 92 |
    | 3 | 2.164, 202, 163 | 3.069, 350, 499 |
    | 4 | 2.628, 202, 111 | 1.377, 252, 209 |
    | 5 | 0.293, 202, 283 | 0.564, 252, 556 |
    | 6 | 1.631, 202, 122 | 1.212, 252, 342 |
    | 7 | 1.294, 202, 181 | 2.880, 315, 513 |

    Isolated single-stage cells mostly run at the lowest operating point of each GPU (Pixel 202 MHz, Samsung
    252 MHz); only the radix sort (stage 2) raises it.

## 2. Range records (profiling/cost-table: ranges)

- [x] 2.1 Add `schemas/profiling-range.schema.json` (`bt.profiling-range/v0`: `start_stage`, `end_stage`, `gap_ms`, `submissions`, timing, provenance with clock samples) and a loader for `range-*.jsonl` that the per-stage loader never reads. Verify: pytest checks that an old store loads unchanged, and that a store with range files yields the same per-stage table as without them.
  - *Evidence:* `schemas/profiling-range.schema.json` (`schema` const `bt.profiling-range/v0`, so a range record
    misfiled as `run-*.jsonl` fails the per-stage schema instead of being read as stage `start_stage`).
    `profiling_loader.load_ranges` globs `range-*.jsonl` only and returns, per range, t0 (`gap_ms` 0) and t1
    (`gap_ms` > 0), each count-weighted over the runs, with the scenario's `min_runs` gate on both cadences;
    `data_loader.load_range_timings` gives the solver `{(i, j): (t0, t1)}` (0-based). The per-stage loaders are
    unchanged (they glob `run-*.jsonl`).
  - `optimizer/tests/test_range_store.py`: `test_old_store_loads_unchanged` (no range files: the per-stage table
    loads, the range loader yields `{}`), `test_range_files_leave_the_per_stage_table_unchanged` (`load_profiling`
    and `load_stage_timings`, both scenarios, equal before and after adding range files),
    `test_range_loader_reads_both_cadences`, `test_interference_ranges_need_two_runs`,
    `test_range_record_is_not_a_per_stage_record`. Local: `uv run pytest -q optimizer` → 38 passed (5 of them
    here); `ctest --test-dir build/pc -L omp` → 100 % of 11 passed.
- [x] 2.2 Build `bm-ranges-<app>-vk` from `bm_phases_vulkan.hpp` (D2): every range, interleaved, `gap 0` and `gap = t0`, calibrated budget, the interference background under `BT_PROF_SCENARIO=interference`, JSONL to stdout. Verify: on the Pixel, the tree tool emits 56 records (28 ranges x 2 gaps) that validate against the schema, and its [1, 7] record reports 2 submissions.
  - *Evidence:* `profiler/bm_ranges_vulkan.hpp` + `profiler/{tree,cifar-dense,cifar-sparse}-vk/bm_ranges.cpp`
    (design D2 "As built"). `bm_prof_common.hpp` gives it `base_provenance` and `timing_json`, split out of
    `emit_jsonl` (bm-prof's records are unchanged).
  - Pixel 7a, Android build of this group (`cmake --preset android -DBT_BUILD_BENCHMARKS=ON`), pushed to
    `/data/local/tmp/bt-costmodel`: `BT_PROF_SCENARIO=isolated BT_PROF_RUN=1 ./bm-ranges-tree-vk --device
    3A021JEHN02756` took 41 s and wrote **56 records** (28 ranges, 28 with `gap_ms` 0), **0 schema errors**
    (`jsonschema` Draft 2020-12 against the new schema). [1, 7]: `submissions` **2** at both gaps; p50 22.88 ms at
    gap 0 and 24.59 ms after a 22.88 ms gap, 12 samples each, `gpu_clock_mhz` 202.
- [x] 2.3 Collect range and per-stage tables for tree, cifar-dense and cifar-sparse on both phones (isolated and interference, 2 runs each) under `data/profiling/`, and record the time each sweep took. Verify: every cell has both kinds of files, the loaders accept them, and design.md records the sweep durations.
  - *Evidence:* store `data/profiling/{3A021JEHN02756,R5CY21Y3VEV}/{tree,cifar-dense,cifar-sparse}/vulkan/{isolated,interference}/`
    (not committed), collected 2026-10-16 22:42 to 2026-10-17 00:18 (PDT) with the Android build of this group, the
    commands of 06 Step 1, `--dest /data/local/tmp/bt-costmodel`; records carry `git_sha` 12e895f (the build's HEAD;
    the tree held this group uncommitted). The tables an interrupted earlier run had left there were all replaced.
  - Every one of the 12 (phone, app, scenario) directories holds `run-001/002.jsonl` and `range-001/002.jsonl`
    (28 / 44 / 44 per-stage and 56 / 132 / 132 range records per file for tree / dense / sparse), every run reported
    `ok`. Loaders: `load_profiling`, `load_stage_timings` and `load_ranges` accept all 12 (schema-validated); the
    solver gets every range (`load_range_timings`: 28 for tree, 66 for the cifar apps, in all 12).
  - Range records: all 2,560 have a positive `gpu_clock_mhz`; none abandoned; samples per record at least 10 (tree),
    3 (cifar-dense) and 2 (cifar-sparse under interference, cut by `BT_PROF_MAX_CELL_S`). Median t1/t0 per cell:
    Pixel 1.04-1.22, Samsung tree 2.34 isolated / 1.54 interference, Samsung cifar 1.00-1.51.
  - Durations: design.md "Sweep durations (task 2.3)" (a whole cell: tree ~7 min, cifar-dense ~17 min, cifar-sparse
    72 min on the Pixel and 60 min on the Samsung).
- [x] 2.4 Document the range store and `bm-ranges-*-vk` in `docs/instruction-for-ai/05-profiling.md` and `06-end-to-end-scheduling.md` (Step 1). Verify: the documented commands reproduce one cell of 2.3.
  - *Evidence:* 05 "GPU stage ranges as whole chunks (`bm-ranges-<app>-vk`)" (what is measured, the two cadences,
    the knobs, the output and where the store keeps it, a by-hand adb recipe, the clock in provenance); 06 the store
    layout and Step 1's `01 --tools prof|ranges` commands. Those two commands, with `<serial>`, `<app>` and
    `<dir>` = `bt-costmodel` filled in, are what produced every cell of 2.3 (`/tmp` driver running them per phone
    for tree, cifar-dense, cifar-sparse); the by-hand recipe of 05 is the one used for 2.2's check.

## 3. Solver cost models (optimizer/cost-model)

- [x] 3.1 Add a `--cost-model {per-stage,range-hot,range-idle}` option to `02_gen_schedule_merged.py` (default `per-stage`) and record it in each schedule's `metrics.cost_model`. Verify: pytest shows that `per-stage` on a store with range files gives byte-identical schedule JSON to today's, apart from the new field.
  - *Evidence:* `02 --cost-model` (choices `constraints.COST_MODELS`, default `per-stage`). Every schedule gets
    `metrics.cost_model` and `metrics.table` (design D6). A range model reads `load_range_timings` of the scenario
    and writes `schedules_<table>_<mode>_<model>.json` (`Case.schedule_path`); without range files it warns and
    prices per stage. Under `per-stage` the range records are never loaded and `add_chunk_time_constraint` emits
    today's constraints.
  - `test_gen_schedule_per_stage_is_unchanged_on_a_store_with_ranges`: `02 --cost-model per-stage` as a subprocess
    on a store with range files; with the two new fields removed, its JSON equals byte for byte what the pre-change
    load → solve → dump path writes. `test_per_stage_ignores_range_records`: per-stage with range costs, and the
    range models without any, give today's candidates.
  - Also on the real phone tables (2026-10-16 store of 2.3, tree, both phones x both tables): HEAD's (12e895f)
    `02_gen_schedule_merged.py`, exported with `git archive`, against this one with `--cost-model per-stage`,
    `--num_solutions 10`: all four outputs byte-identical apart from `metrics.cost_model`/`metrics.table`.
- [x] 3.2 Implement `range-hot`: a GPU maximal segment costs t0 from its range record, with no stage sum and no GPU overhead, and the constraint applies to maximal segments only (D4). CPU chunks are unchanged. Verify: pytest on a hand-made table with a non-monotone range (a sub-range dearer than its parent) gets the expected optimum, and a mixed schedule's CPU chunk keeps its per-stage price.
  - *Evidence:* `constraints.add_chunk_time_constraint(cost_model="range-hot", range_costs=...)`: for the GPU,
    `maximal(i..j) -> t0(i, j) <= T_max` and `>= T_min`; the per-sub-segment constraint is not emitted for it; a
    GPU range without a record keeps its per-stage price on its maximal segment. The solver reports the price
    (`gpu_prices`, evaluated in the model) as the chunk's `time`; `reprice_solution` reproduces it.
  - `test_range_hot_prices_the_chunk_not_its_sub_ranges`: [1, 7] at 6 ms, its sub-range [1, 3] at 9 ms; the
    optimum is `GPU[1-7]` at 6 ms (the per-stage model's optimum on the same table: 12 ms).
    `test_range_hot_cpu_chunks_keep_their_per_stage_price_and_gpu_drops_its_overhead`: with fitted overheads on Big
    and the GPU, every Big chunk of 15 candidates (mixed ones included) costs stage sum + Big overhead, every GPU
    chunk its t0. `test_range_price_does_not_depend_on_per_stage_records`: the [4, 7] chunk's price is the same for
    GPU stage times of 1 and 50 ms and lies in [t0, t1] (spec "Range data present").
- [x] 3.3 Implement `range-idle` (D3's piecewise-linear interpolation between t0 and t1 with `g = T_max - t`). Verify: pytest checks a single-GPU-chunk case at the bottleneck (price t0) and one with slack (price above t0, at most t1), and solve time on the tree phone tables stays within 2x of `per-stage` (recorded).
  - *Evidence:* price `min(t1, (t0 + k*T_max)/(1 + k))`, `k = (t1 - t0)/t0` (`constraints.idle_price`), at t0
    when the GPU has several chunks. It is not asserted as a constraint: `price <= T_max` holds exactly when
    `t0 <= T_max`, and `price >= T_min` is `lin >= T_min and t1 >= T_min`, so everything stays linear.
  - `test_range_idle_at_the_bottleneck_is_t0` (t0 5, t1 10, Big 3 ms: price 5, period 5);
    `test_range_idle_with_slack_rises_toward_t1` (Big 8 ms: price 6.5 = `idle_price(5, 10, 8)`, period 8; Big 40 ms:
    price 10 = t1); `test_reprice_matches_the_solver_under_both_range_models`.
  - Solve time, `solve_optimization_problem(..., 10 solutions, tmax)` in-process on the 2.3 tree tables, median of 5
    (`/tmp` script, build box):

    | device | table | per-stage ms | range-hot ms | range-idle ms | range-idle / per-stage |
    |---|---|---|---|---|---|
    | Pixel 7a | btpm | 61.7 | 55.7 | 68.2 | 1.11x |
    | Pixel 7a | isolated | 91.3 | 89.4 | 107.5 | 1.18x |
    | Samsung | btpm | 89.0 | 90.1 | 150.0 | 1.69x |
    | Samsung | isolated | 104.2 | 112.2 | 134.0 | 1.29x |

    Whole `02` invocations (Python start included): 0.15-0.24 s for every model.
  - A consequence of D3, recorded in design.md (D3 "As built"): the price stays below the period whenever t0 does,
    so in `tmax` mode `range-idle` finds the same optimum and predicts the same period as `range-hot`.
- [x] 3.4 Price several GPU chunks by the sum of their maximal segments' t0 under `--allow-gpu-multi-chunk` (D4). Verify: pytest with a two-GPU-chunk schedule checks the period is at least the sum.
  - *Evidence:* for a shared GPU under a range model, `sum over (i, j) of If(maximal(i..j), t0(i, j), 0) <= T_max`
    (a range without a record adds its per-stage price); `range-idle` prices each chunk at t0 when there are
    several. `test_two_gpu_chunks_bound_the_period_by_their_sum`: GPU [1, 2] at 3 ms and [5, 7] at 4 ms around
    `Big[3-4]`; under both range models the optimum is `GPU[1-2] | Big[3-4] | GPU[5-7]` with chunk times 3 and 4
    and a period of 7, and every one of the 10 candidates has a period >= the sum of its GPU chunks. Without the
    option the best schedule has one GPU chunk.
- [x] 3.5 Document the option and the models in `06-end-to-end-scheduling.md` (Step 2). Verify: the doc names the default and how to compare models.
  - *Evidence:* Step 2 "GPU cost model (`--cost-model`, default `per-stage`)": a table of the three models, the CPU,
    maximal-segment, fallback and multi-chunk rules, the output file names and recorded fields, the measured solve
    times, and "To compare models on a cell": solve each on the same table, execute (Step 3), read the "Prediction
    accuracy" section of `speedup_summary.py` (or `validate_btpm.py`), with the commands.

## 4. Accuracy reporting and the acceptance gate (optimizer/cost-model)

- [ ] 4.1 Move `validate_btpm.py`'s measured/predicted join into a library function, and add the per-schedule ratio table and the per-(device, app, backend, cost model, table) median absolute error to `speedup_summary.py`. Keep the script as a CLI. Verify: pytest on fixture schedule JSON and fixture `sched_logs` gives the expected ratios and medians.
- [ ] 4.2 For each phone and app, solve with all three models on both tables and execute the candidates with `bm-gen-logs` (top-10; cifar-sparse top-5). Verify: the summary lists every model's error and best measured period for every cell.
- [ ] 4.3 Apply the spec's gate. Make the best range variant the default only if, on both phones and for all three apps, its median absolute error is lower than `per-stage`'s on the same table and its best measured period is not worse beyond noise. Verify: a decision record with every table is in `docs/reports-for-human/`, the default in `02_gen_schedule_merged.py` matches it, and local passes.

## 5. Final gate

- [ ] 5.1 Final gate at the last commit: local and phone gate green, and the summary from 4.2 regenerated from the committed code. Verify: the results are listed in the PR.
