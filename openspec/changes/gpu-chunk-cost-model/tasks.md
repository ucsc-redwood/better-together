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

- [ ] 2.1 Add `schemas/profiling-range.schema.json` (`bt.profiling-range/v0`: `start_stage`, `end_stage`, `gap_ms`, `submissions`, timing, provenance with clock samples) and a loader for `range-*.jsonl` that the per-stage loader never reads. Verify: pytest checks that an old store loads unchanged, and that a store with range files yields the same per-stage table as without them.
- [ ] 2.2 Build `bm-ranges-<app>-vk` from `bm_phases_vulkan.hpp` (D2): every range, interleaved, `gap 0` and `gap = t0`, calibrated budget, the interference background under `BT_PROF_SCENARIO=interference`, JSONL to stdout. Verify: on the Pixel, the tree tool emits 56 records (28 ranges x 2 gaps) that validate against the schema, and its [1, 7] record reports 2 submissions.
- [ ] 2.3 Collect range and per-stage tables for tree, cifar-dense and cifar-sparse on both phones (isolated and interference, 2 runs each) under `data/profiling/`, and record the time each sweep took. Verify: every cell has both kinds of files, the loaders accept them, and design.md records the sweep durations.
- [ ] 2.4 Document the range store and `bm-ranges-*-vk` in `docs/instruction-for-ai/05-profiling.md` and `06-end-to-end-scheduling.md` (Step 1). Verify: the documented commands reproduce one cell of 2.3.

## 3. Solver cost models (optimizer/cost-model)

- [ ] 3.1 Add a `--cost-model {per-stage,range-hot,range-idle}` option to `02_gen_schedule_merged.py` (default `per-stage`) and record it in each schedule's `metrics.cost_model`. Verify: pytest shows that `per-stage` on a store with range files gives byte-identical schedule JSON to today's, apart from the new field.
- [ ] 3.2 Implement `range-hot`: a GPU maximal segment costs t0 from its range record, with no stage sum and no GPU overhead, and the constraint applies to maximal segments only (D4). CPU chunks are unchanged. Verify: pytest on a hand-made table with a non-monotone range (a sub-range dearer than its parent) gets the expected optimum, and a mixed schedule's CPU chunk keeps its per-stage price.
- [ ] 3.3 Implement `range-idle` (D3's piecewise-linear interpolation between t0 and t1 with `g = T_max - t`). Verify: pytest checks a single-GPU-chunk case at the bottleneck (price t0) and one with slack (price above t0, at most t1), and solve time on the tree phone tables stays within 2x of `per-stage` (recorded).
- [ ] 3.4 Price several GPU chunks by the sum of their maximal segments' t0 under `--allow-gpu-multi-chunk` (D4). Verify: pytest with a two-GPU-chunk schedule checks the period is at least the sum.
- [ ] 3.5 Document the option and the models in `06-end-to-end-scheduling.md` (Step 2). Verify: the doc names the default and how to compare models.

## 4. Accuracy reporting and the acceptance gate (optimizer/cost-model)

- [ ] 4.1 Move `validate_btpm.py`'s measured/predicted join into a library function, and add the per-schedule ratio table and the per-(device, app, backend, cost model, table) median absolute error to `speedup_summary.py`. Keep the script as a CLI. Verify: pytest on fixture schedule JSON and fixture `sched_logs` gives the expected ratios and medians.
- [ ] 4.2 For each phone and app, solve with all three models on both tables and execute the candidates with `bm-gen-logs` (top-10; cifar-sparse top-5). Verify: the summary lists every model's error and best measured period for every cell.
- [ ] 4.3 Apply the spec's gate. Make the best range variant the default only if, on both phones and for all three apps, its median absolute error is lower than `per-stage`'s on the same table and its best measured period is not worse beyond noise. Verify: a decision record with every table is in `docs/reports-for-human/`, the default in `02_gen_schedule_merged.py` matches it, and local passes.

## 5. Final gate

- [ ] 5.1 Final gate at the last commit: local and phone gate green, and the summary from 4.2 regenerated from the committed code. Verify: the results are listed in the PR.
