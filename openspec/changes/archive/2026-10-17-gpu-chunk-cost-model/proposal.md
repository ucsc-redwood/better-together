# Proposal

## Why

The solver predicts a GPU chunk's time as the sum of its stages, each profiled alone, plus a fitted
per-chunk and per-stage constant. On the phones that sum misses effects that only exist when stages
run together as one chunk, at a pipeline's cadence. So the solver chooses schedules from numbers that
do not describe what the pipeline will do.

Measured this campaign (2026-10-15/16, Pixel 7a and Samsung; report
`docs/reports-for-human/2026-10-16-vk-data-flow-profile.md`):

- **A whole chunk costs about twice its stages' sum.** Tree stages 1-7 run as one GPU chunk on the
  Pixel take 42.6 ms, against 20.5 ms for the same stages timed alone. Three things are missing from
  the sum:
  - two submissions: the host reads `n_unique` between stages 3 and 4;
  - host cache maintenance (11.4 ms);
  - the GPU running at a lower clock after the host round trip. Stages 1-3 take 17.8 ms inside the
    chunk against 9.5 ms alone.
- **Idling slows the next chunk.** On the Radeon 780M, tree stages 1-3 take 1.09 ms back-to-back,
  1.48 / 2.54 / 2.98 ms after 1 / 5 / 20 ms of idle (vk-engine-runtime design D1).
- **The two cost tables disagree on the GPU.** The same GPU stage measures very differently in the
  isolated and the interference (btpm) scenario, because a background load keeps the GPU clock up:
  - Samsung stage 7: 2.72 ms isolated vs 0.36 ms btpm;
  - Samsung stage 4: 1.37 vs 0.44 ms;
  - Pixel stage 2: 6.93 vs 4.93 ms.

  The solver's choice depends on which table it is given.
- **The clock is not recorded on the phones.** `gpu_clock_mhz` is -1 in every phone record, although
  the Pixel exposes devfreq nodes. So a run cannot be checked for throttling or for the operating
  point.
- **Consequence.** In the last fleet summary (2026-07-04) the pipelined tree is slower than one PU
  alone on the Pixel (0.61x), on the 780M (0.73x) and on duck-stable with Vulkan (0.91x). Those runs
  predate this campaign's kernel and engine changes, but no measurement since has shown the model
  became predictive.

## What Changes

- **GPU stage-range records.** The profiler records, for every contiguous stage range of an app, the
  time of that range dispatched as one GPU chunk through the same path the pipeline uses: one
  multi-stage dispatch with its submissions, cache maintenance and any read-back. Each range is
  measured back-to-back and after an idle gap before each submission. The ranges are additive records
  in the profiling store; per-stage records stay.
- **GPU clock in provenance** wherever the device exposes it (Mali/Exynos devfreq, Tegra, amdgpu).
  Records state when it is unknown, as today.
- **The solver uses range costs for GPU chunks** when range records exist for the cell: a GPU chunk's
  cost is its measured range cost, without the per-stage sum and without a fitted GPU overhead. CPU
  chunks keep the per-stage sum and the fitted CPU overhead. A flag selects the old model, for
  comparison.
- **Prediction accuracy is reported.** For every executed schedule the run reports measured period /
  predicted period. The fleet summary gives the median absolute error per (device, app, backend, cost
  model). It extends the existing `validate_btpm.py` check from a manual script to part of the
  summary.
- **Non-goals:**
  - locking clocks on the phones (no root);
  - changing how CPU stages are profiled;
  - changing the z3 objective;
  - new profiling on rocky-ryzen (the owner's instruction: no profiler runs there). Its existing
    tables stay as they are.

## Capabilities

### New Capabilities
- `profiling/cost-table`: what the profiling store records and with what provenance: per-stage
  records as today, GPU stage-range records measured as whole chunks, and the GPU operating frequency
  where the device exposes it.
- `optimizer/cost-model`: how the solver predicts a chunk's time from the cost table (range costs for
  GPU chunks, per-stage sums for CPU chunks), how several GPU chunks of one schedule are priced, and
  how the accuracy of the prediction is measured and reported against executed runs.

### Modified Capabilities
<!-- none in openspec/specs/. runtime/gpu-chunks ("Optimizer models GPU time-sharing") lives in the
     unarchived vk-engine-runtime change; design.md states how range costs keep that rule. -->

## Impact

- **Code:**
  - `profiler/` — a range-profiling harness, built on `bm_phases_vulkan.hpp`'s single-chunk loop;
  - `profiler/bm_prof_common.hpp` — provenance clock readers;
  - `schemas/profiling-table.schema.json` — optional range fields, backward compatible;
  - `optimizer/smt/{profiling_loader,data_loader,constraints,solver,solution_analyzer}.py`;
  - `optimizer/orchestrate/{01_collect_profiling,02_gen_schedule_merged}.py`;
  - `optimizer/analysis/{validate_btpm,speedup_summary}.py`;
  - their tests under `optimizer/tests/`.
- **Data:** range records add one file per (device, app, backend, scenario) next to the per-stage runs.
  Existing stores remain valid.
- **Gates:**
  - local `uv run pytest -q optimizer` and `ctest -L omp`;
  - profiling and schedule runs on the Pixel 7a and the Samsung, for tree, cifar-dense and
    cifar-sparse;
  - Jetson optional.
  - Nothing is profiled on rocky.
