# Design

## Context

See proposal.md (Why). The relevant current state:

- **Profiling store.** `run-NNN.jsonl` files under `data/profiling/<device>/<app>/<backend>/<scenario>/`,
  one record per (stage, PU), schema `schemas/profiling-table.schema.json` (`bt.profiling/v0`).
  Loaders glob `run-*.jsonl` (`optimizer/orchestrate/case.py`).
- **GPU clock reading.** `bm_prof_common.hpp::read_gpu_clock_mhz()` knows amdgpu `pp_dpm_sclk`, kgsl
  and generic devfreq `gpu` nodes, but neither Mali nor Exynos. It reads once per run, before
  measuring. On the phones it records -1. Checked 2026-10-16, both phones expose a readable node to the
  `shell` user:
  - Pixel 7a: `/sys/class/misc/mali0/device/cur_freq` (kHz; 202000 idle);
  - Samsung: `/sys/kernel/gpu/gpu_clock` and devfreq `22200000.sgpu/cur_freq`, both 0 while the GPU
    is idle and power-gated.
- **Solver.** `optimizer/smt/constraints.py::add_chunk_time_constraint` encodes, for every PU c and
  every contiguous segment [i, j]:
  - `assigned(c, i..j) -> sum(stage cells) + per_chunk + n*per_stage <= T_max`;
  - `maximal(c, i..j) -> ... >= T_min`;
  - for a PU shared by several chunks (`allow_gpu_multi_chunk`), the sum of its stage cells plus
    per-chunk overheads `<= T_max`.

  The `<= T_max` side holds for every sub-segment, which is sound only because a per-stage sum is
  monotone in the segment.
- **Overhead fit and validation.** `optimizer/analysis/fit_overhead.py` fits the per-chunk and
  per-stage constants per PU class from executed runs. `validate_btpm.py` already computes measured /
  predicted per schedule (makespan = max chunk average from `data/sched_logs`), but it is a manual
  script, outside the summary.
- **One-chunk timing harness.** `profiler/bm_phases_vulkan.hpp` runs one stage range as a chunk
  through the dispatcher's real path (`dispatch_multi_stage`), with `--gap-ms` idle before each chunk.
  Its whole-chunk time is `dispatch_wall`, which includes the tree's two submissions and its read-back.

## Goals / Non-Goals

**Goals:**
- GPU chunk predictions that include what only a whole chunk shows: several submissions, read-backs,
  cache maintenance, the clock after idling.
- An accuracy number for every cost model on every executed run, so model changes are judged by
  prediction error, not by inspection.
- No loss for stores and tools that know nothing of ranges.

**Non-Goals:**
- CPU range profiling. The CPU tiers' per-stage sums were not shown to be the problem; that can be
  revisited with the accuracy numbers this change produces.
- Clock locking, power models, or thermals.
- The minipc/rocky cell: it is not re-profiled.

## Decisions

### D1. Range records live in their own files, with their own schema
Range records are written to `range-NNN.jsonl` next to `run-NNN.jsonl`, with schema
`bt.profiling-range/v0`. Their fields are the per-stage fields, with `start_stage`/`end_stage` in place
of `stage`, plus `gap_ms`, `submissions` and the clock samples.

Existing globs (`run-*.jsonl`) and readers are untouched, so old tools read new stores exactly as
before (spec "Range records do not break existing stores").

*Alternative:* optional `end_stage` / `gap_ms` fields inside `run-*.jsonl`. Rejected because an old
reader would take a range record for stage `start_stage`, silently.

### D2. Harness: sweep every range through the pipeline's dispatch path
A `bm-ranges-<app>-vk` tool is built from `bm_phases_vulkan.hpp`'s loop (one chunk shape, a pool of
items, `dispatch_multi_stage`, calibrated sample count). It sweeps all n(n+1)/2 ranges:
- ranges are interleaved, not run in order, so drift spreads across them;
- it uses the same time-budget sampling as `bm-prof` (`BT_PROF_BUDGET_S` and friends);
- under `BT_PROF_SCENARIO=interference` it runs `bm-prof`'s background load on the CPU tiers.

A range's time is the median of whole-chunk wall times, `dispatch_wall`, so a two-submission tree
range is priced with its read-back.

*As built (task 2.2):*
- each range runs on its own `Sequence` (`make_seq`, as a pipeline chunk) through
  `submit_multi_stage` + `complete_multi_stage`, the path of `inflight_worker`; a sample is the
  host wall time of that pair (submit to completion, the gap not included); `submissions` is the
  difference of consecutive tickets;
- each range gets its own pool of `--items` (4) items placed for a GPU that runs exactly [a, b]
  (`make_placed_dataset`), and they are primed for stages 1..a-1 **on the GPU**: priming with
  OMP on a big core would take minutes per cifar-sparse range, and in the measured loop no
  host write touches those buffers either way;
- before sampling, every item's chunk runs once (command-buffer recording) and the GPU is warmed
  back-to-back for `BT_PROF_GPU_WARM_MS` (150), as in `bm-prof`;
- interference: one background thread per present CPU tier runs for the whole sweep, on its own
  fully primed item, cycling one stage per dispatch through the stages of the range currently
  measured; a range switch therefore takes effect after one stage of the previous range.

### D3. Two cadences, and how the solver combines them
Each range is measured back-to-back (`gap 0`, giving t0) and after an idle gap equal to its own
back-to-back time (`gap = t0`, giving t1: a 50 % GPU duty cycle). The gap is chosen per range
because the right idle time depends on the schedule's period, which is unknown at profiling time; a
gap proportional to the range keeps both cadences meaningful from 1-ms tree ranges to 1-s cifar-sparse
ranges.

The solver prices a GPU chunk in one of two ways, selected by a flag. Which one becomes the range
model's default is decided by the accuracy gate (task 4.3):

- **hot:** price = t0. Simple and exact when the GPU chunk is the bottleneck, since it then never
  idles.
- **idle-aware:** the chunk idles `g = T_max - t` per period. Its time is interpolated as
  `t = t0 + k * min(g, t0)` with `k = (t1 - t0) / t0`, clamped to [t0, t1]. With `g = T_max - t` this
  is linear in (t, T_max) piecewise, so it stays in z3's linear real arithmetic:
  `t >= t0`, `t >= (t0 + k*T_max) / (1 + k)` while below t1, `t <= T_max`.

*As built (task 3.3):* the price is never asserted as such. `t <= T_max` holds exactly when
`t0 <= T_max`, so the encoding is `maximal -> t0 <= T_max` plus `min(lin, t1) >= T_min` as two
linear constraints; the price is evaluated in the model for the schedule's chunk time. One
consequence: the idle-aware price never exceeds the period when t0 does not, so with
`--minimize_mode tmax` `range-idle` reaches the same optimum, and predicts the same period for
an assignment, as `range-hot`. It only changes the GPU chunk's reported time (and so `min_time`
and the gap objective).

*Alternatives:*
- Fixed gaps (1, 5, 20 ms, as in vk-engine-runtime D1). Rejected because they mean nothing for a 1-s
  chunk.
- Fitting an exponential clock-decay model. More parameters than two points can support.

### D4. Range costs constrain only maximal segments
A measured range time need not be monotone in the range (noise; the tree's 3|4 split makes [3, 4]
cost more than [3, 3] + [4, 4] would suggest). The range model therefore applies its cost only to
**maximal** GPU segments, the actual chunks: `maximal(gpu, i..j) -> t(i..j) <= T_max`. It does not
apply it to every assigned sub-segment.

The per-stage model keeps today's encoding unchanged (spec "Range data absent": identical output).

For several GPU chunks, the time-sharing bound becomes `sum over maximal GPU segments of t(seg) <=
T_max`, using t0. Two chunks on one GPU alternate, so neither idles waiting for the other in the way
D3 models.

### D5. Clock readers per record, sampled while the kernel runs
Add readers for:
- Mali `/sys/class/misc/mali0/device/cur_freq` (kHz to MHz);
- Exynos `/sys/kernel/gpu/gpu_clock` and devfreq `*sgpu*/cur_freq`.

Sample the reader during the measured loop of each GPU cell and each range, every Nth iteration, and
record the median of the non-zero samples. If all samples are zero or no reader exists, record the
unknown marker (-1, as today). Sampling during the loop avoids the idle and power-gated 0 seen on the
Samsung.

### D6. Accuracy in the summary
Schedules record `metrics.cost_model` (`per-stage` | `range-hot` | `range-idle`) and the table they
were solved on. The validation logic of `validate_btpm.py` becomes a library function used by
`speedup_summary.py`. The fleet summary gains a table per (device, app, backend, cost model, table)
with the number of executed schedules, the median `|measured/predicted - 1|`, and the best measured
period. The script stays as a thin CLI.

### D7. Acceptance experiment
Per phone and app:
1. Collect per-stage and range tables, isolated and interference, 2+ runs each (interference needs 2
   by the loader's gate).
2. Solve top-10 candidates with each model on each table.
3. Execute them with `bm-gen-logs` (cifar-sparse: top-5, because of its run time on the Pixel).
4. Compare the summary's errors and best measured periods.

This is the spec's gate. If neither range variant beats the per-stage model, the per-stage model stays
the default, and the numbers are recorded.

## Risks / Trade-offs

- **[Ranges measured alone still miss pipeline interference]** → The interference scenario applies to
  ranges too, and the accuracy numbers show what remains.
- **[Long sweeps drift with phone temperature]** → Interleave ranges, cool down between runs, record
  the clock per range, and repeat each table at least twice.
- **[cifar-sparse range sweep on the Pixel takes long]** (66 ranges, up to ~1 s per chunk) → The
  calibrated budget caps samples per range. Estimate ~10-15 min per scenario and run, measured in task
  2.3 before committing to more runs.
- **[The idle-aware encoding slows z3]** → The piecewise constraints exist for GPU maximal segments
  only (n(n+1)/2 per app). If solve time grows noticeably, it is reported, and `range-hot` remains
  available.
- **[The existing GPU per-chunk overhead becomes double-counted]** → The range model drops the fitted
  GPU overhead (spec). `fit_overhead.py` keeps fitting it for the per-stage model.

## Sweep durations (task 2.3)

Wall time of one `01_collect_profiling.py` run (deploy excluded, 20 s cooldowns not counted), on
2026-10-16/17, with the commands of `06-end-to-end-scheduling.md` Step 1 (`bm-ranges` with
`BT_PROF_MIN_ITERS=3 BT_PROF_ABANDON_S=2.5`). The two phones were collected at the same time from
the build box, each phone's runs one after another. Seconds per run (run 1 / run 2):

| tool | scenario | Pixel tree | Pixel dense | Pixel sparse | Samsung tree | Samsung dense | Samsung sparse |
|---|---|---|---|---|---|---|---|
| bm-prof | isolated | 15 / 14 | 30 / 29 | 125 / 106 | 22 / 21 | 35 / 30 | 99 / 89 |
| bm-prof | interference | 21 / 21 | 81 / 82 | 631 / 635 | 17 / 19 | 62 / 64 | 459 / 457 |
| bm-ranges | isolated | 39 / 40 | 137 / 131 | 476 / 474 | 41 / 41 | 114 / 114 | 303 / 309 |
| bm-ranges | interference | 42 / 44 | 186 / 176 | 846 / 861 | 42 / 41 | 234 / 239 | 852 / 845 |

A whole cell (both tools, both scenarios, 2 runs each, cooldowns included): Pixel tree 6.6 min,
cifar-dense 16.9 min, cifar-sparse 72 min; Samsung tree 6.8 min, cifar-dense 17.5 min, cifar-sparse
59.6 min. The cifar-sparse range sweep is within the 10-15 min per scenario and run estimated above
isolated (5-8 min), and above it under interference (14 min; not analysed, presumably the CPU
background load slowing the host side of each chunk).

## Migration Plan

Additive. New files and a new schema; the old model stays available and stays the default until the
gate passes. Rollback means selecting `per-stage`. No existing store or schedule file changes meaning.

## Open Questions

- ~~The exact sampling stride for the clock reader (D5), if sysfs reads turn out to cost measurable time
  on short ranges. It is measured in task 1.2; it does not change the approach.~~ **Resolved in task 1.2:**
  - one `read_gpu_clock_mhz()` costs **30.3 us** on the Pixel 7a (`test-gpu-clock-omp`
    `GpuClockDevice.ReadCost`, 2000 reads, 2026-10-16; it includes the failed probes of the sources
    listed before Mali). The Samsung skips that test: its GPU reads 0 while idle;
  - the reader therefore runs on its own thread (`GpuClockSampler`), every **5 ms**
    (`BT_PROF_CLOCK_PERIOD_MS`), for the duration of a GPU cell's or range's measured loop, instead of
    every Nth iteration of that loop. The measuring thread blocks on the GPU fence, so a read taken
    there lands between kernels, when the Samsung's GPU may already read 0. At 5 ms the reads take
    about 0.6 % of one core, outside every timed region.
