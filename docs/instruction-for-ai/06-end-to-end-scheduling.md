# End-to-end: profile → schedule → run → compare (the steps)

> The reproducible procedure for taking one `(app × hardware × backend)` cell from
> raw measurement to a scheduled run and a speedup-vs-baseline number. This is the
> **how**, not the results. Tools live in [`optimizer/`](../../optimizer/);
> device access + gotchas in [`01-hardware.md`](01-hardware.md); builds in
> [`02-building.md`](02-building.md); the profiler design in
> [`05-profiling.md`](05-profiling.md).

The pipeline is the paper's "three tools talking through files":

```
bm_prof ─► JSONL store ─► 02 (z3) ─► schedule JSON ─► 03 (run) ─► logs ─► 04 (parse)
 (C++)     data/profiling           data/schedules_btpm        data/sched_logs
```

02 reads the JSONL store directly (schema-validated, count-weighted across runs); there
is no longer a wide-CSV export step.

Canonical store layout (one record per line, schema
[`schemas/profiling-table.schema.json`](../../schemas/profiling-table.schema.json)):
`data/profiling/<device>/<app>/<backend>/<scenario>/run-NNN.jsonl`. GPU stage-range records
(`bm-ranges-<app>-vk`, schema
[`schemas/profiling-range.schema.json`](../../schemas/profiling-range.schema.json)) sit beside
them as `range-NNN.jsonl`; only the range cost models of Step 2 read them.

---

## Step 0 — build the binaries per target

Three presets, three architectures. Build only the targets a step needs.

```bash
cmake --preset vulkan && cmake --build --preset vulkan --target bm-prof-<app>-vk bm-gen-logs-<app>-vk bm-baseline-<app>-vk   # x86 iGPU (minipc)
# Jetson (arm64, CUDA+Vulkan): cross container — see 02-building.md
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/workspace/build -v "$PWD:/workspace" -w /workspace bt-cross:7.2-glslc \
  bash -lc 'cmake --build --preset jetson --target bm-prof-<app>-<cu|vk> bm-gen-logs-<app>-<cu|vk> bm-baseline-<app>-<cu|vk>'
# Android (arm64, Vulkan): NDK
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/29.0.14206865
cmake --preset android && cmake --build --preset android --target bm-prof-<app>-vk bm-gen-logs-<app>-vk bm-baseline-<app>-vk
```

`bm-prof-*-vk` exist only for the x86 vulkan, jetson, and android presets — build the
one matching the device. (A common slip: forgetting `bm-gen-logs-tree-vk` for jetson.)

## Step 1 — collect the profiling table (isolated + interference)

Deploy the binary, run with `--device <id>`, capture **stdout only** (it is pure
JSONL; logs go to stderr) to the store path. Knobs: `BT_PROF_SCENARIO=isolated|interference`,
`BT_PROF_GPU_WALLCLOCK=1` (host wall-clock for the GPU PU, captures dispatch tax),
`BT_PROF_RUN=N`, plus the calibrated-sampling knobs (see 05-profiling.md).

```bash
# ssh device (jetsons=doremy@duck-{stable,naughty}, minipc=rocky-ryzen) — rocky's login shell is fish ⇒ bash -s
ssh <host> 'cd /tmp/bt && LD_LIBRARY_PATH=. BT_PROF_SCENARIO=interference BT_PROF_RUN=1 \
  ./bm-prof-<app>-<be> --device <dev> 2>/dev/null' > data/profiling/<dev>/<app>/<bedir>/interference/run-001.jsonl
# android (adb on rocky for BOTH phones; adb -s <serial>) — suffix every adb shell with </dev/null; strip CR with tr -d '\r'
```

**Step 01 does the deploy, the runs and the store layout** (it pushes only the binaries: the
device directory must already hold `libc++_shared.so` on a phone, and the weights in
`<dest>/weights`, `scripts/deploy-weights.sh`). `--tools ranges` collects the GPU stage-range
records (every range as one GPU chunk, back-to-back and after an idle gap; see
[`05-profiling.md`](05-profiling.md) "GPU stage ranges as whole chunks") into `range-NNN.jsonl`
beside the per-stage runs; `--tools prof` (the default) the per-stage `run-NNN.jsonl`. Each
invocation replaces only the kind it collects. `BT_PROF_*` knobs set for 01 reach the device.
The tables of the gpu-chunk-cost-model change (2026-10-16) were collected per phone and app with:

```bash
uv run optimizer/orchestrate/01_collect_profiling.py --device <serial> --app <app> --backend vk \
  --adb-serial <serial> --build-dir build/android --dest /data/local/tmp/<dir> \
  --tools prof --runs 2 --scenarios isolated,interference --cooldown-s 20
BT_PROF_MIN_ITERS=3 BT_PROF_ABANDON_S=2.5 uv run optimizer/orchestrate/01_collect_profiling.py \
  --device <serial> --app <app> --backend vk --adb-serial <serial> --build-dir build/android \
  --dest /data/local/tmp/<dir> --tools ranges --runs 2 --scenarios isolated,interference --cooldown-s 20
```

(`BT_PROF_ABANDON_S=2.5` lets a range up to 2.5 s long take the `BT_PROF_MIN_ITERS` samples
instead of one; at the default 0.25 s most cifar-sparse ranges on the Pixel would be sampled once.)

**Parallelism rule:** Jetson is independent → run it concurrently with the rocky
side, but **serialize everything on rocky-ryzen — MiniPC (Vulkan) ↔ Pixel ↔ Samsung**.
All three now hang off rocky (the iGPU runs + both phones' adb); concurrent runs
contend (CPU + USB) and produce empty/erratic output.

Validate + view:

```bash
uv run optimizer/smt/profiling_loader.py --device <dev> --app <app> --backend <bedir>   # schema-validate + aggregate (+ ranges, if any)
uv run optimizer/analysis/render_isolated_table.py --scenario interference --metric p50       # per-app table
uv run optimizer/analysis/coverage.py --scenario interference                                 # what's collected vs MISSING
```

## Step 2 — generate schedules (z3)

```bash
uv run optimizer/orchestrate/02_gen_schedule_merged.py --profiling_root data/profiling \
  --device <dev> --app <app> --backend <vk|cu> --table_type btpm --minimize_mode tmax \
  --num_solutions 10 --output_folder data/schedules_btpm
# table_type isolated -> isolated/ scenario, btpm -> interference/ scenario
# -> data/schedules_btpm/<dev>/<app>/<be>/schedules_btpm_tmax.json
#    (array of {uid, chunks:[{core_type,start_stage,end_stage,hardware?}], metrics})
```

**Framework-overhead term (2026-07-02).** If `data/profiling/<dev>/overhead.json`
exists, `02` prices every chunk as `Σ stage times + per_chunk_ms(PU) +
n_stages·per_stage_ms(PU)` — the fitted per-chunk SPSC/dispatch + per-stage GPU
submit/fence tax that the kernel-sum model misses (it made z3 pick pipelines that
LOSE on tiny apps; fixing it took samsung cifar-dense from 0.75x to 1.50x).
Fit/refresh the constants **after any measured sweep** with

```bash
uv run optimizer/analysis/fit_overhead.py        # reads data/sched_logs -> writes overhead.json per device
```

(model-selected per PU class — classes whose residuals a constant can't explain fall
back to zero; see the script docstring).

**The phones ship no `overhead.json` (cpu-chunk-costs, 2026-10-18).** Fitted on the 2026-10-17
executions, the Pixel 7a's CPU class came out zero and the Samsung's at 0.553 ms per chunk (no GPU
overhead on either). Neither passed the held-out check, so `02` prices both phones' chunks as
plain stage sums. The fits are kept in `data/archive/2026-10-18-cpu-chunk-costs-rejected-overhead/`, and
the measurements are in
[`../reports-for-human/2026-10-18-cpu-chunk-costs-gate.md`](../reports-for-human/2026-10-18-cpu-chunk-costs-gate.md).
A refit has to pass the same check before it goes into the store:

1. Fit on one set of executions (`fit_overhead.py --dry-run --sched-logs <set A>`).
2. With the file in place, solve and execute a second, new set.
3. Compute the second set's median CPU-chunk `|measured / predicted - 1|` with and without the
   constants. Ship only if "with" is lower.

`02` also runs a **union sweep**: it solves
under both cost models, re-prices the plain-model extras under the overhead model,
and emits the union sorted by predicted makespan — the measured top-K sweep in step 3
then picks the true winner, hedging either model's blind spots. `--no-overhead`
opts out entirely.

**GPU in several chunks (opt-in, default off).** `02 … --allow-gpu-multi-chunk` lets z3
place the GPU in more than one chunk of a schedule. The Vulkan runtime runs each with its own
`Sequence`; CUDA schedules with two CUDA chunks are still rejected (`first_concurrent_gpu_chunk`)
and `bm-gen-logs` skips them. The chunks share one device, so the solver bounds the period by
the **sum** of the GPU's chunks (each with its per-chunk overhead), and the emitted
`metrics.max_time` is the largest per-PU sum. Without the flag every PU gets one contiguous chunk,
as before.

**GPU cost model (`--cost-model`, default `per-stage`).** How `02` prices a GPU chunk
(gpu-chunk-cost-model; `optimizer/smt/constraints.py`):

| model | a GPU chunk [a, b] costs | needs |
|---|---|---|
| `per-stage` (default) | the sum of its stages + the fitted GPU overhead, as above | `run-*.jsonl` |
| `range-hot` | t0: the measured back-to-back time of [a, b] as one chunk; no stage sum, no GPU overhead | `range-*.jsonl` of the scenario |
| `range-idle` | t0 raised toward t1 (the time after an idle gap of t0) by the chunk's idle time per period, `min(t1, (t0 + k·T)/(1 + k))`, `k = (t1 - t0)/t0` | same |

- CPU chunks are priced per stage, with their overhead, under every model.
- Under the range models only the actual chunks (maximal GPU segments) are constrained: a
  measured range time need not be monotone in the range. A range without a record, and every
  chunk of a store without range files, falls back to its per-stage price (02 warns).
- With `--allow-gpu-multi-chunk` the GPU's chunks are bounded by the sum of their t0.
- `range-idle` cannot raise the predicted period above `range-hot`'s: the price stays below the
  period whenever t0 does. It changes the GPU chunk's reported time, and so the gap objective.
- A range model writes `schedules_<table>_<mode>_<model>.json` beside the per-stage file; every
  schedule records `metrics.cost_model` and `metrics.table`.
- z3 time on the phones' tree tables (2026-10-16): `range-hot` 0.9-1.1x and `range-idle`
  1.1-1.7x the `per-stage` solve.

To compare models on a cell, solve each one on the same table, execute their candidates
(Step 3; the accuracy join matches runs to schedules by chunk assignment, so an assignment two
models propose runs once), and read the "Prediction accuracy" section of `speedup_summary.py`:
the median `|measured / predicted - 1|` and the best measured period per (device, app, backend,
cost model, table). `validate_btpm.py` prints the same join per cost model.

```bash
for m in per-stage range-hot range-idle; do
  uv run optimizer/orchestrate/02_gen_schedule_merged.py --profiling_root data/profiling \
    --device <dev> --app <app> --backend vk --table_type btpm --minimize_mode tmax \
    --num_solutions 10 --output_folder data/schedules_btpm --cost-model $m
done
uv run optimizer/analysis/speedup_summary.py --sched-logs data/sched_logs --schedules-root data
```

## Step 3 — run the schedule(s) on the device

```bash
uv run optimizer/orchestrate/03_run_schedule.py --device <dev> --app <app> --backend <vk|cu> \
  --ssh-host <host> --build-dir <build/jetson|build/vulkan> \           # or --adb-serial <s> --adb-host rocky-ryzen (phones)
  --table-type btpm --minimize-mode tmax --log-folder data/sched_logs/<dev>_<app>_<be> \
  --repeat 1 --n-schedules-to-run 0                                     # 0 = all candidates
```

Engine knobs reach the device runs: the transport forwards every `BT_VK_*` variable set for
the orchestrator (`00`/`01`/`03`) to the remote binary. For example `BT_VK_INFLIGHT=2 uv run
optimizer/orchestrate/00_run_fleet.py …` runs the Vulkan GPU chunks with two items in flight.
The knobs added by vk-engine-runtime, all unset by default (details in
[`03-unit-testing.md`](03-unit-testing.md) "Engine knobs"):

| knob | default | effect |
|---|---|---|
| `BT_VK_BARRIER_ALL=1` | unset: barriers only where a stage touches a buffer written (or read) since the last one | a barrier before every dispatch |
| `BT_VK_WEIGHTS_PLACEMENT=default\|device_local\|uncached` | unset: `uncached` on the Pixel 7a, host-cached elsewhere | memory type of the weights buffer class |
| `BT_VK_GPU_LOCAL_PLACEMENT=default\|device_local\|uncached` | unset: `uncached` on the Pixel 7a, host-cached elsewhere | memory type of the GPU-local buffer class |
| `BT_VK_BOUNDARY_PLACEMENT=default\|device_local\|uncached` | unset: host-cached | memory type of the boundary class (experiment; see `03-unit-testing.md`) |
| `BT_VK_QUEUE=universal\|compute` | `universal` | queue family; `compute` falls back with a warning where none exists |
| `BT_VK_QUEUE_PRIORITY=low\|medium\|high` | the driver's default priority | global queue priority; a refusal warns and keeps the default |
| `BT_VK_INFLIGHT=<n>` | `1` | items in flight per Vulkan GPU chunk |
| `BT_VK_PHASES=1` | off | times every submission's phases (record, submit, wait, …) and verifies the timestamp calibration first; `bm-phases-{tree,cifar-dense,cifar-sparse}-vk` turns it on and prints them per chunk, with the bytes flushed and invalidated |
| `BT_GPU_THREAD_CORES=none\|auto\|<id>[,<id>...]` | unset: the OS places the thread | where the thread that drives a Vulkan GPU chunk runs; `auto` = a free medium/big tier (opt-in, measured device-dependent) |
| `--allow-gpu-multi-chunk` (`02`, solver `allow_gpu_multi_chunk`) | off: one chunk per PU | the GPU may take several chunks (above) |

At `BT_VK_INFLIGHT` > 1 a chunk's `Start`/`End` ticks bracket submit → complete, so they include
the time an item waits behind the others in flight. Its chunk durations, the "max chunk"
makespan of `speedup_summary.py` and the residuals `fit_overhead.py` fits are then latency, not
throughput: compare depths by the per-task makespan of step 4 (vk-engine-runtime 4.5).

`03` deploys the prebuilt binary + schedule JSON, runs the executor with
`--schedule-file` (local file; **no HTTP/curl** — that path is retired), and writes
`schedule_run_<i>.log`. It runs the executor with `check=False` so a Jetson-VK
teardown segfault (bugs §9) doesn't discard the already-flushed records.

## Step 4 — parse / measure

```bash
uv run optimizer/orchestrate/04_parse_schedules.py data/sched_logs/<dev>_<app>_<be>   # per-schedule, per-chunk timing
```
Per-task makespan of a schedule = `(max End − min Start) / n_tasks` over its
`Task=… Start=… End=…` ticks (`Frequency=` gives the tick rate). The best schedule
is the min makespan across candidates.

## Step 5 — baseline & speedup

The no-framework baseline (all 9 stages on one PU, no pipeline):

```bash
ssh <host> 'cd /tmp/bt && LD_LIBRARY_PATH=. ./bm-baseline-<app>-<be> --device <dev> --benchmark_min_time=1s 2>/dev/null'
```
Reports `OMP/.../<tier>` and `VK|CUDA` per-task times. Fastest single-PU baseline =
min over present PUs. **Speedup = fastest_baseline / best_schedule_makespan** (both
ms/task; comparing steady-state throughput, which is what the pipeline targets).

---

## What BTPM (interference) actually measures (operative definition)

The paper's formal BTPM definition is not checked into this repo; this is the **operative
definition the code implements** (`profiler/bm_prof_common.hpp`, the `interfere` branch):

> While timing the *target* PU running stage `s`, **every other present PU is saturated
> with the *same* stage `s`** in a busy loop on disjoint app data (one bg thread keeps the
> GPU busy when the target is a CPU tier; one bg thread per other CPU tier; never two GPU
> users). The per-(stage, PU) timing measured under that load is the BTPM/"interference"
> cost; z3 uses it as a **static, per-cell replacement** for the isolated cost — there is
> no co-execution/overlap term (`optimizer/smt/constraints.py`).

Known limits (interference audit, 2026-06-20 — see the audit memory):
- **Same-stage proxy ≠ real pipeline mix.** A real pipeline runs *different* stages on
  different PUs concurrently; same-stage-everywhere is a worst-case memory-bandwidth proxy,
  not the true co-execution mix, and the static cost model can't represent which stages
  actually co-run.
- **DVFS confound** corrupts the GPU column on clock-scaling devices (see the gotcha
  below) — mitigated in the loader by the DVFS floor + `min_runs>=2`, and in the harness
  by warming the GPU identically in both scenarios + a settle barrier + a saturating bg
  load.
- **Validate, don't assume:** the paper's central claim is that the contended (BTPM)
  table predicts the real pipeline makespan better than the isolated table. Check it with
  `optimizer/analysis/validate_btpm.py` (predicted z3 `max_time` vs measured makespan from
  `data/sched_logs/`, btpm vs isolated) before trusting BTPM over isolated on a device.

## Gotchas that cost time here (read before re-running)

- **scp multiple sources into one destination makes a *directory* on the target.**
  `scp a b c host:/tmp/x.json` creates `/tmp/x.json/` containing a,b,c → the executor
  fails with `load_schedule_json … "Is a directory"`. scp one file per dest.
- **Feed z3 the table matching the deployment, and distrust the BTPM GPU column.** BTPM
  (interference) is meant for when apps actually contend; use the **isolated** table for
  a single app running alone. CRUCIAL CORRECTION (interference audit, 2026-06-20): on
  clock-scaling GPUs the BTPM GPU column is corrupted by **DVFS, not contention** — the
  background GPU load keeps the iGPU/GPU *boosted*, while the isolated measurement runs
  gappy at a low clock, so the GPU measures *faster* under load (physically impossible
  for true contention) and z3 sees a too-**cheap** GPU. (This is the opposite sign from
  what an earlier note claimed about "inflating" GPU time.) The loader now clamps any GPU
  cell measuring below its isolated value up to that floor (`data_loader.py` DVFS floor)
  and requires `min_runs>=2`; until the harness controls/records GPU clocks, BTPM GPU
  numbers remain suspect.
- **Running every candidate schedule can be impractical.** On phones, sparse
  candidates that land heavy stages on little/medium cores take minutes each (100
  tasks × multi-second stages). Run only the best-predicted schedule (filter the JSON
  to the min-`max_time` entry) when full enumeration is too slow.
- **Warmup must use a present CPU tier.** The executor's warmup uses
  `device_registry.hpp::first_present_cpu_type()` (was hardcoded Little → threw on the Big-only
  MiniPC).
- **Jetson VK teardown segfault (bugs §1/§9):** records are valid (flushed before the
  crash); CUDA outputs are partially wrong (managed-mem) but timings hold. `03`
  tolerates the non-zero exit.
- **fish login shell** (rocky; the reflashed Jetsons are bash) → `ssh host bash -s`
  for every ssh target; **adb eats stdin** → suffix `</dev/null`; **adb adds CR** →
  `tr -d '\r'`. Samsung's adb runs on rocky.
