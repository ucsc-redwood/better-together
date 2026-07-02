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
`data/profiling/<device>/<app>/<backend>/<scenario>/run-NNN.jsonl`.

---

## Step 0 — build the binaries per target

Three presets, three architectures. Build only the targets a step needs.

```bash
cmake --preset vulkan && cmake --build --preset vulkan --target bm-prof-<app>-vk bm-gen-logs-<app>-vk bm-baseline-<app>-vk   # x86 iGPU (minipc)
# Jetson (arm64, CUDA+Vulkan): cross container — see 02-building.md
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/workspace/build -v "$PWD:/workspace" -w /workspace bt-cross:6.1 \
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

**Parallelism rule:** Jetson is independent → run it concurrently with the rocky
side, but **serialize everything on rocky-ryzen — MiniPC (Vulkan) ↔ Pixel ↔ Samsung**.
All three now hang off rocky (the iGPU runs + both phones' adb); concurrent runs
contend (CPU + USB) and produce empty/erratic output.

Validate + view:

```bash
uv run optimizer/smt/profiling_loader.py --device <dev> --app <app> --backend <bedir>   # schema-validate + aggregate
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

## Step 3 — run the schedule(s) on the device

```bash
uv run optimizer/orchestrate/03_run_schedule.py --device <dev> --app <app> --backend <vk|cu> \
  --ssh-host <host> --build-dir <build/jetson|build/vulkan> \           # or --adb-serial <s> --adb-host rocky-ryzen (phones)
  --table-type btpm --minimize-mode tmax --log-folder data/sched_logs/<dev>_<app>_<be> \
  --repeat 1 --n-schedules-to-run 0                                     # 0 = all candidates
```

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
