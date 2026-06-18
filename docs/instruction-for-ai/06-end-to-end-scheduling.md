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
# ssh device (jetson=duck-naughty, minipc=rocky-ryzen) — fish login shell ⇒ bash -s
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

## Gotchas that cost time here (read before re-running)

- **scp multiple sources into one destination makes a *directory* on the target.**
  `scp a b c host:/tmp/x.json` creates `/tmp/x.json/` containing a,b,c → the executor
  fails with `load_schedule_json … "Is a directory"`. scp one file per dest.
- **Feed z3 the table matching the deployment.** BTPM (interference) is correct when
  apps actually contend. For a single app running ALONE, the BTPM table inflates the
  shared-memory iGPU's GPU time, so z3 over-splits onto the CPU and a tiny GPU-bound
  app (tree) can run *slower* than pure GPU. Use the **isolated** table for
  single-app-alone scheduling.
- **Running every candidate schedule can be impractical.** On phones, sparse
  candidates that land heavy stages on little/medium cores take minutes each (100
  tasks × multi-second stages). Run only the best-predicted schedule (filter the JSON
  to the min-`max_time` entry) when full enumeration is too slow.
- **Warmup must use a present CPU tier.** The executor's warmup uses
  `app.hpp::first_present_cpu_type()` (was hardcoded Little → threw on the Big-only
  MiniPC).
- **Jetson VK teardown segfault (bugs §1/§9):** records are valid (flushed before the
  crash); CUDA outputs are partially wrong (managed-mem) but timings hold. `03`
  tolerates the non-zero exit.
- **fish login shells** (jetson, rocky) → `ssh host bash -s`; **adb eats stdin** →
  suffix `</dev/null`; **adb adds CR** → `tr -d '\r'`. Samsung's adb runs on rocky.
