# Session progress — 2026-06-17

Consolidated handoff for the big working session on branch `dev` (33 commits,
`ce64890..fdf78b1`, all pushed to `origin/dev`). Pairs with the per-area docs:
[`code-improvement-plan.md`](code-improvement-plan.md) (Phases 1–4),
[`runtime-test-suite-plan.md`](runtime-test-suite-plan.md) (the runtime-test roadmap),
[`bugs-found.md`](bugs-found.md) (§1/§9 statuses).

## 1. Code-improvement plan — Phases 1–4 (DONE except one perf sub-item)

| Phase | What | Status |
|---|---|---|
| **1** (7 items) | baselines derived from `isolated.csv`; `05_timeline` regex; device-specs + git-sha generated at build; `log_parser` div-by-zero guard; vulkan/android presets build benchmarks; tier-5 dead code | ✅ DONE — omp 5/5 + new unit tests + vk/cu build |
| **2** (6 items) | schedule-vs-device skip+warn; worker try/catch (no terminate); Logger clamp; `g_vma_allocator` null-on-destroy; `do_allocate` alignment + cudaError text; per-launch `CheckCudaLaunch` | ✅ DONE — omp 5/5 + rocky vk oracle + cu cross |
| **3** (2 families) | `bm_fully` → `bm_fully_common.hpp` (−1852/+354); `bm_gen_log` → `bm_gen_log_common.hpp` (−1461/+274, warmup→validated, cross-backend skip); removed 5 dead pre-migration vk executors | ✅ DONE — E2E numeric match on rocky+Jetson |
| **4 §1** CUDA managed-mem | Already fixed pre-plan (commit `4161664`, zero-copy pinned). Plan's "RED" baseline was stale. | ✅ Re-verified GREEN on Jetson `ctest -L cuda` |
| **4 §9** kiss-vk teardown | `~BaseEngine` now waitIdle + destroy allocator→device→instance | ✅ DONE — Jetson VK binaries exit 0 (were 139) |
| **4 §3.1** tree/cuda cub | temp storage `cudaFree`→`g_allocator.DeviceFree` (cache works + fixes latent double-free) | ✅ DONE — **3.0–3.5× on cuda cub stages** (2/3/6), oracle green |
| **4 §3.2/§3.3** kiss-vk perf | flush-subset + cmd record-once | ⛔ NOT DONE — payoff+risk only on Mali (rocky flush is a no-op); can reintroduce §7 47× regression. Needs Pixel+Samsung gate. Risk-assessed in the plan. |

**Data-refresh owed:** §3.1 made the `jetson/tree/cu` profiling tables (isolated+btpm)
+ derived schedules stale (cuda stages now ~3× faster) → re-collect (ideally all cells
together for a consistent committed dataset).

## 2. Framework runtime test suite (NEW)

Until this session there was **zero** automated coverage of the framework runtime
(concurrent scheduling / pipeline / sync / unified-memory). Only per-stage kernel
differential tests existed. Now:

- **`builtin-apps/pipeline/pipeline_test_runner.hpp`** — backend-agnostic
  `run_pipeline()`: spawns the REAL `worker()` (reused from `pipe/pipeline_common.hpp`)
  one std::thread per chunk over an `SPSCQueue` ring, with a GPU branch (`gpu_em`).
  Hardened: **completion-edge assertion** (last chunk records every finished item;
  assert count == n_items + all pool objects appear → catches a later-cycle drop the
  golden check would miss) and a **300s deadlock watchdog** (abort with diagnostic
  instead of hanging). Item distinguishability is free — `tree_appdata.cpp` uses a
  static advancing `mt19937(114514)`, so each pool item has a distinct golden.
- **Tests** (`test-pipeline-e2e-{app}-{omp,vk,cu}`): drive a hybrid OMP|GPU schedule
  (CPU chunk ∥ GPU chunk over shared UMA) through the ring; per-item check = the app's
  differential oracle (`CheckStage7` / `CheckFinalPipeline`) + a §1/§7 all-zero detector.

### Coverage table (all verified on real HW)
| dimension | OMP | Vulkan | CUDA |
|---|---|---|---|
| per-stage kernel correctness | ✅ | ✅ | ✅ |
| end-to-end sequential | ✅ | ✅ | ✅ |
| **concurrent pipeline runtime** | ✅ tree | ✅ tree+dense+sparse | ✅ tree+dense+sparse |
| **CPU affinity / tier binding** | ✅ (pc big/little) | — | — |
| **CPU+GPU concurrent execution** | — | ✅ (3 apps) | ✅ (3 apps) |
| **unified-memory coherency** | — | ✅ (3 apps, **Mali = real non-coherent test**) | ✅ (3 apps, §1 pinned) |

Verified: OMP on pc; VK on rocky-ryzen (RADV) + Pixel 7a + Samsung (Mali, non-coherent
— the only place §7 flush/invalidate actually runs); CUDA on Jetson (§1 pinned UMA).

### Gate for future fence/memory refactors
- Vulkan coherency: `cmake --build build/android --target test-pipeline-e2e-*-vk &&
  bash scripts/run-mali-oracle.sh test-pipeline-e2e-tree-vk` — **MUST run on Mali**
  (rocky/RADV is coherent → flush is a no-op → blind to §7 regressions).
  `scripts/run-mali-oracle.sh` (new) stages to rocky + runs both phones via `adb -s`.
- CUDA §1: cross-build `test-pipeline-e2e-*-cu` in `bt-cross:6.1`, run on Jetson.

## 3. Findings & open items

- **🐛 GPU re-entry into the tree octree** (`DISABLED_AlternatingBoundary`) — **this
  triage was WRONG; corrected below in §4 / [`bugs-found.md`](bugs-found.md) §10.** The
  real root cause is a concurrent shared-command-buffer race, not a stale octree count;
  now fixed by a guard. Left here for the record.
- **Hardware:** both phones (Pixel `3A021JEHN02756` + Samsung `R5CY21Y3VEV`) moved to
  rocky-ryzen's adb this session (`adb -s <serial>`); docs synced. Serialize all
  rocky-side work (minipc Vulkan ↔ Pixel ↔ Samsung).
- **Not yet done (optional):** §3.2/§3.3 kiss-vk perf (Mali-gated); the §3.1 profiling
  re-collection; cifar-sparse runtime test on Mali; actually fixing the octree-re-entry
  limitation.
- **Uncommitted in the tree:** `docs/reports-for-human/perf-results/` + a README.md
  edit (the RGA static-analysis write-up) — pre-existing, not from this session's work;
  left for the owner to commit.

## 4. Continued session — full-matrix completion + corrected octree triage

Picked up the question "have we run the full unit tests on all HWs?" → swept the
whole matrix and closed the gaps. **Commits `a39eff6`, `e767b71`, `2e86d41` on `dev`.**

**Matrix now complete (apps × backends × HW), every supported cell run on real HW with
logs committed under [`perf-results/test-runs/`](perf-results/test-runs/):**
- per-stage differential: OMP (pc + Jetson + both phones), CUDA (Jetson), Vulkan
  (Jetson — **newly run**, rocky, both phones). Jetson Vulkan per-stage was the one
  blank cell; filled.
- runtime hetero-pipeline (`test-pipeline-e2e-*`): CUDA ×3 apps (Jetson), Vulkan ×3 apps
  (Jetson — **newly cross-built+run**, rocky, both phones incl. **cifar-sparse on Mali**,
  the §3 gap). OMP column completed: new `test-pipeline-e2e-cifar-{dense,sparse}-omp`,
  plus a mobile `big|medium|little` **medium-tier** case + a `sched_getaffinity`
  read-back check on the tree OMP test.

**🎯 `DISABLED_AlternatingBoundary` — root cause found (overturns §3 triage).**
GPU-assisted validation on rocky (validation layers installed this session) shows
`VUID-vkBeginCommandBuffer-commandBuffer-00049`: **not** octree re-entry / a stale
count (VkAppData_Safe's counts are `const`-correct) but a **concurrent
command-buffer race** — all Vulkan chunks share one `VulkanDispatcher`/`Sequence`, and
≥2 Vulkan chunks run as concurrent worker threads recording into the one buffer →
`VK_ERROR_DEVICE_LOST`. Crash tracks the *number of concurrent GPU chunks*, not octree
re-entry. **Fix:** `first_concurrent_gpu_chunk()` guard (`pipeline/schedule.hpp`)
rejects any >1-GPU-chunk schedule (z3 never emits one); the ex-DISABLED test is now
`PipelineE2EVk.RejectsMultiGpuChunkSchedule` + `ScheduleGpuReuse.*` static unit tests.
**No DISABLED tests remain.** Full detail: [`bugs-found.md`](bugs-found.md) §10.

**Also added** (pc robustness): `LoggerRejectsOverflowChunk` (>kMaxChunks → clean
`out_of_range`). `ctest -L omp` now **8/8**.

**Not done (deliberately deferred — needs HW iteration / heavy infra, not matrix gaps):**
GPU-bottleneck concurrent-visibility stress (Jetson+Mali); TSan (blocked on GCC
libgomp not being TSan-instrumented → needs archer or clang+TSan-OMP runtime; `next_uid`
still non-atomic); mobile DVFS timing; watchdog self-test (needs a configurable timeout).

## 5. Pipeline-hygiene + cross-tool-contract pass (continued session)

After the matrix work, a second pass on repo hygiene and the three-tool seams.
Commits on `dev` (main still at `e767b71`, dev ahead by 5): `8d45084`, `3ea1b74`,
`89d615e`, `80a0b98`, `24c5b2d`.

- **data/ out of git** (`8d45084`): 383 files / ~396MB of regenerable experiment
  output (profiling tables, schedules, exec logs, CIFAR dataset) un-tracked
  (`git rm --cached`, kept on disk) + `data/` gitignored. `.gitignore` had been
  inconsistent (ignored the CIFAR tarball/`saved_params`/`Testing` but committed the
  rest). **New model: data is regenerable, not versioned** — no committed snapshot
  (owner's choice); regenerate via BT-Profiler → export → 02 → 03.
- **Legacy purge** (`3ea1b74`): removed the iiswc2025-era cluster — `scripts/misc/`
  (param/print helpers that only read the retired `data/exe_logs_*`), `justfile.old`,
  and on-disk `data/exe_logs_*`+`data/bm_logs` (~52MB) + `.xmake/`. `utility/` kept
  (live CMake tools + intentionally-retained volk/NNAPI).
- **Schedule contract hardened** (`89d615e`, the one cross-tool artifact that lacked a
  schema): added `schemas/schedule.schema.json` (core_type enum = single PU-string
  source; GPU chunk requires `hardware`); dropped the dead `stage_assignments` field;
  chunks now carry explicit **1-based-inclusive `start_stage`/`end_stage`** instead of
  a 0-based `stages[]` + the C++ `+1` blanket shift (the old off-by-one). Producer
  validates against the schema before writing; consumer fails fast.
- **Round-trip contract test** (`80a0b98`, Lever B): C++ `ScheduleContract` consumes
  the committed `tests/fixtures/schedule.contract.json`; Python `test_schedule_contract.py`
  validates the fixture + live producer output against the schema. Producer/consumer
  drift now reds CI.
- **Case path builder** (`24c5b2d`, Seam 3): `scripts/collect/case.py` owns the
  `<root>/<device>/<app>/<backend>/...` layout (+ cu/vk↔cuda/vulkan naming); 02/03/
  export call it instead of hand-joining paths.
- **Profiling refresh** (jetson/tree/cu only — the cell §3.1's cub fix staled): 6 runs
  (isolated+interference) at the post-fix sha; cuda cub stages dropped ~3× (sort 2.98×,
  unique 3.10×, prefix-sum 3.45×). New schedule: **GPU 1-6 + Little 7** (z3 offloads
  the GPU-slow octree to CPU). Local-only (data/ gitignored).

**Noted, not changed:** 03's default `--schedules-root data/schedules_btpm` ≠ where 02
writes (`data/schedules`) — a real path mismatch, left for a deliberate fix.

### Remaining cross-tool seams (next-session backlog, session-sized each)
- **Seam 1** — kill the legacy wide-CSV bridge: `export_btpm_csv.py` is a
  self-described shim; make `smt/data_loader.py` read the JSONL store directly →
  removes the 5-hardcoded-column CSV + the fragile `absent = 0.0` convention
  (`data_loader.py:52 use_cuda = avg_df["cuda"].sum() > 0` misreads a real 0.0).
- **Lever A** — shared vocabulary codegen (à la `embed_device_specs.py`): PU enum +
  app→stage count (`baselines.py:get_num_stages_for_app` vs C++ `kNumStages`) +
  backend names, generated once for Python + C++ (the "add a PU = touch 1 place" the
  EdgeTPU/NNAPI roadmap needs).
- **Seam 2** — schedule identity header `{device, app, n_stages, pu_vocab, table_type,
  mode}` so the Implementer rejects a wrong-device/wrong-app schedule at load, not deep
  in a worker. Design choice: wrap as `{meta, schedules}` (breaks the array contract +
  config_reader + my consumer tests, once) vs per-schedule fields.
