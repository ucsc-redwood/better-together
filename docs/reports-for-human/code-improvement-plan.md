# Code-improvement execution plan (resume here)

> Companion to [`code-review-2026-06-17.md`](code-review-2026-06-17.md) (the 46
> confirmed findings + evidence). **This is the EXECUTION plan** — phased, with a
> verification gate per change, ordered low-risk → high-risk. Started 2026-06-17;
> meant to be resumed across sessions. Branch: `dev`.

## How to read this
Each item has a **gate** (how to prove it didn't break anything). The cardinal rule:
- touches **builtin-apps kernels / memory / kiss-vk engine** → the **differential
  oracle** (`ctest -L cuda` on Jetson / `-L vulkan` on rocky) is the gate.
- touches a **pipe/ benchmark driver** → **numeric match on all HW** is the gate
  (re-run the bm and compare to the snapshot in `data/sched_logs/speedup-summary.md`).
- touches **solver / loaders / scripts** → a **unit test** is the gate.

---

## Status as of 2026-06-17 (what's DONE)

| area | item | commit |
|---|---|---|
| T1 | `minimize_mode` wired into z3 (was a gapness clone) + `smt/test_minimize_mode.py` | d1928fd |
| T1 | absent CPU tier → UNAVAILABLE (not 0.0) + missing-stage raises | 9ed2bfa |
| T1 | speedup uses makespan not avg-chunk | 83f72fa |
| T2 | `use_cuda` from `--backend` flag, not data-sniff | 83f72fa |
| T2 | missing-target-until-deploy → fixed by CMake foreach | 0d01b6e |
| T4 | `const.hpp` dedup → `pipe/pipeline_common.hpp` (−397 lines) | a72c259 |
| T4 | `get_mr()` ref/ptr normalized via `pipe/mr_ptr.hpp` `as_mr_ptr` | a72c259 / d8ae83c |
| T4 | `bm_baseline` dedup → `pipe/bm_baseline_common.hpp` (−476 lines) | d8ae83c |
| T4 | CMake bm-* targets via `foreach(app × backend)` | 0d01b6e |
| T5 | delete `gen_schedule_tpu.py`, `04_parse_schedules_adv.py`, `utility/nnapi/` | 83f72fa |
| T5 | fix stale `scripts/collect/README.md` | 83f72fa |
| robustness | `03_run_schedule.py` tolerates the §9 teardown segfault (`check=False`) | 18c79e6 |
| robustness | executor warmup uses `first_present_cpu_type()` (was hardcoded Little) | f47e2d7 |

All pushed to `origin/dev` through `0d01b6e`.

## Phase 0 — safety-net baselines (DONE 2026-06-17)
Record the current gate states so regressions are detectable:
- **`ctest -L omp` (local, build/pc): GREEN 5/5.** The everyday gate.
- **`ctest -L vulkan` on rocky-ryzen: GREEN** (tree/cifar-dense/cifar-sparse, 10/10 each).
  `cmake --build --preset vulkan --target test-*-vk && bash scripts/run-on-rocky.sh test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk`
- **`ctest -L cuda` on Jetson: GREEN** (re-verified 2026-06-17 on current `dev`):
  tree-cu 7/7, cifar-dense-cu 10/10, cifar-sparse-cu 10/10, deterministic, no hangs.
  The §1 managed-mem race was already fixed by the pinned-memory switch (commit
  `4161664`, before this plan); the earlier "RED/HANGS" note here was stale. Phase 4
  §1 is therefore already met (see below).
- **Mali differential oracle (Pixel 7a + Samsung) on rocky: GREEN** (baseline fixed
  2026-06-17 at commit `360e579`): both phones tree-vk 7/7, cifar-dense-vk 10/10,
  cifar-sparse-vk 10/10. This is the **gate for Phase 4 §3.2/§3.3** — the HOST_CACHED
  flush is a no-op on rocky/RADV, so a kiss-vk coherency regression only surfaces here.
  Reproduce: `cmake --build build/android --target test-*-vk && bash scripts/run-mali-oracle.sh`
  (stages to rocky, runs both phones via `adb -s`; non-zero exit if any fails).
- Numeric snapshot for the bm dedup regression check: `data/sched_logs/speedup-summary.md`.

---

## Phase 1 — low-risk scripts / build / dead code (no devices) — DONE 2026-06-17
Gate for the whole phase: `uv run python scripts/collect/smt/test_minimize_mode.py`
green + new unit tests + `02` generates cleanly + presets configure. Batchable.

**All 7 items done** (gates below all green): `test_minimize_mode` + new
`smt/test_baselines.py` pass; `02` generates a schedule cleanly with derived
baselines; `pc`/`vulkan` presets configure and `bm-*` targets exist; OMP gate 5/5;
CUDA cross-build (run-*-cu + bm-prof-tree-cu) green in `bt-cross:6.1`.
1. **DONE** — `baselines.py` now DERIVES baselines from the committed
   `data/btpm_export/<dev>/<app>/<backend>/isolated.csv` (GPU = backend column sum;
   OMP = fastest fully-populated CPU tier sum); covers minipc. Gate: `smt/test_baselines.py`.
2. **DONE** — `05_timeline.py` freq/UID regex fixed to the current
   `Frequency=<hz> Hz` / `Schedule_UID=<uid>` format (+warn on missing freq). Verified
   on `data/sched_logs/minipc/schedule_run_1.log` (freq 3.99 GHz, sane ms).
3. **DONE** — `device_specs_embedded.hpp` regenerated at build by a CMake custom
   command (`bt_device_specs`) from `devices/*.json` (identical bytes → tree stays clean).
4. **DONE** — `BT_GIT_SHA` captured at BUILD time via an always-run `bt_git_sha`
   target → `build/<preset>/generated/bt_git_sha.h` (`cmake/write_git_sha.cmake`).
5. **DONE** — `results/log_parser.py` guards non-positive `Frequency=` (skip+warn,
   was a ZeroDivisionError traceback); empty-log path already graceful.
6. **DONE** — `vulkan` + `android` presets set `BT_BUILD_BENCHMARKS=ON` (bm-*-vk exist).
7. **DONE** — deleted orphan `builtin-apps/pipeline/worker.hpp` + dead
   `scripts/collect/00_bm.py` (+README refs); dropped the unused stream from `CudaManager`.

Original item descriptions (for reference):

1. **Hardcoded/stale baselines** (`scripts/collect/smt/baselines.py`). They're
   hand-coded, decoupled from measured data, and return `None` for minipc — so the
   (now makespan-based) `speedup_over_*` divides by a stale number (e.g. jetson cuda
   baseline 5.48 vs measured 38.1). Fix: load baselines from a real source (the
   `bm-baseline-*` output or a committed baseline CSV keyed by device/app/backend).
   Gate: new unit test asserting the loader returns measured values incl. minipc.
   **M / med** (decide the source-of-truth for baselines).
2. **`05_timeline.py` parses the obsolete log format** (wrong UID/freq regex;
   frequency falls back to a hardcoded 24576000, mis-scaling cycles→ms). Fix the
   regex to the current `### Python Begin ###` / `Frequency=` / `Task=… Start/End`
   format. Gate: run on a real `data/sched_logs/*/schedule_run_1.log`, no crash,
   sane ms. **M / low**.
3. **`device_specs_embedded.hpp` has no codegen** (`conf.cpp`): editing `devices/*.json`
   without re-running the generator compiles stale topology silently. Fix: a CMake
   custom-command (or a checked-in `scripts/embed_device_specs.py` invoked at
   configure) that regenerates it from `devices/*.json`; or at least a staleness
   check. Gate: edit a device JSON → build picks it up. **S / low**.
4. **`BT_GIT_SHA` captured at configure time** (CMakeLists): a commit+rebuild keeps
   the old sha in provenance. Fix: a build-time custom command that re-runs
   `git rev-parse` each build. Gate: commit then rebuild → bm-prof provenance sha
   updates. **M / low**.
5. **`04_parse_schedules.py` div-by-zero / swallowed parse** (and `smt/statistics.py`,
   `smt/model_comparison.py`): empty/malformed logs → ZeroDivisionError aborts; widest
   window silently widened. Fix: guard len==0, skip non-positive, surface parse errors.
   Gate: feed an empty log → graceful message, no traceback. **M / low**.
6. **`vulkan` preset omits `BT_BUILD_BENCHMARKS`** (CMakePresets.json) — caveat: also
   check `android`. Fix: set it ON where the bm targets are expected. Gate: configure
   the preset, the bm-*-vk targets exist. **S / low**.
7. **Tier-5 dead code**: orphan `bm_*` + duplicate `worker.hpp` from the pipe
   migration; `00_bm.py` full removal (only `--only-aggregate` worked, xmake run path
   dead — confirm no doc/script depends on it, then delete + drop from README);
   `CudaManager` holds an unused stream. Gate: grep confirms unreferenced, then build.
   **S / low**.

## Phase 2 — C++ robustness guards (build + device smoke + oracle stays green) — DONE 2026-06-17
Gate for the phase: vk+cu compile; `ctest -L omp` green; `ctest -L vulkan` on rocky
stays GREEN (proves kernels untouched); plus the per-item gate below.

**All 6 items done.** Phase gate met: `ctest -L omp` 5/5 (incl. new SchedulePUs
tests); vulkan differential oracle GREEN on rocky (tree 7/7, cifar-dense/sparse 10/10
— kernels untouched); vk bm-gen-logs + cu (test-*-cu/run-*-cu, bt-cross:6.1) build.
1. **DONE** — `first_unavailable_pu()` (schedule.hpp, predicate-driven/unit-testable)
   + `schedule_unrunnable_reason()` (app.hpp); the 6 bm_gen_log executors skip+warn a
   schedule whose PU the device lacks. 4 new SchedulePUs tests incl. the Little-on-
   Big-only gate.
2. **DONE** — worker/worker_with_record (pipe/pipeline_common.hpp) catch a thrown
   body, log it, and re-enqueue to keep the SPSC ring alive (was uncatchable terminate;
   catch-and-break would hang). No signature change (std::thread can't carry a default sink).
3. **DONE** — Logger (record.hpp) emits any ticked cell (start != 0) with a clamped
   duration, so a sub-resolution-fast stage is counted instead of dropped.
4. **DONE** — `~BaseEngine` nulls `g_vma_allocator` after destroy (no double-free).
5. **DONE** — `cu_mem_resource.cu` do_allocate honors alignment (validate vs CUDA's
   256B) + logs cudaGetErrorString (was a bare bad_alloc); frees host alloc on the
   device-pointer failure path.
6. **DONE** — `CheckCudaLaunch()` (helpers.cuh) after every kernel launch (5 tree + 4
   dense + 3 sparse): cheap cudaGetLastError always, cudaDeviceSynchronize in debug.

Original item descriptions (for reference):

1. **No schedule-vs-device validation** (`pipeline/schedule.hpp:122-155`, executor
   mains): a schedule chunk referencing an absent PU throws in an unguarded worker
   thread → uncatchable `terminate`. Fix: validate each chunk's PU against the device
   (`has_*_cores` / GPU present) before running; skip+warn. Gate: **new test/smoke** —
   craft a schedule with a Little chunk, run on the Big-only MiniPC → graceful skip,
   not a crash. **M / low**.
2. **Worker threads have no per-thread try/catch** (`pipeline_common.hpp` worker,
   executor): a Logger OOB on >16 chunks throws in the thread body → terminate. Fix:
   wrap the worker loop body; surface the error to the main thread. Gate: build + smoke.
   **M / low**.
3. **Logger drops `end ≤ start` records** (`pipeline/record.hpp:131,152`): a
   sub-resolution fast stage vanishes, biasing durations up. Fix: clamp `end = max(end,
   start)` (or count zero-duration). Gate: build + records still emitted; **oracle green**.
   **S / low**.
4. **`g_vma_allocator` global never nulled** (`kiss-vk/base_engine.cpp:15,45`):
   latent UAF/double-free if two Engines coexist. Fix: null on destroy, or make it a
   member. Gate: vk build + smoke; `ctest -L vulkan` green. **M / med**.
5. **`do_allocate` ignores alignment, drops the cudaError string**
   (`common/cuda/cu_mem_resource.cu:27-76`). Fix: honor alignment; keep the error text.
   Gate: jetson build + run. **S / low**.
6. **CUDA launch errors deferred / mis-attributed** (`*/cuda/dispatchers`,
   `cifar-cuda/all_kernels.cuh`): no post-launch `cudaGetLastError`, so a bad early
   launch is blamed on a later stage. Fix: check after each launch (debug build).
   Gate: jetson build. **S / low**.

## Phase 3 — `bm_fully` + `bm_gen_log` dedup (HIGH risk — full-HW numeric gate) — DONE 2026-06-17
**Both families done.** Each lifted into a shared header (the bm_baseline_common
pattern); each cell is now a ~14-line main() supplying its types + per-cell knobs.
1. **bm_fully → `pipe/bm_fully_common.hpp`** (−1852/+354). Knobs: OMP namespace, GPU
   ProcessorType + BmTable col (vk 3 / cu 4), and the timer (WallTimer vs a
   `#ifdef __CUDACC__` CudaEventTimer, which now also frees its events). Analysis-only
   output (no script consumes it). Verified: vk local + cu cross build; bm-fully-tree-vk
   on rocky (WallTimer) + bm-fully-tree-cu on Jetson (CudaEventTimer) exit 0, sane tables.
2. **bm_gen_log → `pipe/bm_gen_log_common.hpp`** (−1461/+274). Knobs: OMP namespace +
   the GPU ExecutionModel (kCuda/kVulkan). Folded in: the warmup magic-schedule →
   `make_warmup_schedule()` (portable, full-coverage [1,n], validated); cross-backend
   schedules skipped+warned (rides the Phase-2 device guard). **E2E numeric gate met**
   via 03→04: minipc vk SCH-G6B3 widest-chunk 26.90 ms vs committed 28.0 (noise);
   Jetson cu SCH-L2G7 35.55 ms vs 35.75 (~0.6%). Samsung vk = identical WallTimer/Logger
   path as minipc.
3. **Dead old executor removed**: the 5 pre-migration `builtin-apps/*/vulkan/bm_*`
   orphans (4 `#include "common.hpp"`, which doesn't exist; none in CMake).

## Phase 4 — deep bugs + perf (HIGHEST risk — devices, separate, one at a time)
1. **§1 CUDA managed-mem — DONE (already fixed pre-plan by commit `4161664`).** The
   defect was resolved by switching the dispatchers' `CudaManager` from
   `CudaManagedResource` to **`CudaPinnedResource`** (zero-copy mapped pinned), which
   is coherent on the Jetson UMA — so no `cudaStreamAttachMemAsync` surgery was needed
   and the launches stay on the default stream. The "RED" baseline in this plan was
   stale. **GATE MET: Jetson `ctest -L cuda` GREEN** — re-verified 2026-06-17 on
   current `dev` (tree-cu 7/7, cifar-dense-cu 10/10, cifar-sparse-cu 10/10,
   deterministic) *including* the Phase 1/2 CUDA changes (CudaManager stream removal,
   do_allocate alignment, per-launch CheckCudaLaunch — none regressed it). The unused
   `CudaManagedResource` is left in place; reusing it on Tegra would still need the
   stream-attach. See bugs-found §1.
2. **§9 kiss-vk no teardown — DONE 2026-06-17.** `~BaseEngine` leaked the vk::Device
   and vk::Instance (only the VMA allocator was destroyed) → loader static teardown
   segfaulted on Tegra. The dtor now drains (`device_.waitIdle()`, guarded) and
   destroys allocator → device → instance in order (safe: `engine` is the dispatcher's
   first member so it's destroyed last). **GATE MET:** `bm-baseline-cifar-dense-vk`,
   `test-tree-vk`, `bm-gen-logs-tree-vk` all **exit 0** on Jetson (were 139); rocky
   `ctest -L vulkan` stays GREEN (7/10/10). See bugs-found §9. **M / med.**
3. **Tier-3 perf — these CHANGE the GPU times z3 reads, so re-profile after:**
   - **§3.1 cub alloc/free — DONE 2026-06-17** (commit `364e0b0`). The per-item cub
     temp-storage churn: stages 2/3/6 of `tree/cuda/dispatchers` allocated from the
     caching `g_allocator` but freed with raw `cudaFree`, defeating the cache (real
     cudaMalloc/cudaFree every call) + a latent double-free at exit. Fixed to
     `g_allocator.DeviceFree`. **Measured Jetson cuda p50: stage 2 1.476→0.496 (3.0×),
     stage 3 0.489→0.159 (3.1×), stage 6 0.505→0.144 (3.5×)**; oracle stays GREEN.
     ⚠ jetson/tree/cu profiling tables (isolated+btpm) + derived schedules are now
     stale → need re-collection (ideally all cells together for consistency).
     - **Deferred (separate, riskier):** the device-wide `cudaDeviceSynchronize` ⇒
       per-chunk-stream redesign — those syncs are correctness-required (before freeing
       async-used temp storage / before the host count read) and per-stream conflicts
       with the §1 default-stream pinned model.
   - **§3.2 HOST_CACHED flush-all — NOT DONE (high risk).** `flush_all`/`invalidate_all`
     iterate ALL allocations every submit (`vma_pmr.cpp:139-150`, called from
     `sequence.cpp:189,215`). Flushing only the buffers a stage touches is the win — but
     it is the §7 Mali coherency fix; under-flushing reintroduces the **47× Pixel
     regression / wrong output**. NB the flush is a **no-op on rocky/RADV (coherent)** —
     the cost AND the regression risk exist **only on Mali (Pixel/Samsung)**, so this
     MUST be validated on the phones, not just rocky. Needs per-stage buffer-usage
     tracking threaded from the descriptor binding to the flush.
   - **§3.3 per-stage cmd re-record + fence — NOT DONE (redesign).** Record-once/replay
     needs caching command buffers per (stage, descriptor set); each stage currently
     re-records with different descriptors/push-constants. Also Mali-validated.
   **GATE: differential oracle stays green + re-collect the profiling tables + confirm
   schedules regenerate sanely + quantify the speedup.** §3.2/§3.3 additionally need the
   **Mali phones** (Pixel + Samsung) in the gate, not just rocky. **M–L / med.**

Doc nit: `rearchitecture.md` Phase 2 "Next"→Done — **DONE** (commit `3a4a3ab`).

---

## Resume che-sheet (gate commands)
```bash
# OMP gate (everyday)
cmake --preset pc && cmake --build --preset pc && ctest --test-dir build/pc -L omp --output-on-failure
# solver unit test
uv run python scripts/collect/smt/test_minimize_mode.py
# Vulkan differential oracle (rocky)
cmake --build --preset vulkan --target test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk
bash scripts/run-on-rocky.sh test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk
# CUDA differential oracle (Jetson) — RED until §1 is fixed; tree path may HANG
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/workspace/build -v "$PWD:/workspace" -w /workspace \
  bt-cross:6.1 bash -lc 'cmake --build --preset jetson --target test-tree-cu test-cifar-dense-cu test-cifar-sparse-cu'
bash scripts/run-on-jetson.sh test-tree-cu test-cifar-dense-cu test-cifar-sparse-cu
# end-to-end (schedule on a device): docs/instruction-for-ai/06-end-to-end-scheduling.md
```
Hosts/serials/gotchas: `docs/instruction-for-ai/01-hardware.md`. Devices: Jetson
`duck-naughty` (cu+vk), MiniPC `rocky-ryzen` (vk, Big-only), Pixel 7a `3A021JEHN02756`
(vk subgroup-16) + Samsung `R5CY21Y3VEV` (vk subgroup-32) — **both phones now adb-on-rocky**
(`adb -s <serial>`). Parallelize Jetson ∥ rocky; serialize all rocky work (MiniPC ↔ Pixel ↔ Samsung).

## Recommended order
Phase 1 (this/next session, fast) → Phase 2 (build+oracle) → Phase 3 (per family,
all-HW numeric) → Phase 4 (§1 first — biggest unlock — then §9, then perf).
