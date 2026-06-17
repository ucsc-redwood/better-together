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

- **🐛 GPU re-entry into the tree octree** (`DISABLED_AlternatingBoundary`, triaged):
  a schedule where Vulkan runs early stages, a CPU chunk runs the middle, then Vulkan
  re-enters for octree (stage 7) alone reads a STALE data-dependent count
  (`n_brt_nodes`) → wrong output, or an OOB octree write → SIGSEGV. Minimal repro
  `{VK 1-3, OMP 4-6, VK 7}`. Contiguous GPU chunks always pass. **Not** a generic
  SPSC/concurrency bug, and z3 only emits one contiguous chunk per PU so it never
  produces a re-entry schedule. Fix = dispatcher re-reads the stage-7 count on entry,
  or an executor guard rejecting GPU re-entry. cifar is feed-forward → not affected.
- **Hardware:** both phones (Pixel `3A021JEHN02756` + Samsung `R5CY21Y3VEV`) moved to
  rocky-ryzen's adb this session (`adb -s <serial>`); docs synced. Serialize all
  rocky-side work (minipc Vulkan ↔ Pixel ↔ Samsung).
- **Not yet done (optional):** §3.2/§3.3 kiss-vk perf (Mali-gated); the §3.1 profiling
  re-collection; cifar-sparse runtime test on Mali; actually fixing the octree-re-entry
  limitation.
- **Uncommitted in the tree:** `docs/reports-for-human/perf-results/` + a README.md
  edit (the RGA static-analysis write-up) — pre-existing, not from this session's work;
  left for the owner to commit.
