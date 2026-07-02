# Session 2026-07-01/02 — JetPack 7.2 fleet refresh, fresh-start baseline, CUDA-13 port, and the P2 overhead cost model

One arc, four acts: the Jetsons were reflashed and became new devices; every old
result was archived and the whole differential matrix re-verified from scratch; the
CUDA-13 toolchain gap was closed (CUB port + a matching official cross image, now the
default); and the z3 cost model gained its first fitted framework-overhead term,
measurably improving fleet-wide scheduling decisions.

## 1. Fleet refresh (2026-07-01)

- Both Orin Nano Super devkits reflashed to **JetPack 7.2** (L4T R39.2.0, kernel
  6.8-tegra, Ubuntu 24.04, CUDA 13.2, MAXN_SUPER, user `doremy`, login shell bash).
- Re-registered as new devices **`duck-stable`** (coverage-gated primary) and
  **`duck-naughty`** (benchmark-only twin); JetPack-6 id `jetson` retired.
- **Fresh-start policy:** ALL pre-2026-07 results archived
  ([`perf-results/test-runs/archive-pre-2026-07/`](perf-results/test-runs/archive-pre-2026-07/);
  profiling snapshot at commit `8d45084^`). Rationale: new hardware era + one clean
  baseline date beats mixing eras.
- The old i9/RTX build box left the workflow: **rocky-ryzen builds everything**
  (x86 Vulkan natively; Jetson via podman cross images; Android built on a laptop
  with NDK 29 and deployed through rocky's adb).

## 2. Matrix re-verified — COVERAGE GREEN 12/12 (2026-07-02)

Differential suites (tree 7 / cifar-dense 10 / cifar-sparse 10) pass on:
duck-stable + duck-naughty (CUDA), minipc/RADV + pixel/Mali-sg16 + samsung/sg32
(Vulkan). Logs in `perf-results/test-runs/`. The sparse suites now exercise the
REAL shipped CSR (bug §5 resolved by `2540be6`; docs updated this session).

## 3. CUDA-13 toolchain closed (2026-07-02)

- **CUB port** (`apps/tree/cuda`): `cub::CachingDeviceAllocator` → grow-only device
  scratch; `CubDebugExit` → `CheckCuda`; `cub::DivideAndRoundUp` → local helper.
  Same code builds under CUDA 12.6 AND 13.2; differential suites green from both.
- **Cross toolchain finding:** NGC's official cross container has **no 7.x tag**,
  but the official toolchain IS public — **`cuda-cross-sbsa-13-2`** apt debs
  (JetPack 7 is SBSA-aligned; same component SDK Manager ships). No sysroot rsync
  needed; nvcc auto-selects the sbsa target dir from the aarch64 `-ccbin`.
- **`bt-cross:7.2`** (`Dockerfile.cross-7.2`: ubuntu:24.04 + cuda-nvcc-13-2 +
  cuda-cross-sbsa-13-2 + arm64 multiarch libvulkan) is the **DEFAULT** cross image
  — all six `test-*-{cu,vk}` suites green on duck-stable from it. `bt-cross:6.1`
  stays as the JetPack-6 legacy image (its binaries also verified on 7.2).
- **CUDA 12.6 vs 13.2 performance: parity.** Per-stage p50 A/B on duck-stable
  (tree incl. the CUB radix sort, cifar-dense) shows ±2-3% — noise. The switch
  buys toolchain/device alignment, not speed.

## 4. Fresh-start baseline (21 cells, `speedup-summary-2026-07-02-fresh-start.md`)

Geomean **1.244x**, 16 wins / 5 losses. Twin consistency duck-stable vs
duck-naughty within ~3%. cifar-sparse (first honest numbers: real CSR) is the most
consistent win: 1.61-1.62x on all four duck cells. The 5 losses had a pattern:
tiny-app x GPU dispatch tax (tree x VK on ducks 0.90/0.96, minipc tree 0.75) plus
one true anomaly (samsung dense 0.75).

## 5. P2: fitted per-chunk framework-overhead term (`d688d63`)

Chunk cost is now `Σ stage kernel times + per_chunk_ms(PU) + n·per_stage_ms(PU)`,
constants fitted from the fleet's own measured runs (888 chunk samples) by
`analysis/fit_overhead.py` with per-class **model selection** (zero / median
intercept / lstsq slope — classes whose residuals a constant cannot explain, mostly
CPU co-execution effects, honestly fall back toward zero). Fitted constants are a
result in themselves:

| device | gpu_cuda /chunk | gpu_vulkan /chunk | note |
|---|---|---|---|
| duck-stable / naughty | ~0.21-0.22 ms | ~0.67-0.68 ms | clean submit/fence tax |
| minipc (RADV) | — | 1.4 ms | moderate |
| samsung | — | 12.4 ms | explains the dense 0.75x anomaly |
| pixel (Mali) | — | 24.4 + 9.5/stage ms | huge dispatch tax; capped its cifar wins |

**A/B (same profiling tables, re-solved + re-measured):** geomean 1.244 → **1.298**.
Fixed: samsung dense **0.75 → 1.50**, samsung sparse 0.96 → 1.42, pixel dense
1.01 → 1.25, pixel sparse 1.02 → 1.37. Regressed: phone x tree (samsung 1.50 → 0.98,
pixel 1.24 → 0.85) — the constants, fitted mostly on large cifar chunks,
over-penalize tree's sub-ms chunks, and the old true winners fell out of the top-K.

## 6. P2 v2: union-candidate sweep

`02_gen_schedule_merged` now solves under BOTH cost models, re-prices the
plain-model extras under the overhead model (`reprice_solution` — one consistent
prediction semantics per file), and emits the union sorted by predicted makespan.
The measured top-K sweep picks the true winner, hedging either model's blind spots.

**Validation (full re-solve + re-measure):** geomean **1.300** (baseline 1.244,
v1 1.298), losses 5 → 4. The v1 regressions recovered exactly as intended —
samsung tree 0.98 → **1.76** (beats even the 1.50 baseline), pixel tree
0.85 → 1.18 — while keeping the v1 fixes (samsung dense 1.51). Residual
scatter on the phones (pixel dense measured 0.79 in this sweep with even the
GPU-only candidate slower than its own baseline; samsung sparse 1.12 vs v1's
1.42) is **thermal/DVFS drift across long sweeps**, not a scheduling decision —
the "chaos over lab" reality the framework deliberately measures. Three summaries
archived side by side in `perf-results/`: `...-fresh-start.md`,
`...-overhead-model.md`, `...-union-sweep.md`.

## 7. Open items

- **Co-execution/overlap term** (the remaining CPU residual): deliberately deferred;
  the fitter documents why constants can't model it.
- **P1 CI gating**: rocky self-hosted runner + fail-loud `check-fleet` + fix the
  promotion gate — everything verified this session is still manually enforced.
- **9→11-stage AlexNetCIFAR migration** (P3) unchanged.
- Native on-device CUDA 13.2 builds: expected to work post-port, not yet exercised.
- dev→main promotion PR for this whole arc.
