# Session 2026-06-20 — Fleet benchmark orchestrator + DVFS/interference audit & fixes

## TL;DR

1. **Built + shipped the fleet benchmark orchestrator** (`optimizer/orchestrate/00_run_fleet.py`)
   — one command runs the whole scheduler benchmark e2e (build → profile → z3 schedule →
   run → speedup) across all hardware concurrently, with live progress. Merged to `dev` and
   promoted to `main` (now `main == dev`).
2. **Found the minipc-tree-<1× loss is GPU DVFS, not framework overhead** — the iGPU
   downclocks (800 vs 2799 MHz) when execution is gappy. See [[vk-bench-dvfs-artifact]].
3. **Audited the interference (BTPM) profiling** — the GPU column is corrupted by DVFS on
   most of the fleet (GPU measures *faster* under load → impossible for contention →
   inverts the signal), and z3 has no co-execution term. See [[interference-btpm-audit]].
4. **Fixed it** on branch `fix/interference-dvfs-and-btpm` (not yet PR'd): loader guards,
   harness changes, docs, and a validator — calibrated to the paper's *chaotic-environment*
   thesis (do NOT lock clocks).
5. **Two research findings for the paper**: (Q1) the z3 cost model is accurate on heavy
   workloads but under-predicts tiny ones 2–6× because it ignores per-chunk framework
   overhead; (Q2) running **top-10 is empirically justified** — z3's #1 pick is the true
   optimum only 38% of the time; sweeping top-10 is up to **69% faster** than #1.

---

## 1. Fleet benchmark orchestrator (shipped → `main`)

Turned the ad-hoc per-device shell scripts into one reusable tool:

- `optimizer/orchestrate/00_run_fleet.py` — concurrent (one worker/device) e2e with a live
  `rich` progress table (Detail column: profiling run N/M, z3 solution K/10, build %);
  non-TTY → plain logs. Phases selectable (`--phases build,profile,schedule,run,summary`),
  `--fresh` to start clean, `--repeat N` to sample run-to-run variance.
- `optimizer/orchestrate/01_collect_profiling.py` — the previously-manual `bm-prof` step.
- `optimizer/orchestrate/transport.py` — shared ssh/adb/local deploy channels.
- `optimizer/analysis/speedup_summary.py` — one canonical method → `data/sched_logs/speedup-summary.md`.
- `fleet.json` — per-device transport/build/caps/runs (fixed the stale jetson host →
  `yanwen@duck-stable`).
- `scripts/build-bench-jetson.sh` — Jetson bench cross-build on rocky (podman).
- justfile: `build-bench-*`, `fleet-bench`, `bench-clean`.

**Clean-replace data semantics** (so re-runs don't silently mix old + new implementation):
`01`/`03` wipe a cell's stale files before writing; `00 --fresh` wipes a device's results.
Loaders aggregate a whole directory and don't filter by `git_sha`, so this was a real trap.

Merge path this session: PR #11 (orchestrator → dev), #13 (reconcile main→dev), #12
(dev→main), #14 (per-device `runs`). `main == dev`.

## 2. DVFS root cause (minipc tree < 1×)

A debug agent traced the only sub-1× fleet cell. **Cause = GPU DVFS, not overhead.** The
Radeon 780M runs the gappy pipeline at the lowest DPM state (800 MHz) while the tight-loop
baseline stays boosted (2799 MHz) — a ~3.5× clock penalty. submit/fence latency (~40–95 µs)
is second-order. Implication: the tight-loop baseline gets a free boost the pipeline doesn't,
so measured speedups are *conservative for the pipeline* fleet-wide.

## 3. Interference (BTPM) audit

The interference table feeds z3's contended cost matrix. Audit verdict: **not sound for the
GPU on DVFS devices.** On Samsung/Pixel/Jetson the GPU measures *faster* under interference
(Samsung tree 7/7 stages faster, median 0.43×, min 0.09×) — impossible for true contention;
it's the bg GPU load keeping the clock boosted. Only minipc (clock-parked at 800) gives a
correct slowdown. Also: same-stage saturation ≠ the real pipeline mix, and `constraints.py`
has **no co-execution/overlap term** (it statically sums per-stage contended costs). The
paper's formal BTPM definition is not in the repo.

## 4. Fixes (branch `fix/interference-dvfs-and-btpm`)

- **Loader guards** (`smt/data_loader.py`): min_runs≥2 on interference (kills "survives on
  one lucky run"); a gate-dropped CPU cell is demoted to UNAVAILABLE (z3 avoids it) instead
  of fatally failing the app's schedule; an **opt-in, off-by-default DVFS floor** that clamps
  a too-cheap-under-load GPU up to its isolated value.
- **Harness** (`profiler/bm_prof_common.hpp`): record `gpu_clock_mhz` in provenance
  (best-effort; works on AMD, recorded minipc isolated@800 vs interference@2420); a **settle
  barrier** (exclude the bg ramp from samples); a GPU warm before sampling (measured
  insufficient alone — `time_once()` fence-waits).
- `scripts/lock-gpu-clocks.sh` — optional clock-lock (see §5 — NOT the default path).
- Docs: an operative BTPM definition + corrected the wrong-sign iGPU caveat in
  `06-end-to-end-scheduling.md`.
- `optimizer/analysis/validate_btpm.py` — predicted-vs-measured validator.

## 5. Decision: chaos over lab (do NOT lock clocks)

The user's correction, adopted: the paper's thesis is finding the optimum in the **real
chaotic environment**, so DVFS/contention/thermal are *part of reality, not artifacts to
sanitize* — and that chaos is exactly *why* interference-aware profiling is needed.
Therefore:
- The **DVFS floor is off by default** (z3 sees raw chaotic interference costs); it's an
  opt-in conservative guard only.
- **`--repeat N` samples the chaotic distribution** (median over runs); it does **not**
  control the environment. Clock-locking is a separate sensitivity tool, never the headline.
- `settle`, `min_runs≥2`, and `gpu_clock_mhz` recording stay — they reduce ramp noise /
  require statistical agreement / *record* reality; they don't sanitize chaos.

## 6. E2E validation (with fixes, no clock-lock)

- **Prediction (run-internal, the clean metric):** with the fixes, the BTPM table predicts
  the measured makespan **≥ isolated on every cell (8 strictly better, 7 tie, 0 worse)** —
  vs 8/1/5 before; the fix removed the one cell where the corrupted BTPM lost to isolated.
- **Interference-aware vs isolated scheduling** (z3 #1 pick from each table, measured, 3×
  averaged, chaotic): **btpm-pick faster on the contended heavy workloads by 13–25%**
  (pixel cifar-dense 25%, samsung cifar-sparse 22%, samsung tree 15%, pixel cifar-sparse
  13%); **geomean 0.965**. One honest counterexample — **pixel tree: btpm 39% slower**,
  the tiny-Mali case where raw interference's GPU-boost optimism makes z3 over-commit to the
  GPU (exactly what the opt-in floor guards). 10 ties (no contention to exploit).
- Note: absolute speedup numbers move run-to-run (chaotic + baseline is itself a GPU run);
  the *within-run* comparisons above are the reliable signal, not before/after deltas.

## 7. Research findings for the paper

**Q1 — How good is the optimizer's estimate vs measured?** z3's #1-pick `measured/predicted`:
- **Accurate (≤3%) on heavy compute-bound** cells (cifar-dense everywhere, jetson cifar-sparse-cu).
- **Under-predicts tiny/overhead-bound cells 2–6×** (minipc tree 6.0×, jetson tree-vk 4.1×,
  samsung tree 2.5×). Root cause: the cost model **sums per-stage kernel times and ignores
  the per-chunk framework overhead** (submit/fence), which dominates tiny kernels.
  → To predict tiny workloads, the cost model needs a per-chunk overhead term.

**Q2 — Is the true optimum z3's #1, or in the top-10?** Run all 10 candidates, take the
best-measured:
- True optimum = z3's **#1 on only 5/13** cells; it lives at **rank #2–#10 on 8/13**.
- Best-of-top-10 vs only-#1: **avg 14% faster, up to 69%** (jetson tree-vk #4 = 69%, minipc
  tree #3 = 59%, pixel tree #5 = 27%). The big wins are exactly the tree cells z3 mis-predicts.
- → **top-k is a hedge against cost-model inaccuracy**: where the model is strong (heavy),
  #1 is optimal (top-k free); where it's weak (tiny), top-k recovers the real optimum.
  This empirically justifies the top-k recommendation.

**Methodology note:** the headline `speedup-summary.md` numbers are **best-of-top-10**
(oracle / with-sweep), NOT z3's #1 pick. The paper should distinguish "best-of-top-k (needs
a sweep)" from "deploy z3 #1 (no sweep)"; Q2 quantifies the gap (the value of the sweep).

## 8. State + open items

- `main == dev`; orchestrator shipped.
- `fix/interference-dvfs-and-btpm` pushed, **not yet PR'd to dev** (5 commits: loader guards,
  btpm docs+validator, harness, floor-off-default+`--repeat`, + 2 user cleanup commits).
- Open / future: (a) PR the fix branch to dev; (b) add a **per-chunk overhead term** to the
  z3 cost model (Q1) and a **co-execution/overlap term** (audit) — the two modeling gaps;
  (c) device-specific GPU-clock sysfs paths for Mali/Tegra (only AMD works today); (d) the
  bg GPU load doesn't truly saturate (one cmd buffer + one fence — deferred).
