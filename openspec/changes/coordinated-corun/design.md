# Design

## Context

`llm-phase-affinity` (decision record `docs/reports-for-human/2026-10-21-llm-phase-affinity.md`) measured co-runs as
two independent processes started back to back, each pinned with `taskset` and running its phase in a loop, with no
coordination (`data/llm-phase-affinity/drivers/lpa_corun.sh`). Samsung, llama.cpp 1B, ratios to alone:

| Pair | GPU side | CPU side |
|---|---|---|
| GPU prefill (P 512) + CPU decode | 0.46 | 0.41 |
| GPU prefill (P 2048) + CPU decode | 0.54 | 0.34 |
| GPU decode + CPU decode (data-parallel) | 0.74 | 0.21 |

Read bandwidth: CPU 36.4 GB/s alone, 7.6 next to a streaming GPU; GPU 47.4 alone, 40.3 next to a streaming CPU.

Why pacing could favour the specialized pipeline: its period is max(prefill time, decode time). Where CPU decode is
the bottleneck (most mixes), the GPU sits idle part of every period anyway, so spreading its prefill over that slack
costs the pipeline nothing while it may return bandwidth to the CPU decode. In data-parallel execution every PU's
throughput counts, so slowing the GPU costs directly. Whether this shows up depends on how the CPU decode rate
responds to GPU duty, which is what this study measures.

Constraints found while planning: the GPU clock caps are not writable from `adb shell` (Samsung
`/sys/kernel/gpu/gpu_max_clock` is `system`-owned, mode 644; Pixel Mali `devfreq` nodes are root-only), so pacing is
the only GPU throttle. `llama-bench` has `-ub` (chunk size) but no per-ubatch or per-token gap (`--delay` is between
whole tests), so a small harness is needed.

## Goals / Non-Goals

**Goals:**
- Trade-off curves (GPU rate vs CPU rate) under pacing, chunk size and decode width, for the specialized and the
  data-parallel pairs, Samsung llama.cpp 1B; 3B at the best settings.
- The `llm-phase-affinity` D5 grid recomputed with the best coordinated settings on both sides, and a go/no-go.

**Non-Goals:**
- A runtime, a feedback controller, single-process execution, or KV hand-off code.
- Changing llama.cpp or ExecuTorch sources, or any clock/governor setting.
- The Pixel's llama.cpp path (its GPU loses both phases, so there is nothing to specialize).

## Decisions

### D1. Paced harness `lpa_pace`
One C++ file, `data/coordinated-corun/drivers/lpa_pace.cpp`, using only the public `llama.h` API of llama.cpp
`8345f33` (the version of `llm-phase-affinity`), built with the Android NDK and the same CMake flags as
`build/lpa/llamacpp-android-s24` (link that build's libraries, or rebuild out of tree with the same flags).
- Arguments: model, `ngl` (0 with no GPU device, as `-dev none`, or 99), threads, mode `prefill|decode`, prompt or
  depth P, ubatch C, target duty u, duration (s).
- `prefill`: clear the KV memory, decode P tokens in batches of C; `decode`: fill depth P once (unmeasured), then
  decode one token at a time, clearing back to depth P every 128 tokens.
- After each ubatch or token: `llama_synchronize`, timestamp, then sleep for t_step x (1/u - 1), with t_step the
  measured time of that step (u = 1 means no sleep). Output: one JSONL line per step (epoch start, end, tokens), and a
  summary line (rate over compute time, rate over wall time, achieved duty).
- Token ids are a fixed pseudo-random sequence, as `llama-bench` uses.
Alternative considered: patching `llama-bench` with a gap option. Rejected: the sources must not be modified.

### D2. Knobs and staged sweep (Samsung, 1B)
The GPU host thread stays on core 3 (mask `8`). Stages:
- **A, GPU pacing, specialized**: GPU prefill paced at u = 1.0, 0.8, 0.67, 0.5, 0.33 (C = 512) next to a CPU decode on
  `3f0`; P = 512 and 2048.
- **B, decode width**: CPU decode on `3f0` (cores 4-9), `1f0` (4-8), `1c0` (6-8) and `180` (7-8), alone and next to
  the GPU prefill at u = 1.0 and at the best u of stage A. Each mask's thread placement is checked first (one thread
  per core, as `llm-phase-affinity` did for `380` vs `180`); a mask the scheduler packs is dropped and recorded.
- **C, chunk size**: C = 128 and 256 at the best (u, mask) of A-B.
- **D, data-parallel coordinated**: GPU decode paced at the same five duties next to a CPU decode on `3f0` and on the
  best mask of B.
- **E, reverse pair**: CPU prefill (`3f0`) next to a GPU decode at u = 1.0 and 0.5, for completeness.
- **F, 3B**: the best specialized and the best data-parallel settings, plus u = 1.0, for 3B.
Also measured: each paced phase alone at every duty, so pacing's own cost is separated from interference.

### D3. Co-run procedure
Reuse `lpa_corun.sh` and the `llm-phase-affinity` thermal policy unchanged: wait for thermal status 0, 20 s between
runs, a run ending at status >= 1 repeated once after 120 s, GPU clock and CPU policy frequencies sampled every 0.5 s.
Each co-run: both loops at least 30 s, overlapped interval only, 3 trials, both start orders. With the per-step
timestamps of D1, the overlap is computed exactly, without the duty extrapolation `llm-phase-affinity` needed for
ExecuTorch.

### D4. Prediction and gate
Reuse `optimizer/analysis/llm_phase_affinity.py` (D5 there), adding `optimizer/analysis/coordinated_corun.py`:
- specialized, per mix: the maximum over measured settings s of 1 / (max(P/p'_s, D/d'_s) + h(P));
- data-parallel, per mix: the maximum over measured data-parallel settings of the sum of per-PU request rates with
  the co-run decode rates at that setting (prefill at its alone rate, as D5 does), and also with co-run prefill;
- the baseline is the larger of coordinated data-parallel, uncoordinated data-parallel and the best single PU.
Gate: specialized / baseline >= 1.2 on some mix.

### D5. Pixel ExecuTorch (optional)
`llama_main` cannot be paced unmodified, so only the CPU decode width varies: XNNPACK decode on `f0`, `c0` and `30`
next to the Vulkan 8da4w prefill at P = 512, plus the data-parallel pair at the same masks. Run only after the Samsung
part, if time allows; skipped otherwise, with the reason recorded.

## Risks / Trade-offs

- **The GPU clock drops while paced** (DVFS sees idle time) → the paced rate falls below u x the alone rate. That is
  part of the real effect; the clock samples show it, and the paced-alone runs of D2 measure it separately.
- **The harness differs from `llama-bench`** (batch, ubatch, warm-up) → the calibration requirement; if it misses by
  more than 5%, the harness's own alone rates are the reference for every ratio.
- **Sweep length**: about 40 configurations x 3 trials on one phone, roughly 10-15 hours with thermal retries.
  Stages run in the order A, D, B, C, E, F, so the gate's inputs come first; data is committed per stage.
- **Open-loop pacing is not a scheduler** → it shows whether a trade-off exists, not the best controller; a positive
  result motivates the runtime, a negative one bounds what coordination of this kind can buy.

## Migration Plan

None: a study. Results are additive under `data/coordinated-corun/` and `docs/reports-for-human/`.

## Open Questions

- None blocking. Whether `build/lpa/llamacpp-android-s24` built shared libraries decides between linking it and an
  out-of-tree rebuild (D1); either is acceptable.
