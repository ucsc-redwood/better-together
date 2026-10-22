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

## Deviations and choices made while running the study

Nobody was available to answer questions during the run. Where the plan left a choice open, the conservative option
was taken, meaning the one that makes the baseline stronger. Raw evidence for each item is under
`data/coordinated-corun/`.

- **Harness build (D1).** `lpa_pace.cpp` links the static libraries of `llm-phase-affinity`'s Samsung build
  (`build/lpa/llamacpp-android-s24` of the main checkout, read only: llama.cpp `8345f33`,
  `GGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16`, Vulkan on, OpenMP and KleidiAI off). It is compiled with that
  build's flags by `drivers/build_pace.sh`. Besides `llama.h` it uses `ggml-cpu.h`, also a public header, for the
  thread pool that llama-bench attaches (`ggml_threadpool_new`, poll 50). Without one, ggml makes a throwaway pool per
  graph. It mirrors llama-bench's `test_prompt`/`test_gen`: unseeded `std::rand` token ids, BOS first, logits for the
  last token of a batch, one warm-up run, and `n_ctx` = P (prefill) or P + 128 (decode). It differs in two ways. Each
  ubatch is its own `llama_decode` call (`n_batch` = `n_ubatch` = C), followed by a `llama_synchronize`, so it can
  pause between ubatches; llama-bench submits the whole prompt in one call (`n_batch` 2048, `n_ubatch` 512). A
  run's wall time includes the gap after its last step, so achieved duty = compute / wall is not biased upward over
  a few long steps. The 30-s calibration runs used a build of the source before `clang-format`; every later run uses
  the build of the committed source. The two differ only in formatting.
- **Flash attention.** As in `llm-phase-affinity`'s co-runs, the GPU prefill runs with `-fa auto` at P <= 512 and
  `-fa 0` at P = 2048 (auto collapses prefill at 2048 on the Xclipse). GPU decode uses auto at every depth, which was
  the faster configuration at every depth in `llm-phase-affinity` (1B).
- **Thread placement (task 1.4).** Masks `3f0`, `1f0`, `1c0` and `180` all keep one compute thread per core under the
  harness's CPU decode (`R5CY21Y3VEV/diag-thread-placement.txt`: 94-99% busy on each core of the mask, under 13% on
  the others). No mask is dropped. On the Pixel, `f0`, `c0` and `30` do the same for `llama_main`'s XNNPACK decode
  (`3A021JEHN02756/diag-thread-placement.txt`).
- **Calibration and the reference rates (task 1.3).** The GPU rates match `llm-phase-affinity` within 2.5%. The CPU
  decode on `3f0` does not: it is 12% lower over 30 s (`R5CY21Y3VEV/calib/`), because a sustained CPU decode slows as
  its clocks fall. The per-3-s rate drops from 55.5 to 43.7 tok/s while policy 9 falls from 3.07 to 2.02 GHz and
  policy 7 from 2.78 to 2.02 GHz, with the thermal status still 0. `llm-phase-affinity`'s alone references were
  llama-bench repetitions of 2-5 s; its co-run loops lasted about 100 s. Its CPU decode co-run ratios therefore
  include this sustained-load slowdown as well as interference. Following the spec, the harness's own alone rates
  are the reference for every ratio and for the grid. Alone runs last 40 s, the length of a co-run's GPU side and
  about the CPU side's counted span. The 30-s calibration runs stay in `calib/` and serve as calibration only.
- **Co-run timing (D3).** `lpa_corun.sh` and the thermal policy are reused unchanged. The host drivers `run_plan.sh`
  and `run_corun.sh` are copies of `llm-phase-affinity`'s with the phone directory changed to `/data/local/tmp/ccr`,
  and they log the battery level. In stages A-D the GPU side (A) runs 40 s and the CPU side (B) 70 s, so B covers
  A's last paced repetition (a 2048-token prefill at duty 0.33 takes about 14 s of wall time). In stage E the CPU
  prefill (B) runs 40 s and the GPU decode (A) 70 s. A side counts only its units that lie wholly inside the
  partner's measured interval (first step's start to last step's end): a whole prefill repetition (all its ubatches,
  with the gap after each) or one decode token. Its rate is over wall time, so a paced side's rate includes the gaps.
- **Prediction (D4).** Each co-run ratio is the co-run wall rate over the harness's duty-1 alone rate of the same
  phase. Pacing's own cost is therefore in the ratio, and the paced-alone runs report it separately. As in D5, the
  grid applies each ratio to the alone rate at prompt P or KV depth P, and a prompt of 128 uses the ratios measured
  at 512. Choices that make the baseline stronger:
  - the CPU prefill of data-parallel, of the CPU-prefill pipeline and of the single CPU always runs on `3f0`, the
    fastest mask, whatever mask that setting decodes on;
  - the baseline is the largest of: the best coordinated data-parallel over every measured setting; the uncoordinated
    data-parallel bound of `llm-phase-affinity` (its grid, unchanged); and the best single PU of either study.
  The gate is set against that baseline. The literal ratio of the spec, against the best coordinated data-parallel,
  is reported beside it.
- **Stage order.** Stage E needs no setting from A-D, so it runs right after D. B and C come next, once the analysis
  of A and D has picked their settings.
- **Pixel ExecuTorch part (D5) in parallel.** It runs on the Pixel at the same time as the Samsung sweep, not after
  it. The two phones share only the host's adb. `llm-phase-affinity` also ran both phones at once.
- **Commits.** This session could not write the main checkout's `.git` (index, objects and refs are outside its
  writable area). Every commit therefore goes to a bare clone sharing the main repository's objects,
  `build/ccr-git.git` in this worktree, on branch `openspec/coordinated-corun`. Fetch it from there.

## Migration Plan

None: a study. Results are additive under `data/coordinated-corun/` and `docs/reports-for-human/`.

## Open Questions

- None blocking. Whether `build/lpa/llamacpp-android-s24` built shared libraries decides between linking it and an
  out-of-tree rebuild (D1); either is acceptable.
