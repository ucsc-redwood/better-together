# LLM phase affinity: the go/no-go (2026-10-21)

**Question.** Can a pipeline that prefills on one processing unit (PU) and decodes on the other beat CPU+GPU
data-parallel execution of on-device LLM inference? The spec
(`openspec/changes/llm-phase-affinity/specs/studies/llm-phase-affinity/spec.md`) sets the gate. The answer is
"go" only if some specialized pipeline is predicted to reach at least 1.2x the co-run-corrected data-parallel
bound, on at least one phone and one request mix (prompt P = 128/512/2048 tokens, output D = 32/128/512 tokens).

**Decision: go, by a narrow margin, in one cell.** Every grid in scope was assessed with its own co-run rates,
measured with the two runs overlapping for the whole measured interval. One specialized pipeline passes the gate.

- **The qualifying cell: 1.23.**
  - Configuration: Samsung S24+, ExecuTorch, Llama 3.2 3B, P = 2048, D = 512. Prefill runs on the CPU (XNNPACK
    8da4w, cores 4-9) and decode on the GPU (Vulkan 4w).
  - It delivers 0.0241 requests/s against a co-run data-parallel bound of 0.0196.
  - The three trials give 1.20, 1.26 and 1.24. Taking the least favourable ratio of every input across trials
    gives 1.15.
- **Why it passes.**
  - In data-parallel both PUs decode, and two 3B decodes interfere heavily on this phone: the CPU decode keeps
    0.43 of its rate and the GPU decode 0.67.
  - In the split pipeline the GPU decode keeps 0.92 next to a CPU prefill. It sets the period (41.5 s for 512
    tokens), with the CPU prefill (30.3 s) hidden behind it. The KV hand-off adds 19 ms.
- **The caveat a PoC must face.** At this cell the co-run data-parallel bound is below the GPU working alone
  (0.0205 requests/s), because running the CPU's decode alongside costs the GPU more than the CPU adds.
  - Against the GPU alone the pipeline is 1.18x.
  - Against the better of data-parallel and one PU, with each phase slowed only while the other runs, it is 1.20.
  - The gate passes as the spec defines it, against the co-run data-parallel bound. Its advantage over the
    best single PU is just under 1.2.
- **Nowhere else is close.** The next cells are 1.03:
  - ExecuTorch 3B at P = 512, D = 128;
  - the Pixel's ExecuTorch 1B at P = 512, D = 32, whose CPU decode keeps 0.65 next to a GPU prefill where 0.76
    would pass.
  The Samsung's ExecuTorch 1B reaches 0.96 and llama.cpp at most 0.79. No other cell reaches 1.2 however its
  measured interference is priced (sections 5-6).
- **The phase affinity is real.** Where the GPU path is strong, the GPU prefills 1.4-2.9x faster than the CPU and,
  at P <= 512, the CPU decodes 1.2-2x faster (except ExecuTorch 3B, where they decode alike). Co-running costs
  both PUs, though. A split pipeline beats data-parallel only where data-parallel suffers more, as with 3B here
  (section 3).
- **The prompt-heavy agent corner (P = 2048, D = 32) is the weakest**: 0.41-0.62. At P = 2048 the GPU decodes as
  fast as the CPU or faster (except on the Pixel's llama.cpp, whose GPU is slower at everything), so there is
  no decode to hand to the CPU, and the GPU prefill keeps 0.44-0.68 of its rate next to a CPU decode.

## 1. What was run

All timing ran on the two phones (2026-10-20 15:08 to 2026-10-21 21:00, PDT); nothing was timed on rocky-ryzen
or the build box. Choices made during the run, each with its reason, are in
`openspec/changes/llm-phase-affinity/design.md` ("Deviations and choices"). The ones that shape the numbers:

- **Phones.** Pixel 7a `3A021JEHN02756` (Tensor G2, Mali-G710) and Samsung S24+ `R5CY21Y3VEV` (Exynos 2400,
  Xclipse 940), Android 16, clocks left to the governors. Llama 3.2 1B on both, 3B on the Samsung, Llama 3.1 8B on
  the Samsung as a single-backend reference.
- **llama.cpp** `8345f33`, built per phone ISA (Samsung `armv8.6-a+dotprod+i8mm+fp16`, Pixel
  `armv8.2-a+dotprod+fp16`), Vulkan on, KleidiAI and OpenMP off; the owner's Q4_0 GGUFs. Rates from `llama-bench`:
  prefill = `pp` at prompt 128/512/2048, decode = `tg128` at KV depth 0/128/512/2048 (the decode rate of a request
  with prompt P is the one at depth P).
- **ExecuTorch** `e229c58784`: one `llama_main` with XNNPACK and Vulkan built from an independent clone; the owner's
  Vulkan 4w and 8da4w `.pte` files (ctx 2560) and XNNPACK 8da4w `.pte` files exported for this study with the same
  flags. Prompt `" the"` x P, 128 new tokens, greedy, EOS ignored, `--warmup`; rates from the runner's
  PyTorchObserver record of the measured (second) run.
- **Processing units** (design D2). CPU: llama.cpp `-ngl 0 -dev none`, ExecuTorch XNNPACK; one tier (Samsung cores
  7-8, mask `180`; Pixel 6-7, `c0`) and a wider set (Samsung 4-9, `3f0`; Pixel 4-7, `f0`), one thread per core. GPU:
  llama.cpp `-ngl 99` with the host thread on core 3 (mask `8`), per point the faster of flash attention auto and
  off; ExecuTorch Vulkan with the host thread on core 3, per phase the faster of the 4w and 8da4w files. The CPU
  unit of the prediction and of the co-runs is the wider set. It is faster at every prefill and at every decode
  the grid uses (KV depth 128-2048), on both phones and runtimes; only the Pixel's llama.cpp `c0` decodes 3%
  faster at depth 0.
- **Discipline.** Each value is the median of 3 runs with its min-max. The driver waits for thermal status 0
  before every run and leaves 20 s between runs. A run that ends at thermal status 1 or more is repeated once
  after a 120 s cooldown, and the repeat replaces it. The GPU clock and every CPU policy frequency are sampled
  every 0.5 s during each run (`*.clk`).
- **Co-runs** (design D3): two pinned processes started back to back, both orders, 3 trials per pair. Every
  counted interval lies wholly inside the partner's compute, so the two runs overlap for the whole measured
  interval (spec "Co-run interference is measured").
  - llama.cpp loops a phase with `llama-bench -r`. A repetition counts only if the other process's
    repetitions cover it wholly.
  - ExecuTorch's `llama_main` cannot repeat a phase, so whole processes loop, and model loads leave gaps. A
    prefill counts only if a partner 512-token decode window covers it. Decode is measured separately, in warm
    8-token windows that count only inside a partner prefill or decode. Each side is compared with the same
    loop run alone, and the counted / total intervals per trial are reported.
  - A first ExecuTorch decode estimate extrapolated partly overlapped windows to full overlap. It does not meet
    the spec and is kept only as a labelled diagnostic (design.md).
- **Bandwidth** (design D4): a pinned streaming read of 256 MiB (`cpu_readbw`, one thread per core of the wider
  set) and the owner's `igpu-roofline` `mem_read_v4` kernel, alone and together, 40 s per case, 3 trials.
- **Prediction** (design D5): `optimizer/analysis/llm_phase_affinity.py`; its pytest checks a hand-computed mix.

## 2. Phase affinity: each PU alone

GPU / CPU ratio per phase. Prefill above 1 means the GPU prefills faster, decode below 1 means the CPU decodes
faster: a split pipeline needs both. Decode is at KV depth P. Medians of 3 runs; the spreads are in the appendix.

| Phone | Runtime | Model | Prefill GPU / CPU at P = 128, 512, 2048 | Decode GPU / CPU at P = 128, 512, 2048 |
|---|---|---|---|---|
| Samsung S24+ | llama.cpp | 1B | 1.55, 1.65, 1.98 | 0.65, 0.69, 0.95 |
| Samsung S24+ | llama.cpp | 3B | 1.45, 1.59, 2.10 | 0.63, 0.80, 1.18 |
| Samsung S24+ | ExecuTorch | 1B | 1.37, 1.84, 1.99 | 0.79, 0.82, 1.30 |
| Samsung S24+ | ExecuTorch | 3B | 1.70, 1.92, 2.13 | 0.99, 1.16, 2.12 |
| Pixel 7a | llama.cpp | 1B | 0.42, 0.62, 0.76 | 0.50, 0.56, 0.77 |
| Pixel 7a | ExecuTorch | 1B | 2.05, 2.90, 2.49 | 0.50, 0.59, 0.96 |

- **The split exists where the GPU path is strong**: the Samsung with either runtime and the Pixel with ExecuTorch.
  There the GPU prefills 1.37-2.90x faster than the CPU, and at P <= 512 the CPU decodes faster (GPU / CPU
  0.50-0.82). ExecuTorch 3B is the exception, decoding alike on both (0.99, 1.16).
- **llama.cpp's Vulkan path on the Mali loses both phases** (prefill 0.42-0.76, decode 0.50-0.77 of the CPU's),
  so the Pixel llama.cpp pipeline has nothing to specialize.
- **At P = 2048 the decode preference turns toward the GPU**: Samsung llama.cpp 0.95 (1B) and 1.18 (3B), ExecuTorch
  1.30 and 2.12, Pixel ExecuTorch 0.96. From KV depth 128 to 2048 the CPU decode loses 37-65% of its rate, the GPU
  decode 8-26%.
- **Alone, specialization would pay.** With every phase at its alone rate ("gain, no interference" in section 5),
  a split pipeline would beat the co-run data-parallel bound by up to 1.74 (Samsung llama.cpp 3B) and 1.59
  (Pixel ExecuTorch 1B).

## 3. Interference: each phase co-running

Rate while the other PU runs the other phase (or, last column, both decode), divided by the rate alone. Every
counted interval is wholly overlapped by the partner's compute (section 1). llama.cpp: against the alone sweep.
ExecuTorch: against the same loop run alone; counted / total intervals per trial are in the appendix.

| Phone | Runtime | P | GPU prefill next to CPU decode | CPU decode next to GPU prefill | CPU prefill next to GPU decode | GPU decode next to CPU prefill | both decoding: CPU, GPU |
|---|---|---|---|---|---|---|---|
| Samsung S24+ | llama.cpp 1B | 512 | 0.46 | 0.41 | 0.56 | 0.55 | 0.21, 0.74 |
| Samsung S24+ | llama.cpp 1B | 2048 | 0.54 | 0.34 | 0.65 | 0.52 | ″ |
| Samsung S24+ | llama.cpp 3B | 512 | 0.48 | 0.36 | 0.60 | 0.61 | 0.20, 0.76 |
| Samsung S24+ | llama.cpp 3B | 2048 | 0.45 | 0.34 | 0.68 | 0.38 | ″ |
| Samsung S24+ | ExecuTorch 1B | 512 | 0.69 | 0.74 | 0.79 | 1.02 | 0.65, 0.95 |
| Samsung S24+ | ExecuTorch 1B | 2048 | 0.54 | 0.63 | 0.85 | 0.96 | ″ |
| Samsung S24+ | ExecuTorch 3B | 512 | 0.48 | 0.73 | 0.71 | 0.92 | 0.43, 0.67 |
| Samsung S24+ | ExecuTorch 3B | 2048 | 0.44 | 0.60 | 0.75 | 0.92 | ″ |
| Pixel 7a | llama.cpp 1B | 512 | 0.57 | 0.62 | 0.56 | 0.66 | 0.58, 0.63 |
| Pixel 7a | llama.cpp 1B | 2048 | 0.48 | 0.59 | 0.63 | 0.61 | ″ |
| Pixel 7a | ExecuTorch 1B | 512 | 0.89 | 0.65 | 0.84 | 1.07 | 0.74, 0.99 |
| Pixel 7a | ExecuTorch 1B | 2048 | 0.68 | 0.66 | 0.84 | 1.06 | ″ |

- **The CPU decode always pays.** Next to a GPU prefill it keeps 0.34-0.74 of its rate, next to a GPU decode
  0.20-0.74. Each prefill loses too: the GPU's keeps 0.44-0.89 next to a CPU decode, the CPU's 0.56-0.85 next to
  a GPU decode.
- **The GPU decode depends on the runtime and the partner.** ExecuTorch's Vulkan decode is barely touched by a
  CPU prefill (0.92-1.07), or by a CPU decode with 1B (0.95-0.99). Next to a 3B CPU decode it keeps 0.67.
  llama.cpp's GPU decode loses either way (0.38-0.76).
- **On the Samsung a prefill partner is quieter for the CPU decode than a decode partner; on the Pixel it is
  not.** CPU decode next to a GPU prefill vs next to a GPU decode:
  - Samsung: llama.cpp 1B 0.41 vs 0.21, ExecuTorch 1B 0.74 vs 0.65, ExecuTorch 3B 0.60-0.73 vs 0.43;
  - Pixel: llama.cpp 0.62 vs 0.58, ExecuTorch 0.65 vs 0.74.

  For the GPU decode the picture is mixed. Next to a CPU prefill vs a CPU decode it keeps: ExecuTorch 3B 0.92 vs
  0.67, ExecuTorch 1B 0.96-1.07 vs 0.95-0.99, llama.cpp on the Samsung 0.38-0.61 vs 0.74-0.76.
- **Only ExecuTorch 3B on the Samsung has both conditions.** Each decode is quieter next to a prefill than next to
  the other decode, and data-parallel pays heavily for running two decodes. That is the cell that passes.
  - With 1B ExecuTorch, the GPU decode keeps 0.95-0.99 next to a CPU decode, so data-parallel loses little. The
    split pipeline's critical phase pays in full: the CPU decode next to a GPU prefill (0.63-0.74), or the GPU
    prefill next to a CPU decode (0.54-0.89).
  - With llama.cpp on the Samsung, data-parallel pays heavily (CPU decode 0.20-0.21), but so does the split: the
    GPU prefill keeps 0.45-0.54 next to a CPU decode, the GPU decode 0.38-0.61 next to a CPU prefill.
- **On the Samsung, llama.cpp's CPU loses when the GPU streams.** Next to a GPU decode its CPU decode keeps 0.20-0.21
  and the GPU 0.74-0.76. Its streaming read falls from 36 to 7.6 GB/s while the GPU's falls from 47 to 40
  (section 4). The Pixel splits the loss evenly (0.58, 0.63; reads 24 to 14.5 and 34 to 22.6 GB/s).

## 4. Memory bandwidth

Read bandwidth (GB/s, median over 40 s of 3 trials):

| Phone | CPU alone | GPU alone | CPU, both streaming | GPU, both streaming |
|---|---|---|---|---|
| Samsung S24+ | 36.4 [33.9-36.5] | 47.4 [47.4-48.0] | 7.6 [7.5-7.6] | 40.3 [39.9-40.3] |
| Pixel 7a | 24.0 [24.0-24.0] | 33.7 [33.6-33.7] | 14.5 [14.2-14.6] | 22.6 [22.4-22.6] |

Decode (llama.cpp, depth 0) as a fraction of the bytes-per-token bound of that bandwidth (bytes per token = the
Q4_0 model size, since Llama 3.2 ties the output matrix to the embedding):

| Phone | Model | CPU alone | GPU alone | CPU, both decoding | GPU, both decoding |
|---|---|---|---|---|---|
| Samsung S24+ | 1B | 1.06 | 0.52 | 1.07 | 0.45 |
| Samsung S24+ | 3B | 1.06 | 0.50 | 1.02 | 0.44 |
| Pixel 7a | 1B | 0.94 | 0.37 | 0.90 | 0.35 |

- **The CPU decodes at its bandwidth; the GPU does not.** CPU decode reaches 0.90-1.07 of the bound (above 1 on
  the Samsung because its 40-s read settles below the burst a few-second decode sees: the first second reads
  43 GB/s, design.md). GPU decode reaches 0.35-0.52, so it is limited by per-token submit and kernel cost, not
  by DRAM. This is why the CPU wins decode at short context despite less bandwidth.
- **Co-running, the CPU decode tracks the bandwidth it keeps** (Samsung 11.7 tok/s on 7.6 GB/s, 1.07 of the
  bound; Pixel 0.90). The Samsung's memory system gives the GPU priority.

## 5. Prediction grid

Request throughput (requests/s) by design D5. "Data-parallel, co-run" is the gate's reference; "gain" is the
best specialized pipeline over it. The KV hand-off is copied at the slower PU's alone read bandwidth (1B, P = 2048:
64 MiB of fp16 K and V read and written, 4-8 ms; ExecuTorch writes it as fp32). The last six columns are
diagnostics, not the gate:
- the gain with the specialized pair at its alone rates;
- the gain without the KV hand-off;
- the gain if the two phases slowed each other only while both run: the shorter phase at its co-run rate, the
  longer one at its co-run rate for that long and alone after. This is set against the better of data-parallel
  and the best single PU;
- the gain against a data-parallel bound whose prefill also runs at its co-run rate (design D5 keeps prefill at
  its alone rate, which favours data-parallel);
- ExecuTorch only: the gain with the earlier, duty-extrapolated co-run estimate, which does not meet the spec's
  overlap requirement (design.md);
- for the best pair, the decode co-run ratio used and the one that would reach 1.2.

Every grid uses its own model's co-run rates.

### Samsung S24+ (R5CY21Y3VEV): llamacpp 1b: predicted request throughput (requests/s)

| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run | GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference | gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) | gain vs data-parallel with co-run prefill | gain, previous duty-extrapolated estimate | best pair: decode ratio used / needed for the gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 1.0391 (CPU) | 0.8929 | 1.9018 | 0.9869 | 0.6795 | 0.5912 | **0.69** | 1.67 | 0.69 | 1.07 | 0.79 | — | 0.41 / 0.72 |
| 128 | 128 | 0.3600 (CPU) | 0.2706 | 0.6134 | 0.2761 | 0.1699 | 0.1478 | **0.62** | 1.50 | 0.62 | 1.02 | 0.64 | — | 0.41 / 0.80 |
| 128 | 512 | 0.0996 (CPU) | 0.0714 | 0.1658 | 0.0712 | 0.0425 | 0.0370 | **0.60** | 1.45 | 0.60 | 1.01 | 0.60 | — | 0.41 / 0.83 |
| 512 | 32 | 0.5329 (GPU) | 0.5486 | 0.9892 | 0.6704 | 0.4990 | 0.3652 | **0.74** | 1.60 | 0.74 | 0.84 | 1.05 | — | 0.41 / unreachable |
| 512 | 128 | 0.2410 (CPU) | 0.2268 | 0.4533 | 0.2384 | 0.1576 | 0.1453 | **0.66** | 1.61 | 0.66 | 1.09 | 0.76 | — | 0.41 / 0.75 |
| 512 | 512 | 0.0835 (CPU) | 0.0678 | 0.1458 | 0.0667 | 0.0394 | 0.0363 | **0.59** | 1.44 | 0.59 | 1.03 | 0.62 | — | 0.41 / 0.84 |
| 2048 | 32 | 0.1777 (GPU) | 0.0632 | 0.2769 | 0.2405 | 0.1172 | 0.0715 | **0.49** | 0.90 | 0.49 | 0.70 | 0.76 | — | 0.34 / unreachable |
| 2048 | 128 | 0.1153 (GPU) | 0.0544 | 0.1924 | 0.1359 | 0.0892 | 0.0715 | **0.66** | 1.59 | 0.66 | 0.78 | 0.87 | — | 0.34 / unreachable |
| 2048 | 512 | 0.0479 (GPU) | 0.0349 | 0.0888 | 0.0500 | 0.0223 | 0.0318 | **0.64** | 1.30 | 0.64 | 0.95 | 0.72 | — | 0.52 / 0.98 |

### Samsung S24+ (R5CY21Y3VEV): llamacpp 3b: predicted request throughput (requests/s)

| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run | GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference | gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) | gain vs data-parallel with co-run prefill | gain, previous duty-extrapolated estimate | best pair: decode ratio used / needed for the gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 0.3973 (CPU) | 0.1679 | 0.7161 | 0.3694 | 0.2321 | 0.2477 | **0.67** | 1.74 | 0.67 | 1.03 | 0.77 | — | 0.61 / 1.10 |
| 128 | 128 | 0.1394 (CPU) | 0.0464 | 0.2342 | 0.1042 | 0.0580 | 0.0619 | **0.59** | 1.54 | 0.59 | 1.01 | 0.62 | — | 0.61 / 1.24 |
| 128 | 512 | 0.0388 (CPU) | 0.0119 | 0.0636 | 0.0269 | 0.0145 | 0.0155 | **0.57** | 1.49 | 0.57 | 1.00 | 0.58 | — | 0.61 / 1.28 |
| 512 | 32 | 0.1834 (GPU) | 0.1253 | 0.3339 | 0.2271 | 0.1657 | 0.1306 | **0.73** | 1.52 | 0.73 | 0.75 | 1.02 | — | 0.36 / unreachable |
| 512 | 128 | 0.0784 (CPU) | 0.0426 | 0.1546 | 0.0831 | 0.0442 | 0.0598 | **0.72** | 1.48 | 0.72 | 1.00 | 0.83 | — | 0.61 / 1.02 |
| 512 | 512 | 0.0269 (CPU) | 0.0117 | 0.0497 | 0.0236 | 0.0111 | 0.0150 | **0.64** | 1.30 | 0.64 | 1.02 | 0.66 | — | 0.61 / 1.16 |
| 2048 | 32 | 0.0636 (GPU) | 0.0445 | 0.0965 | 0.0831 | 0.0351 | 0.0252 | **0.42** | 0.93 | 0.42 | 0.66 | 0.72 | — | 0.34 / unreachable |
| 2048 | 128 | 0.0414 (GPU) | 0.0236 | 0.0663 | 0.0469 | 0.0261 | 0.0252 | **0.56** | 1.62 | 0.56 | 0.67 | 0.80 | — | 0.34 / unreachable |
| 2048 | 512 | 0.0173 (GPU) | 0.0082 | 0.0298 | 0.0173 | 0.0065 | 0.0086 | **0.50** | 1.29 | 0.50 | 0.83 | 0.58 | — | 0.38 / 0.93 |

### Samsung S24+ (R5CY21Y3VEV): et 1b: predicted request throughput (requests/s)

| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run | GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference | gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) | gain vs data-parallel with co-run prefill | gain, previous duty-extrapolated estimate | best pair: decode ratio used / needed for the gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 1.1215 (CPU) | — | 2.1339 | 1.7781 | 1.1698 | 1.2774 | **0.72** | 0.89 | 0.72 | 0.80 | 0.77 | 0.59 | 1.02 / 1.70 |
| 128 | 128 | 0.3596 (CPU) | — | 0.6557 | 0.5218 | 0.2925 | 0.3195 | **0.61** | 0.76 | 0.61 | 0.74 | 0.62 | 0.51 | 1.02 / 2.00 |
| 128 | 512 | 0.0967 (CPU) | — | 0.1740 | 0.1364 | 0.0731 | 0.0799 | **0.59** | 0.73 | 0.59 | 0.72 | 0.59 | 0.49 | 1.02 / 2.09 |
| 512 | 32 | 0.6562 (GPU) | — | 1.1783 | 1.0694 | 1.0243 | 0.6600 | **0.96** | 1.30 | 0.96 | 0.97 | 1.12 | 0.84 | 0.74 / unreachable |
| 512 | 128 | 0.2457 (CPU) | — | 0.4873 | 0.4081 | 0.2564 | 0.2919 | **0.72** | 0.85 | 0.72 | 0.78 | 0.76 | 0.57 | 1.02 / 1.71 |
| 512 | 512 | 0.0788 (CPU) | — | 0.1473 | 0.1177 | 0.0641 | 0.0730 | **0.62** | 0.74 | 0.62 | 0.72 | 0.63 | 0.50 | 1.02 / 1.97 |
| 2048 | 32 | 0.2146 (GPU) | — | 0.3315 | 0.3199 | 0.1471 | 0.1166 | **0.46** | 0.85 | 0.46 | 0.68 | 0.66 | 0.50 | 0.63 / unreachable |
| 2048 | 128 | 0.1302 (GPU) | — | 0.2102 | 0.1916 | 0.1200 | 0.1166 | **0.63** | 0.99 | 0.63 | 0.67 | 0.80 | 0.72 | 0.63 / unreachable |
| 2048 | 512 | 0.0506 (GPU) | — | 0.0860 | 0.0736 | 0.0300 | 0.0596 | **0.81** | 0.84 | 0.81 | 0.83 | 0.90 | 0.61 | 0.96 / 1.42 |

### Samsung S24+ (R5CY21Y3VEV): et 3b: predicted request throughput (requests/s)

| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run | GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference | gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) | gain vs data-parallel with co-run prefill | gain, previous duty-extrapolated estimate | best pair: decode ratio used / needed for the gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 0.4440 (GPU) | — | 0.8317 | 0.5243 | 0.4157 | 0.5226 | **1.00** | 1.09 | 1.00 | 1.03 | 1.12 | — | 0.92 / 1.11 |
| 128 | 128 | 0.1323 (GPU) | — | 0.2598 | 0.1490 | 0.1040 | 0.1307 | **0.88** | 0.96 | 0.88 | 0.94 | 0.91 | — | 0.92 / 1.26 |
| 128 | 512 | 0.0347 (GPU) | — | 0.0693 | 0.0386 | 0.0260 | 0.0327 | **0.85** | 0.92 | 0.85 | 0.91 | 0.86 | — | 0.92 / 1.31 |
| 512 | 32 | 0.2790 (GPU) | — | 0.4614 | 0.3426 | 0.2768 | 0.2124 | **0.81** | 1.35 | 0.81 | 0.89 | 1.06 | — | 0.73 / unreachable |
| 512 | 128 | 0.1095 (GPU) | — | 0.1932 | 0.1212 | 0.0848 | 0.1249 | **1.03** | 1.11 | 1.03 | 1.06 | 1.15 | — | 0.92 / 1.08 |
| 512 | 512 | 0.0319 (GPU) | — | 0.0584 | 0.0338 | 0.0212 | 0.0312 | **0.92** | 1.00 | 0.92 | 0.99 | 0.95 | — | 0.92 / 1.20 |
| 2048 | 32 | 0.0766 (GPU) | — | 0.1127 | 0.0994 | 0.0414 | 0.0330 | **0.42** | 0.94 | 0.42 | 0.65 | 0.69 | — | 0.60 / unreachable |
| 2048 | 128 | 0.0495 (GPU) | — | 0.0728 | 0.0546 | 0.0297 | 0.0330 | **0.61** | 0.91 | 0.61 | 0.72 | 0.85 | — | 0.92 / unreachable |
| 2048 | 512 | 0.0205 (GPU) | — | 0.0301 | 0.0196 | 0.0074 | 0.0241 | **1.23** | 1.34 | 1.23 | 1.20 | 1.42 | — | 0.92 / 0.90 |

### Pixel 7a (3A021JEHN02756): llamacpp 1b: predicted request throughput (requests/s)

| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run | GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference | gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) | gain vs data-parallel with co-run prefill | gain, previous duty-extrapolated estimate | best pair: decode ratio used / needed for the gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 0.6331 (CPU) | 0.2370 | 0.9304 | 0.6593 | 0.4019 | 0.3389 | **0.61** | 1.07 | 0.61 | 0.72 | 0.74 | — | 0.62 / unreachable |
| 128 | 128 | 0.2211 (CPU) | 0.0732 | 0.3297 | 0.2086 | 0.1591 | 0.0847 | **0.76** | 1.22 | 0.76 | 0.93 | 0.81 | — | 0.62 / 0.98 |
| 128 | 512 | 0.0614 (CPU) | 0.0194 | 0.0920 | 0.0559 | 0.0398 | 0.0212 | **0.71** | 1.14 | 0.71 | 0.98 | 0.72 | — | 0.62 / 1.05 |
| 512 | 32 | 0.2488 (CPU) | 0.1400 | 0.3991 | 0.3358 | 0.1214 | 0.1930 | **0.57** | 1.02 | 0.57 | 0.71 | 0.84 | — | 0.66 / unreachable |
| 512 | 128 | 0.1363 (CPU) | 0.0620 | 0.2159 | 0.1533 | 0.1214 | 0.0838 | **0.79** | 1.39 | 0.79 | 0.84 | 0.96 | — | 0.62 / unreachable |
| 512 | 512 | 0.0485 (CPU) | 0.0192 | 0.0761 | 0.0483 | 0.0353 | 0.0210 | **0.73** | 1.17 | 0.73 | 0.99 | 0.78 | — | 0.62 / 1.03 |
| 2048 | 32 | 0.0515 (CPU) | 0.0343 | 0.0905 | 0.0858 | 0.0205 | 0.0356 | **0.41** | 0.65 | 0.41 | 0.61 | 0.69 | — | 0.61 / unreachable |
| 2048 | 128 | 0.0411 (CPU) | 0.0249 | 0.0724 | 0.0616 | 0.0205 | 0.0356 | **0.58** | 0.91 | 0.58 | 0.71 | 0.86 | — | 0.61 / unreachable |
| 2048 | 512 | 0.0228 (CPU) | 0.0119 | 0.0402 | 0.0289 | 0.0205 | 0.0180 | **0.71** | 1.33 | 0.71 | 0.77 | 0.87 | — | 0.59 / unreachable |

### Pixel 7a (3A021JEHN02756): et 1b: predicted request throughput (requests/s)

| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run | GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference | gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) | gain vs data-parallel with co-run prefill | gain, previous duty-extrapolated estimate | best pair: decode ratio used / needed for the gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 0.5348 (CPU) | — | 0.9275 | 0.8341 | 0.5977 | 0.4963 | **0.72** | 1.11 | 0.72 | 0.97 | 0.75 | 0.73 | 0.65 / 1.08 |
| 128 | 128 | 0.1954 (CPU) | — | 0.3060 | 0.2605 | 0.1495 | 0.1241 | **0.57** | 0.89 | 0.57 | 0.86 | 0.58 | 0.57 | 0.65 / 1.35 |
| 128 | 512 | 0.0552 (CPU) | — | 0.0838 | 0.0697 | 0.0374 | 0.0310 | **0.54** | 0.83 | 0.54 | 0.82 | 0.54 | 0.53 | 0.65 / 1.45 |
| 512 | 32 | 0.2932 (GPU) | — | 0.5027 | 0.4822 | 0.4964 | 0.2415 | **1.03** | 1.59 | 1.03 | 1.16 | 1.11 | 1.08 | 0.65 / 0.76 |
| 512 | 128 | 0.1152 (CPU) | — | 0.2146 | 0.1937 | 0.1242 | 0.1212 | **0.64** | 0.99 | 0.64 | 0.91 | 0.67 | 0.66 | 0.65 / 1.21 |
| 512 | 512 | 0.0411 (CPU) | — | 0.0684 | 0.0587 | 0.0311 | 0.0303 | **0.53** | 0.82 | 0.53 | 0.80 | 0.54 | 0.54 | 0.65 / 1.47 |
| 2048 | 32 | 0.1031 (GPU) | — | 0.1524 | 0.1501 | 0.0936 | 0.0466 | **0.62** | 0.92 | 0.62 | 0.79 | 0.80 | 0.67 | 0.66 / unreachable |
| 2048 | 128 | 0.0584 (GPU) | — | 0.0948 | 0.0905 | 0.0694 | 0.0466 | **0.77** | 1.16 | 0.77 | 0.84 | 0.90 | 0.97 | 0.66 / unreachable |
| 2048 | 512 | 0.0213 (GPU) | — | 0.0392 | 0.0356 | 0.0174 | 0.0266 | **0.75** | 0.74 | 0.75 | 0.73 | 0.80 | 0.63 | 1.06 / 1.69 |

## 6. Decision

The best cell of every grid. "Gate" says whether the grid uses that model's own co-run rates (all do). The
last three columns are the diagnostics of section 5: the best gain with no interference, the best gain against
data-parallel with co-run prefill, and the best cell's decode ratio used / needed. The other diagnostics are in
the grids.

| Phone | Runtime | Model | Gate | Best P / D | Prefill → decode | Gain | No interference | vs DP, co-run prefill | Decode ratio used / needed |
|---|---|---|---|---|---|---|---|---|---|
| Samsung S24+ | llama.cpp | 1B | yes | 512 / 32 | GPU → CPU | **0.74** | 1.67 | 1.05 | 0.41 / unreachable |
| Samsung S24+ | llama.cpp | 3B | yes | 512 / 32 | GPU → CPU | **0.73** | 1.74 | 1.02 | 0.36 / unreachable |
| Samsung S24+ | ExecuTorch | 1B | yes | 512 / 32 | GPU → CPU | **0.96** | 1.30 | 1.12 | 0.74 / unreachable |
| Samsung S24+ | ExecuTorch | 3B | yes | 2048 / 512 | CPU → GPU | **1.23** | 1.35 | 1.42 | 0.92 / 0.90 |
| Pixel 7a | llama.cpp | 1B | yes | 512 / 128 | GPU → CPU | **0.79** | 1.39 | 0.96 | 0.62 / unreachable |
| Pixel 7a | ExecuTorch | 1B | yes | 512 / 32 | GPU → CPU | **1.03** | 1.59 | 1.11 | 0.65 / 0.76 |

**Go.** One cell reaches the gate: the Samsung S24+ with ExecuTorch, Llama 3.2 3B, P = 2048, D = 512, prefill on
the CPU and decode on the GPU, at 1.23. This is the configuration for the proof of concept:
- **Prefill:** XNNPACK 8da4w on cores 4-9 (mask `3f0`, 6 threads), 90 tok/s alone and 0.75 of that next to a GPU
  decode: 30.3 s per 2048-token prompt.
- **Decode:** Vulkan 4w with the host thread on core 3, 13.4 tok/s alone at depth 2048 and 0.92 of that next to a
  CPU prefill: 41.5 s per 512 tokens. This sets the period, 0.0241 requests/s.
- **Data-parallel bound:** each PU serves whole requests, and both decode at once. Two 3B decodes interfere: CPU
  0.43, GPU 0.67. The bound is 0.0196 requests/s, the GPU contributing 0.0148 and the CPU 0.0047.

**How sure.** Every co-run rate is measured at full overlap:
- 10-11 of 11 GPU decode windows per trial lie inside a CPU prefill, and 10-11 of 15 CPU / 6 of 8 GPU decode
  windows per trial inside the partner's decode.
- Only 1-2 of 3 CPU prefills per trial lie wholly inside a GPU decode window (a 2048-token CPU prefill takes
  about 30 s). Their ratio is consistent, 0.745-0.757. It also does not set the period unless it falls below
  0.55.
- Paired by trial, the gain is 1.20, 1.26 and 1.24. The least favourable ratio of every input across trials
  gives 1.15. The margin over 1.2 is smaller than the trial-to-trial spread.
- The alone decode rate is the one at depth 2048; the 512 tokens decoded here reach depth 2560, the model's
  context limit, where every PU decodes a little slower (design.md, "Decode rate at prompt length P").

**What the gate does not say.** At this cell the co-run data-parallel bound (0.0196) is below the GPU alone
(0.0205). Adding the CPU's decode costs the GPU more than the CPU contributes.
- Against the GPU alone the pipeline is 1.18x. With each phase slowed only while the other runs (column
  "overlap only while both run", against the better of data-parallel and one PU) it is 1.20.
- So the pipeline beats the data-parallel bound the spec defines by 1.23, and the best single PU by just under
  1.2. The proof of concept should report both.
- The same happens in 12 llama.cpp cells, nine on the Samsung, where a GPU decode leaves the CPU decode 0.20-0.21
  of its rate. Their gate gains are at most 0.76.

**Everything else is short of the gate.**
- The next cells are 1.03:
  - ExecuTorch 3B at P = 512, D = 128;
  - the Pixel's ExecuTorch 1B at P = 512, D = 32, where the CPU decode keeps 0.65 next to a GPU prefill (trials
    0.64-0.67, gains 1.01-1.07) and 0.76 would pass.
- The Samsung's ExecuTorch 1B reaches 0.96; llama.cpp at most 0.79 (Pixel) and 0.74 / 0.73 (Samsung 1B / 3B).
- Giving data-parallel prefill its co-run slowdown, or slowing each phase only while the other runs, lifts no
  other cell to 1.2. The largest such value elsewhere is 1.16 (Pixel ExecuTorch 1B, partial overlap).
- For the Pixel's best cell: counting a decode window only inside the partner's measured prefill, not its
  identical warm-up prefill, gives 0.63 instead of 0.65. Every 1B decode ratio moves by at most 0.02.
- The earlier duty-extrapolated ExecuTorch estimate, which did not meet the spec's overlap requirement, put the
  Pixel cell at 1.08. It is kept as a diagnostic column. A first run of the 8-token loops without warm-up was
  discarded, because cold first decode steps dominated it (design.md).

## 7. 8B reference (Samsung, single backend)

Llama 3.1 8B (Q4_0 for llama.cpp, the owner's Vulkan `.pte` files for ExecuTorch) on the Samsung's GPU only:
two copies do not fit in memory, so it has no pipeline and no grid (design Non-Goals). Tokens/s, median
[min-max] of 3 runs; decode at KV depth P.

| Runtime | Config | Prefill at P = 128, 512, 2048 | Decode at P = 128, 512, 2048 |
|---|---|---|---|
| llama.cpp | gpu | 72.1 [71.5-74.7], 68.6 [67.4-69.2], 55.6 [55.5-56.7] | 6.1 [6.1-6.1], 6.0 [6.0-6.0], 5.5 [5.4-5.5] |
| llama.cpp | gpufa0 | 70.9 [70.6-72.2], 68.0 [67.4-68.1], 62.3 [61.9-62.7] | 6.1 [6.1-6.1], 5.9 [5.9-5.9], 5.4 [5.4-5.4] |
| ExecuTorch | vk4w | 81.2 [81.1-81.6], 95.1 [94.9-95.7], 76.1 [75.1-76.4] | 9.1 [9.1-9.1], 8.8 [8.8-8.9], 7.1 [5.7-7.2] |
| ExecuTorch | vk8da4w | 125.9 [122.5-127.4], 132.7 [131.6-133.2], 96.3 [95.5-96.4] | 8.8 [8.8-8.9], 8.6 [8.6-8.6], 6.9 [6.9-7.0] |

- ExecuTorch's Vulkan path beats llama.cpp's at 8B in both phases. Prefill is 1.5-1.9x faster with the 8da4w
  file, decode 1.3-1.5x faster with the 4w file.
- llama.cpp's 8B decode at depth 0 (6.2 tok/s) reaches 0.59 of the bound of the GPU's read bandwidth (10.5
  tok/s), more than its 1B (0.52) or 3B (0.50) does. Per-token overhead weighs less in a larger model.

## 8. Reproduce

```bash
uv run optimizer/analysis/llm_phase_affinity.py --json /tmp/llm-phase-affinity.json   # grids + appendix tables
uv run pytest -q optimizer/tests/test_llm_phase_affinity.py                            # D5 on a hand-computed mix
```

Raw files: `data/llm-phase-affinity/<serial>/` with `alone-llamacpp{,-extra,-8b}/`, `alone-et{,-8b}/`
(`<model>_<config>_{pp,tg|p<P>}_r<k>[_retry].{out,err,clk,meta}`), `corun-llamacpp{,-3b}/`, `corun-et/`
(`<pair>_t<k>[_retry]/{A,B}.{out,err}, corun.{meta,clk}`), `bandwidth/<case>_t<k>[_retry]/`, and each
directory's `driver.log`. The drivers and plan files are in `data/llm-phase-affinity/drivers/`. Every row of the
JSON lists the raw files behind it (`sources`). A `_retry` replaces the run it repeats. The summary tables of
sections 2-4 and 6 are condensed from the grids and the appendix.

## Appendix: full tables

### Samsung S24+ (R5CY21Y3VEV)

#### Read bandwidth (GB/s)

| case | CPU | GPU |
|---|---|---|
| cpu | 36.40 [33.90-36.54] | — |
| gpu | — | 47.39 [47.37-47.96] |
| both | 7.60 [7.51-7.60] | 40.29 [39.89-40.30] |

#### llamacpp: alone rates (tok/s, median [min-max] of 3 runs)

| model | config | mask | pp128 | pp512 | pp2048 | tg@0 | tg@128 | tg@512 | tg@2048 | GPU MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| 1b | cpu180 | 180 | 156.5 [156.5-156.6] | 144.8 [139.1-144.8] | 80.4 [77.6-98.2] | 38.5 [38.5-44.2] | 37.1 [36.6-42.6] | 30.8 [26.8-35.5] | 19.8 [19.8-21.7] | — |
| 1b | cpu3f0 | 3f0 | 358.5 [357.5-374.2] | 332.7 [332.6-338.5] | 224.6 [218.5-224.7] | 55.6 [54.4-55.8] | 52.9 [51.2-53.8] | 49.0 [42.2-49.1] | 33.3 [32.4-33.4] | — |
| 1b | gpu | 8 | 556.2 [551.1-571.0] | 549.7 [549.4-559.2] | 98.0 [89.4-98.4] | 35.2 [34.8-35.2] | 34.4 [34.3-34.9] | 33.9 [33.7-33.9] | 31.5 [31.3-31.7] | 1095 |
| 1b | gpufa0 | 8 | 537.1 [536.4-543.7] | 513.7 [510.5-517.9] | 444.1 [444.0-444.7] | 35.1 [34.6-35.1] | 34.4 [34.2-34.7] | 32.7 [32.7-33.2] | 29.9 [29.9-30.1] | 1095 |
| 1b | partial | 3f0 | 489.6 [480.4-491.3] | 533.0 [532.1-539.0] | 100.4 [93.7-100.6] | 28.1 [16.5-45.8] | 28.4 [20.2-42.4] | 32.3 [26.3-35.3] | 35.5 [32.6-36.1] | 1095 |
| 1b | partialnh | 3f0 | 242.2 [235.6-249.2] | 382.5 [338.1-402.4] | 136.8 [132.9-137.4] | 54.0 [52.9-54.3] | 37.3 [31.3-49.0] | 37.1 [34.6-47.1] | 37.4 [36.5-37.8] | 1095 |
| 3b | cpu180 | 180 | 44.6 [40.3-44.6] | 35.8 [35.1-37.4] | 28.5 [28.4-28.6] | 16.2 [16.1-16.5] | 13.1 [12.5-13.4] | 10.9 [10.7-10.9] | 8.5 [8.5-8.5] | — |
| 3b | cpu3f0 | 3f0 | 132.7 [131.3-134.1] | 111.1 [111.1-111.4] | 75.5 [73.9-75.6] | 21.3 [21.1-21.3] | 20.6 [19.0-20.6] | 15.7 [15.6-16.4] | 9.7 [9.6-9.8] | — |
| 3b | gpu | 8 | 192.0 [183.6-193.5] | 176.7 [173.4-178.4] | 135.4 [131.0-135.7] | 13.1 [13.0-13.1] | 13.0 [12.8-13.0] | 12.5 [12.5-12.6] | 11.4 [11.2-11.4] | 1095 |
| 3b | gpufa0 | 8 | 181.9 [180.8-184.3] | 176.4 [175.2-179.4] | 158.4 [157.6-164.3] | 13.0 [12.9-13.0] | 12.9 [12.8-12.9] | 12.2 [12.2-12.4] | 10.9 [10.8-11.0] | 1095 |
| 3b | partial | 3f0 | 169.8 [169.0-171.3] | 181.5 [177.0-182.8] | 129.1 [128.6-130.8] | 7.4 [4.2-8.7] | 6.1 [5.9-6.4] | 5.3 [4.9-5.5] | 4.7 [4.7-4.8] | 1095 |
| 3b | partialnh | 3f0 | 154.8 [154.2-155.4] | 138.2 [131.1-138.4] | 94.8 [94.4-95.4] | 7.2 [7.0-8.0] | 5.8 [2.3-5.8] | 6.2 [5.3-6.4] | 4.8 [4.7-4.9] | 1095 |
| 8b | gpu | 8 | 72.1 [71.5-74.7] | 68.6 [67.4-69.2] | 55.6 [55.5-56.7] | 6.2 [6.2-6.2] | 6.1 [6.1-6.1] | 6.0 [6.0-6.0] | 5.5 [5.4-5.5] | 1095 |
| 8b | gpufa0 | 8 | 70.9 [70.6-72.2] | 68.0 [67.4-68.1] | 62.3 [61.9-62.7] | 6.1 [6.1-6.1] | 6.1 [6.1-6.1] | 5.9 [5.9-5.9] | 5.4 [5.4-5.4] | 1095 |

#### llamacpp: phase affinity (tok/s; CPU = wider set)

| model | P | prefill CPU | prefill GPU | GPU / CPU | decode CPU | decode GPU | GPU / CPU |
|---|---|---|---|---|---|---|---|
| 1b | 128 | 358.5 [357.5-374.2] | 556.2 [551.1-571.0] | 1.55 | 52.9 [51.2-53.8] | 34.4 [34.2-34.7] | 0.65 |
| 1b | 512 | 332.7 [332.6-338.5] | 549.7 [549.4-559.2] | 1.65 | 49.0 [42.2-49.1] | 33.9 [33.7-33.9] | 0.69 |
| 1b | 2048 | 224.6 [218.5-224.7] | 444.1 [444.0-444.7] | 1.98 | 33.3 [32.4-33.4] | 31.5 [31.3-31.7] | 0.95 |
| 3b | 128 | 132.7 [131.3-134.1] | 192.0 [183.6-193.5] | 1.45 | 20.6 [19.0-20.6] | 13.0 [12.8-13.0] | 0.63 |
| 3b | 512 | 111.1 [111.1-111.4] | 176.7 [173.4-178.4] | 1.59 | 15.7 [15.6-16.4] | 12.5 [12.5-12.6] | 0.80 |
| 3b | 2048 | 75.5 [73.9-75.6] | 158.4 [157.6-164.3] | 2.10 | 9.7 [9.6-9.8] | 11.4 [11.2-11.4] | 1.18 |

#### llamacpp 1b: co-run (tok/s; ratio = co-run / alone)

| pair | P | A: phase on PU | A co-run | A alone | A ratio | B: phase on PU | B co-run | B alone | B ratio | kept reps A / B per trial |
|---|---|---|---|---|---|---|---|---|---|---|
| cpuD_gpuD | — | decode on CPU | 11.7 [11.7-11.8] | 55.6 [54.4-55.8] | 0.21 | decode on GPU | 26.2 [26.1-26.2] | 35.2 [34.8-35.2] | 0.74 | 11/27 · 27/27, 11/27 · 27/27, 11/27 · 27/27 |
| cpuP_gpuD_p2048 | 2048 | prefill on CPU | 146.6 [144.6-147.9] | 224.6 [218.5-224.7] | 0.65 | decode on GPU | 18.2 [17.0-19.3] | 35.2 [34.8-35.2] | 0.52 | 7/7 · 13/31, 7/7 · 14/31, 7/7 · 13/31 |
| cpuP_gpuD_p512 | 512 | prefill on CPU | 187.1 [185.1-187.4] | 332.7 [332.6-338.5] | 0.56 | decode on GPU | 19.3 [19.1-21.3] | 35.2 [34.8-35.2] | 0.55 | 36/39 · 15/27, 37/39 · 15/27, 37/39 · 15/27 |
| gpuP_cpuD_p2048 | 2048 | prefill on GPU | 240.2 [223.5-253.0] | 444.1 [444.0-444.7] | 0.54 | decode on CPU | 19.0 [18.1-19.6] | 55.6 [54.4-55.8] | 0.34 | 13/13 · 14/44, 13/13 · 16/44, 13/13 · 17/44 |
| gpuP_cpuD_p512 | 512 | prefill on GPU | 255.6 [249.5-264.4] | 549.7 [549.4-559.2] | 0.46 | decode on CPU | 22.9 [22.7-23.0] | 55.6 [54.4-55.8] | 0.41 | 65/65 · 22/42, 65/65 · 21/42, 65/65 · 23/42 |

#### llamacpp 3b: co-run (tok/s; ratio = co-run / alone)

| pair | P | A: phase on PU | A co-run | A alone | A ratio | B: phase on PU | B co-run | B alone | B ratio | kept reps A / B per trial |
|---|---|---|---|---|---|---|---|---|---|---|
| cpuD_gpuD | — | decode on CPU | 4.3 [4.3-4.4] | 21.3 [21.1-21.3] | 0.20 | decode on GPU | 9.9 [9.6-10.0] | 13.1 [13.0-13.1] | 0.76 | 4/10 · 10/10, 4/10 · 9/10, 4/10 · 10/10 |
| cpuP_gpuD_p2048 | 2048 | prefill on CPU | 51.6 [51.3-51.9] | 75.5 [73.9-75.6] | 0.68 | decode on GPU | 5.0 [5.0-5.2] | 13.1 [13.0-13.1] | 0.38 | 3/3 · 4/16, 3/3 · 4/16, 3/3 · 4/16 |
| cpuP_gpuD_p512 | 512 | prefill on CPU | 66.9 [66.5-67.8] | 111.1 [111.1-111.4] | 0.60 | decode on GPU | 8.0 [7.5-8.1] | 13.1 [13.0-13.1] | 0.61 | 13/14 · 5/11, 13/14 · 6/11, 13/14 · 6/11 |
| gpuP_cpuD_p2048 | 2048 | prefill on GPU | 72.0 [69.8-75.3] | 158.4 [157.6-164.3] | 0.45 | decode on CPU | 7.3 [7.2-7.8] | 21.3 [21.1-21.3] | 0.34 | 5/5 · 7/20, 5/5 · 8/20, 5/5 · 7/20 |
| gpuP_cpuD_p512 | 512 | prefill on GPU | 84.9 [84.3-90.8] | 176.7 [173.4-178.4] | 0.48 | decode on CPU | 7.7 [7.1-7.8] | 21.3 [21.1-21.3] | 0.36 | 21/21 · 6/17, 21/21 · 7/17, 21/21 · 7/17 |

#### Decode against the bandwidth bound

| model | PU | bandwidth GB/s | bound tok/s | decode tok/s | fraction |
|---|---|---|---|---|---|
| 1b | CPU alone | 36.40 [33.90-36.54] | 52.3 | 55.6 [54.4-55.8] | 1.06 |
| 1b | GPU alone | 47.39 [47.37-47.96] | 68.1 | 35.2 [34.8-35.2] | 0.52 |
| 1b | CPU both decoding | 7.60 [7.51-7.60] | 10.9 | 11.7 [11.7-11.8] | 1.07 |
| 1b | GPU both decoding | 40.29 [39.89-40.30] | 57.9 | 26.2 [26.1-26.2] | 0.45 |
| 3b | CPU alone | 36.40 [33.90-36.54] | 20.1 | 21.3 [21.1-21.3] | 1.06 |
| 3b | GPU alone | 47.39 [47.37-47.96] | 26.2 | 13.1 [13.0-13.1] | 0.50 |
| 8b | GPU alone | 47.39 [47.37-47.96] | 10.5 | 6.2 [6.2-6.2] | 0.59 |

#### et: alone rates (tok/s, median [min-max] of 3 runs)

| model | config | mask | pp128 | pp512 | pp2048 | tg@0 | tg@128 | tg@512 | tg@2048 | GPU MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| 1b | vk4w | 8 | 492.3 [479.4-492.3] | 577.2 [577.2-595.3] | 458.3 [456.6-459.7] | — | 40.2 [37.3-48.6] | 36.7 [36.5-37.2] | 31.8 [31.4-32.1] | 1095 |
| 1b | vk8da4w | 8 | 670.2 [636.8-727.3] | 785.3 [767.6-810.1] | 560.9 [553.2-565.6] | — | 36.0 [36.0-47.2] | 35.1 [34.9-37.1] | 29.9 [29.1-30.1] | 1095 |
| 1b | xnn180 | 180 | 203.8 [203.5-211.2] | 193.1 [179.1-193.2] | 101.3 [101.1-102.8] | — | 41.7 [41.5-42.0] | 34.2 [30.5-34.5] | 14.4 [14.3-15.0] | — |
| 1b | xnn3f0 | 3f0 | 488.5 [484.8-496.1] | 427.7 [385.3-435.7] | 282.5 [279.6-332.7] | — | 50.8 [50.8-51.1] | 44.5 [44.5-44.9] | 24.4 [24.3-25.6] | — |
| 3b | vk4w | 8 | 174.6 [169.8-178.8] | 221.0 [217.5-221.5] | 157.9 [157.7-158.3] | — | 18.1 [17.9-18.2] | 17.3 [17.2-17.5] | 13.4 [13.4-13.5] | 1095 |
| 3b | vk8da4w | 8 | 264.5 [261.8-279.5] | 295.3 [293.7-298.0] | 192.0 [191.7-193.6] | — | 17.6 [17.3-17.7] | 16.7 [16.7-16.8] | 13.0 [12.9-13.1] | 1095 |
| 3b | xnn180 | 180 | 57.3 [52.8-57.8] | 48.7 [48.6-48.7] | 37.9 [37.8-38.0] | — | 12.1 [12.0-12.1] | 9.6 [9.6-9.6] | 4.5 [4.5-4.6] | — |
| 3b | xnn3f0 | 3f0 | 155.3 [149.7-161.4] | 153.7 [143.8-157.0] | 90.2 [88.9-93.7] | — | 18.2 [15.9-18.7] | 14.9 [14.7-15.4] | 6.3 [6.3-6.4] | — |
| 8b | vk4w | 8 | 81.2 [81.1-81.6] | 95.1 [94.9-95.7] | 76.1 [75.1-76.4] | — | 9.1 [9.1-9.1] | 8.8 [8.8-8.9] | 7.1 [5.7-7.2] | 1095 |
| 8b | vk8da4w | 8 | 125.9 [122.5-127.4] | 132.7 [131.6-133.2] | 96.3 [95.5-96.4] | — | 8.8 [8.8-8.9] | 8.6 [8.6-8.6] | 6.9 [6.9-7.0] | 1095 |

#### et: phase affinity (tok/s; CPU = wider set)

| model | P | prefill CPU | prefill GPU | GPU / CPU | decode CPU | decode GPU | GPU / CPU |
|---|---|---|---|---|---|---|---|
| 1b | 128 | 488.5 [484.8-496.1] | 670.2 [636.8-727.3] | 1.37 | 50.8 [50.8-51.1] | 40.2 [37.3-48.6] | 0.79 |
| 1b | 512 | 427.7 [385.3-435.7] | 785.3 [767.6-810.1] | 1.84 | 44.5 [44.5-44.9] | 36.7 [36.5-37.2] | 0.82 |
| 1b | 2048 | 282.5 [279.6-332.7] | 560.9 [553.2-565.6] | 1.99 | 24.4 [24.3-25.6] | 31.8 [31.4-32.1] | 1.30 |
| 3b | 128 | 155.3 [149.7-161.4] | 264.5 [261.8-279.5] | 1.70 | 18.2 [15.9-18.7] | 18.1 [17.9-18.2] | 0.99 |
| 3b | 512 | 153.7 [143.8-157.0] | 295.3 [293.7-298.0] | 1.92 | 14.9 [14.7-15.4] | 17.3 [17.2-17.5] | 1.16 |
| 3b | 2048 | 90.2 [88.9-93.7] | 192.0 [191.7-193.6] | 2.13 | 6.3 [6.3-6.4] | 13.4 [13.4-13.5] | 2.12 |

#### et 1b: co-run (tok/s; ratio = co-run / solo loop)

| loop | A rate, counted | A counted / total per trial | B rate, counted | B counted / total per trial | A, B rate over all intervals (duty) |
|---|---|---|---|---|---|
| cpuD_gpuD | — | 0/6, 0/5, 0/6 | — | 0/5, 0/5, 0/5 | 22.4 [22.2-23.1] (0.78 [0.76-0.79]); 21.0 [20.9-22.6] (0.88 [0.74-0.89]) |
| cpuDs_gpuD | 30.8 [30.8-30.9] | 30/42, 30/41, 30/41 | — | 0/7, 0/7, 0/7 | 32.9 [32.1-33.2] (0.78 [0.77-0.78]); 34.8 [34.8-34.9] (0.18 [0.18-0.18]) |
| cpuP_gpuD_p2048 | 226.6 [223.1-258.9] | 4/5, 3/5, 2/5 | — | 0/5, 0/5, 0/5 | 219.5 [218.9-224.4] (0.88 [0.88-0.89]); 24.8 [24.6-25.8] (0.71 [0.71-0.72]) |
| cpuP_gpuD_p512 | 293.7 [293.1-294.2] | 12/19, 13/18, 13/18 | — | 0/5, 0/6, 0/6 | 303.2 [300.3-303.6] (0.75 [0.75-0.79]); 27.7 [26.2-27.8] (0.44 [0.44-0.54]) |
| cpuP_gpuDs_p2048 | — | 0/6, 0/6, 0/6 | 32.8 [32.2-36.9] | 15/16, 15/15, 16/16 | 251.9 [250.5-254.6] (0.16 [0.16-0.16]); 33.1 [32.2-36.9] (1.00 [0.94-1.00]) |
| cpuP_gpuDs_p512 | — | 0/22, 0/22, 0/21 | 34.8 [33.3-35.8] | 10/17, 10/17, 10/17 | 353.5 [353.2-358.2] (0.24 [0.21-0.27]); 35.2 [32.9-35.9] (0.65 [0.65-0.67]) |
| gpuDs_cpuD | 32.4 [31.9-33.0] | 10/12, 9/12, 10/12 | — | 0/7, 0/7, 0/7 | 33.3 [33.0-33.3] (0.85 [0.76-0.87]); 32.8 [32.4-33.0] (0.17 [0.17-0.19]) |
| gpuP_cpuD_p2048 | 302.2 [286.3-305.6] | 5/6, 4/6, 4/6 | — | 0/6, 0/6, 0/6 | 293.9 [273.5-308.3] (0.93 [0.91-0.96]); 25.0 [24.5-25.2] (0.63 [0.61-0.64]) |
| gpuP_cpuD_p512 | 547.3 [547.0-562.3] | 13/14, 13/15, 10/15 | — | 0/7, 0/7, 0/7 | 529.5 [520.2-537.4] (0.92 [0.82-0.99]); 30.2 [30.1-30.6] (0.21 [0.18-0.22]) |
| gpuP_cpuDs_p2048 | — | 0/9, 0/8, 0/8 | 30.0 [29.9-31.5] | 25/42, 25/40, 27/41 | 451.1 [450.0-462.6] (0.22 [0.19-0.23]); 32.3 [31.9-34.0] (0.70 [0.65-0.73]) |
| gpuP_cpuDs_p512 | — | 0/21, 0/21, 0/21 | 35.2 [35.0-35.5] | 12/50, 11/50, 13/49 | 761.9 [759.4-768.8] (0.27 [0.23-0.32]); 41.9 [41.1-42.0] (0.33 [0.32-0.35]) |
| solo_cpuD | 33.4 [33.3-33.6] | 8/8, 8/8, 8/8 | — | — | 33.4 [33.3-33.6] (0.00 [0.00-0.00]) |
| solo_cpuDs | 47.7 [47.6-47.9] | 48/48, 49/49, 48/48 | — | — | 47.7 [47.6-47.9] (0.00 [0.00-0.00]) |
| solo_cpuP_p2048 | 268.0 [267.1-268.0] | 6/6, 6/6, 6/6 | — | — | 268.0 [267.1-268.0] (0.00 [0.00-0.00]) |
| solo_cpuP_p512 | 371.4 [357.1-373.5] | 22/22, 22/22, 21/21 | — | — | 371.4 [357.1-373.5] (0.00 [0.00-0.00]) |
| solo_gpuD | 35.5 [35.4-35.5] | 7/7, 7/7, 7/7 | — | — | 35.5 [35.4-35.5] (0.00 [0.00-0.00]) |
| solo_gpuDs | 34.1 [33.7-35.2] | 16/16, 16/16, 16/16 | — | — | 34.1 [33.7-35.2] (0.00 [0.00-0.00]) |
| solo_gpuP_p2048 | 562.3 [561.2-565.2] | 9/9, 9/9, 9/9 | — | — | 562.3 [561.2-565.2] (0.00 [0.00-0.00]) |
| solo_gpuP_p512 | 792.4 [789.0-795.4] | 19/19, 19/19, 19/19 | — | — | 792.4 [789.0-795.4] (0.00 [0.00-0.00]) |

#### et 1b: co-run ratios

| pair | P | prefill ratio | prefills counted per trial | decode ratio | decode windows counted per trial | previous estimate: decode ratio (duty) |
|---|---|---|---|---|---|---|
| GPU prefill → CPU decode | 512 | 0.69 | 13/14, 13/15, 10/15 | 0.74 | 12/50, 11/50, 13/49 | 0.55 (0.21) |
| CPU prefill → GPU decode | 512 | 0.79 | 12/19, 13/18, 13/18 | 1.02 | 10/17, 10/17, 10/17 | 0.51 (0.44) |
| GPU prefill → CPU decode | 2048 | 0.54 | 5/6, 4/6, 4/6 | 0.63 | 25/42, 25/40, 27/41 | 0.60 (0.63) |
| CPU prefill → GPU decode | 2048 | 0.85 | 4/5, 3/5, 2/5 | 0.96 | 15/16, 15/15, 16/16 | 0.58 (0.71) |
| both decoding (CPU, GPU) | — | — | — | 0.65, 0.95 | 30/42, 30/41, 30/41; 10/12, 9/12, 10/12 | 0.67, 0.59 (as measured) |

#### et 3b: co-run (tok/s; ratio = co-run / solo loop)

| loop | A rate, counted | A counted / total per trial | B rate, counted | B counted / total per trial | A, B rate over all intervals (duty) |
|---|---|---|---|---|---|
| cpuDs_gpuD | 8.1 [7.8-8.6] | 10/15, 11/15, 11/15 | — | 0/5, 0/5, 0/5 | 9.1 [8.8-9.4] (0.82 [0.79-0.84]); 14.6 [14.4-14.7] (0.12 [0.12-0.14]) |
| cpuP_gpuD_p2048 | 68.2 [67.7-68.8] | 2/3, 1/3, 1/3 | 14.7 [12.2-15.4] | 1/4, 1/4, 1/4 | 72.0 [69.3-72.5] (0.90 [0.90-0.95]); 10.1 [9.9-10.2] (0.75 [0.75-0.76]) |
| cpuP_gpuD_p512 | 92.9 [91.9-95.0] | 8/10, 9/10, 8/10 | — | 0/4, 0/4, 0/4 | 95.3 [94.6-95.7] (0.90 [0.90-0.91]); 10.8 [10.7-11.0] (0.49 [0.49-0.50]) |
| cpuP_gpuDs_p2048 | — | 0/4, 0/4, 0/4 | 16.7 [16.7-17.5] | 11/11, 10/11, 10/11 | 88.3 [86.8-89.9] (0.14 [0.13-0.15]); 16.7 [15.0-17.5] (0.93 [0.93-1.00]) |
| cpuP_gpuDs_p512 | — | 0/12, 0/12, 0/13 | 16.8 [16.7-17.2] | 7/12, 7/11, 8/12 | 117.3 [116.8-118.4] (0.14 [0.13-0.14]); 17.2 [17.1-17.3] (0.70 [0.69-0.77]) |
| gpuDs_cpuD | 12.2 [11.9-13.3] | 6/8, 6/8, 6/8 | — | 0/5, 0/5, 0/5 | 11.8 [11.5-13.0] (0.73 [0.72-0.73]); 11.8 [11.7-11.8] (0.08 [0.08-0.08]) |
| gpuP_cpuD_p2048 | 84.8 [79.6-88.9] | 3/3, 3/3, 3/3 | — | 0/4, 0/4, 0/4 | 84.8 [79.6-88.9] (1.00 [1.00-1.00]); 9.6 [9.5-9.8] (0.62 [0.61-0.62]) |
| gpuP_cpuD_p512 | 146.3 [143.6-146.7] | 9/10, 10/11, 9/11 | — | 0/4, 0/4, 0/4 | 141.3 [141.3-143.2] (0.90 [0.89-0.93]); 10.5 [10.5-10.7] (0.38 [0.34-0.38]) |
| gpuP_cpuDs_p2048 | — | 0/5, 0/5, 0/5 | 11.3 [11.3-11.7] | 13/16, 12/16, 11/16 | 140.8 [135.3-147.4] (0.09 [0.07-0.15]); 12.0 [11.6-12.4] (0.82 [0.73-0.86]) |
| gpuP_cpuDs_p512 | — | 0/17, 0/17, 0/17 | 13.8 [13.7-13.8] | 4/19, 5/18, 6/19 | 294.5 [294.5-294.6] (0.20 [0.19-0.21]); 14.1 [14.0-14.1] (0.33 [0.25-0.38]) |
| solo_cpuDs | 18.9 [18.8-19.1] | 19/19, 20/20, 20/20 | — | — | 18.9 [18.8-19.1] (0.00 [0.00-0.00]) |
| solo_cpuP_p2048 | 90.9 [90.8-91.8] | 4/4, 4/4, 4/4 | — | — | 90.9 [90.8-91.8] (0.00 [0.00-0.00]) |
| solo_cpuP_p512 | 131.2 [130.8-133.1] | 13/13, 13/13, 13/13 | — | — | 131.2 [130.8-133.1] (0.00 [0.00-0.00]) |
| solo_gpuDs | 18.2 [18.2-18.3] | 12/12, 12/12, 12/12 | — | — | 18.2 [18.2-18.3] (0.00 [0.00-0.00]) |
| solo_gpuP_p2048 | 191.9 [191.2-192.8] | 6/6, 6/6, 6/6 | — | — | 191.9 [191.2-192.8] (0.00 [0.00-0.00]) |
| solo_gpuP_p512 | 304.3 [303.8-304.5] | 18/18, 18/18, 19/19 | — | — | 304.3 [303.8-304.5] (0.00 [0.00-0.00]) |

#### et 3b: co-run ratios

| pair | P | prefill ratio | prefills counted per trial | decode ratio | decode windows counted per trial | previous estimate: decode ratio (duty) |
|---|---|---|---|---|---|---|
| GPU prefill → CPU decode | 512 | 0.48 | 9/10, 10/11, 9/11 | 0.73 | 4/19, 5/18, 6/19 | — |
| CPU prefill → GPU decode | 512 | 0.71 | 8/10, 9/10, 8/10 | 0.92 | 7/12, 7/11, 8/12 | — |
| GPU prefill → CPU decode | 2048 | 0.44 | 3/3, 3/3, 3/3 | 0.60 | 13/16, 12/16, 11/16 | — |
| CPU prefill → GPU decode | 2048 | 0.75 | 2/3, 1/3, 1/3 | 0.92 | 11/11, 10/11, 10/11 | — |
| both decoding (CPU, GPU) | — | — | — | 0.43, 0.67 | 10/15, 11/15, 11/15; 6/8, 6/8, 6/8 | — |

### Pixel 7a (3A021JEHN02756)

#### Read bandwidth (GB/s)

| case | CPU | GPU |
|---|---|---|
| cpu | 24.02 [24.02-24.02] | — |
| gpu | — | 33.66 [33.64-33.67] |
| both | 14.49 [14.22-14.62] | 22.61 [22.37-22.64] |

#### llamacpp: alone rates (tok/s, median [min-max] of 3 runs)

| model | config | mask | pp128 | pp512 | pp2048 | tg@0 | tg@128 | tg@512 | tg@2048 | GPU MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| 1b | cpuc0 | c0 | 138.7 [137.7-144.8] | 118.7 [117.6-126.8] | 78.9 [77.8-91.0] | 33.3 [33.0-33.7] | 31.3 [31.1-31.5] | 26.6 [23.5-27.4] | 17.6 [17.5-20.0] | — |
| 1b | cpuf0 | f0 | 213.8 [211.8-215.0] | 175.7 [175.0-177.4] | 115.0 [114.2-118.9] | 32.4 [27.1-32.8] | 32.6 [32.5-32.6] | 28.9 [26.4-29.1] | 19.7 [19.6-19.8] | — |
| 1b | gpu | 8 | 90.5 [90.4-90.7] | 109.3 [108.2-109.8] | 72.5 [72.3-73.1] | 17.8 [17.7-17.8] | 16.4 [16.4-16.4] | 16.2 [16.2-16.4] | 15.1 [15.1-15.1] | 848 |
| 1b | gpufa0 | 8 | 86.9 [86.4-87.2] | 100.5 [100.5-100.9] | 87.1 [87.1-87.4] | 17.6 [17.6-17.7] | 16.4 [16.3-16.5] | 15.5 [15.5-15.6] | 14.7 [14.7-14.8] | 848 |
| 1b | partial | f0 | 80.3 [79.9-80.4] | 103.4 [103.3-103.4] | 69.1 [69.0-69.6] | 1.7 [1.7-1.8] | 2.7 [2.6-2.7] | 2.9 [2.8-2.9] | 2.7 [2.7-2.7] | 740 |
| 1b | partialnh | f0 | 119.4 [119.1-120.2] | 123.4 [123.1-125.0] | 80.4 [79.8-80.9] | 10.2 [9.3-10.9] | 10.2 [10.0-10.2] | 10.7 [10.1-10.8] | 8.7 [8.5-9.0] | 618 |

#### llamacpp: phase affinity (tok/s; CPU = wider set)

| model | P | prefill CPU | prefill GPU | GPU / CPU | decode CPU | decode GPU | GPU / CPU |
|---|---|---|---|---|---|---|---|
| 1b | 128 | 213.8 [211.8-215.0] | 90.5 [90.4-90.7] | 0.42 | 32.6 [32.5-32.6] | 16.4 [16.4-16.4] | 0.50 |
| 1b | 512 | 175.7 [175.0-177.4] | 109.3 [108.2-109.8] | 0.62 | 28.9 [26.4-29.1] | 16.2 [16.2-16.4] | 0.56 |
| 1b | 2048 | 115.0 [114.2-118.9] | 87.1 [87.1-87.4] | 0.76 | 19.7 [19.6-19.8] | 15.1 [15.1-15.1] | 0.77 |

#### llamacpp 1b: co-run (tok/s; ratio = co-run / alone)

| pair | P | A: phase on PU | A co-run | A alone | A ratio | B: phase on PU | B co-run | B alone | B ratio | kept reps A / B per trial |
|---|---|---|---|---|---|---|---|---|---|---|
| cpuD_gpuD | — | decode on CPU | 18.8 [18.8-18.8] | 32.4 [27.1-32.8] | 0.58 | decode on GPU | 11.3 [11.2-11.4] | 17.8 [17.7-17.8] | 0.63 | 13/16 · 7/14, 13/16 · 7/14, 13/16 · 7/14 |
| cpuP_gpuD_p2048 | 2048 | prefill on CPU | 72.9 [72.6-74.3] | 115.0 [114.2-118.9] | 0.63 | decode on GPU | 10.8 [10.6-10.9] | 17.8 [17.7-17.8] | 0.61 | 4/4 · 9/19, 4/4 · 9/19, 4/4 · 8/19 |
| cpuP_gpuD_p512 | 512 | prefill on CPU | 98.8 [97.0-101.5] | 175.7 [175.0-177.4] | 0.56 | decode on GPU | 11.7 [11.1-12.1] | 17.8 [17.7-17.8] | 0.66 | 19/21 · 8/14, 19/21 · 9/14, 19/21 · 9/14 |
| gpuP_cpuD_p2048 | 2048 | prefill on GPU | 41.9 [39.1-44.7] | 87.1 [87.1-87.4] | 0.48 | decode on CPU | 19.2 [19.0-19.3] | 32.4 [27.1-32.8] | 0.59 | 3/3 · 22/35, 3/3 · 20/35, 3/3 · 21/35 |
| gpuP_cpuD_p512 | 512 | prefill on GPU | 62.1 [59.7-64.0] | 109.3 [108.2-109.8] | 0.57 | decode on CPU | 20.2 [19.6-21.1] | 32.4 [27.1-32.8] | 0.62 | 13/13 · 17/26, 13/13 · 16/26, 13/13 · 16/26 |

#### Decode against the bandwidth bound

| model | PU | bandwidth GB/s | bound tok/s | decode tok/s | fraction |
|---|---|---|---|---|---|
| 1b | CPU alone | 24.02 [24.02-24.02] | 34.5 | 32.4 [27.1-32.8] | 0.94 |
| 1b | GPU alone | 33.66 [33.64-33.67] | 48.4 | 17.8 [17.7-17.8] | 0.37 |
| 1b | CPU both decoding | 14.49 [14.22-14.62] | 20.8 | 18.8 [18.8-18.8] | 0.90 |
| 1b | GPU both decoding | 22.61 [22.37-22.64] | 32.5 | 11.3 [11.2-11.4] | 0.35 |

#### et: alone rates (tok/s, median [min-max] of 3 runs)

| model | config | mask | pp128 | pp512 | pp2048 | tg@0 | tg@128 | tg@512 | tg@2048 | GPU MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| 1b | vk4w | 8 | 132.8 [129.3-134.0] | 168.0 [166.9-168.1] | 143.7 [143.5-143.9] | — | 14.8 [14.6-14.8] | 14.5 [14.2-14.5] | 12.9 [12.9-13.0] | 848 |
| 1b | vk8da4w | 8 | 334.2 [333.3-340.4] | 428.1 [426.7-439.9] | 283.9 [282.4-284.2] | — | 14.7 [14.7-14.9] | 14.4 [14.1-14.4] | 12.9 [12.8-13.2] | 848 |
| 1b | xnnc0 | c0 | 104.8 [102.6-105.5] | 99.3 [98.8-108.6] | 78.3 [76.5-78.3] | — | 28.4 [27.6-28.6] | 23.3 [23.2-25.8] | 11.8 [10.7-12.3] | — |
| 1b | xnnf0 | f0 | 162.6 [160.0-163.1] | 147.5 [146.7-149.6] | 114.2 [114.0-114.5] | — | 29.5 [29.0-29.9] | 24.6 [24.4-24.6] | 13.5 [13.3-13.5] | — |

#### et: phase affinity (tok/s; CPU = wider set)

| model | P | prefill CPU | prefill GPU | GPU / CPU | decode CPU | decode GPU | GPU / CPU |
|---|---|---|---|---|---|---|---|
| 1b | 128 | 162.6 [160.0-163.1] | 334.2 [333.3-340.4] | 2.05 | 29.5 [29.0-29.9] | 14.8 [14.6-14.8] | 0.50 |
| 1b | 512 | 147.5 [146.7-149.6] | 428.1 [426.7-439.9] | 2.90 | 24.6 [24.4-24.6] | 14.5 [14.2-14.5] | 0.59 |
| 1b | 2048 | 114.2 [114.0-114.5] | 283.9 [282.4-284.2] | 2.49 | 13.5 [13.3-13.5] | 12.9 [12.9-13.0] | 0.96 |

#### et 1b: co-run (tok/s; ratio = co-run / solo loop)

| loop | A rate, counted | A counted / total per trial | B rate, counted | B counted / total per trial | A, B rate over all intervals (duty) |
|---|---|---|---|---|---|
| cpuD_gpuD | — | 0/5, 0/5, 0/5 | — | 0/3, 0/3, 0/3 | 19.1 [19.1-19.6] (0.86 [0.86-0.87]); 9.6 [9.5-9.6] (0.72 [0.69-0.73]) |
| cpuDs_gpuD | 26.3 [26.2-26.6] | 24/31, 25/31, 24/31 | — | 0/3, 0/3, 0/3 | 27.0 [26.8-27.2] (0.82 [0.79-0.83]); 13.3 [12.9-13.9] (0.16 [0.15-0.16]) |
| cpuP_gpuD_p2048 | 84.1 [83.5-84.9] | 2/2, 2/2, 2/2 | — | 0/3, 0/3, 0/3 | 84.1 [83.5-84.9] (1.00 [1.00-1.00]); 11.2 [11.1-11.2] (0.56 [0.56-0.57]) |
| cpuP_gpuD_p512 | 118.1 [117.7-121.5] | 6/9, 6/9, 6/9 | — | 0/3, 0/3, 0/3 | 119.5 [119.1-121.9] (0.83 [0.82-0.87]); 11.6 [11.5-11.7] (0.49 [0.48-0.50]) |
| cpuP_gpuDs_p2048 | — | 0/3, 0/3, 0/3 | 15.3 [15.2-15.4] | 9/10, 10/10, 9/10 | 91.7 [90.8-93.2] (0.19 [0.19-0.20]); 15.2 [15.2-15.4] (0.99 [0.97-1.00]) |
| cpuP_gpuDs_p512 | — | 0/10, 0/9, 0/10 | 15.5 [15.2-15.8] | 8/11, 9/10, 10/11 | 132.8 [130.2-133.6] (0.22 [0.22-0.34]); 15.4 [15.2-15.8] (0.90 [0.73-0.91]) |
| gpuDs_cpuD | 14.2 [14.1-14.3] | 8/8, 8/8, 8/8 | — | 0/5, 0/5, 0/5 | 14.2 [14.1-14.3] (1.00 [1.00-1.00]); 23.2 [23.0-23.2] (0.26 [0.26-0.27]) |
| gpuP_cpuD_p2048 | 192.0 [187.2-201.6] | 3/4, 3/4, 3/4 | — | 0/5, 0/5, 0/5 | 162.1 [159.7-181.3] (0.92 [0.92-0.98]); 19.2 [19.1-19.3] (0.66 [0.63-0.68]) |
| gpuP_cpuD_p512 | 388.1 [385.6-392.0] | 8/9, 6/8, 8/9 | — | 0/5, 0/5, 0/5 | 385.5 [381.4-392.4] (0.89 [0.81-0.91]); 21.6 [21.4-21.7] (0.19 [0.16-0.20]) |
| gpuP_cpuDs_p2048 | — | 0/4, 0/4, 0/4 | 23.4 [23.2-23.5] | 22/29, 23/30, 22/30 | 206.6 [194.0-217.6] (0.23 [0.23-0.27]); 24.6 [24.3-24.8] (0.82 [0.78-0.84]) |
| gpuP_cpuDs_p512 | — | 0/11, 0/12, 0/11 | 22.9 [22.5-23.7] | 8/31, 9/31, 7/31 | 435.0 [433.7-442.1] (0.25 [0.18-0.31]); 27.4 [27.2-27.4] (0.33 [0.31-0.39]) |
| solo_cpuD | 23.4 [23.4-23.4] | 6/6, 6/6, 6/6 | — | — | 23.4 [23.4-23.4] (0.00 [0.00-0.00]) |
| solo_cpuDs | 35.4 [35.4-35.5] | 36/36, 36/36, 36/36 | — | — | 35.4 [35.4-35.5] (0.00 [0.00-0.00]) |
| solo_cpuP_p2048 | 100.5 [100.1-102.1] | 3/3, 3/3, 3/3 | — | — | 100.5 [100.1-102.1] (0.00 [0.00-0.00]) |
| solo_cpuP_p512 | 140.8 [140.0-143.3] | 10/10, 10/10, 10/10 | — | — | 140.8 [140.0-143.3] (0.00 [0.00-0.00]) |
| solo_gpuD | 14.4 [14.3-14.4] | 3/3, 3/3, 3/3 | — | — | 14.4 [14.3-14.4] (0.00 [0.00-0.00]) |
| solo_gpuDs | 14.4 [14.4-14.5] | 12/12, 12/12, 12/12 | — | — | 14.4 [14.4-14.5] (0.00 [0.00-0.00]) |
| solo_gpuP_p2048 | 284.0 [283.0-284.6] | 5/5, 5/5, 5/5 | — | — | 284.0 [283.0-284.6] (0.00 [0.00-0.00]) |
| solo_gpuP_p512 | 437.6 [435.9-438.0] | 12/12, 12/12, 12/12 | — | — | 437.6 [435.9-438.0] (0.00 [0.00-0.00]) |

#### et 1b: co-run ratios

| pair | P | prefill ratio | prefills counted per trial | decode ratio | decode windows counted per trial | previous estimate: decode ratio (duty) |
|---|---|---|---|---|---|---|
| GPU prefill → CPU decode | 512 | 0.89 | 8/9, 6/8, 8/9 | 0.65 | 8/31, 9/31, 7/31 | 0.59 (0.19) |
| CPU prefill → GPU decode | 512 | 0.84 | 6/9, 6/9, 6/9 | 1.07 | 8/11, 9/10, 10/11 | 0.60 (0.49) |
| GPU prefill → CPU decode | 2048 | 0.68 | 3/4, 3/4, 3/4 | 0.66 | 22/29, 23/30, 22/30 | 0.73 (0.66) |
| CPU prefill → GPU decode | 2048 | 0.84 | 2/2, 2/2, 2/2 | 1.06 | 9/10, 10/10, 9/10 | 0.61 (0.56) |
| both decoding (CPU, GPU) | — | — | — | 0.74, 0.99 | 24/31, 25/31, 24/31; 8/8, 8/8, 8/8 | 0.82, 0.67 (as measured) |
