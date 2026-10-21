# LLM phase affinity: the go/no-go (2026-10-21)

**Question.** Can a pipeline that prefills on one processing unit (PU) and decodes on the other beat CPU+GPU
data-parallel execution of on-device LLM inference? The spec
(`openspec/changes/llm-phase-affinity/specs/studies/llm-phase-affinity/spec.md`) sets the gate. The answer is
"go" only if some specialized pipeline is predicted to reach at least 1.2x the co-run-corrected data-parallel
bound, on at least one phone and one request mix (prompt P = 128/512/2048 tokens, output D = 32/128/512 tokens).

**Decision: no-go.** No specialized pipeline reaches 1.2x on any phone, model, runtime or mix.

- **Best margin: 1.08.** Pixel 7a, ExecuTorch, Llama 3.2 1B, P = 512, D = 32, prefill on the GPU (Vulkan 8da4w)
  and decode on the CPU (XNNPACK, cores 4-7): 0.453 requests/s against a data-parallel bound of 0.419.
  Everywhere else the gain is at most 0.97 (that phone and runtime), 0.84 (Samsung ExecuTorch), 0.79 (Pixel
  llama.cpp), 0.74 (Samsung llama.cpp 1B) and 0.73 (3B).
- **The limiting effect is co-run interference.** A phase keeps 0.34-0.89 of its alone rate while the other PU
  runs the other phase, and the specialized pair pays for that on both of its sides. At the best cell, the CPU
  decode keeps 0.59 of its rate next to a GPU prefill, where 0.66 would reach the gate, and the GPU prefill keeps
  0.89. With every phase at its alone rate, the same mix would reach 1.83, and the Samsung's llama.cpp mixes up
  to 1.74.
- **The phase affinity is real.** Where the GPU path is strong, the GPU prefills 1.4-2.9x faster than the CPU and,
  at P <= 512, the CPU decodes 1.2-2x faster (except ExecuTorch 3B, where they decode alike). A pipeline that
  exploits it beats the best single PU, by 1.54x at the best cell. Data-parallel execution uses the same two PUs,
  though, and co-running the two phases costs about as much as co-running two decodes (section 3).
- **The margin at the best cell is narrow and depends on how the pipeline is priced.** The KV hand-off costs milliseconds.
  - Giving data-parallel prefill the co-run slowdown too, where design D5 keeps it at its alone rate, lifts the
    cell to 1.17.
  - Its three co-run trials, taken one at a time, would give 0.93-1.17.
  - D5 charges the decode its co-run rate for the whole period, though in this pipeline the GPU prefills only
    1.35 s of each ~1.8 s period. If the decode slowed only while the GPU prefills, the cell would reach 1.29.
    That is the one cell, of all grids, where this changes the verdict (section 6).
- **The prompt-heavy agent corner (P = 2048, D = 32) is the weakest**: 0.41-0.67. At P = 2048 the GPU decodes as
  fast as the CPU or faster (except on the Pixel's llama.cpp, whose GPU is slower at everything), so there is
  no decode to hand to the CPU, and the GPU prefill keeps 0.45-0.68 of its rate next to a CPU decode.

## 1. What was run

All timing ran on the two phones (2026-10-20 15:08 to 2026-10-21 10:19, PDT); nothing was timed on rocky-ryzen
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
- **Co-runs** (design D3): two pinned processes started back to back, both orders, 3 trials per pair. llama.cpp
  loops a phase with `llama-bench -r`. A rate counts only the repetitions that the other process's repetitions
  cover wholly. ExecuTorch loops whole `llama_main` processes. Each side is compared with the same loop run
  alone, and decode windows carry their duty (section 3).
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
  a split pipeline would beat the co-run data-parallel bound by up to 1.83 (Pixel ExecuTorch 1B) and 1.74
  (Samsung llama.cpp 3B).

## 3. Interference: each phase co-running

Rate while the other PU runs the other phase, divided by the rate alone. llama.cpp: against the alone sweep.
ExecuTorch: against the same loop run alone; the decode ratio is extrapolated to a fully overlapped window
(design.md, "ExecuTorch co-runs"), and the both-decoding ratios are as measured.

| Phone | Runtime | P | GPU prefill next to CPU decode | CPU decode next to GPU prefill | CPU prefill next to GPU decode | GPU decode next to CPU prefill | both decoding: CPU, GPU |
|---|---|---|---|---|---|---|---|
| Samsung S24+ | llama.cpp 1B | 512 | 0.46 | 0.41 | 0.56 | 0.55 | 0.21, 0.74 |
| Samsung S24+ | llama.cpp 1B | 2048 | 0.54 | 0.34 | 0.65 | 0.52 | ″ |
| Samsung S24+ | llama.cpp 3B | 512 | 0.48 | 0.36 | 0.60 | 0.61 | 0.20, 0.76 |
| Samsung S24+ | llama.cpp 3B | 2048 | 0.45 | 0.34 | 0.68 | 0.38 | ″ |
| Samsung S24+ | ExecuTorch 1B | 512 | 0.69 | 0.55 | 0.79 | 0.51 | 0.67, 0.59 |
| Samsung S24+ | ExecuTorch 1B | 2048 | 0.54 | 0.60 | 0.85 | 0.58 | ″ |
| Pixel 7a | llama.cpp 1B | 512 | 0.57 | 0.62 | 0.56 | 0.66 | 0.58, 0.63 |
| Pixel 7a | llama.cpp 1B | 2048 | 0.48 | 0.59 | 0.63 | 0.61 | ″ |
| Pixel 7a | ExecuTorch 1B | 512 | 0.89 | 0.59 | 0.84 | 0.60 | 0.82, 0.67 |
| Pixel 7a | ExecuTorch 1B | 2048 | 0.68 | 0.73 | 0.84 | 0.61 | ″ |

- **No pairing is free.** In a specialized pair each phase keeps 0.34-0.89 of its rate; two decodes keep
  0.20-0.82. Prefill is not the quiet partner the proposal hoped for:
  - on the Pixel, a llama.cpp decode loses as much next to a prefill as next to another decode (CPU 0.62 vs 0.58,
    GPU 0.66 vs 0.63);
  - on the Samsung, a GPU prefill hurts the CPU decode less than a GPU decode does (0.41 vs 0.21). But the GPU
    decode loses more next to a CPU prefill (0.55 vs 0.74), and each prefill keeps only about half its rate
    (GPU 0.46, CPU 0.56 at P = 512).
- **On the Samsung the CPU loses when the GPU streams.** Next to a GPU decode the CPU's llama.cpp decode keeps 0.21
  and the GPU 0.74. Its streaming read falls from 36 to 7.6 GB/s while the GPU's falls from 47 to 40 (section 4).
  The Pixel splits the loss evenly (0.58, 0.63; reads 24 to 14.5 and 34 to 22.6 GB/s).
- **On the Samsung, ExecuTorch spares the CPU more than llama.cpp does.** Its CPU decode keeps 0.55 next to a GPU
  prefill (llama.cpp 0.41) and 0.67 next to a GPU decode (0.21); its GPU decode keeps 0.51-0.59 (llama.cpp
  0.52-0.74). Part of the gap is method: the ExecuTorch loops leave gaps for model loads, and the both-decoding
  ratios are as measured, at duty 0.78 and 0.88.

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
64 MiB of fp16 K and V read and written, 4-8 ms; ExecuTorch writes it as fp32). The last five columns are
diagnostics, not the gate:
- the gain with the specialized pair at its alone rates;
- the gain without the KV hand-off;
- the gain if the two phases slowed each other only while both run: the shorter phase at its co-run rate, the
  longer one at its co-run rate for that long and alone after. This is set against the better of data-parallel
  and the best single PU;
- the gain against a data-parallel bound whose prefill also runs at its co-run rate (design D5 keeps prefill at
  its alone rate, which favours data-parallel);
- for the best pair, the decode co-run ratio used and the one that would reach 1.2.

A 3B grid built on 1B co-run ratios is a transfer and is not counted for the gate.

### Samsung S24+ (R5CY21Y3VEV): llamacpp 1b: predicted request throughput (requests/s)

| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run | GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference | gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) | gain vs data-parallel with co-run prefill | best pair: decode ratio used / needed for the gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 1.0391 (CPU) | 0.8929 | 1.9018 | 0.9869 | 0.6795 | 0.5912 | **0.69** | 1.67 | 0.69 | 1.07 | 0.79 | 0.41 / 0.72 |
| 128 | 128 | 0.3600 (CPU) | 0.2706 | 0.6134 | 0.2761 | 0.1699 | 0.1478 | **0.62** | 1.50 | 0.62 | 1.02 | 0.64 | 0.41 / 0.80 |
| 128 | 512 | 0.0996 (CPU) | 0.0714 | 0.1658 | 0.0712 | 0.0425 | 0.0370 | **0.60** | 1.45 | 0.60 | 1.01 | 0.60 | 0.41 / 0.83 |
| 512 | 32 | 0.5329 (GPU) | 0.5486 | 0.9892 | 0.6704 | 0.4990 | 0.3652 | **0.74** | 1.60 | 0.74 | 0.84 | 1.05 | 0.41 / unreachable |
| 512 | 128 | 0.2410 (CPU) | 0.2268 | 0.4533 | 0.2384 | 0.1576 | 0.1453 | **0.66** | 1.61 | 0.66 | 1.09 | 0.76 | 0.41 / 0.75 |
| 512 | 512 | 0.0835 (CPU) | 0.0678 | 0.1458 | 0.0667 | 0.0394 | 0.0363 | **0.59** | 1.44 | 0.59 | 1.03 | 0.62 | 0.41 / 0.84 |
| 2048 | 32 | 0.1777 (GPU) | 0.0632 | 0.2769 | 0.2405 | 0.1172 | 0.0715 | **0.49** | 0.90 | 0.49 | 0.70 | 0.76 | 0.34 / unreachable |
| 2048 | 128 | 0.1153 (GPU) | 0.0544 | 0.1924 | 0.1359 | 0.0892 | 0.0715 | **0.66** | 1.59 | 0.66 | 0.78 | 0.87 | 0.34 / unreachable |
| 2048 | 512 | 0.0479 (GPU) | 0.0349 | 0.0888 | 0.0500 | 0.0223 | 0.0318 | **0.64** | 1.30 | 0.64 | 0.95 | 0.72 | 0.52 / 0.98 |

### Samsung S24+ (R5CY21Y3VEV): llamacpp 3b: predicted request throughput (requests/s)

| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run | GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference | gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) | gain vs data-parallel with co-run prefill | best pair: decode ratio used / needed for the gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 0.3973 (CPU) | 0.1679 | 0.7161 | 0.3694 | 0.2321 | 0.2477 | **0.67** | 1.74 | 0.67 | 1.03 | 0.77 | 0.61 / 1.10 |
| 128 | 128 | 0.1394 (CPU) | 0.0464 | 0.2342 | 0.1042 | 0.0580 | 0.0619 | **0.59** | 1.54 | 0.59 | 1.01 | 0.62 | 0.61 / 1.24 |
| 128 | 512 | 0.0388 (CPU) | 0.0119 | 0.0636 | 0.0269 | 0.0145 | 0.0155 | **0.57** | 1.49 | 0.57 | 1.00 | 0.58 | 0.61 / 1.28 |
| 512 | 32 | 0.1834 (GPU) | 0.1253 | 0.3339 | 0.2271 | 0.1657 | 0.1306 | **0.73** | 1.52 | 0.73 | 0.75 | 1.02 | 0.36 / unreachable |
| 512 | 128 | 0.0784 (CPU) | 0.0426 | 0.1546 | 0.0831 | 0.0442 | 0.0598 | **0.72** | 1.48 | 0.72 | 1.00 | 0.83 | 0.61 / 1.02 |
| 512 | 512 | 0.0269 (CPU) | 0.0117 | 0.0497 | 0.0236 | 0.0111 | 0.0150 | **0.64** | 1.30 | 0.64 | 1.02 | 0.66 | 0.61 / 1.16 |
| 2048 | 32 | 0.0636 (GPU) | 0.0445 | 0.0965 | 0.0831 | 0.0351 | 0.0252 | **0.42** | 0.93 | 0.42 | 0.66 | 0.72 | 0.34 / unreachable |
| 2048 | 128 | 0.0414 (GPU) | 0.0236 | 0.0663 | 0.0469 | 0.0261 | 0.0252 | **0.56** | 1.62 | 0.56 | 0.67 | 0.80 | 0.34 / unreachable |
| 2048 | 512 | 0.0173 (GPU) | 0.0082 | 0.0298 | 0.0173 | 0.0065 | 0.0086 | **0.50** | 1.29 | 0.50 | 0.83 | 0.58 | 0.38 / 0.93 |

### Samsung S24+ (R5CY21Y3VEV): et 1b: predicted request throughput (requests/s)

| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run | GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference | gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) | gain vs data-parallel with co-run prefill | best pair: decode ratio used / needed for the gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 1.1215 (CPU) | — | 2.1339 | 1.4814 | 0.8758 | 0.6388 | **0.59** | 1.07 | 0.59 | 0.90 | 0.62 | 0.55 / 1.12 |
| 128 | 128 | 0.3596 (CPU) | — | 0.6557 | 0.4276 | 0.2190 | 0.1597 | **0.51** | 0.93 | 0.51 | 0.88 | 0.52 | 0.55 / 1.29 |
| 128 | 512 | 0.0967 (CPU) | — | 0.1740 | 0.1113 | 0.0548 | 0.0399 | **0.49** | 0.89 | 0.49 | 0.88 | 0.49 | 0.55 / 1.35 |
| 512 | 32 | 0.6562 (GPU) | — | 1.1783 | 0.9108 | 0.7671 | 0.5834 | **0.84** | 1.53 | 0.84 | 0.96 | 0.96 | 0.55 / unreachable |
| 512 | 128 | 0.2457 (CPU) | — | 0.4873 | 0.3348 | 0.1919 | 0.1460 | **0.57** | 1.04 | 0.57 | 0.91 | 0.60 | 0.55 / 1.16 |
| 512 | 512 | 0.0788 (CPU) | — | 0.1473 | 0.0957 | 0.0480 | 0.0365 | **0.50** | 0.91 | 0.50 | 0.88 | 0.51 | 0.55 / 1.32 |
| 2048 | 32 | 0.2146 (GPU) | — | 0.3315 | 0.2953 | 0.1471 | 0.1166 | **0.50** | 0.93 | 0.50 | 0.73 | 0.69 | 0.60 / unreachable |
| 2048 | 128 | 0.1302 (GPU) | — | 0.2102 | 0.1618 | 0.1147 | 0.1166 | **0.72** | 1.18 | 0.72 | 0.78 | 0.87 | 0.58 / unreachable |
| 2048 | 512 | 0.0506 (GPU) | — | 0.0860 | 0.0583 | 0.0287 | 0.0358 | **0.61** | 1.06 | 0.61 | 0.87 | 0.66 | 0.58 / 1.13 |

### Samsung S24+ (R5CY21Y3VEV): et 3b: predicted request throughput (requests/s) — co-run ratios transferred from 1b, not gate-eligible

| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run | GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference | gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) | gain vs data-parallel with co-run prefill | best pair: decode ratio used / needed for the gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 0.4440 (GPU) | — | 0.8317 | 0.5780 | 0.3142 | 0.2878 | **0.54** | 0.99 | 0.54 | 0.84 | 0.58 | 0.55 / 1.22 |
| 128 | 128 | 0.1323 (GPU) | — | 0.2598 | 0.1687 | 0.0786 | 0.0720 | **0.47** | 0.84 | 0.47 | 0.81 | 0.47 | 0.55 / 1.42 |
| 128 | 512 | 0.0347 (GPU) | — | 0.0693 | 0.0441 | 0.0196 | 0.0180 | **0.45** | 0.81 | 0.45 | 0.80 | 0.45 | 0.55 / 1.48 |
| 512 | 32 | 0.2790 (GPU) | — | 0.4614 | 0.3584 | 0.2560 | 0.2371 | **0.71** | 1.29 | 0.72 | 0.85 | 0.82 | 0.55 / unreachable |
| 512 | 128 | 0.1095 (GPU) | — | 0.1932 | 0.1319 | 0.0641 | 0.0688 | **0.52** | 1.02 | 0.52 | 0.80 | 0.55 | 0.51 / 1.17 |
| 512 | 512 | 0.0319 (GPU) | — | 0.0584 | 0.0376 | 0.0160 | 0.0172 | **0.46** | 0.90 | 0.46 | 0.84 | 0.46 | 0.51 / 1.33 |
| 2048 | 32 | 0.0766 (GPU) | — | 0.1127 | 0.1011 | 0.0503 | 0.0372 | **0.50** | 0.93 | 0.50 | 0.68 | 0.71 | 0.60 / unreachable |
| 2048 | 128 | 0.0495 (GPU) | — | 0.0728 | 0.0562 | 0.0298 | 0.0372 | **0.66** | 0.88 | 0.66 | 0.70 | 0.82 | 0.58 / unreachable |
| 2048 | 512 | 0.0205 (GPU) | — | 0.0301 | 0.0203 | 0.0074 | 0.0151 | **0.75** | 1.29 | 0.75 | 0.99 | 0.81 | 0.58 / 0.93 |

### Pixel 7a (3A021JEHN02756): llamacpp 1b: predicted request throughput (requests/s)

| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run | GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference | gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) | gain vs data-parallel with co-run prefill | best pair: decode ratio used / needed for the gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 0.6331 (CPU) | 0.2370 | 0.9304 | 0.6593 | 0.4019 | 0.3389 | **0.61** | 1.07 | 0.61 | 0.72 | 0.74 | 0.62 / unreachable |
| 128 | 128 | 0.2211 (CPU) | 0.0732 | 0.3297 | 0.2086 | 0.1591 | 0.0847 | **0.76** | 1.22 | 0.76 | 0.93 | 0.81 | 0.62 / 0.98 |
| 128 | 512 | 0.0614 (CPU) | 0.0194 | 0.0920 | 0.0559 | 0.0398 | 0.0212 | **0.71** | 1.14 | 0.71 | 0.98 | 0.72 | 0.62 / 1.05 |
| 512 | 32 | 0.2488 (CPU) | 0.1400 | 0.3991 | 0.3358 | 0.1214 | 0.1930 | **0.57** | 1.02 | 0.57 | 0.71 | 0.84 | 0.66 / unreachable |
| 512 | 128 | 0.1363 (CPU) | 0.0620 | 0.2159 | 0.1533 | 0.1214 | 0.0838 | **0.79** | 1.39 | 0.79 | 0.84 | 0.96 | 0.62 / unreachable |
| 512 | 512 | 0.0485 (CPU) | 0.0192 | 0.0761 | 0.0483 | 0.0353 | 0.0210 | **0.73** | 1.17 | 0.73 | 0.99 | 0.78 | 0.62 / 1.03 |
| 2048 | 32 | 0.0515 (CPU) | 0.0343 | 0.0905 | 0.0858 | 0.0205 | 0.0356 | **0.41** | 0.65 | 0.41 | 0.61 | 0.69 | 0.61 / unreachable |
| 2048 | 128 | 0.0411 (CPU) | 0.0249 | 0.0724 | 0.0616 | 0.0205 | 0.0356 | **0.58** | 0.91 | 0.58 | 0.71 | 0.86 | 0.61 / unreachable |
| 2048 | 512 | 0.0228 (CPU) | 0.0119 | 0.0402 | 0.0289 | 0.0205 | 0.0180 | **0.71** | 1.33 | 0.71 | 0.77 | 0.87 | 0.59 / unreachable |

### Pixel 7a (3A021JEHN02756): et 1b: predicted request throughput (requests/s)

| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run | GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference | gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) | gain vs data-parallel with co-run prefill | best pair: decode ratio used / needed for the gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 0.5348 (CPU) | — | 0.9275 | 0.7488 | 0.5450 | 0.2784 | **0.73** | 1.23 | 0.73 | 1.06 | 0.76 | 0.59 / 0.97 |
| 128 | 128 | 0.1954 (CPU) | — | 0.3060 | 0.2390 | 0.1363 | 0.0696 | **0.57** | 0.97 | 0.57 | 0.93 | 0.58 | 0.59 / 1.24 |
| 128 | 512 | 0.0552 (CPU) | — | 0.0838 | 0.0646 | 0.0341 | 0.0174 | **0.53** | 0.89 | 0.53 | 0.88 | 0.53 | 0.59 / 1.34 |
| 512 | 32 | 0.2932 (GPU) | — | 0.5027 | 0.4186 | 0.4526 | 0.2415 | **1.08** | 1.83 | 1.08 | 1.29 | 1.17 | 0.59 / 0.66 |
| 512 | 128 | 0.1152 (CPU) | — | 0.2146 | 0.1705 | 0.1132 | 0.0680 | **0.66** | 1.12 | 0.66 | 1.02 | 0.69 | 0.59 / 1.07 |
| 512 | 512 | 0.0411 (CPU) | — | 0.0684 | 0.0529 | 0.0283 | 0.0170 | **0.54** | 0.91 | 0.54 | 0.88 | 0.54 | 0.59 / 1.32 |
| 2048 | 32 | 0.1031 (GPU) | — | 0.1524 | 0.1394 | 0.0936 | 0.0466 | **0.67** | 0.99 | 0.67 | 0.87 | 0.85 | 0.73 / unreachable |
| 2048 | 128 | 0.0584 (GPU) | — | 0.0948 | 0.0790 | 0.0765 | 0.0466 | **0.97** | 1.33 | 0.97 | 1.02 | 1.10 | 0.73 / unreachable |
| 2048 | 512 | 0.0213 (GPU) | — | 0.0392 | 0.0305 | 0.0191 | 0.0153 | **0.63** | 0.86 | 0.63 | 0.80 | 0.66 | 0.73 / 1.39 |

## 6. Decision

The best cell of every grid. "Gate" says whether the grid uses that model's own co-run rates. The last three
columns are the diagnostics of section 5: the best gain with no interference, the best gain against
data-parallel with co-run prefill, and the best cell's decode ratio used / needed. The overlap diagnostic is in
the grids.

| Phone | Runtime | Model | Gate | Best P / D | Prefill → decode | Gain | No interference | vs DP, co-run prefill | Decode ratio used / needed |
|---|---|---|---|---|---|---|---|---|---|
| Samsung S24+ | llama.cpp | 1B | yes | 512 / 32 | GPU → CPU | **0.74** | 1.67 | 1.05 | 0.41 / unreachable |
| Samsung S24+ | llama.cpp | 3B | yes | 512 / 32 | GPU → CPU | **0.73** | 1.74 | 1.02 | 0.36 / unreachable |
| Samsung S24+ | ExecuTorch | 1B | yes | 512 / 32 | GPU → CPU | **0.84** | 1.53 | 0.96 | 0.55 / unreachable |
| Samsung S24+ | ExecuTorch | 3B | no (1B ratios) | 2048 / 512 | CPU → GPU | **0.75** | 1.29 | 0.82 | 0.58 / 0.93 |
| Pixel 7a | llama.cpp | 1B | yes | 512 / 128 | GPU → CPU | **0.79** | 1.39 | 0.96 | 0.62 / unreachable |
| Pixel 7a | ExecuTorch | 1B | yes | 512 / 32 | GPU → CPU | **1.08** | 1.83 | 1.17 | 0.59 / 0.66 |

**No-go.** The largest margin is 1.08, on the Pixel 7a with ExecuTorch, 1B, P = 512, D = 32: GPU prefill and CPU
decode. Interference limits it:
- The pipeline's period at that mix is the CPU decode, 32 tokens at 24.6 tok/s x 0.59 = 2.21 s, against 1.35 s
  for the GPU prefill (428 tok/s x 0.89).
- Data-parallel gets 0.419 requests/s from the same two PUs. The Pixel's CPU decodes at 0.82 of its rate next to
  the GPU's decode (measured, duty 0.86), so data-parallel loses less to interference than the split pipeline.

**How sure.** This is the only cell whose measurement uncertainty spans the gate.
- Its CPU decode ratio (0.59) is extrapolated from decode windows that the GPU prefill loop overlapped only 19%
  of the time, because model loads dominate the loop. The extrapolation 1 - (1 - r)/u turns r = 0.92 into 0.59
  and magnifies noise five-fold. The three trials give 0.51, 0.55 and 0.64, gains 0.93, 1.00 and 1.17.
- The P = 2048 co-run of the same pair overlaps 66% of the time and extrapolates to 0.73. At 0.73 this cell
  would reach 1.34; at 0.66, 1.2.
- The verdict rests on the estimate at the cell's own prompt length. The method was fixed before any ExecuTorch
  co-run was taken (design.md). After review, the duty was weighted by window length, as the rate already was,
  which lowered the cell from 1.10.

**How conservative D5 is.** D5 prices the specialized period with co-run rates, as if both phases ran
side by side for the whole period (spec scenario "A specialized pipeline's period").
- At this mix the GPU prefill takes 1.35 s of a ~1.8 s period, so the CPU decode would run alone for the rest.
- With the decode slowed only while the GPU prefills, and the GPU prefill slowed throughout, the cell would
  reach 1.29 against the better of data-parallel and the best single PU (column "overlap only while both run").
- Nowhere else does this exceed 1.09. The comparison uses the better of the two because the co-run
  data-parallel bound falls below the best single PU in 12 llama.cpp cells. Nine of them are on the Samsung,
  where a GPU decode leaves the CPU decode 0.21 of its rate. A pipeline that idles the GPU then "beats"
  data-parallel without beating one PU. These cells have gate gains of at most 0.76, so the verdict does not
  depend on them.
- The gate is the spec's, and by it this cell is a no-go. Under the less conservative pricing it would be a
  "go".

**What would settle it.** Measure the decode this pipeline would see, on the Pixel, at this one mix. Two
processes suffice, as in this study, if `llama_main` can repeat `generate()` without reloading its model:
- Vulkan 8da4w prefills a 512-token prompt once per pipeline period (about 1.8-2.2 s);
- XNNPACK on cores 4-7 decodes 32-token requests back to back.

That covers both doubts above, the interference at full duty and the overlap a real period has. The runner
change is small but in ExecuTorch, outside this study. Nothing else in the study is close enough to the gate for
a better measurement to matter.

Everything else is clear of the gate:
- every other gate-eligible cell is at most 0.97;
- the best cell of every other gate-eligible grid cannot reach 1.2 at its measured prefill ratio, whatever its
  decode does ("unreachable");
- neither the data-parallel bound with co-run prefill nor the partial-overlap pricing lifts any other cell to 1.2.

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

| loop | A rate | A duty | B rate | B duty |
|---|---|---|---|---|
| cpuD_gpuD | 22.4 [22.2-23.1] | 0.78 [0.76-0.79] | 21.0 [20.9-22.6] | 0.88 [0.74-0.89] |
| cpuP_gpuD_p2048 | 226.6 [223.1-258.9] | 1.00 [1.00-1.00] | 24.8 [24.6-25.8] | 0.71 [0.71-0.72] |
| cpuP_gpuD_p512 | 293.7 [293.1-294.2] | 1.00 [1.00-1.00] | 27.7 [26.2-27.8] | 0.44 [0.44-0.54] |
| gpuP_cpuD_p2048 | 302.2 [286.3-305.6] | 1.00 [1.00-1.00] | 25.0 [24.5-25.2] | 0.63 [0.61-0.64] |
| gpuP_cpuD_p512 | 547.3 [547.0-562.3] | 1.00 [1.00-1.00] | 30.2 [30.1-30.6] | 0.21 [0.18-0.22] |
| solo_cpuD | 33.4 [33.3-33.6] | 0.00 [0.00-0.00] | — | — |
| solo_cpuP_p2048 | 268.0 [267.1-268.0] | 0.00 [0.00-0.00] | — | — |
| solo_cpuP_p512 | 371.4 [357.1-373.5] | 0.00 [0.00-0.00] | — | — |
| solo_gpuD | 35.5 [35.4-35.5] | 0.00 [0.00-0.00] | — | — |
| solo_gpuP_p2048 | 562.3 [561.2-565.2] | 0.00 [0.00-0.00] | — | — |
| solo_gpuP_p512 | 792.4 [789.0-795.4] | 0.00 [0.00-0.00] | — | — |

#### et 1b: co-run ratios

| pair | P | prefill ratio | decode ratio, measured | decode duty | decode ratio, full overlap |
|---|---|---|---|---|---|
| GPU prefill → CPU decode | 512 | 0.69 | 0.90 | 0.21 | 0.55 |
| CPU prefill → GPU decode | 512 | 0.79 | 0.78 | 0.44 | 0.51 |
| GPU prefill → CPU decode | 2048 | 0.54 | 0.75 | 0.63 | 0.60 |
| CPU prefill → GPU decode | 2048 | 0.85 | 0.70 | 0.71 | 0.58 |
| both decoding (CPU, GPU) | — | — | 0.67, 0.59 | 0.78, 0.88 | as measured |

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

| loop | A rate | A duty | B rate | B duty |
|---|---|---|---|---|
| cpuD_gpuD | 19.1 [19.1-19.6] | 0.86 [0.86-0.87] | 9.6 [9.5-9.6] | 0.72 [0.69-0.73] |
| cpuP_gpuD_p2048 | 84.1 [83.5-84.9] | 1.00 [1.00-1.00] | 11.2 [11.1-11.2] | 0.56 [0.56-0.57] |
| cpuP_gpuD_p512 | 118.1 [117.7-121.5] | 1.00 [1.00-1.00] | 11.6 [11.5-11.7] | 0.49 [0.48-0.50] |
| gpuP_cpuD_p2048 | 192.0 [187.2-201.6] | 1.00 [1.00-1.00] | 19.2 [19.1-19.3] | 0.66 [0.63-0.68] |
| gpuP_cpuD_p512 | 388.1 [385.6-392.0] | 1.00 [1.00-1.00] | 21.6 [21.4-21.7] | 0.19 [0.16-0.20] |
| solo_cpuD | 23.4 [23.4-23.4] | 0.00 [0.00-0.00] | — | — |
| solo_cpuP_p2048 | 100.5 [100.1-102.1] | 0.00 [0.00-0.00] | — | — |
| solo_cpuP_p512 | 140.8 [140.0-143.3] | 0.00 [0.00-0.00] | — | — |
| solo_gpuD | 14.4 [14.3-14.4] | 0.00 [0.00-0.00] | — | — |
| solo_gpuP_p2048 | 284.0 [283.0-284.6] | 0.00 [0.00-0.00] | — | — |
| solo_gpuP_p512 | 437.6 [435.9-438.0] | 0.00 [0.00-0.00] | — | — |

#### et 1b: co-run ratios

| pair | P | prefill ratio | decode ratio, measured | decode duty | decode ratio, full overlap |
|---|---|---|---|---|---|
| GPU prefill → CPU decode | 512 | 0.89 | 0.92 | 0.19 | 0.59 |
| CPU prefill → GPU decode | 512 | 0.84 | 0.81 | 0.49 | 0.60 |
| GPU prefill → CPU decode | 2048 | 0.68 | 0.82 | 0.66 | 0.73 |
| CPU prefill → GPU decode | 2048 | 0.84 | 0.78 | 0.56 | 0.61 |
| both decoding (CPU, GPU) | — | — | 0.82, 0.67 | 0.86, 0.72 | as measured |
