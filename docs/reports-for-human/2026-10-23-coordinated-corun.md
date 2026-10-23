# Coordinated co-run: the go/no-go (2026-10-23)

**Question.** `llm-phase-affinity` measured CPU/GPU co-run interference with two independent processes running flat
out. Is that interference intrinsic, or an artifact of no coordination? If the CPU and GPU are coordinated (the GPU
paced in software, the prefill chunk size and the CPU decode width chosen), can a phase-specialized pipeline (prefill
on one processing unit, decode on the other) beat an equally coordinated data-parallel baseline? The spec
(`openspec/changes/coordinated-corun/specs/studies/coordinated-corun/spec.md`) sets the gate: "go" only if a
coordinated specialized pipeline is predicted to reach at least 1.2x the best coordinated data-parallel throughput
on at least one request mix (prompt P = 128/512/2048, output D = 32/128/512 tokens).

**Decision: no-go.** No coordinated specialized pipeline reaches 1.2x the baseline on any request mix, for either
model or on either phone. Coordination raises both sides, and the baseline enough that the best gain falls from
1.09 with this study's uncoordinated settings to 0.99.

- **Best margin: 0.99** (Samsung, llama.cpp, 1B, P = 512, D = 128).
  - The pipeline: CPU prefill on cores 4-9, GPU decode at duty 1.0. That is an uncoordinated setting, 0.306
    requests/s.
  - The baseline: data-parallel at duty 1.0 with the CPU decoding on cores 7-8 (mask `180`), 0.309 requests/s.
  - Trial spread 0.91-1.03. The most favourable trial anywhere in the grid gives 1.08.
- **3B: at most 0.85** (same cell and pipeline). **Pixel ExecuTorch 1B: at most 0.95** (P = 512, D = 32), where
  `llm-phase-affinity` had 1.03.
- **What limits it.**
  - The GPU-prefill pipeline is bounded by the CPU decode, which never keeps more than 0.66 of its alone rate next to
    a GPU prefill at any duty, mask or ubatch. Its best cell (P = 512, D = 32) would need 0.87 of the GPU prefill and
    0.72 of the CPU decode at once.
  - The CPU-prefill pipeline is bounded by the GPU decode. That pipeline is just the GPU decoding at its best rate
    (1.13 of alone next to a CPU prefill). Data-parallel gets nearly the same GPU decode next to a 2-core CPU decode
    (1.12) and adds the CPU's own requests. To pass, the pipeline's GPU decode would need 1.37 of its alone rate.
- **The interference is partly an artifact, but not one that favours specialization.**
  - Pacing trades GPU rate for CPU decode rate. From duty 1.0 to 0.33 the GPU prefill falls from 0.74 to 0.30 of its
    alone rate, and the CPU decode beside it rises from 0.41 to 0.64 (section 4). With all three knobs, the
    GPU-prefill pipeline rises by 22-28% where it is the best pipeline (1B), and data-parallel by 2-5% there.
  - The strongest knob is not pacing but keeping the CPU decode off the prime core (and off the medium cores). With
    core 9 decoding, a 2048-token GPU prefill runs at 450 MHz instead of 1095, which points to a shared power and
    thermal budget. Narrow masks lift both sides of both comparisons. Data-parallel gains the most, and it becomes
    the strongest baseline at most mixes.
- **`llm-phase-affinity`'s interference was overstated, for both sides.** Its CPU references were 2-5-s bursts,
  11-17% above a sustained 40-s CPU decode. Its ~100-s co-run loops ran in a deeper throttle, with the GPU at 315-700
  MHz in every pair. Measured the same way for both sides, the 1B gains here range from 0.66 to 0.99, against
  0.49-0.74 there, and all stay below the gate.

## 1. What was run

All timing ran on the phones (2026-10-21 21:46 to 2026-10-23 04:15, PDT); nothing was timed on rocky-ryzen or the
build box, and no clock or governor setting was changed. Choices made during the run, each with its reason, are in
`openspec/changes/coordinated-corun/design.md` ("Deviations and choices"). The ones that shape the numbers:

- **Harness.** `lpa_pace` (design D1, `data/coordinated-corun/drivers/lpa_pace.cpp`) runs a pinned prefill loop in
  ubatches of C tokens, or a decode loop one token at a time. After every step it synchronizes, takes a timestamp and
  sleeps t_step x (1/u - 1), so the loop computes a fraction u (the duty) of the wall time. It is built against
  `llm-phase-affinity`'s llama.cpp `8345f33` libraries for the Samsung (i8mm build), uses only public headers, and
  mirrors llama-bench's loops.
- **Phone and model.** Samsung S24+ `R5CY21Y3VEV` (Exynos 2400: cores 0-3 little, 4-6 medium, 7-8 big, 9 prime;
  Xclipse 940), Llama 3.2 1B and 3B Q4_0 (the owner's GGUFs). GPU: `-ngl 99`, host thread on core 3, flash attention
  auto at P <= 512 and off at 2048. CPU: `-ngl 0 -dev none`, one thread per core of the mask.
- **Knobs** (design D2): GPU duty u = 1.0, 0.8, 0.67, 0.5, 0.33; CPU decode mask `3f0` (cores 4-9), `1f0` (4-8),
  `1c0` (6-8), `180` (7-8), each placing one thread per core (`R5CY21Y3VEV/diag-thread-placement.txt`); prefill
  ubatch C = 512, 256, 128. Stages: A (GPU prefill paced, CPU decode on 3f0), D (GPU decode paced, the data-parallel
  pair), E (CPU prefill next to a paced GPU decode), B (the narrower masks, and the data-parallel sweep repeated on
  the best masks), C (ubatch), F (3B at the best settings and at duty 1.0 on 3f0).
- **Discipline** (design D3, `llm-phase-affinity`'s thermal policy unchanged). Every value is the median of 3 runs or
  trials with its min-max. Before every run the phone must report thermal status 0, and runs are 20 s apart. A run
  that ends at status >= 1 is repeated once after 120 s, and the repeat replaces it. The GPU clock and every CPU
  policy frequency are sampled every 0.5 s. Co-runs: both orders, the GPU side 40 s, the CPU side 70 s (100 s for
  3B), so the CPU side covers the GPU side's last repetition. A side counts only its units (a whole prefill
  repetition with its gaps, or a decode token) that lie wholly inside the partner's measured interval. Rates are over
  wall time, so a paced side's rate includes its gaps.
- **Reference rates.** The harness's own 40-s alone runs, not `llm-phase-affinity`'s llama-bench medians (section 2).
- **Prediction** (design D4, `optimizer/analysis/coordinated_corun.py`). The prediction model is
  `llm-phase-affinity`'s D5. Each co-run ratio (co-run wall rate over the duty-1 alone rate, so pacing's own cost is
  in it) is applied to the alone rate at prompt or KV depth P, and a prompt of 128 uses the ratios measured at 512.
  Each pipeline takes, per mix, the measured setting that maximizes its throughput. The baseline is the largest of
  the best coordinated data-parallel throughput, `llm-phase-affinity`'s uncoordinated data-parallel bound, and the
  best single PU of either study. The CPU prefill of data-parallel and of the single CPU always runs on 3f0, the
  fastest mask. These choices make the baseline stronger.
- **Pixel ExecuTorch** (design D5, optional): Pixel 7a `3A021JEHN02756`, `llm-phase-affinity`'s `llama_main` and
  `.pte` files. The CPU decode width is the only knob (masks `f0`, `c0`, `30`), measured with that study's co-run
  method. It ran in parallel with the Samsung sweep.

## 2. Calibration, and why the references are the harness's own

The harness against `llm-phase-affinity`'s llama-bench medians (`R5CY21Y3VEV/calib/`, 30-s runs; spec "The paced
harness matches the reference rates"):

| Phase (1B) | Harness, 30 s (tok/s) | llm-phase-affinity | Ratio |
|---|---|---|---|
| GPU prefill, P = 512 | 550.4 [550.0-551.0] | 549.7 | 1.001 |
| GPU prefill, P = 2048 | 444.3 [443.8-445.5] | 444.1 | 1.001 |
| GPU decode, depth 0 | 35.9 [35.8-36.0] | 35.2 | 1.021 |
| CPU decode, depth 0, 3f0 | 49.2 [48.5-49.4] | 55.6 | **0.886** |
| GPU prefill, P = 512, duty 0.5 | 269.9 [269.7-271.6], achieved duty 0.499 | — | — |

- **The GPU rates match within 2.5%; the CPU decode does not.** A sustained CPU decode slows as its clocks fall, with
  the thermal status still 0: per 5 s, 55 tok/s in the first seconds (llama-bench's 55.6), then steps to 44 and in
  some runs to 32 within 40 s. Meanwhile the prime core falls from 2.3-2.6 GHz (peaks of 3.2) in the first seconds to
  1.8-2.0 GHz (`R5CY21Y3VEV/alone/1b_cpuD_d0_m3f0_r*.{out,clk}`). `llm-phase-affinity`'s references were llama-bench
  repetitions of 2-5 s, but its co-run loops lasted about 100 s. Its CPU decode co-run ratios therefore include this
  slowdown as well as interference.
- **So every ratio here is against the harness's own alone runs** (the spec's fallback), of 40 s, the length of a
  co-run's GPU side. Against `llm-phase-affinity`'s medians these are 0.86-0.90 for the CPU decode at depth <= 512,
  0.92 for the CPU prefill at P <= 512, and 0.99-1.03 for everything else.
- **The baseline keeps `llm-phase-affinity`'s numbers anyway.** Its uncoordinated data-parallel bound and best single
  PU rest on those burst rates and are higher. They stay in the baseline because that makes it stronger. A column of
  the grid also gives the gain against this study's own numbers only.

## 3. Pacing alone

Each paced phase alone, wall rate over its duty-1 rate (`R5CY21Y3VEV/alone/`, 1B):

| Phase | u = 0.8 | 0.67 | 0.5 | 0.33 |
|---|---|---|---|---|
| GPU prefill, P = 512 | 0.789 | 0.654 | 0.490 | 0.324 |
| GPU prefill, P = 2048 | 0.781 | 0.651 | 0.487 | 0.323 |
| GPU decode, depth 0 | 0.778 | 0.687 | 0.548 | 0.216 |
| achieved duty (prefill / decode) | 0.80 / 0.79 | 0.67 / 0.66 | 0.50 / 0.50 | 0.33 / 0.33 |

- **Pacing a prefill costs only its duty**, plus 1-3% (each ubatch computes 1-3% slower after a pause). The GPU clock
  stays at 1095 MHz.
- **A paced decode behaves differently.** Its tokens compute faster at duty 0.5-0.67 (36.7-39.2 tok/s over compute
  time, against 35.4 at duty 1), but 34% slower at duty 0.33 (23.4). So at duty 0.33 the wall rate is 0.22 of the
  duty-1 rate, not 0.33.

## 4. Trade-off curves (1B)

Co-run rate over the duty-1 alone rate of the same phase: the left number of each pair is the GPU side, the right
the CPU side. GPU clock over the overlap in MHz. Medians of 3 trials (`R5CY21Y3VEV/corun-*/`); the min-max, the
counted units per trial and the CPU clocks are in the appendix.

**GPU prefill paced, next to a CPU decode on 3f0** (stage A):

| P | u = 1.0 | 0.8 | 0.67 | 0.5 | 0.33 |
|---|---|---|---|---|---|
| 512 | 0.74, 0.41 (1000) | 0.59, 0.50 | 0.53, 0.54 | 0.41, 0.58 | 0.30, 0.64 |
| 2048 | 0.69, 0.34 (450) | 0.56, 0.40 (450) | 0.48, 0.46 (450) | 0.38, 0.51 | 0.28, 0.58 |

**GPU decode paced, next to a CPU decode** (the data-parallel pair; stage D on 3f0, stage B on the others):

| CPU mask | u = 1.0 | 0.8 | 0.67 | 0.5 | 0.33 |
|---|---|---|---|---|---|
| 3f0 (4-9) | 0.95, 0.26 | 0.75, 0.38 | 0.66, 0.49 | 0.51, 0.65 | 0.25, 0.77 |
| 1f0 (4-8) | 1.02, 0.28 | 0.80, 0.41 | 0.70, 0.50 | 0.53, 0.64 | 0.27, 0.79 |
| 1c0 (6-8) | 1.06, 0.29 | — | — | — | — |
| 180 (7-8) | 1.12, 0.38 | 0.91, 0.55 | 0.77, 0.65 | 0.57, 0.76 | 0.32, 0.91 |

**The CPU decode width, next to a GPU prefill** (stage B; 3f0 from stage A):

| P, duty | 3f0 | 1f0 | 1c0 | 180 |
|---|---|---|---|---|
| 512, 1.0 | 0.74, 0.41 | 0.82, 0.45 | 0.87, 0.48 | 0.93, 0.58 |
| 512, 0.8 | 0.59, 0.50 | 0.67, 0.49 | 0.76, 0.53 | 0.79, 0.66 |
| 2048, 1.0 | 0.69, 0.34 (450) | 0.76, 0.34 | 0.80, 0.35 | 0.87, 0.45 |
| 2048, 0.8 | 0.56, 0.40 (450) | 0.67, 0.41 | 0.69, 0.41 | 0.76, 0.52 |

**CPU prefill on 3f0, next to a paced GPU decode** (stage E): the CPU prefill first, the GPU decode second.

| P | u = 1.0 | 0.5 |
|---|---|---|
| 512 | 0.81, 1.13 | 0.90, 0.61 |
| 2048 | 0.75, 0.80 | 0.91, 0.60 |

**Prefill ubatch** (stage C, at the best duty and mask of A-B per P):

| Setting | C = 512 | 256 | 128 |
|---|---|---|---|
| P = 512, duty 0.8, CPU decode on 1f0 | 0.67, 0.49 | 0.62, 0.53 | 0.58, 0.62 |
| P = 2048, duty 1.0, CPU decode on 180 | 0.87, 0.45 | 0.88, 0.46 | 0.81, 0.55 |

- **Pacing trades one PU's rate for the other's.** From duty 1.0 to 0.33 the GPU prefill falls from 0.74 to 0.30 and
  the CPU decode beside it rises from 0.41 to 0.64 (P = 512). Paced alone, the prefill already falls to 0.32, so the
  GPU prefill is almost untouched by the CPU decode at low duty (0.93 of its paced-alone rate). The CPU decode is
  never left alone. Even with the GPU computing a third of the time it keeps 0.58-0.64, and next to a GPU decode at
  duty 0.33, 0.77-0.91.
- **A smaller ubatch is a finer-grained version of the same trade.** Smaller ubatches cost GPU prefill rate and
  return CPU decode rate, about as pacing does.
- **The CPU decode width matters more than the duty.** Leaving out the prime core 9 (1f0), and then the medium cores
  too (1c0, 180), raises the GPU side at every duty. With 180 the GPU prefill keeps 0.87-0.93 of its rate instead of
  0.69-0.74, and its clock stays at 1095 MHz. With 3f0 at P = 2048 it runs at 450 MHz (the CPU clusters run at
  1.7-2.0 GHz in every co-run, the level an alone CPU decode settles to; appendix). The clocks point to a shared
  power and thermal budget, of which the prime core spends a large part. A narrower decode is also slower alone (33.9
  tok/s on 180 against 47.6 on 3f0), so it reads less memory too; these runs do not separate the two causes. On 180
  the CPU decode also keeps more of its own alone rate: 0.58 against 0.41 at P = 512.
- **Data-parallel gains from the same knobs.** The GPU decode next to a 2-core CPU decode keeps 1.12 of its alone
  rate, more than alone. Next to a CPU prefill it keeps 1.13. The data-parallel pair at duty 1.0 on 180 becomes the
  strongest baseline at most mixes.

## 5. Prediction grid, 1B

Request throughput (requests/s) by design D4. "Gain" is the best coordinated specialized pipeline over the baseline
(the largest of the best coordinated data-parallel, `llm-phase-affinity`'s uncoordinated data-parallel, and the best
single PU). "Spread" uses every co-run rate at its least and its most favourable trial. The last column is
`llm-phase-affinity`'s uncoordinated gain for the same mix. Setting names: `gpuP_cpuD` is GPU prefill and CPU
decode, `gpuD_cpuP` is CPU prefill and GPU decode, `gpuD_cpuD` is data-parallel; `u` is the GPU duty x 100, `m` the
CPU mask, `c` the ubatch.

| P | D | best specialized (setting) | best data-parallel (setting) | baseline (from) | **gain** | spread | gain, duty-1 3f0 only | llm-phase-affinity gain |
|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 1.2451 (`gpuD_cpuP_p512_u100_m3f0`) | 1.2958 (`gpuD_cpuD_u100_m180`) | 1.2958 (coordinated data-parallel) | **0.96** | 0.88-0.99 | 1.06 | 0.69 |
| 128 | 128 | 0.3113 (`gpuD_cpuP_p512_u100_m3f0`) | 0.3808 (`gpuD_cpuD_u100_m180`) | 0.3808 (coordinated data-parallel) | **0.82** | 0.75-0.85 | 0.86 | 0.62 |
| 128 | 512 | 0.0778 (`gpuD_cpuP_p512_u100_m3f0`) | 0.0996 (`gpuD_cpuD_u100_m180`) | 0.0996 (coordinated data-parallel) | **0.78** | 0.72-0.80 | 0.78 | 0.60 |
| 512 | 32 | 0.6596 (`gpuP_cpuD_p512_u80_m1f0_c256`) | 0.7807 (`gpuD_cpuD_u100_m180`) | 0.7807 (coordinated data-parallel) | **0.84** | 0.79-1.08 | 0.73 | 0.74 |
| 512 | 128 | 0.3060 (`gpuD_cpuP_p512_u100_m3f0`) | 0.3085 (`gpuD_cpuD_u100_m180`) | 0.3085 (coordinated data-parallel) | **0.99** | 0.91-1.03 | 1.09 | 0.66 |
| 512 | 512 | 0.0765 (`gpuD_cpuP_p512_u100_m3f0`) | 0.0907 (`gpuD_cpuD_u100_m180`) | 0.0907 (coordinated data-parallel) | **0.84** | 0.78-0.87 | 0.92 | 0.59 |
| 2048 | 32 | 0.1909 (`gpuP_cpuD_p2048_u100_m180_c256`) | 0.2578 (`gpuD_cpuD_u100_m1f0`) | 0.2578 (coordinated data-parallel) | **0.74** | 0.72-0.77 | 0.59 | 0.49 |
| 2048 | 128 | 0.1065 (`gpuP_cpuD_p2048_u80_m1f0`) | 0.1606 (`gpuD_cpuD_u100_m1f0`) | 0.1606 (coordinated data-parallel) | **0.66** | 0.64-0.67 | 0.57 | 0.66 |
| 2048 | 512 | 0.0503 (`gpuD_cpuP_p2048_u100_m3f0`) | 0.0659 (`gpuD_cpuD_u100_m180`) | 0.0659 (coordinated data-parallel) | **0.76** | 0.76-0.84 | 0.83 | 0.64 |

- **Coordination raises both sides, and data-parallel enough to keep the gate shut.** Within this study (the column
  "duty-1 3f0"), coordination raises the GPU-prefill pipeline by 22-28% where it is the best pipeline (P = 512,
  D = 32; P = 2048, D = 32 and 128). Data-parallel rises by 2-5% at those cells, so the gain there rises, from 0.73
  to 0.84, 0.59 to 0.74 and 0.57 to 0.66. Where the CPU-prefill pipeline is the best one, coordination gives it
  nothing, while data-parallel gains 8-13%, mostly from the 180 mask. The gain at those cells falls: 1.09 to 0.99
  (P = 512, D = 128) and 1.06 to 0.96 (P = 128, D = 32). The pipeline passes the gate nowhere. Its best cell is 0.99
  (spread 0.91-1.03).
- **Against `llm-phase-affinity` both sides are higher.** The best pipeline is 19-94% higher per mix, data-parallel
  7-40%, so the gain rises from that study's 0.49-0.74 to 0.66-0.99. Part of that is the 40-s regime itself, not
  coordination: next to a CPU prefill the GPU decode keeps 1.13 here, at 1095 MHz, against 0.55 in that study's
  ~100-s loops at 545-650 MHz.
- **The best pipeline is mostly the CPU-prefill one at duty 1.0**: an uncoordinated setting. A CPU prefill leaves the
  GPU decode faster than alone (1.13), so the pipeline's period is the GPU decode at its best rate. Data-parallel
  gets nearly the same from its own GPU decode next to a 2-core CPU decode (1.12), and adds the CPU's requests.
- **The GPU-prefill pipeline peaks at 0.84** (P = 512, D = 32: duty 0.8, CPU decode on 1f0, ubatch 256). Its period
  is the CPU decode next to the GPU, which never keeps more than 0.66 of its rate.

## 6. Stage F: 3B at the best settings

Llama 3.2 3B, co-run at the best 1B settings and at duty 1.0 on 3f0 (`R5CY21Y3VEV/corun-F/`; ratios as in section
4, the GPU side first):

| Pair | Setting | Ratios | GPU MHz |
|---|---|---|---|
| GPU prefill → CPU decode | P = 512, duty 1.0, 3f0 | 0.59, 0.39 | 450 |
| | P = 512, duty 0.8, 1f0, ubatch 256 | 0.54, 0.44 | 675 |
| | P = 2048, duty 1.0, 3f0 | 0.56, 0.37 | 450 |
| | P = 2048, duty 1.0, 180, ubatch 256 | 0.62, 0.42 | 450 |
| CPU prefill → GPU decode (CPU first) | P = 512, duty 1.0 | 0.79, 0.90 | 1095 |
| | P = 2048, duty 1.0 | 0.73, 0.42 | 450 |
| data-parallel | duty 1.0, 3f0 / 1f0 / 180 | 0.88, 0.26 / 0.95, 0.26 / 1.03, 0.34 | 1095 |

| P | D | best specialized (setting) | best data-parallel (setting) | baseline (from) | **gain** | spread | gain, duty-1 3f0 only | llm-phase-affinity gain |
|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 0.3709 (`gpuD_cpuP_p512_u100_m3f0`) | 0.4447 (`gpuD_cpuD_u100_m180`) | 0.4447 (coordinated data-parallel) | **0.83** | 0.80-0.92 | 0.92 | 0.67 |
| 128 | 128 | 0.0927 (`gpuD_cpuP_p512_u100_m3f0`) | 0.1308 (`gpuD_cpuD_u100_m180`) | 0.1394 (single PU (CPU, llm-phase-affinity)) | **0.67** | 0.64-0.71 | 0.67 | 0.59 |
| 128 | 512 | 0.0232 (`gpuD_cpuP_p512_u100_m3f0`) | 0.0342 (`gpuD_cpuD_u100_m180`) | 0.0388 (single PU (CPU, llm-phase-affinity)) | **0.60** | 0.58-0.64 | 0.60 | 0.57 |
| 512 | 32 | 0.1940 (`gpuP_cpuD_p512_u80_m1f0_c256`) | 0.2662 (`gpuD_cpuD_u100_m180`) | 0.2662 (coordinated data-parallel) | **0.73** | 0.71-0.75 | 0.71 | 0.73 |
| 512 | 128 | 0.0902 (`gpuD_cpuP_p512_u100_m3f0`) | 0.1059 (`gpuD_cpuD_u100_m180`) | 0.1059 (coordinated data-parallel) | **0.85** | 0.82-0.94 | 0.94 | 0.72 |
| 512 | 512 | 0.0226 (`gpuD_cpuP_p512_u100_m3f0`) | 0.0312 (`gpuD_cpuD_u100_m180`) | 0.0312 (coordinated data-parallel) | **0.72** | 0.69-0.80 | 0.81 | 0.64 |
| 2048 | 32 | 0.0487 (`gpuP_cpuD_p2048_u100_m180_c256`) | 0.0903 (`gpuD_cpuD_u100_m180`) | 0.0903 (coordinated data-parallel) | **0.54** | 0.53-0.55 | 0.50 | 0.42 |
| 2048 | 128 | 0.0321 (`gpuP_cpuD_p2048_u100_m3f0`) | 0.0567 (`gpuD_cpuD_u100_m180`) | 0.0567 (coordinated data-parallel) | **0.57** | 0.53-0.58 | 0.60 | 0.56 |
| 2048 | 512 | 0.0097 (`gpuD_cpuP_p2048_u100_m3f0`) | 0.0231 (`gpuD_cpuD_u100_m180`) | 0.0231 (coordinated data-parallel) | **0.42** | 0.39-0.45 | 0.46 | 0.50 |

- **3B co-runs throttle harder.** The GPU sits at 450-675 MHz in every GPU-prefill pair and in the 2048-token
  CPU-prefill pair. For 1B it held 1095 MHz except next to a 3f0 decode at P = 2048. Neither pacing nor a narrower
  mask lifts the GPU prefill above 0.62.
- **The best 3B cell is 0.85** (P = 512, D = 128: CPU prefill into GPU decode at duty 1.0), against data-parallel at
  duty 1.0 on 180, again the strongest baseline. At P = 128 and D >= 128 the baseline is `llm-phase-affinity`'s
  single CPU, computed from burst rates, which is above this study's best data-parallel throughput there.


## 7. Pixel ExecuTorch (design D5)

Pixel 7a, ExecuTorch 1B (`3A021JEHN02756/`): co-run ratios against the same loop run alone, per CPU decode mask, and
the grid at P = 128 and 512 (co-runs at P = 512 only).

| CPU decode mask | GPU prefill next to CPU decode | CPU decode next to GPU prefill | both decoding: CPU, GPU |
|---|---|---|---|
| f0 (cores 4-7, 4 threads) | 0.90 | 0.60 | 0.75, 1.01 |
| c0 (X1 6-7, 2 threads) | 0.95 | 0.62 | 0.71, 1.03 |
| 30 (A78 4-5, 2 threads) | 0.96 | 0.64 | 0.80, 1.06 |

| P | D | best specialized (mask) | best data-parallel (mask) | gain | llm-phase-affinity gain |
|---|---|---|---|---|---|
| 128 | 32 | 0.6141 (f0) | 0.8955 (f0) | **0.69** | 0.72 |
| 128 | 128 | 0.1536 (f0) | 0.2802 (f0) | **0.55** | 0.57 |
| 128 | 512 | 0.0384 (f0) | 0.0750 (f0) | **0.51** | 0.54 |
| 512 | 32 | 0.4887 (f0) | 0.5136 (30) | **0.95** | 1.03 |
| 512 | 128 | 0.1223 (f0) | 0.2044 (30) | **0.60** | 0.64 |
| 512 | 512 | 0.0306 (f0) | 0.0613 (f0) | **0.50** | 0.53 |

- **Narrowing the decode does not help on the Pixel.** Next to a GPU prefill the CPU decode keeps 0.60-0.64 on any
  mask. The narrower masks also decode slower alone, so the 4-core decode remains the best pipeline at every mix.
- **The best cell is 0.95** (P = 512, D = 32), below `llm-phase-affinity`'s 1.03 for the same cell. In this session
  the CPU decode next to the GPU prefill kept 0.60, against 0.65 in that study's, and data-parallel was stronger
  (0.51 against 0.48 requests/s).

## 8. Decision

**No-go.** The spec's gate requires a coordinated specialized pipeline at >= 1.2x the best coordinated data-parallel
throughput on at least one request mix. Every grid falls short, against every form of the baseline:

| Grid | Best cell | Pipeline (setting) | Gain (spread) | vs coordinated data-parallel only | vs this study's own baseline |
|---|---|---|---|---|---|
| Samsung, llama.cpp 1B | P = 512, D = 128 | CPU prefill → GPU decode, duty 1.0 | **0.99** (0.91-1.03) | 0.99 | 0.99 |
| Samsung, llama.cpp 3B | P = 512, D = 128 | CPU prefill → GPU decode, duty 1.0 | **0.85** (0.82-0.94) | 0.85 | 0.85 |
| Pixel, ExecuTorch 1B | P = 512, D = 32 | GPU prefill → CPU decode on f0 | **0.95** | 0.95 | — |

- **Coordination helps both sides** (spec scenario "Coordination helps both sides equally"). The gains are computed
  against the coordinated baseline. With 1B, coordination raises the GPU-prefill pipelines more than data-parallel at
  the cells where they are best, by 22-28% against 2-5% (3B: 9-10% against 3-6%). It does not raise the CPU-prefill
  pipeline at all, while it raises data-parallel by 8-13% at those cells (1B). The net effect on the best gain is
  1.09 → 0.99 (1B) and 0.94 → 0.85 (3B).
- **The best margin and its setting:** 0.99, Samsung, 1B, P = 512, D = 128, CPU prefill on 3f0 with the GPU decoding
  at duty 1.0, against data-parallel at duty 1.0 with the CPU decoding on 180.
- **What limits it:** the CPU decode next to any GPU work (at most 0.66 of its alone rate beside a prefill), and the
  GPU decode, which a split pipeline can only use at the same rate data-parallel already gets (section 5).

What this means for BetterTogether 2:
- **A scheduler that coordinates the CPU and GPU of this SoC should pick core sets before duty.** Keeping a CPU
  decode off the prime core raised the GPU prefill's co-run rate by 10-26% here, and its clock from 450 to 1095 MHz
  in the 2048-token prefill pairs. That benefit is not specific to phase specialization: it applies to any CPU + GPU
  co-schedule.
- **Phase specialization is not where the gain is on these phones with these runtimes.** In this study's measurements
  the best split is a GPU decoding next to a CPU prefill. That only matches data-parallel, it does not beat it.
- **Not covered here:** closed-loop control (open-loop pacing bounds what coordination of this kind buys), and the
  deeper throttle of loops much longer than 40-70 s, where `llm-phase-affinity` saw both PUs slow further. A runtime
  that runs for minutes would operate there.

## 9. Reproduce

```bash
uv run optimizer/analysis/coordinated_corun.py --json /tmp/coordinated-corun.json   # every table and grid below
uv run pytest -q optimizer/tests/test_coordinated_corun.py                         # D4 on a hand-computed mix
```

Raw files under `data/coordinated-corun/`:
- `R5CY21Y3VEV/`:
  - `calib/`, `alone/`, `alone-C/`, `alone-3b/`: `<model>_<cfg>_r<k>[_retry].{out,err,clk,meta}`;
  - `corun-{A,B,B2,C,D,E,F}/`: `<model>_<pair>_..._t<k>[_retry]/{A,B}.{out,err}, corun.{meta,clk}`;
  - `diag-thread-placement.txt`, `queue.log`, `supervise.log`, `battery_guard.log`.
- `3A021JEHN02756/`: `alone-et/`, `corun-et/`, `diag-thread-placement.txt`.

The drivers, plans and harness source are in `data/coordinated-corun/drivers/`; `lpa_pace` is rebuilt with
`drivers/build_pace.sh`. Every grid row in the JSON lists the raw files behind it (`sources`). A `_retry` replaces
the run it repeats.

## Appendix: full tables

The output of `uv run optimizer/analysis/coordinated_corun.py` (Samsung, then the Pixel). Co-run columns: each
side's wall rate, ratio to its duty-1 alone rate and to its paced-alone rate, the GPU side's achieved duty, the GPU
and CPU-cluster clocks over the overlap, the counted units per trial (A, then B), the counted seconds, and the
thermal status after each trial.

### Samsung S24+ (R5CY21Y3VEV)

#### Calibration

| config | target duty | calibration, 30 s (tok/s) | achieved duty | llm-phase-affinity median | calibration / that | 40-s alone reference | calibration / that |
|---|---|---|---|---|---|---|---|
| 1b_cpuD_d0_m3f0 | 1.00 | 49.2 [48.5-49.4] | 1.000 [1.000-1.000] | 55.6 [54.4-55.8] | 0.886 | 47.6 [47.6-47.6] | 1.034 |
| 1b_gpuD_d0_u100 | 1.00 | 35.9 [35.8-36.0] | 0.999 [0.999-0.999] | 35.2 [34.8-35.2] | 1.021 | 35.4 [35.3-35.8] | 1.015 |
| 1b_gpuP_p2048_u100 | 1.00 | 444.3 [443.8-445.5] | 1.000 [1.000-1.000] | 444.1 [444.0-444.7] | 1.001 | 443.6 [437.3-447.7] | 1.002 |
| 1b_gpuP_p512_u100 | 1.00 | 550.4 [550.0-551.0] | 1.000 [1.000-1.000] | 549.7 [549.4-559.2] | 1.001 | 549.4 [548.4-549.5] | 1.002 |
| 1b_gpuP_p512_u50 | 0.50 | 269.9 [269.7-271.6] | 0.499 [0.499-0.499] | — | — | 269.3 [269.0-269.7] | 1.002 |

#### Alone rates, lpa_pace (median [min-max] of 3 runs)

| config | target duty | rate over wall (tok/s) | rate over compute | achieved duty | wall / duty-1 rate | llm-phase-affinity median | harness / that | GPU MHz | thermal after |
|---|---|---|---|---|---|---|---|---|---|
| 1b_cpuD_d0_m180 | 1.00 | 33.9 [33.5-34.5] | 33.9 [33.5-34.5] | 1.000 [1.000-1.000] | 1.000 | 38.5 [38.5-44.2] | 0.879 | — | 0,0,0 |
| 1b_cpuD_d0_m1c0 | 1.00 | 44.9 [44.9-45.9] | 44.9 [44.9-45.9] | 1.000 [1.000-1.000] | 1.000 | — | — | — | 0,0,0 |
| 1b_cpuD_d0_m1f0 | 1.00 | 46.5 [46.4-47.3] | 46.5 [46.4-47.3] | 1.000 [1.000-1.000] | 1.000 | — | — | — | 0,0,0 |
| 1b_cpuD_d0_m3f0 | 1.00 | 47.6 [47.6-47.6] | 47.6 [47.6-47.7] | 1.000 [1.000-1.000] | 1.000 | 55.6 [54.4-55.8] | 0.857 | — | 0,0,1 |
| 1b_cpuD_d128_m180 | 1.00 | 32.4 [32.2-32.6] | 32.4 [32.2-32.6] | 1.000 [1.000-1.000] | 1.000 | 37.1 [36.6-42.6] | 0.872 | — | 0,0,0 |
| 1b_cpuD_d128_m1c0 | 1.00 | 40.6 [40.4-41.1] | 40.6 [40.4-41.1] | 1.000 [1.000-1.000] | 1.000 | — | — | — | 0,0,0 |
| 1b_cpuD_d128_m1f0 | 1.00 | 45.3 [45.2-46.0] | 45.3 [45.3-46.0] | 1.000 [1.000-1.000] | 1.000 | — | — | — | 0,0,0 |
| 1b_cpuD_d128_m3f0 | 1.00 | 47.2 [46.6-47.4] | 47.2 [46.7-47.4] | 1.000 [1.000-1.000] | 1.000 | 52.9 [51.2-53.8] | 0.893 | — | 0,1,0 |
| 1b_cpuD_d2048_m180 | 1.00 | 19.9 [19.8-20.1] | 19.9 [19.8-20.1] | 1.000 [1.000-1.000] | 1.000 | 19.8 [19.8-21.7] | 1.005 | — | 0,0,0 |
| 1b_cpuD_d2048_m1c0 | 1.00 | 26.8 [26.6-27.6] | 26.8 [26.6-27.6] | 1.000 [1.000-1.000] | 1.000 | — | — | — | 0,0,0 |
| 1b_cpuD_d2048_m1f0 | 1.00 | 32.9 [32.3-33.1] | 32.9 [32.3-33.1] | 1.000 [1.000-1.000] | 1.000 | — | — | — | 0,0,0 |
| 1b_cpuD_d2048_m3f0 | 1.00 | 33.1 [32.4-33.8] | 33.2 [32.4-33.8] | 1.000 [1.000-1.000] | 1.000 | 33.3 [32.4-33.4] | 0.995 | — | 1,1,1 |
| 1b_cpuD_d512_m180 | 1.00 | 27.6 [27.5-28.1] | 27.6 [27.5-28.1] | 1.000 [1.000-1.000] | 1.000 | 30.8 [26.8-35.5] | 0.898 | — | 0,0,0 |
| 1b_cpuD_d512_m1c0 | 1.00 | 36.1 [35.0-38.8] | 36.1 [35.0-38.9] | 1.000 [1.000-1.000] | 1.000 | — | — | — | 0,0,0 |
| 1b_cpuD_d512_m1f0 | 1.00 | 41.4 [41.0-42.1] | 41.5 [41.0-42.1] | 1.000 [1.000-1.000] | 1.000 | — | — | — | 0,0,0 |
| 1b_cpuD_d512_m3f0 | 1.00 | 42.7 [41.8-43.6] | 42.7 [41.8-43.6] | 1.000 [1.000-1.000] | 1.000 | 49.0 [42.2-49.1] | 0.870 | — | 1,0,1 |
| 1b_cpuP_p128_m3f0 | 1.00 | 329.2 [325.9-330.5] | 329.2 [325.9-330.6] | 1.000 [1.000-1.000] | 1.000 | 358.5 [357.5-374.2] | 0.918 | — | 0,0,0 |
| 1b_cpuP_p2048_m3f0 | 1.00 | 223.2 [220.5-225.2] | 223.2 [220.5-225.2] | 1.000 [1.000-1.000] | 1.000 | 224.6 [218.5-224.7] | 0.994 | — | 1,0,0 |
| 1b_cpuP_p512_m3f0 | 1.00 | 305.2 [302.2-307.4] | 305.3 [302.2-307.4] | 1.000 [1.000-1.000] | 1.000 | 332.7 [332.6-338.5] | 0.917 | — | 0,0,0 |
| 1b_gpuD_d0_u100 | 1.00 | 35.4 [35.3-35.8] | 35.4 [35.4-35.8] | 0.999 [0.999-0.999] | 1.000 | 35.2 [34.8-35.2] | 1.006 | 1095 | 0,0,0 |
| 1b_gpuD_d0_u33 | 0.33 | 7.6 [7.6-7.7] | 23.4 [23.2-23.6] | 0.327 [0.327-0.327] | 0.216 | — | — | 1095 | 0,0,0 |
| 1b_gpuD_d0_u50 | 0.50 | 19.4 [19.3-19.5] | 39.2 [39.1-39.4] | 0.495 [0.495-0.495] | 0.548 | — | — | 1095 | 0,0,0 |
| 1b_gpuD_d0_u67 | 0.67 | 24.3 [24.3-24.7] | 36.7 [36.6-37.3] | 0.662 [0.662-0.662] | 0.687 | — | — | 1095 | 0,0,0 |
| 1b_gpuD_d0_u80 | 0.80 | 27.5 [27.4-27.8] | 34.8 [34.6-35.1] | 0.792 [0.792-0.792] | 0.778 | — | — | 1095 | 0,0,0 |
| 1b_gpuD_d128_u100 | 1.00 | 35.3 [35.2-35.8] | 35.3 [35.3-35.8] | 0.999 [0.999-0.999] | 1.000 | 34.4 [34.3-34.9] | 1.026 | 1095 | 0,0,0 |
| 1b_gpuD_d2048_u100 | 1.00 | 32.1 [32.0-32.4] | 32.1 [32.1-32.4] | 0.999 [0.999-0.999] | 1.000 | 31.5 [31.3-31.7] | 1.017 | 1095 | 0,0,0 |
| 1b_gpuD_d512_u100 | 1.00 | 34.7 [34.5-35.1] | 34.7 [34.5-35.1] | 0.999 [0.999-0.999] | 1.000 | 33.9 [33.7-33.9] | 1.025 | 1095 | 0,0,0 |
| 1b_gpuP_p128_u100 | 1.00 | 557.4 [556.2-559.1] | 557.6 [556.5-559.3] | 1.000 [1.000-1.000] | 1.000 | 556.2 [551.1-571.0] | 1.002 | 1095 | 0,0,0 |
| 1b_gpuP_p2048_u100 | 1.00 | 443.6 [437.3-447.7] | 443.7 [437.4-447.8] | 1.000 [1.000-1.000] | 1.000 | 444.1 [444.0-444.7] | 0.999 | 1095 | 0,0,0 |
| 1b_gpuP_p2048_u100_c128 | 1.00 | 433.0 [429.4-433.3] | 433.1 [429.5-433.4] | 1.000 [1.000-1.000] | 1.000 | — | — | 1095 | 0,0,0 |
| 1b_gpuP_p2048_u100_c256 | 1.00 | 441.1 [439.9-443.7] | 441.2 [440.0-443.8] | 1.000 [1.000-1.000] | 1.000 | — | — | 1095 | 0,0,0 |
| 1b_gpuP_p2048_u33 | 0.33 | 143.2 [141.6-143.6] | 434.2 [429.3-435.3] | 0.330 [0.330-0.330] | 0.323 | — | — | 1095 | 0,0,0 |
| 1b_gpuP_p2048_u50 | 0.50 | 216.2 [215.6-217.2] | 432.7 [431.4-434.7] | 0.500 [0.500-0.500] | 0.487 | — | — | 1095 | 0,0,0 |
| 1b_gpuP_p2048_u67 | 0.67 | 288.9 [287.0-290.8] | 431.6 [428.7-434.5] | 0.669 [0.669-0.669] | 0.651 | — | — | 1095 | 0,0,0 |
| 1b_gpuP_p2048_u80 | 0.80 | 346.4 [345.0-347.3] | 433.4 [431.8-434.7] | 0.799 [0.799-0.799] | 0.781 | — | — | 1095 | 0,0,0 |
| 1b_gpuP_p512_u100 | 1.00 | 549.4 [548.4-549.5] | 549.6 [548.6-549.7] | 1.000 [1.000-1.000] | 1.000 | 549.7 [549.4-559.2] | 1.000 | 1095 | 0,0,0 |
| 1b_gpuP_p512_u33 | 0.33 | 177.8 [177.8-178.3] | 539.2 [539.0-540.7] | 0.330 [0.330-0.330] | 0.324 | — | — | 1095 | 0,0,0 |
| 1b_gpuP_p512_u50 | 0.50 | 269.3 [269.0-269.7] | 539.3 [538.5-540.1] | 0.499 [0.499-0.499] | 0.490 | — | — | 1095 | 0,0,0 |
| 1b_gpuP_p512_u67 | 0.67 | 359.1 [359.0-360.2] | 536.6 [536.6-538.2] | 0.669 [0.669-0.669] | 0.654 | — | — | 1095 | 0,0,0 |
| 1b_gpuP_p512_u80 | 0.80 | 433.6 [433.0-434.0] | 543.0 [542.1-543.4] | 0.799 [0.798-0.799] | 0.789 | — | — | 1095 | 0,0,0 |
| 1b_gpuP_p512_u80_c128 | 0.80 | 400.4 [399.3-401.9] | 502.4 [501.2-504.5] | 0.797 [0.797-0.797] | — | — | — | 1095 | 0,0,0 |
| 1b_gpuP_p512_u80_c256 | 0.80 | 420.9 [420.8-424.1] | 527.6 [527.5-531.6] | 0.798 [0.798-0.798] | — | — | — | 1095 | 0,0,0 |
| 3b_cpuD_d0_m180 | 1.00 | 14.2 [14.1-14.8] | 14.2 [14.1-14.8] | 1.000 [1.000-1.000] | 1.000 | 16.2 [16.1-16.5] | 0.876 | — | 0,0,0 |
| 3b_cpuD_d0_m1f0 | 1.00 | 17.9 [16.8-18.1] | 17.9 [16.8-18.1] | 1.000 [1.000-1.000] | 1.000 | — | — | — | 0,1,1 |
| 3b_cpuD_d0_m3f0 | 1.00 | 17.4 [17.4-18.1] | 17.4 [17.4-18.1] | 1.000 [1.000-1.000] | 1.000 | 21.3 [21.1-21.3] | 0.818 | — | 1,1,1 |
| 3b_cpuD_d128_m180 | 1.00 | 12.3 [12.2-12.9] | 12.3 [12.3-12.9] | 1.000 [1.000-1.000] | 1.000 | 13.1 [12.5-13.4] | 0.937 | — | 0,0,0 |
| 3b_cpuD_d128_m1f0 | 1.00 | 17.1 [16.5-18.0] | 17.1 [16.5-18.0] | 1.000 [1.000-1.000] | 1.000 | — | — | — | 0,1,0 |
| 3b_cpuD_d128_m3f0 | 1.00 | 16.3 [16.1-18.8] | 16.3 [16.1-18.8] | 1.000 [1.000-1.000] | 1.000 | 20.6 [19.0-20.6] | 0.793 | — | 1,1,1 |
| 3b_cpuD_d2048_m180 | 1.00 | 8.5 [8.5-8.5] | 8.5 [8.5-8.5] | 1.000 [1.000-1.000] | 1.000 | 8.5 [8.5-8.5] | 0.998 | — | 0,0,0 |
| 3b_cpuD_d2048_m1f0 | 1.00 | 10.9 [10.9-11.4] | 10.9 [10.9-11.4] | 1.000 [1.000-1.000] | 1.000 | — | — | — | 1,1,1 |
| 3b_cpuD_d2048_m3f0 | 1.00 | 11.0 [11.0-11.7] | 11.0 [11.0-11.7] | 1.000 [1.000-1.000] | 1.000 | 9.7 [9.6-9.8] | 1.137 | — | 1,1,1 |
| 3b_cpuD_d512_m180 | 1.00 | 11.0 [10.9-11.7] | 11.0 [10.9-11.8] | 1.000 [1.000-1.000] | 1.000 | 10.9 [10.7-10.9] | 1.016 | — | 0,0,0 |
| 3b_cpuD_d512_m1f0 | 1.00 | 15.6 [15.2-16.2] | 15.7 [15.2-16.2] | 1.000 [1.000-1.000] | 1.000 | — | — | — | 0,1,1 |
| 3b_cpuD_d512_m3f0 | 1.00 | 14.8 [14.6-16.4] | 14.8 [14.6-16.5] | 1.000 [1.000-1.000] | 1.000 | 15.7 [15.6-16.4] | 0.942 | — | 1,1,1 |
| 3b_cpuP_p128_m3f0 | 1.00 | 112.1 [108.7-115.2] | 112.1 [108.7-115.2] | 1.000 [1.000-1.000] | 1.000 | 132.7 [131.3-134.1] | 0.845 | — | 0,1,1 |
| 3b_cpuP_p2048_m3f0 | 1.00 | 71.2 [70.7-75.4] | 71.2 [70.7-75.4] | 1.000 [1.000-1.000] | 1.000 | 75.5 [73.9-75.6] | 0.943 | — | 1,1,1 |
| 3b_cpuP_p512_m3f0 | 1.00 | 101.2 [101.0-107.0] | 101.2 [101.0-107.0] | 1.000 [1.000-1.000] | 1.000 | 111.1 [111.1-111.4] | 0.911 | — | 0,1,1 |
| 3b_gpuD_d0_u100 | 1.00 | 13.3 [13.2-13.4] | 13.3 [13.3-13.4] | 1.000 [1.000-1.000] | 1.000 | 13.1 [13.0-13.1] | 1.020 | 1095 | 0,0,0 |
| 3b_gpuD_d128_u100 | 1.00 | 13.2 [13.2-13.2] | 13.2 [13.2-13.2] | 1.000 [1.000-1.000] | 1.000 | 13.0 [12.8-13.0] | 1.016 | 1095 | 0,0,0 |
| 3b_gpuD_d2048_u100 | 1.00 | 11.7 [11.7-11.8] | 11.7 [11.7-11.8] | 1.000 [1.000-1.000] | 1.000 | 11.4 [11.2-11.4] | 1.028 | 1095 | 0,0,0 |
| 3b_gpuD_d512_u100 | 1.00 | 12.8 [12.8-12.9] | 12.8 [12.8-12.9] | 1.000 [1.000-1.000] | 1.000 | 12.5 [12.5-12.6] | 1.023 | 1095 | 0,0,0 |
| 3b_gpuP_p128_u100 | 1.00 | 190.1 [189.2-192.6] | 190.1 [189.2-192.7] | 1.000 [1.000-1.000] | 1.000 | 192.0 [183.6-193.5] | 0.990 | 1095 | 0,1,1 |
| 3b_gpuP_p2048_u100 | 1.00 | 161.0 [159.6-164.0] | 161.0 [159.6-164.0] | 1.000 [1.000-1.000] | 1.000 | 158.4 [157.6-164.3] | 1.016 | 1095 | 1,1,1 |
| 3b_gpuP_p2048_u100_c256 | 1.00 | 162.6 [162.6-167.6] | 162.7 [162.6-167.6] | 1.000 [1.000-1.000] | 1.000 | — | — | 1095 | 1,1,1 |
| 3b_gpuP_p512_u100 | 1.00 | 184.2 [184.2-186.2] | 184.2 [184.2-186.2] | 1.000 [1.000-1.000] | 1.000 | 176.7 [173.4-178.4] | 1.042 | 1095 | 1,1,1 |
| 3b_gpuP_p512_u80 | 0.80 | 146.7 [145.7-148.3] | 183.5 [182.3-185.5] | 0.800 [0.799-0.800] | 0.797 | — | — | 1095 | 0,0,0 |
| 3b_gpuP_p512_u80_c256 | 0.80 | 146.2 [143.5-146.3] | 183.0 [179.5-183.1] | 0.799 [0.799-0.799] | — | — | — | 1095 | 0,0,0 |

#### Co-runs (wall rates, tok/s; median [min-max] of 3 trials)

| co-run | GPU side (A) rate | / alone duty 1 | / paced alone | A duty | CPU side (B) rate | / alone | GPU MHz | CPU MHz, cores 4-6 / 7-8 / 9 | counted A, B per trial | counted s A, B (median) | thermal after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1b_gpuD_cpuD_u100_m180 | 39.7 [39.3-39.8] | 1.12 | 1.12 | 1.00 [1.00-1.00] | 12.8 [12.6-12.9] | 0.38 | 1095 | 1440 / 2016 / 672 | 1569/1569 · 504/1426, 1589/1589 · 515/1437, 1586/1586 · 511/1430 | 40, 40 | 1,1,1 |
| 1b_gpuD_cpuD_u100_m1c0 | 37.5 [35.9-38.1] | 1.06 | 1.06 | 1.00 [1.00-1.00] | 13.2 [12.4-13.5] | 0.29 | 1095 | 2016 / 2016 / 672 | 1498/1498 · 527/1566, 1522/1522 · 540/1588, 1432/1432 · 495/1537 | 40, 40 | 1,1,1 |
| 1b_gpuD_cpuD_u100_m1f0 | 36.2 [34.8-36.6] | 1.02 | 1.02 | 1.00 [1.00-1.00] | 13.0 [12.2-13.1] | 0.28 | 1095 | 2016 / 2016 / 672 | 1446/1446 · 518/1611, 1389/1389 · 485/1582, 1462/1462 · 524/1611 | 40, 40 | 1,1,1 |
| 1b_gpuD_cpuD_u100_m3f0 | 33.5 [33.0-34.3] | 0.95 | 0.95 | 1.00 [1.00-1.00] | 12.3 [11.9-12.8] | 0.26 | 1095 | 1824 / 1824 / 1824 | 1372/1372 · 514/1659, 1320/1320 · 474/1622, 1337/1337 · 489/1632 | 40, 40 | 2,1,2 |
| 1b_gpuD_cpuD_u33_m180 | 11.3 [11.2-11.4] | 0.32 | 1.49 | 0.33 [0.33-0.33] | 30.7 [30.0-31.1] | 0.91 | 1095 | 1056 / 2016 / 672 | 454/454 · 1240/2276, 456/456 · 1226/2257, 446/446 · 1197/2228 | 40, 40 | 0,0,0 |
| 1b_gpuD_cpuD_u33_m1f0 | 9.4 [9.4-9.8] | 0.27 | 1.24 | 0.33 [0.33-0.33] | 36.8 [36.6-37.5] | 0.79 | 1095 | 2016 / 2016 / 672 | 393/393 · 1495/2581, 377/377 · 1465/2555, 378/378 · 1460/2550 | 40, 40 | 1,1,1 |
| 1b_gpuD_cpuD_u33_m3f0 | 9.0 [8.6-9.2] | 0.25 | 1.18 | 0.33 [0.33-0.33] | 36.7 [35.9-36.8] | 0.77 | 1095 | 1824 / 1824 / 1824 | 361/361 · 1473/2622, 343/343 · 1435/2580, 367/367 · 1466/2620 | 40, 40 | 1,1,1 |
| 1b_gpuD_cpuD_u50_m180 | 20.3 [20.3-20.3] | 0.57 | 1.05 | 0.50 [0.50-0.50] | 25.9 [25.7-26.0] | 0.76 | 1095 | 1152 / 2016 / 672 | 812/812 · 1039/2021, 813/813 · 1034/1973, 814/814 · 1025/1978 | 40, 40 | 1,1,1 |
| 1b_gpuD_cpuD_u50_m1f0 | 18.7 [18.4-18.9] | 0.53 | 0.97 | 0.50 [0.50-0.50] | 29.9 [29.2-30.1] | 0.64 | 1095 | 2016 / 2016 / 672 | 749/749 · 1205/2293, 737/737 · 1164/2262, 756/756 · 1193/2286 | 40, 40 | 1,1,1 |
| 1b_gpuD_cpuD_u50_m3f0 | 17.9 [17.7-18.5] | 0.51 | 0.92 | 0.50 [0.50-0.50] | 30.8 [29.5-31.4] | 0.65 | 1095 | 1824 / 1824 / 1824 | 718/718 · 1230/2373, 708/708 · 1175/2307, 739/739 · 1255/2397 | 40, 40 | 2,2,2 |
| 1b_gpuD_cpuD_u67_m180 | 27.4 [27.2-27.7] | 0.77 | 1.13 | 0.66 [0.66-0.66] | 21.9 [21.6-22.0] | 0.65 | 1095 | 1152 / 2016 / 672 | 1107/1107 · 879/1895, 1094/1094 · 876/1837, 1087/1087 · 862/1799 | 40, 40 | 1,1,1 |
| 1b_gpuD_cpuD_u67_m1f0 | 24.9 [24.5-25.3] | 0.70 | 1.02 | 0.67 [0.67-0.67] | 23.3 [23.0-24.0] | 0.50 | 1095 | 2016 / 2016 / 672 | 1011/1011 · 957/2046, 977/977 · 917/1998, 994/994 · 931/2024 | 40, 40 | 1,1,1 |
| 1b_gpuD_cpuD_u67_m3f0 | 23.5 [22.8-24.1] | 0.66 | 0.96 | 0.67 [0.66-0.67] | 23.1 [21.7-23.5] | 0.49 | 1095 | 1824 / 1824 / 1824 | 961/961 · 938/2077, 938/938 · 922/2065, 911/911 · 867/2016 | 40, 40 | 2,2,1 |
| 1b_gpuD_cpuD_u80_m180 | 32.2 [30.9-33.4] | 0.91 | 1.17 | 0.79 [0.79-0.79] | 18.5 [18.0-19.3] | 0.55 | 1095 | 1248 / 2016 / 672 | 1333/1333 · 771/1806, 1287/1287 · 738/1680, 1235/1235 · 719/1671 | 40, 40 | 0,1,1 |
| 1b_gpuD_cpuD_u80_m1f0 | 28.2 [27.9-28.6] | 0.80 | 1.02 | 0.79 [0.79-0.80] | 18.8 [18.5-19.2] | 0.41 | 1095 | 2016 / 2016 / 672 | 1144/1144 · 764/1856, 1116/1116 · 740/1832, 1126/1126 · 752/1844 | 40, 40 | 1,1,1 |
| 1b_gpuD_cpuD_u80_m3f0 | 26.6 [26.4-26.7] | 0.75 | 0.97 | 0.79 [0.79-0.80] | 18.3 [18.0-18.3] | 0.38 | 1095 | 1824 / 1824 / 1824 | 1053/1053 · 719/1867, 1064/1064 · 729/1874, 1066/1066 · 733/1881 | 40, 40 | 1,2,2 |
| 1b_gpuD_cpuP_p2048_u100_m3f0 | 28.5 [28.4-30.9] | 0.80 | 0.80 | 1.00 [1.00-1.00] | 166.3 [164.8-171.1] | 0.75 | 1095 | 1824 / 1824 / 1824 | 1410/1874 · 4/4, 1394/1887 · 4/4, 1475/1964 · 4/4 | 49, 49 | 2,2,2 |
| 1b_gpuD_cpuP_p2048_u50_m3f0 | 21.4 [21.1-21.4] | 0.60 | 1.10 | 0.50 [0.50-0.50] | 203.9 [200.6-204.6] | 0.91 | 1095 | 2016 / 2016 / 2016 | 857/1470 · 4/4, 860/1454 · 4/4, 855/1151 · 4/4 | 40, 40 | 1,1,2 |
| 1b_gpuD_cpuP_p512_u100_m3f0 | 39.9 [37.0-41.0] | 1.13 | 1.13 | 1.00 [1.00-1.00] | 246.7 [237.5-248.7] | 0.81 | 1095 | 2016 / 2016 / 1824 | 1443/1875 · 18/20, 1493/1982 · 18/20, 1521/2078 · 18/20 | 37, 37 | 2,2,2 |
| 1b_gpuD_cpuP_p512_u50_m3f0 | 21.7 [21.7-22.1] | 0.61 | 1.12 | 0.50 [0.50-0.50] | 275.8 [272.1-276.2] | 0.90 | 1095 | 2016 / 2016 / 2016 | 818/1444 · 20/22, 845/1441 · 21/23, 837/1455 · 20/22 | 38, 38 | 1,1,1 |
| 1b_gpuP_cpuD_p2048_u100_m180 | 386.1 [385.1-404.8] | 0.87 | 0.87 | 1.00 [1.00-1.00] | 15.2 [14.6-15.3] | 0.45 | 1095 | 1536 / 1824 / 672 | 8/8 · 592/1523, 8/8 · 651/1533, 9/9 · 725/1457 | 43, 43 | 2,2,2 |
| 1b_gpuP_cpuD_p2048_u100_m180_c128 | 357.5 [344.4-387.0] | 0.81 | 0.83 | 1.00 [1.00-1.00] | 18.7 [18.2-19.0] | 0.55 | 1095 | 1344 / 1824 / 672 | 8/8 · 902/1670, 7/7 · 750/1708, 8/8 · 771/1681 | 42, 42 | 2,2,2 |
| 1b_gpuP_cpuD_p2048_u100_m180_c256 | 391.2 [347.6-400.6] | 0.88 | 0.89 | 1.00 [1.00-1.00] | 15.7 [15.5-18.3] | 0.46 | 1095 | 1440 / 1824 / 672 | 8/8 · 642/1573, 8/8 · 650/1551, 8/8 · 862/1622 | 42, 42 | 2,2,2 |
| 1b_gpuP_cpuD_p2048_u100_m1c0 | 353.9 [345.1-393.0] | 0.80 | 0.80 | 1.00 [1.00-1.00] | 15.5 [13.8-17.4] | 0.35 | 1095 | 1824 / 1824 / 672 | 8/8 · 574/1602, 7/7 · 704/1804, 8/8 · 737/1604 | 42, 42 | 2,2,2 |
| 1b_gpuP_cpuD_p2048_u100_m1f0 | 337.1 [324.7-346.5] | 0.76 | 0.76 | 1.00 [1.00-1.00] | 15.7 [15.5-17.0] | 0.34 | 1095 | 1824 / 1824 / 672 | 8/8 · 744/1679, 7/7 · 657/1681, 7/7 · 748/1782 | 44, 44 | 2,2,2 |
| 1b_gpuP_cpuD_p2048_u100_m3f0 | 305.8 [304.9-308.8] | 0.69 | 0.69 | 1.00 [1.00-1.00] | 16.1 [15.9-17.0] | 0.34 | 450 | 1728 / 1728 / 1728 | 6/6 · 683/1873, 7/7 · 752/1691, 7/7 · 738/1698 | 46, 46 | 2,2,2 |
| 1b_gpuP_cpuD_p2048_u33_m3f0 | 124.1 [122.8-128.7] | 0.28 | 0.87 | 0.33 [0.33-0.33] | 27.5 [26.0-27.7] | 0.58 | 1095 | 1824 / 1824 / 1824 | 3/3 · 1237/2315, 3/3 · 1241/2290, 3/3 · 1157/2244 | 50, 45 | 2,2,2 |
| 1b_gpuP_cpuD_p2048_u50_m3f0 | 168.3 [163.8-174.4] | 0.38 | 0.78 | 0.50 [0.50-0.50] | 24.2 [23.5-24.3] | 0.51 | 1095 | 1824 / 1824 / 1824 | 4/4 · 1157/2093, 4/4 · 1119/2133, 4/4 · 1047/2069 | 49, 46 | 2,2,2 |
| 1b_gpuP_cpuD_p2048_u67_m3f0 | 211.8 [206.1-220.2] | 0.48 | 0.73 | 0.67 [0.67-0.67] | 21.7 [18.4-21.8] | 0.46 | 450 | 1824 / 1824 / 1824 | 4/5 · 949/1749, 5/5 · 1026/1985, 5/5 · 984/2017 | 47, 47 | 2,2,2 |
| 1b_gpuP_cpuD_p2048_u80_m180 | 337.5 [329.9-345.8] | 0.76 | 0.97 | 0.80 [0.80-0.80] | 17.5 [16.6-18.6] | 0.52 | 1095 | 1440 / 1824 / 672 | 7/7 · 764/1686, 7/7 · 736/1638, 7/7 · 715/1576 | 42, 42 | 2,2,1 |
| 1b_gpuP_cpuD_p2048_u80_m1c0 | 305.2 [305.1-310.3] | 0.69 | 0.88 | 0.80 [0.80-0.80] | 18.5 [17.9-20.2] | 0.41 | 1095 | 1824 / 1824 / 672 | 7/7 · 840/1766, 7/7 · 935/1855, 7/7 · 831/1738 | 47, 46 | 2,2,2 |
| 1b_gpuP_cpuD_p2048_u80_m1f0 | 295.5 [288.9-298.2] | 0.67 | 0.85 | 0.80 [0.80-0.80] | 19.2 [18.4-19.4] | 0.41 | 1095 | 1824 / 1824 / 672 | 6/6 · 787/1924, 7/7 · 903/1761, 6/6 · 787/1937 | 42, 41 | 2,2,2 |
| 1b_gpuP_cpuD_p2048_u80_m3f0 | 248.6 [235.4-249.9] | 0.56 | 0.72 | 0.80 [0.80-0.80] | 19.0 [19.0-19.3] | 0.40 | 450 | 1728 / 1728 / 1728 | 6/6 · 923/1806, 5/5 · 825/1922, 5/5 · 769/1962 | 44, 43 | 2,2,2 |
| 1b_gpuP_cpuD_p512_u100_m180 | 511.8 [476.3-537.3] | 0.93 | 0.93 | 1.00 [1.00-1.00] | 19.5 [18.7-19.6] | 0.58 | 1095 | 1248 / 1824 / 672 | 43/43 · 798/1706, 40/40 · 747/1660, 38/38 · 800/1702 | 41, 41 | 1,2,2 |
| 1b_gpuP_cpuD_p512_u100_m1c0 | 480.7 [451.9-550.5] | 0.87 | 0.87 | 1.00 [1.00-1.00] | 21.5 [19.0-23.6] | 0.48 | 1095 | 1824 / 1824 / 672 | 43/43 · 942/2010, 36/36 · 874/1907, 39/39 · 788/1796 | 41, 41 | 1,2,2 |
| 1b_gpuP_cpuD_p512_u100_m1f0 | 447.9 [434.7-573.9] | 0.82 | 0.82 | 1.00 [1.00-1.00] | 21.1 [19.8-30.0] | 0.45 | 1095 | 1824 / 1824 / 672 | 45/45 · 1205/2557, 36/36 · 816/1891, 35/35 · 868/1942 | 41, 41 | 0,2,2 |
| 1b_gpuP_cpuD_p512_u100_m3f0 | 405.3 [387.4-405.9] | 0.74 | 0.74 | 1.00 [1.00-1.00] | 19.3 [19.1-21.6] | 0.41 | 1000 | 1824 / 1824 / 1728 | 31/31 · 884/2003, 33/33 · 793/1911, 33/33 · 806/1925 | 42, 42 | 2,2,2 |
| 1b_gpuP_cpuD_p512_u33_m3f0 | 165.8 [165.2-166.0] | 0.30 | 0.93 | 0.33 [0.33-0.33] | 30.5 [30.1-32.1] | 0.64 | 1095 | 1824 / 1824 / 1824 | 14/14 · 1266/2457, 13/13 · 1139/2352, 14/14 · 1250/2398 | 43, 39 | 2,2,1 |
| 1b_gpuP_cpuD_p512_u50_m3f0 | 223.3 [217.8-240.4] | 0.41 | 0.83 | 0.50 [0.50-0.50] | 27.6 [26.5-29.4] | 0.58 | 1095 | 1824 / 1824 / 1824 | 19/19 · 1147/2261, 19/19 · 1021/2241, 18/18 · 1185/2343 | 42, 40 | 2,2,2 |
| 1b_gpuP_cpuD_p512_u67_m3f0 | 289.5 [280.4-294.3] | 0.53 | 0.81 | 0.67 [0.67-0.67] | 25.6 [25.6-25.9] | 0.54 | 1095 | 1824 / 1824 / 1824 | 22/22 · 1015/2214, 24/24 · 1060/2179, 24/24 · 1042/2186 | 42, 41 | 2,2,2 |
| 1b_gpuP_cpuD_p512_u80_m180 | 435.1 [427.1-436.6] | 0.79 | 1.00 | 0.80 [0.80-0.80] | 22.3 [21.1-22.7] | 0.66 | 1095 | 1248 / 2016 / 672 | 34/34 · 887/1813, 34/34 · 852/1763, 35/35 · 926/1823 | 41, 40 | 2,1,2 |
| 1b_gpuP_cpuD_p512_u80_m1c0 | 418.2 [373.0-430.8] | 0.76 | 0.96 | 0.80 [0.80-0.80] | 23.7 [21.5-24.7] | 0.53 | 1095 | 1824 / 1824 / 672 | 34/34 · 952/2001, 30/30 · 1006/2044, 33/33 · 863/1918 | 40, 40 | 2,2,2 |
| 1b_gpuP_cpuD_p512_u80_m1f0 | 366.0 [350.9-394.6] | 0.67 | 0.84 | 0.80 [0.80-0.80] | 22.9 [22.4-24.4] | 0.49 | 1095 | 1824 / 1824 / 672 | 31/31 · 910/2029, 30/30 · 929/1994, 29/29 · 1022/2068 | 42, 41 | 2,2,2 |
| 1b_gpuP_cpuD_p512_u80_m1f0_c128 | 317.5 [312.3-331.9] | 0.58 | 0.79 | 0.80 [0.80-0.80] | 28.6 [28.3-29.5] | 0.62 | 1095 | 1824 / 1824 / 672 | 27/27 · 1226/2343, 26/26 · 1195/2306, 25/25 · 1154/2293 | 42, 42 | 2,2,2 |
| 1b_gpuP_cpuD_p512_u80_m1f0_c256 | 337.9 [318.8-353.6] | 0.62 | 0.80 | 0.80 [0.80-0.80] | 24.7 [23.5-25.0] | 0.53 | 1095 | 1824 / 1824 / 672 | 27/27 · 1002/2120, 26/26 · 1036/2144, 29/29 · 982/2091 | 42, 41 | 2,2,2 |
| 1b_gpuP_cpuD_p512_u80_m3f0 | 323.1 [319.3-341.3] | 0.59 | 0.75 | 0.80 [0.80-0.80] | 23.7 [22.6-23.7] | 0.50 | 1095 | 1824 / 1824 / 1728 | 27/27 · 906/2051, 26/26 · 976/2103, 26/26 · 966/2110 | 41, 41 | 2,2,2 |
| 3b_gpuD_cpuD_u100_m180 | 13.7 [13.2-13.8] | 1.03 | 1.03 | 1.00 [1.00-1.00] | 4.9 [4.7-5.0] | 0.34 | 1095 | 1440 / 2016 / 672 | 553/553 · 199/883, 529/529 · 187/877, 550/550 · 193/882 | 40, 40 | 1,1,1 |
| 3b_gpuD_cpuD_u100_m1f0 | 12.6 [12.3-12.8] | 0.95 | 0.95 | 1.00 [1.00-1.00] | 4.6 [4.6-4.9] | 0.26 | 1095 | 1824 / 1824 / 672 | 510/510 · 196/1004, 494/494 · 183/995, 506/506 · 185/998 | 40, 40 | 1,1,1 |
| 3b_gpuD_cpuD_u100_m3f0 | 11.7 [11.6-11.9] | 0.88 | 0.88 | 1.00 [1.00-1.00] | 4.4 [4.4-4.7] | 0.25 | 1095 | 1824 / 1824 / 1824 | 466/466 · 185/1028, 475/475 · 177/1015, 469/469 · 176/1019 | 40, 40 | 2,2,2 |
| 3b_gpuD_cpuP_p2048_u100_m3f0 | 5.6 [5.3-5.9] | 0.42 | 0.42 | 1.00 [1.00-1.00] | 52.1 [51.0-52.8] | 0.73 | 450 | 1728 / 1728 / 1728 | 421/959 · 2/2, 457/1023 · 2/2, 440/1027 · 2/2 | 78, 79 | 2,2,2 |
| 3b_gpuD_cpuP_p512_u100_m3f0 | 12.0 [11.6-12.9] | 0.90 | 0.90 | 1.00 [1.00-1.00] | 80.1 [77.9-82.3] | 0.79 | 1095 | 1824 / 2016 / 1824 | 508/1263 · 6/7, 529/1259 · 6/7, 502/1269 · 6/7 | 42, 38 | 2,2,2 |
| 3b_gpuP_cpuD_p2048_u100_m180_c256 | 99.8 [98.3-101.6] | 0.62 | 0.61 | 1.00 [1.00-1.00] | 5.9 [5.8-6.0] | 0.42 | 450 | 1344 / 1728 / 672 | 3/3 · 370/788, 3/3 · 352/774, 3/3 · 371/807 | 62, 61 | 2,2,2 |
| 3b_gpuP_cpuD_p2048_u100_m3f0 | 90.7 [88.9-94.5] | 0.56 | 0.56 | 1.00 [1.00-1.00] | 6.5 [6.1-6.6] | 0.37 | 450 | 1728 / 1728 / 1728 | 2/2 · 301/962, 2/2 · 265/989, 2/2 · 293/971 | 45, 45 | 2,2,2 |
| 3b_gpuP_cpuD_p512_u100_m3f0 | 109.0 [107.3-120.3] | 0.59 | 0.59 | 1.00 [1.00-1.00] | 6.7 [6.7-7.4] | 0.39 | 450 | 1728 / 1728 / 1728 | 10/10 · 284/1083, 9/9 · 289/1089, 9/9 · 312/1111 | 43, 42 | 2,2,2 |
| 3b_gpuP_cpuD_p512_u80_m1f0_c256 | 99.4 [97.6-100.3] | 0.54 | 0.68 | 0.80 [0.80-0.80] | 7.8 [7.8-8.2] | 0.44 | 675 | 1776 / 1776 / 672 | 8/8 · 330/1141, 9/9 · 364/1090, 8/8 · 311/1131 | 41, 40 | 2,2,2 |

#### 1b: trade-off curve, gpu->cpu

| setting | duty | mask | P | C | GPU prefill ratio | CPU decode ratio |
|---|---|---|---|---|---|---|
| 1b_gpuP_cpuD_p512_u100_m180 | 1.00 | 180 | 512 | 512 | 0.932 | 0.575 |
| 1b_gpuP_cpuD_p512_u80_m180 | 0.80 | 180 | 512 | 512 | 0.792 | 0.659 |
| 1b_gpuP_cpuD_p512_u100_m1c0 | 1.00 | 1c0 | 512 | 512 | 0.875 | 0.478 |
| 1b_gpuP_cpuD_p512_u80_m1c0 | 0.80 | 1c0 | 512 | 512 | 0.761 | 0.528 |
| 1b_gpuP_cpuD_p512_u80_m1f0_c128 | 0.80 | 1f0 | 512 | 128 | 0.578 | 0.616 |
| 1b_gpuP_cpuD_p512_u80_m1f0_c256 | 0.80 | 1f0 | 512 | 256 | 0.615 | 0.531 |
| 1b_gpuP_cpuD_p512_u100_m1f0 | 1.00 | 1f0 | 512 | 512 | 0.815 | 0.454 |
| 1b_gpuP_cpuD_p512_u80_m1f0 | 0.80 | 1f0 | 512 | 512 | 0.666 | 0.493 |
| 1b_gpuP_cpuD_p512_u100_m3f0 | 1.00 | 3f0 | 512 | 512 | 0.738 | 0.406 |
| 1b_gpuP_cpuD_p512_u80_m3f0 | 0.80 | 3f0 | 512 | 512 | 0.588 | 0.498 |
| 1b_gpuP_cpuD_p512_u67_m3f0 | 0.67 | 3f0 | 512 | 512 | 0.527 | 0.537 |
| 1b_gpuP_cpuD_p512_u50_m3f0 | 0.50 | 3f0 | 512 | 512 | 0.406 | 0.579 |
| 1b_gpuP_cpuD_p512_u33_m3f0 | 0.33 | 3f0 | 512 | 512 | 0.302 | 0.640 |
| 1b_gpuP_cpuD_p2048_u100_m180_c128 | 1.00 | 180 | 2048 | 128 | 0.806 | 0.552 |
| 1b_gpuP_cpuD_p2048_u100_m180_c256 | 1.00 | 180 | 2048 | 256 | 0.882 | 0.464 |
| 1b_gpuP_cpuD_p2048_u100_m180 | 1.00 | 180 | 2048 | 512 | 0.870 | 0.449 |
| 1b_gpuP_cpuD_p2048_u80_m180 | 0.80 | 180 | 2048 | 512 | 0.761 | 0.516 |
| 1b_gpuP_cpuD_p2048_u100_m1c0 | 1.00 | 1c0 | 2048 | 512 | 0.798 | 0.346 |
| 1b_gpuP_cpuD_p2048_u80_m1c0 | 0.80 | 1c0 | 2048 | 512 | 0.688 | 0.411 |
| 1b_gpuP_cpuD_p2048_u100_m1f0 | 1.00 | 1f0 | 2048 | 512 | 0.760 | 0.339 |
| 1b_gpuP_cpuD_p2048_u80_m1f0 | 0.80 | 1f0 | 2048 | 512 | 0.666 | 0.414 |
| 1b_gpuP_cpuD_p2048_u100_m3f0 | 1.00 | 3f0 | 2048 | 512 | 0.689 | 0.337 |
| 1b_gpuP_cpuD_p2048_u80_m3f0 | 0.80 | 3f0 | 2048 | 512 | 0.560 | 0.400 |
| 1b_gpuP_cpuD_p2048_u67_m3f0 | 0.67 | 3f0 | 2048 | 512 | 0.478 | 0.457 |
| 1b_gpuP_cpuD_p2048_u50_m3f0 | 0.50 | 3f0 | 2048 | 512 | 0.379 | 0.509 |
| 1b_gpuP_cpuD_p2048_u33_m3f0 | 0.33 | 3f0 | 2048 | 512 | 0.280 | 0.578 |

#### 1b: trade-off curve, cpu->gpu

| setting | duty | mask | P | C | CPU prefill ratio | GPU decode ratio |
|---|---|---|---|---|---|---|
| 1b_gpuD_cpuP_p512_u100_m3f0 | 1.00 | 3f0 | 512 | 512 | 0.808 | 1.129 |
| 1b_gpuD_cpuP_p512_u50_m3f0 | 0.50 | 3f0 | 512 | 512 | 0.904 | 0.614 |
| 1b_gpuD_cpuP_p2048_u100_m3f0 | 1.00 | 3f0 | 2048 | 512 | 0.745 | 0.804 |
| 1b_gpuD_cpuP_p2048_u50_m3f0 | 0.50 | 3f0 | 2048 | 512 | 0.913 | 0.604 |

#### 1b: trade-off curve, dp

| setting | duty | mask | P | C | GPU decode ratio | CPU decode ratio |
|---|---|---|---|---|---|---|
| 1b_gpuD_cpuD_u100_m180 | 1.00 | 180 | — | — | 1.121 | 0.377 |
| 1b_gpuD_cpuD_u80_m180 | 0.80 | 180 | — | — | 0.910 | 0.545 |
| 1b_gpuD_cpuD_u67_m180 | 0.67 | 180 | — | — | 0.774 | 0.647 |
| 1b_gpuD_cpuD_u50_m180 | 0.50 | 180 | — | — | 0.575 | 0.764 |
| 1b_gpuD_cpuD_u33_m180 | 0.33 | 180 | — | — | 0.321 | 0.906 |
| 1b_gpuD_cpuD_u100_m1c0 | 1.00 | 1c0 | — | — | 1.059 | 0.294 |
| 1b_gpuD_cpuD_u100_m1f0 | 1.00 | 1f0 | — | — | 1.023 | 0.279 |
| 1b_gpuD_cpuD_u80_m1f0 | 0.80 | 1f0 | — | — | 0.797 | 0.405 |
| 1b_gpuD_cpuD_u67_m1f0 | 0.67 | 1f0 | — | — | 0.703 | 0.502 |
| 1b_gpuD_cpuD_u50_m1f0 | 0.50 | 1f0 | — | — | 0.529 | 0.643 |
| 1b_gpuD_cpuD_u33_m1f0 | 0.33 | 1f0 | — | — | 0.267 | 0.791 |
| 1b_gpuD_cpuD_u100_m3f0 | 1.00 | 3f0 | — | — | 0.946 | 0.258 |
| 1b_gpuD_cpuD_u80_m3f0 | 0.80 | 3f0 | — | — | 0.752 | 0.383 |
| 1b_gpuD_cpuD_u67_m3f0 | 0.67 | 3f0 | — | — | 0.663 | 0.485 |
| 1b_gpuD_cpuD_u50_m3f0 | 0.50 | 3f0 | — | — | 0.507 | 0.647 |
| 1b_gpuD_cpuD_u33_m3f0 | 0.33 | 3f0 | — | — | 0.255 | 0.772 |

#### 1b: each specialized setting alone, best gain over its mixes

| setting | family | best gain | at P, D |
|---|---|---|---|
| 1b_gpuD_cpuP_p512_u100_m3f0 | cpu->gpu | 0.992 | 512, 128 |
| 1b_gpuP_cpuD_p512_u80_m1f0_c256 | gpu->cpu | 0.845 | 512, 32 |
| 1b_gpuP_cpuD_p512_u80_m1f0 | gpu->cpu | 0.818 | 512, 32 |
| 1b_gpuP_cpuD_p512_u80_m3f0 | gpu->cpu | 0.808 | 512, 32 |
| 1b_gpuP_cpuD_p512_u80_m1f0_c128 | gpu->cpu | 0.794 | 512, 32 |
| 1b_gpuD_cpuP_p2048_u100_m3f0 | cpu->gpu | 0.763 | 2048, 512 |
| 1b_gpuP_cpuD_p512_u80_m1c0 | gpu->cpu | 0.762 | 512, 32 |
| 1b_gpuP_cpuD_p512_u100_m1f0 | gpu->cpu | 0.752 | 512, 32 |
| 1b_gpuP_cpuD_p2048_u100_m180_c256 | gpu->cpu | 0.740 | 2048, 32 |
| 1b_gpuP_cpuD_p2048_u100_m180 | gpu->cpu | 0.731 | 2048, 32 |
| 1b_gpuP_cpuD_p512_u80_m180 | gpu->cpu | 0.729 | 512, 32 |
| 1b_gpuP_cpuD_p512_u33_m3f0 | gpu->cpu | 0.729 | 128, 32 |
| 1b_gpuP_cpuD_p512_u67_m3f0 | gpu->cpu | 0.724 | 512, 32 |
| 1b_gpuP_cpuD_p512_u100_m3f0 | gpu->cpu | 0.693 | 512, 32 |
| 1b_gpuD_cpuP_p512_u50_m3f0 | cpu->gpu | 0.690 | 512, 32 |
| 1b_gpuP_cpuD_p512_u100_m1c0 | gpu->cpu | 0.689 | 512, 32 |
| 1b_gpuP_cpuD_p2048_u100_m180_c128 | gpu->cpu | 0.677 | 2048, 32 |
| 1b_gpuP_cpuD_p2048_u100_m1c0 | gpu->cpu | 0.670 | 2048, 32 |
| 1b_gpuP_cpuD_p2048_u80_m1f0 | gpu->cpu | 0.663 | 2048, 128 |
| 1b_gpuP_cpuD_p512_u50_m3f0 | gpu->cpu | 0.659 | 128, 32 |
| 1b_gpuP_cpuD_p2048_u80_m3f0 | gpu->cpu | 0.644 | 2048, 128 |
| 1b_gpuP_cpuD_p2048_u67_m3f0 | gpu->cpu | 0.644 | 2048, 128 |
| 1b_gpuP_cpuD_p2048_u80_m180 | gpu->cpu | 0.639 | 2048, 32 |
| 1b_gpuP_cpuD_p2048_u100_m1f0 | gpu->cpu | 0.638 | 2048, 32 |
| 1b_gpuP_cpuD_p512_u100_m180 | gpu->cpu | 0.636 | 512, 32 |
| 1b_gpuD_cpuP_p2048_u50_m3f0 | cpu->gpu | 0.619 | 2048, 128 |
| 1b_gpuP_cpuD_p2048_u100_m3f0 | gpu->cpu | 0.579 | 2048, 32 |
| 1b_gpuP_cpuD_p2048_u80_m1c0 | gpu->cpu | 0.578 | 2048, 32 |
| 1b_gpuP_cpuD_p2048_u33_m3f0 | gpu->cpu | 0.567 | 2048, 512 |
| 1b_gpuP_cpuD_p2048_u50_m3f0 | gpu->cpu | 0.512 | 2048, 128 |

#### 1b: predicted request throughput (requests/s)

| P | D | best coordinated specialized (setting) | best coordinated data-parallel (setting) | best single PU | uncoordinated: specialized, data-parallel, gain (llm-phase-affinity) | duty-1 3f0: specialized, data-parallel, gain | baseline (from) | **gain** | gain, trial spread | gain vs coordinated data-parallel | gain vs this study's baseline only | diag.: data-parallel with co-run prefill |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 1.2451 (1b_gpuD_cpuP_p512_u100_m3f0) | 1.2958 (1b_gpuD_cpuD_u100_m180) | 0.9375 (CPU 3f0) | 0.6795, 0.9869, 0.69 | 1.2451, 1.1729, 1.06 | 1.2958 (coordinated data-parallel) | **0.96** | 0.88-0.99 | 0.96 | 0.96 | 1.2156 |
| 128 | 128 | 0.3113 (1b_gpuD_cpuP_p512_u100_m3f0) | 0.3808 (1b_gpuD_cpuD_u100_m180) | 0.3226 (CPU 3f0) | 0.1699, 0.2761, 0.62 | 0.3113, 0.3377, 0.86 | 0.3808 (coordinated data-parallel) | **0.82** | 0.75-0.85 | 0.82 | 0.82 | 0.3734 |
| 128 | 512 | 0.0778 (1b_gpuD_cpuP_p512_u100_m3f0) | 0.0996 (1b_gpuD_cpuD_u100_m180) | 0.0890 (CPU 3f0) | 0.0425, 0.0712, 0.60 | 0.0778, 0.0878, 0.78 | 0.0996 (coordinated data-parallel) | **0.78** | 0.72-0.80 | 0.78 | 0.78 | 0.0991 |
| 512 | 32 | 0.6596 (1b_gpuP_cpuD_p512_u80_m1f0_c256) | 0.7807 (1b_gpuD_cpuD_u100_m180) | 0.5394 (GPU) | 0.4990, 0.6704, 0.74 | 0.5413, 0.7423, 0.73 | 0.7807 (coordinated data-parallel) | **0.84** | 0.79-1.08 | 0.84 | 0.84 | 0.6738 |
| 512 | 128 | 0.3060 (1b_gpuD_cpuP_p512_u100_m3f0) | 0.3085 (1b_gpuD_cpuD_u100_m180) | 0.2164 (GPU) | 0.1576, 0.2384, 0.66 | 0.3060, 0.2820, 1.09 | 0.3085 (coordinated data-parallel) | **0.99** | 0.91-1.03 | 0.99 | 0.99 | 0.2893 |
| 512 | 512 | 0.0765 (1b_gpuD_cpuP_p512_u100_m3f0) | 0.0907 (1b_gpuD_cpuD_u100_m180) | 0.0731 (CPU 3f0) | 0.0394, 0.0667, 0.59 | 0.0765, 0.0812, 0.92 | 0.0907 (coordinated data-parallel) | **0.84** | 0.78-0.87 | 0.84 | 0.84 | 0.0889 |
| 2048 | 32 | 0.1909 (1b_gpuP_cpuD_p2048_u100_m180_c256) | 0.2578 (1b_gpuD_cpuD_u100_m1f0) | 0.1781 (GPU) | 0.1172, 0.2405, 0.49 | 0.1492, 0.2537, 0.59 | 0.2578 (coordinated data-parallel) | **0.74** | 0.72-0.77 | 0.74 | 0.74 | 0.1936 |
| 2048 | 128 | 0.1065 (1b_gpuP_cpuD_p2048_u80_m1f0) | 0.1606 (1b_gpuD_cpuD_u100_m1f0) | 0.1161 (GPU) | 0.0892, 0.1359, 0.66 | 0.0873, 0.1545, 0.57 | 0.1606 (coordinated data-parallel) | **0.66** | 0.64-0.67 | 0.66 | 0.66 | 0.1324 |
| 2048 | 512 | 0.0503 (1b_gpuD_cpuP_p2048_u100_m3f0) | 0.0659 (1b_gpuD_cpuD_u100_m180) | 0.0486 (GPU) | 0.0318, 0.0500, 0.64 | 0.0503, 0.0610, 0.83 | 0.0659 (coordinated data-parallel) | **0.76** | 0.76-0.84 | 0.76 | 0.76 | 0.0602 |

#### 3b: trade-off curve, gpu->cpu

| setting | duty | mask | P | C | GPU prefill ratio | CPU decode ratio |
|---|---|---|---|---|---|---|
| 3b_gpuP_cpuD_p512_u80_m1f0_c256 | 0.80 | 1f0 | 512 | 256 | 0.540 | 0.437 |
| 3b_gpuP_cpuD_p512_u100_m3f0 | 1.00 | 3f0 | 512 | 512 | 0.592 | 0.386 |
| 3b_gpuP_cpuD_p2048_u100_m180_c256 | 1.00 | 180 | 2048 | 256 | 0.620 | 0.419 |
| 3b_gpuP_cpuD_p2048_u100_m3f0 | 1.00 | 3f0 | 2048 | 512 | 0.563 | 0.373 |

#### 3b: trade-off curve, cpu->gpu

| setting | duty | mask | P | C | CPU prefill ratio | GPU decode ratio |
|---|---|---|---|---|---|---|
| 3b_gpuD_cpuP_p512_u100_m3f0 | 1.00 | 3f0 | 512 | 512 | 0.791 | 0.902 |
| 3b_gpuD_cpuP_p2048_u100_m3f0 | 1.00 | 3f0 | 2048 | 512 | 0.732 | 0.421 |

#### 3b: trade-off curve, dp

| setting | duty | mask | P | C | GPU decode ratio | CPU decode ratio |
|---|---|---|---|---|---|---|
| 3b_gpuD_cpuD_u100_m180 | 1.00 | 180 | — | — | 1.032 | 0.343 |
| 3b_gpuD_cpuD_u100_m1f0 | 1.00 | 1f0 | — | — | 0.949 | 0.259 |
| 3b_gpuD_cpuD_u100_m3f0 | 1.00 | 3f0 | — | — | 0.881 | 0.255 |

#### 3b: each specialized setting alone, best gain over its mixes

| setting | family | best gain | at P, D |
|---|---|---|---|
| 3b_gpuD_cpuP_p512_u100_m3f0 | cpu->gpu | 0.852 | 512, 128 |
| 3b_gpuP_cpuD_p512_u80_m1f0_c256 | gpu->cpu | 0.729 | 512, 32 |
| 3b_gpuP_cpuD_p512_u100_m3f0 | gpu->cpu | 0.670 | 512, 32 |
| 3b_gpuP_cpuD_p2048_u100_m3f0 | gpu->cpu | 0.567 | 2048, 128 |
| 3b_gpuP_cpuD_p2048_u100_m180_c256 | gpu->cpu | 0.540 | 2048, 32 |
| 3b_gpuD_cpuP_p2048_u100_m3f0 | cpu->gpu | 0.448 | 2048, 128 |

#### 3b: predicted request throughput (requests/s)

| P | D | best coordinated specialized (setting) | best coordinated data-parallel (setting) | best single PU | uncoordinated: specialized, data-parallel, gain (llm-phase-affinity) | duty-1 3f0: specialized, data-parallel, gain | baseline (from) | **gain** | gain, trial spread | gain vs coordinated data-parallel | gain vs this study's baseline only | diag.: data-parallel with co-run prefill |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 0.3709 (3b_gpuD_cpuP_p512_u100_m3f0) | 0.4447 (3b_gpuD_cpuD_u100_m180) | 0.3317 (CPU 1f0) | 0.2477, 0.3694, 0.67 | 0.3709, 0.4045, 0.92 | 0.4447 (coordinated data-parallel) | **0.83** | 0.80-0.92 | 0.83 | 0.83 | 0.3970 |
| 128 | 128 | 0.0927 (3b_gpuD_cpuP_p512_u100_m3f0) | 0.1308 (3b_gpuD_cpuD_u100_m180) | 0.1158 (CPU 1f0) | 0.0619, 0.1042, 0.59 | 0.0927, 0.1167, 0.67 | 0.1394 (single PU (CPU, llm-phase-affinity)) | **0.67** | 0.64-0.71 | 0.71 | 0.71 | 0.1262 |
| 128 | 512 | 0.0232 (3b_gpuD_cpuP_p512_u100_m3f0) | 0.0342 (3b_gpuD_cpuD_u100_m180) | 0.0321 (CPU 1f0) | 0.0155, 0.0269, 0.57 | 0.0232, 0.0304, 0.60 | 0.0388 (single PU (CPU, llm-phase-affinity)) | **0.60** | 0.58-0.64 | 0.68 | 0.68 | 0.0339 |
| 512 | 32 | 0.1940 (3b_gpuP_cpuD_p512_u80_m1f0_c256) | 0.2662 (3b_gpuD_cpuD_u100_m180) | 0.1895 (GPU) | 0.1657, 0.2271, 0.73 | 0.1783, 0.2518, 0.71 | 0.2662 (coordinated data-parallel) | **0.73** | 0.71-0.75 | 0.73 | 0.73 | 0.2078 |
| 512 | 128 | 0.0902 (3b_gpuD_cpuP_p512_u100_m3f0) | 0.1059 (3b_gpuD_cpuD_u100_m180) | 0.0783 (GPU) | 0.0598, 0.0831, 0.72 | 0.0902, 0.0964, 0.94 | 0.1059 (coordinated data-parallel) | **0.85** | 0.82-0.94 | 0.85 | 0.85 | 0.0944 |
| 512 | 512 | 0.0226 (3b_gpuD_cpuP_p512_u100_m3f0) | 0.0312 (3b_gpuD_cpuD_u100_m180) | 0.0265 (CPU 1f0) | 0.0150, 0.0236, 0.64 | 0.0226, 0.0279, 0.81 | 0.0312 (coordinated data-parallel) | **0.72** | 0.69-0.80 | 0.72 | 0.72 | 0.0301 |
| 2048 | 32 | 0.0487 (3b_gpuP_cpuD_p2048_u100_m180_c256) | 0.0903 (3b_gpuD_cpuD_u100_m180) | 0.0647 (GPU) | 0.0351, 0.0831, 0.42 | 0.0442, 0.0881, 0.50 | 0.0903 (coordinated data-parallel) | **0.54** | 0.53-0.55 | 0.54 | 0.54 | 0.0595 |
| 2048 | 128 | 0.0321 (3b_gpuP_cpuD_p2048_u100_m3f0) | 0.0567 (3b_gpuD_cpuD_u100_m180) | 0.0423 (GPU) | 0.0261, 0.0469, 0.56 | 0.0321, 0.0533, 0.60 | 0.0567 (coordinated data-parallel) | **0.57** | 0.53-0.58 | 0.57 | 0.57 | 0.0422 |
| 2048 | 512 | 0.0097 (3b_gpuD_cpuP_p2048_u100_m3f0) | 0.0231 (3b_gpuD_cpuD_u100_m180) | 0.0178 (GPU) | 0.0086, 0.0173, 0.50 | 0.0097, 0.0208, 0.46 | 0.0231 (coordinated data-parallel) | **0.42** | 0.39-0.45 | 0.42 | 0.42 | 0.0201 |

### Pixel 7a (3A021JEHN02756), ExecuTorch 1B, CPU decode width (design D5)

| CPU decode mask | GPU prefill next to CPU decode | CPU decode next to GPU prefill | both decoding: CPU, GPU |
|---|---|---|---|
| f0 | 0.901 | 0.603 | 0.750, 1.005 |
| c0 | 0.947 | 0.620 | 0.714, 1.028 |
| 30 | 0.960 | 0.639 | 0.795, 1.063 |

| P | D | specialized by CPU mask (f0, c0, 30) | data-parallel by mask (f0, c0, 30) | best single PU | uncoordinated: specialized, data-parallel, gain | baseline (from) | **gain** |
|---|---|---|---|---|---|---|---|
| 128 | 32 | 0.6141, 0.5570, 0.5750 | 0.8955, 0.8479, 0.8940 | 0.5976 (CPU f0) | 0.5977, 0.8341, 0.72 | 0.8955 (coordinated data-parallel) | **0.69** |
| 128 | 128 | 0.1536, 0.1393, 0.1438 | 0.2802, 0.2585, 0.2771 | 0.2164 (CPU f0) | 0.1495, 0.2605, 0.57 | 0.2802 (coordinated data-parallel) | **0.55** |
| 128 | 512 | 0.0384, 0.0348, 0.0359 | 0.0750, 0.0685, 0.0739 | 0.0609 (CPU f0) | 0.0374, 0.0697, 0.54 | 0.0750 (coordinated data-parallel) | **0.51** |
| 512 | 32 | 0.4887, 0.4560, 0.4609 | 0.5071, 0.5002, 0.5136 | 0.2921 (GPU) | 0.4964, 0.4822, 1.03 | 0.5136 (coordinated data-parallel) | **0.95** |
| 512 | 128 | 0.1223, 0.1141, 0.1153 | 0.2034, 0.1953, 0.2044 | 0.1256 (CPU f0) | 0.1242, 0.1937, 0.64 | 0.2044 (coordinated data-parallel) | **0.60** |
| 512 | 512 | 0.0306, 0.0285, 0.0288 | 0.0613, 0.0577, 0.0612 | 0.0439 (CPU f0) | 0.0311, 0.0587, 0.53 | 0.0613 (coordinated data-parallel) | **0.50** |


### Gate

**NO-GO**: best gain 0.99 (gate 1.2) with 1b, P=512, D=128, setting 1b_gpuD_cpuP_p512_u100_m3f0, against coordinated data-parallel.
Pixel ExecuTorch 1B: best gain 0.95 at P=512, D=32 (GPU prefill -> CPU decode on f0), against coordinated data-parallel.
