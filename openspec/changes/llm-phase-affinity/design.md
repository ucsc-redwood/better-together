# Design

## Context

Assets already on the phones (owner's, read-only for this study):
- **Samsung S24+** (`R5CY21Y3VEV`, SM-S926B, Exynos 2400: cores 0-3 little, 4-6 medium, 7-8 big, 9 super; Xclipse 940,
  subgroup 64; 11.5 GB RAM, ~8.3 GB available):
  - `/data/local/tmp/lcpp`: llama.cpp `llama-bench`, `llama-completion`, `llama-tokenize`, Vulkan backend (driver reports
    `int dot: 0`, `matrix cores: none`, UMA), and `models/llama3_2_{1b,3b}_q4_0.gguf`, `llama3_1_8b_q4_0.gguf`.
  - `/data/local/tmp/llama_dev`: ExecuTorch Vulkan `.pte` files for 1B/3B/8B in 4w and 8da4w (buffer, texture and
    `embq` variants, ctx 3072), plus ETDump files.
  - `/data/local/tmp/llama_bench`, `llama_r15`, `llama`, `llama_vk`: further Vulkan `.pte` files and `llama_main`
    runners (baseline, coopmat, vanilla variants).
- **Pixel 7a** (`3A021JEHN02756`: 2 X1 big, 2 A78 medium, 4 A55 little; Mali-G710; 7.6 GB RAM, ~3.9 GB available):
  `/data/local/tmp/llama` holds the 1B Vulkan `.pte` files (4w, 8da4w) and a tokenizer; no llama.cpp build and no
  runner.

Sources outside the phones: `~/llamacpp-compare/llama.cpp` (llama.cpp source); `~/Desktop/executorch` (ExecuTorch
checkout at `e229c58784` on branch `hmz/shader-speedup-test` with uncommitted Vulkan shader edits, so it must not be
touched); `examples/models/llama/config/llama_q8da4w.yaml` there is the XNNPACK 8da4w export recipe.

**Inventory taken 2026-10-20 (task 1.1; raw listings in `data/llm-phase-affinity/<serial>/inventory.txt`):**
- Samsung (One UI build `S926BXXSHDZI1`, Android 16):
  - `lcpp/llama-bench` (40.5 MB): no `--version` flag; its `-o jsonl` records give `build_commit` **c745be2**. Its
    disassembly has `sdot` (dotprod) but no `smmla` (i8mm), no KleidiAI and no OpenMP. Models
    `llama3_2_1b_q4_0.gguf` (703 MB), `llama3_2_3b_q4_0.gguf` (1.82 GB), `llama3_1_8b_q4_0.gguf` (4.53 GB).
  - `llama/`: `llama3_2-{1b,3b}_vulkan_{4w,8da4w}.pte`, `llama3_1-8b_vulkan_{4w,8da4w}.pte`, `tokenizer.model`,
    `prompt_2048.txt`, runners `llama_main` and `llama_main_qshaders`. These `.pte` files are byte-identical in size to
    `/mnt/linux-share/models/<model>/exported/*.pte` (ctx 2560).
  - `llama_dev/` (ctx 3072, `buffer`/`texture`/`embq` variants), `llama_bench/` and `llama_vk/` (8B 4w/8da4w baseline
    and coopmat), `llama_r15/` (1B 4w/8da4w, `llama_main_base`, `llama_main_vanilla`), `et_mv2/` (`executor_runner`
    and MobileNet `.pte`s).
  - **No `llama_main` runner on the phone includes XNNPACK:** none of the 12 contains an `xnn_` symbol string (each has
    a single `XnnpackBackend` type-name string). Only `et_mv2/executor_runner` (not an LLM runner) has XNNPACK. The
    `--help` of `llama_dev/llama_main` lists `--cpu_threads`, `--ignore_eos`, `--warmup`, `--prompt_file`.
- Pixel (Android 16): `llama/` holds `llama3_2-1b_vulkan_{4w,8da4w}.pte` (same sizes as the Samsung's), `tokenizer.model`
  and `prompt_2048.txt` (byte-identical to `" the"` x 2048); `gpu-lab/` and `minibench/` hold tooling only. No llama.cpp
  build and no runner. The phone is rooted; clocks were left to the driver (no `sysfs`/`thermal`/`bcl` lock in
  `clock_info` at the start).
- Both phones also hold `igpu-roofline/` (the owner's Vulkan roofline runner and SPIR-V; used for D4, copied, not run in
  place).

**Checkpoints and export flags (task 1.2):** `/mnt/linux-share/models/llama-3.2-{1b,3b}/original/`
(`consolidated.00.pth`, `params.json`, `tokenizer.model`; found through `~/Desktop/sarc-acl/CLAUDE.md` "Models"). That
file records the export of the `exported/` Vulkan `.pte`s as reconstructed from their metadata (the original command
was not recorded): `-d fp32 --vulkan --vulkan-force-fp16 -qmode <8da4w|4w> -G 128 -E 4,32 --max_seq_length 2560
--max_context_length 2560 -kv --use_sdpa_with_kv_cache --metadata '{"get_bos_id":128000, "get_eos_ids":[128009]}'
--model llama3_2`. The `llama_dev` files (ctx 3072, `embq`) come from a different export, so the study uses the ctx-2560
files in `/data/local/tmp/llama` on both phones. No checkpoint is under `~/.cache/huggingface` or `~/.llama`; the shell
histories hold no export command.

## Goals / Non-Goals

**Goals:**
- Prefill and decode rates per PU, alone and co-running, on both phones, for 1B (both) and 3B (Samsung).
- A prediction of specialized-pipeline vs data-parallel throughput per request mix, and a go/no-go.

**Non-Goals:**
- Any pipelining runtime, KV transfer code, or ExecuTorch change.
- 8B in two-backend form (two copies do not fit). 8B is measured single-backend on the Samsung as a reference only.
- Energy (a later study).

## Decisions

### D1. Two runtimes, llama.cpp first
llama.cpp gives prefill (`pp`) and decode (`tg`) rates per backend in one command (`llama-bench -p <P> -n <D> -ngl
<0|99> -t <threads>`), and its Samsung build already exists. It is measured first. ExecuTorch is measured second, with
the owner's Vulkan `.pte` files and new XNNPACK 8da4w ones, because ExecuTorch is the intended substrate for
BetterTogether 2. A result that holds in only one runtime is reported as such.

### D2. PU configurations
- CPU: llama.cpp `-ngl 0`, ExecuTorch XNNPACK. Pinned with `taskset` (hex mask on the Samsung) to:
  - one tier: Samsung big+super (cores 7-9, mask `380`), Pixel big (cores 6-7, mask `c0`);
  - a wider set: Samsung medium+big+super (cores 4-9, mask `3f0`), Pixel medium+big (cores 4-7, mask `f0`).
  Thread count equals the pinned core count.
- GPU: llama.cpp `-ngl 99` (Vulkan), ExecuTorch Vulkan `.pte` (4w and 8da4w; the faster per phase is used). The host
  thread of a GPU run is pinned to one little core (Samsung core 3, mask `8`; Pixel core 3, mask `8`) so it does not
  take a core from the CPU side.
- Partial offload (`-ngl` = half the layers) is measured as the "layer-split, no overlap" baseline llama.cpp offers.

### D3. Co-run by two processes
Interference is measured with two concurrent processes, each pinned as in D2, each running its phase in a loop for at
least 30 s, starting within 1 s of each other; only the overlapped interval counts. Pairings: GPU-prefill + CPU-decode,
CPU-prefill + GPU-decode, CPU-decode + GPU-decode. With llama.cpp, a prefill-only loop is `-p P -n 0` and a decode-only
loop is `-p 0 -n D` (`llama-bench -r` repeats). With ExecuTorch, the runner's reported prefill and decode rates are used,
with a prompt sized for the phase being measured.

### D4. Bandwidth
CPU: a pinned multi-threaded streaming read (e.g. tinymembench or a small C++ loop built with the NDK). GPU: a Vulkan
buffer-read kernel (the owner's `igpu-roofline` tool if it applies, else a minimal compute shader). Both alone and
together.

### D5. Prediction model
For PU X, prefill rate p_X and decode rate d_X (tokens/s), request (P, D):
- alone: T_X = P/p_X + D/d_X; best single PU = min T_X.
- ideal data-parallel: throughput = sum over X of 1/T_X, with alone rates; co-run-corrected: the same with the
  both-decoding co-run rates (conservative for the prefill part).
- specialized, prefill on A and decode on B: period = max(P/p'_A, D/d'_B) + h(P), with p', d' the specialized co-run
  rates and h(P) the KV hand-off: KV bytes = 2 x layers x kv_heads x head_dim x P x bytes per element, copied at the
  measured bandwidth (an upper bound: on unified memory the copy may become a layout conversion or disappear).
- gain = specialized throughput / co-run-corrected data-parallel throughput. Gate: >= 1.2 for some (phone, model, mix).

### D6. Request mixes
Prompt 128/512/2048 x output 32/128/512, reported as a grid. Agent workloads (long tool output, short answer) sit in
the prompt-heavy corner; it is called out separately.

## Risks / Trade-offs

- **Thermal drift over long sweeps** → 20 s between runs, 3 rounds, both orders of the co-run pairs, and the GPU clock
  recorded; a run that throttles is repeated after cooldown and noted.
- **llama.cpp and ExecuTorch disagree** → reported side by side; the decision names the runtime it rests on.
- **ExecuTorch XNNPACK export needs the original checkpoint** → the owner's checkpoint and the exact export flags of the
  existing Vulkan `.pte` files must be found (task 1.2); if they cannot be, ExecuTorch CPU numbers are skipped and the
  study rests on llama.cpp, stated as such.
- **Pixel memory** → 1B only; 3B would leave too little headroom for the co-run.
- **Two processes are not one pipeline** → co-run rates approximate the specialized pipeline but miss the hand-off
  synchronization; D5's hand-off term bounds it from above.

## Deviations and choices made while running the study

Nobody was available to answer questions during the run; where the plan left a choice open, the conservative option
was taken, meaning the one that makes the data-parallel baseline stronger or the specialized pipeline weaker. Raw
evidence for each item is under `data/llm-phase-affinity/`.

- **Build directories.** This session could write only to the repository and `/tmp`, not to `~`, so the independent
  ExecuTorch clone is `build/lpa/etc/executorch` (gitignored; cloned from `~/Desktop/executorch`, checked out at
  `e229c58784`, submodules from upstream). ExecuTorch requires the clone directory to be named `executorch`. llama.cpp
  was built out of tree in `build/lpa/llamacpp-android{,-s24}` from `~/llamacpp-compare/llama.cpp` at `8345f33`
  (unmodified). Its Vulkan build needs `vulkan.hpp` and `spirv.hpp`, which the NDK lacks. They came from the clone's
  `Vulkan-Headers` submodule and `~/llamacpp-compare/deps/prefix` (SPIRV-Headers), both read only.
- **One llama.cpp version on both phones, built per ISA, instead of the Samsung's existing build.** The owner's Samsung
  `llama-bench` (c745be2) lacks i8mm. On mask 180 it reached 112.7 tok/s pp512 against 144.8 for 8345f33 built with
  `GGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16` (`R5CY21Y3VEV/diag-build-comparison.txt`). A weaker CPU build lowers
  the data-parallel bound and so biases the gate toward "go". Both phones therefore run 8345f33: the Samsung with that
  flag set, the Pixel with `armv8.2-a+dotprod+fp16` (Tensor G2 has no i8mm). Without an arch flag
  (`GGML_NATIVE=OFF` alone) the Pixel's CPU prefill was 5x slower (`3A021JEHN02756/diag-thread-placement.txt`). KleidiAI
  and OpenMP are off on both, as in the owner's build. The GGUFs are the owner's (sha256 `4f69b261fd1da7c5…` for 1B,
  the same file on both phones), read in place on the Samsung and pushed to `/data/local/tmp/lpa/models` on the Pixel.
  The 6 runs already taken with c745be2 are kept in `R5CY21Y3VEV/abandoned-c745be2-partial/` and are not used.
- **CPU configurations use `-dev none`.** With the Vulkan device visible, `-ngl 0` leaves the weights in a Vulkan host
  buffer and lets llama.cpp offload large prefill matmuls to the GPU. CPU decode then fell to 0.16-0.24 tok/s on both
  phones. Hiding the device makes the CPU configuration CPU-only.
- **Samsung one-tier mask is `180` (cores 7-8, big, 2 threads), not `380`.** Under a plain `taskset 380`, the Exynos
  scheduler kept all three compute threads on core 9 for the whole run, with cores 7-8 idle at 672 MHz
  (`R5CY21Y3VEV/diag-ems-thread-placement.txt`: pp512 36.6 tok/s on 380 vs 112.7 on 180). ggml's own per-thread
  affinity (`-C`/`--cpu-strict`) does nothing on Android, and the runners may not be modified. Masks `180`, `3f0`, `c0`
  and `f0` all spread one thread per core.
- **Partial offload** (`-ngl 8` for 1B, `-ngl 14` for 3B) runs its CPU part on the wider mask (`3f0` / `f0`).
- **Decode rate at prompt length P** is llama-bench `tg128` at KV depth P (`-p 0 -n 128 -d 0,128,512,2048`); depth 0
  is kept for the co-run ratios (D3's decode-only loop is `-p 0 -n D` at depth 0). For D = 512 the depth-P rate is
  slightly optimistic for every PU alike.
- **Thermal check.** Before every run the driver waits for `dumpsys thermalservice` "Thermal Status: 0". A run that
  ends with status >= 1 is repeated once after a 120 s cooldown as `<name>_rK_retry`, and the retry replaces it.
  Clocks were not pinned on either phone. The GPU clock (and every CPU policy frequency) is sampled every 0.5 s during
  each run (`*.clk`). Long CPU runs reach status 1 ("light") by themselves, and nearly every retry ends at 1 again.
  The rule was kept because it matters on the Pixel CPU: across the 118 original/retry rate pairs of the two
  llama.cpp sweeps, Pixel CPU retries ran 2-22% faster, GPU retries within ±4%. A retry that ends at status 1 again
  is kept and counted as the run.
- **llama.cpp GPU: flash attention.** The sweep ran the default `-fa auto`. Single-run checks
  (`<serial>/diag-gpu-flags/`) found that auto picks flash attention on and that it collapses prefill at 2048 on the
  Xclipse: 1B pp2048 99 tok/s on, 443 off. The Mali is 71 on vs 87 off; at 512, on is 5-10% faster on both. `gpufa0`
  (`-fa 0`) was therefore added as a 3-round configuration (`alone-llamacpp-extra/`). The llama.cpp GPU unit takes,
  at each prompt length and depth, the faster of `gpu` (auto) and `gpufa0`, the way D2 takes the faster of the two
  ExecuTorch Vulkan files. Co-runs use auto at P=512 and `-fa 0` at P=2048.
- **Partial offload with `--no-host 1`.** On the Pixel, `-ngl 8` decoded at 1.7-2.9 tok/s, the same Vulkan host-buffer
  effect as above. `--no-host 1` gives 11 tok/s, and on the Samsung 51 vs 28 tok/s at depth 0 (but slower pp512).
  `partialnh` was added as a 3-round configuration, and the partial baseline takes the faster of the two per point.
  Partial offload is a reference only; it does not enter the gate.
- **ExecuTorch runner and export (task 1.4).** `llama_main` was built from the clone with the Android preset
  (XNNPACK, weight cache, shared workspace) plus `EXECUTORCH_BUILD_VULKAN=ON`. It is 17.5 MB, statically linked
  against libc++, and has both backends. The XNNPACK `.pte` files came from the legacy CLI
  `python -m examples.models.llama.export_llama` with the flags of task 1.2, `--vulkan --vulkan-force-fp16` replaced
  by `-X --xnnpack-extended-ops`: `-d fp32 -qmode 8da4w -G 128 -E 4,32 --max_seq_length 2560
  --max_context_length 2560 -kv --use_sdpa_with_kv_cache --metadata '{"get_bos_id":128000, "get_eos_ids":[128009]}'
  --model llama3_2`. The result is 803 MB for 1B and 1.91 GB for 3B. The Hydra `export_llm` entry point fails on
  Python 3.14 (argparse "badly formed help string"). `lm_eval` and `torchtune`'s dependencies were skipped because
  `pyarrow` does not build on 3.14; `torchtune` is installed with `--no-deps`. Text check on the Samsung (greedy, 32
  tokens, "The capital of France is"): the XNNPACK 1B file, both Vulkan files on this runner, and both Vulkan files
  on the owner's `llama/llama_main` produce identical text (`R5CY21Y3VEV/smoke-et/text-check.txt`). This runner is
  used for every ExecuTorch run, Vulkan included. It decodes 20-25% faster than the owner's runner, which logs at
  Info level during generation.
- **3B co-runs on the Samsung (added to task 3.2).** The analysis first borrowed 1B co-run ratios for 3B. A draft
  of it took the GPU unit's alone prefill at P=2048 from `gpu` (flash attention auto, 98 tok/s) only, not the
  faster `gpufa0` (444 tok/s) as decided above. The 2048 co-runs use `-fa 0`, so the GPU prefill co-run ratio came
  out as 2.58 and the transferred 3B grid reached 1.29 at P=2048, D=32. With the GPU unit fixed, the 3B transferred
  grid peaks at 0.72. The 3B co-runs (`plan_corun_samsung_llamacpp_3b.txt`, `R5CY21Y3VEV/corun-llamacpp-3b/`) are
  still measured, last in the Samsung queue, because the spec asks for every model's prediction from co-run rates.
  A grid built on transferred ratios is labelled as such and does not count for the gate.
- **ExecuTorch co-runs: loops of whole processes, and on both phones.** `llama_main` cannot repeat a phase inside
  one process, and the runner is not modified, so each side of an ExecuTorch co-run is `lpa_loop.sh` running
  `llama_main` again and again (prefill loops 90 s, decode loops 120 s). A prefill loop uses prompt P, `--warmup`
  and 1 new token: two prefills per process, the second (warm, like the alone runs) recorded. A decode loop uses a
  16-token prompt and 512 new tokens, without warmup. Each side's reference is the same loop run alone
  (`solo_<tag>`), not the alone sweep, so model load and a cold first run cancel in the ratio. Model load (3.4 s
  for a 1B Vulkan file) leaves gaps in which the partner runs alone. So a prefill counts only when a partner decode
  window covers it wholly. A decode window counts with its duty, the fraction of it during which the partner
  computed (warm-up included). For the specialized pairs the decode ratio r at duty u is extrapolated to a fully
  overlapped window, 1 - (1 - r)/u. That assumes all interference falls in the partner's compute phases. Model
  load also streams memory, so the extrapolation overstates interference, which is conservative for the
  specialized pipeline. The both-decoding ratio is used as measured: its duty is near 1, and any gap makes the
  data-parallel bound larger. The tasks put ExecuTorch co-runs on the Samsung only. They were added on the Pixel
  because its ExecuTorch alone rates show the strongest phase split of the study: Vulkan 8da4w prefill 284-440
  tok/s against 65-163 for XNNPACK, but XNNPACK decode about twice Vulkan's at short context. Without them the
  Pixel ExecuTorch grid could not be computed. Plans: `plan_corun_{pixel,samsung}_et.txt`. Each GPU side uses the
  faster Vulkan file for its phase.
- **No ExecuTorch 3B co-runs.** The Samsung's ExecuTorch 3B grid uses the 1B co-run ratios, a transfer that is
  labelled and does not count for the gate. Its own co-runs (about 3 h) were not measured because they could not
  change the decision. Each 3B mix would need a decode co-run ratio of at least 0.93 to reach 1.2, or cannot reach
  it at the measured prefill ratio. No co-run in the study comes close: the highest measured decode ratio is 0.92,
  at 20% duty, and the highest at full overlap is 0.73. Even with no interference, 3B ExecuTorch tops out at 1.29.
- **Analysis after an independent review.** A read-only review of the analysis script found no defect that
  changes the verdict. It led to four changes:
  - the duty of a decode window is now weighted by window length, as its rate is. The extrapolation is then
    exact, and the best cell moves from 1.10 to 1.08;
  - an ExecuTorch hand-off writes fp32, since the XNNPACK export is `-d fp32`;
  - the per-trial spread of the best cell's decode ratio is reported in the decision record;
  - the grid gains a diagnostic, outside the gate, pricing a pipeline whose phases slow each other only while
    both run. D5 assumes they always do. The diagnostic is set against the better of data-parallel and the best
    single PU, because on the Samsung the co-run data-parallel bound falls below one PU alone in several cells.
  On the Pixel the ExecuTorch co-runs decode with the 8da4w file. The alone maximum the ratios are applied to
  is 4w at every depth, faster by under 0.5%.
- **Queue order on the Samsung after the 1B llama.cpp co-runs:** ExecuTorch alone, ExecuTorch co-runs, 3B llama.cpp
  co-runs, then the 8B reference runs (task 2.4), which do not enter the gate.
- **Read bandwidth is the median over 40 s, a sustained figure.** The Samsung CPU read starts at 43 GB/s and
  settles to 34-36 GB/s within the 40 s (`R5CY21Y3VEV/bandwidth/cpu_t*/cpu.out`), while a `tg128` run lasts 2-5 s. So
  llama.cpp CPU decode reaches 1.06x the bytes-per-token bound of the sustained figure (0.89x of the first-second
  43 GB/s). The fractions are reported against the sustained median as measured; a fraction above 1 means the decode
  ran on burst bandwidth, not that it beat the memory system.
- **Interruption and resume.** A host reboot (~20:57) killed the drivers during task 3.2. The phones kept running
  and nothing on them was lost. Each phone had one trial in flight (Samsung `gpuP_cpuD_p512_t2_retry`, Pixel
  `cpuD_gpuD_t2`), killed by PID. Those partial trials were not pulled and were redone. `run_plan.sh` and
  `run_corun.sh` were made resumable (a run with a complete `.meta`/`corun.meta` on the host is skipped), and
  `run_plan.sh` now also repeats once a run that exits non-zero, as a `_retry`. If the retry fails too, the
  configuration is recorded as failed.

## Migration Plan

None: a study. Results are additive under `data/llm-phase-affinity/` and `docs/reports-for-human/`.

## Open Questions

- ~~Where the Llama 3.2 1B/3B checkpoints and the export flags of the existing Vulkan `.pte` files are~~: resolved,
  see Context (task 1.2).
- ~~Whether `llama_main` on the Samsung includes the XNNPACK backend~~: none does. A runner with both backends was
  built from the independent clone (task 1.4).
