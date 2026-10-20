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

## Migration Plan

None: a study. Results are additive under `data/llm-phase-affinity/` and `docs/reports-for-human/`.

## Open Questions

- Where the Llama 3.2 1B/3B checkpoints and the export flags of the existing Vulkan `.pte` files are (needed for
  matching XNNPACK exports).
- Whether `llama_main` on the Samsung includes the XNNPACK backend (if not, a runner with both is built).
