# Proposal

## Why

BetterTogether 2 wants to show "1+1>2" on a real model: a pipeline that gives each processing unit (PU) the part of
the work it is good at should beat the best CPU+GPU data-parallel execution, not only the best single PU. The chosen
workload is on-device LLM inference (Llama 3.2 1B / 3B, 4-bit weights), the core of agentic apps.

An LLM is almost all linear layers, so it may simply favour the GPU everywhere, as the cifar apps did (specialization
gained only 1.04-1.2x over the ideal data-parallel bound there). The case for specialization rests on the two phases
behaving differently:
- **prefill** is a matrix-matrix product per layer: compute-bound;
- **decode** is a matrix-vector product per token: memory-bandwidth-bound, with a fixed GPU submit cost per token, on
  DRAM the CPU and GPU share.

If the phases prefer different PUs, or if a compute-bound GPU prefill and a bandwidth-bound CPU decode interfere less
than two decodes do, a prefill-on-A / decode-on-B pipeline over a stream of requests can beat the sum of the PUs'
throughputs. This change measures whether that holds on our phones **before** any pipelining runtime is built.

## What Changes

A measurement study with a go/no-go decision. No framework code is written.

- Per-PU prefill and decode rates for Llama 3.2 1B (Pixel 7a and Samsung S24+) and 3B (Samsung), at prompt lengths
  128/512/2048, on:
  - **llama.cpp** (`llama-bench`, Q4_0): CPU pinned to one tier and to big+medium tiers, and Vulkan with all layers
    offloaded; partial offload (`-ngl` between) recorded as a baseline;
  - **ExecuTorch**: the existing Vulkan `.pte` files (4w and 8da4w) and new XNNPACK 8da4w `.pte` files of the same
    models.
- **Co-run interference**: each phase's rate while the other PU runs the other phase (GPU prefill with CPU decode, and
  the reverse), and while both PUs decode (the data-parallel case).
- **Memory bandwidth** reachable by each PU alone and together, to explain the decode rates.
- **Prediction**: for request mixes (prompt 128/512/2048 tokens x output 32/128/512 tokens), the throughput of the best
  single PU, the ideal data-parallel bound (sum of per-PU request rates, with and without co-run correction), and the
  best specialized pipeline including the KV-cache hand-off cost. Gate: a specialized pipeline is predicted to beat the
  co-run-corrected data-parallel bound by >= 1.2x on at least one phone and one request mix.
- A decision record in `docs/reports-for-human/` with every table and the go/no-go.

Out of scope: building the pipelining runtime, KV-cache transfer code, ExecuTorch partitioner changes, Llama 3.1 8B in
two-backend form (it does not fit twice in memory; it is measured single-backend on the Samsung only as a reference),
and any profiling on rocky-ryzen.

## Capabilities

### New Capabilities
- `studies/llm-phase-affinity`: the measurements, prediction model and decision gate that establish whether
  phase-specialized CPU/GPU pipelining of on-device LLM inference can beat data-parallel execution.

### Modified Capabilities
- None.

## Impact

- **Devices:** Pixel 7a (`3A021JEHN02756`, 7.6 GB RAM) and Samsung S24+ (`R5CY21Y3VEV`, 11.5 GB, Xclipse 940). Phones
  only.
- **Existing assets reused:** on the Samsung, `/data/local/tmp/lcpp` (llama.cpp `llama-bench` with Vulkan and Q4_0
  GGUFs of 1B/3B/8B) and `/data/local/tmp/llama_dev`, `llama_bench`, `llama_r15` (ExecuTorch Vulkan `.pte` files and
  `llama_main` runners); on the Pixel, `/data/local/tmp/llama` (1B Vulkan `.pte`). These are the owner's; they are
  read, not modified or deleted.
- **To build:** llama.cpp for the Pixel (from `~/llamacpp-compare/llama.cpp`, Android NDK); XNNPACK 8da4w `.pte` and an
  XNNPACK-capable `llama_main` from a separate ExecuTorch worktree (the owner's `~/Desktop/executorch` checkout has
  uncommitted work and is not modified).
- **New files:** raw results under `data/llm-phase-affinity/`, an analysis script under `optimizer/analysis/`, the
  decision record.
