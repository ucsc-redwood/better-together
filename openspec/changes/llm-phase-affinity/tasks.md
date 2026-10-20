# Tasks

Rules for every task:
- **Phones only** for timing: Pixel 7a `3A021JEHN02756`, Samsung S24+ `R5CY21Y3VEV`. Nothing is timed on rocky-ryzen.
- The study's own phone directory is `/data/local/tmp/lpa` on each phone; it is removed at the end. The owner's existing
  directories (`lcpp`, `llama*`, `et_mv2`, ...) are read from, never modified or deleted.
- Each measured value: median of 3 runs, 20 s apart, with min-max, cores (taskset mask) and GPU clock
  (`/sys/kernel/gpu/gpu_clock` Samsung, `/sys/class/misc/mali0/device/cur_freq` Pixel) recorded.
- Raw outputs go to `data/llm-phase-affinity/<serial>/`.

## 1. Setup

- [ ] 1.1 Inventory the owner's assets on both phones (files, sizes, `llama-bench --version`, `llama_main --help`) and record them in design.md "Context". Verify: the list matches the phones and names which runners include XNNPACK.
- [ ] 1.2 Find the Llama 3.2 1B/3B checkpoints and tokenizer on the build box and the export flags of the existing Vulkan `.pte` files (shell history, export logs, `llama_dev` file names: ctx 3072, `embq`). Verify: the paths and flags are recorded in design.md, or the absence is recorded and ExecuTorch CPU numbers are dropped (design Risks).
- [ ] 1.3 Build llama.cpp for Android arm64 with Vulkan from `~/llamacpp-compare/llama.cpp` (NDK) and push `llama-bench` plus the 1B Q4_0 GGUF to the Pixel's `/data/local/tmp/lpa`. Verify: `llama-bench -m ... -p 32 -n 8` runs on the Pixel with `-ngl 0` and with `-ngl 99`.
- [ ] 1.4 In a separate ExecuTorch worktree at the owner's commit `e229c58784` (never in `~/Desktop/executorch`), export XNNPACK 8da4w `.pte` for 1B and 3B with the flags of 1.2, and build an Android `llama_main` with XNNPACK and Vulkan. Verify: on the Samsung, the XNNPACK 1B `.pte` generates text and its output for a fixed prompt matches the Vulkan `.pte` within the tokens the owner's runs show (or any difference is recorded).

## 2. Per-PU rates (alone)

- [ ] 2.1 llama.cpp on the Samsung, 1B and 3B: CPU on mask `380` and `3f0`, Vulkan `-ngl 99` (host on mask `8`), partial `-ngl` at half the layers; `-p 128,512,2048 -n 128`. Verify: a table of pp/tg tokens/s per configuration, with spreads and clocks.
- [ ] 2.2 llama.cpp on the Pixel, 1B: CPU on `c0` and `f0`, Vulkan `-ngl 99`, partial `-ngl`. Verify: same table.
- [ ] 2.3 ExecuTorch on the Samsung (1B, 3B) and the Pixel (1B): Vulkan 4w and 8da4w, XNNPACK 8da4w on the same masks; prefill and decode rates from the runner at prompt 128/512/2048. Verify: same table; failures recorded with the error.
- [ ] 2.4 llama.cpp and ExecuTorch, Llama 3.1 8B on the Samsung, single backend each (reference only). Verify: rates or the recorded failure.

## 3. Co-run interference and bandwidth

- [ ] 3.1 Measure CPU and GPU read bandwidth on each phone, alone and together (design D4). Verify: GB/s per case, and each PU's decode rate (from 2.x) as a fraction of its bandwidth bound.
- [ ] 3.2 Co-run pairs with llama.cpp (design D3) on both phones, 1B, prompt 512 and 2048: GPU-prefill + CPU-decode, CPU-prefill + GPU-decode, CPU-decode + GPU-decode. Verify: each rate co-running and its ratio to alone, for each pair.
- [ ] 3.3 The same pairs with ExecuTorch on the Samsung (1B), if 1.4 succeeded. Verify: same table, or the skip recorded.

## 4. Prediction and decision

- [ ] 4.1 Write `optimizer/analysis/llm_phase_affinity.py` implementing design D5 over the raw results, with a pytest on a hand-computed case (one mix, known rates). Verify: `uv run pytest -q optimizer` passes.
- [ ] 4.2 Produce the grid (phone x model x runtime x mix): best single PU, ideal data-parallel (alone and co-run), every specialized pipeline, and the gain. Verify: the grid is in the decision record with every input traceable to a raw file.
- [ ] 4.3 Write `docs/reports-for-human/<date>-llm-phase-affinity.md`: the affinity table, interference, bandwidth, prediction grid, and the go/no-go per the spec; if "go", the best (phone, model, runtime, mix, PU assignment) for the PoC; if "no-go", the best margin and the limiting effect. Verify: `openspec validate llm-phase-affinity --strict` passes, the record states the decision, and `/data/local/tmp/lpa` is removed from both phones.
