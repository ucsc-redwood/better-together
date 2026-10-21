# Proposal

## Why

The `llm-phase-affinity` study ended "no-go": the best specialized pipeline (prefill on one PU, decode on the other)
reached 1.10x the co-run-corrected data-parallel bound, against a 1.2x gate. The phase affinity is real: alone, the GPU
prefills 1.4-2.9x faster and the CPU decodes 1.2-2x faster at short context, and with no interference the same
pipelines would reach 1.67-1.83x. What sank it was co-run interference (each phase kept 0.34-0.89 of its alone rate).

That interference was measured **uncoordinated**: two independent pinned processes, each running flat out, with the
memory system arbitrating. On the Samsung the arbitration is lopsided: while the GPU decodes, the CPU keeps 7.6 of its
36 GB/s and its decode drops to 0.21 of its alone rate, while the GPU keeps 40 of 47 GB/s and 0.74 of its rate. The
GPU's decode only uses 0.35-0.52 of its own bandwidth bound, so it is not even using what it takes.

So the open question is whether the interference is intrinsic or an artifact of no coordination. If slowing the GPU a
little (by pacing its work in software) returns more CPU decode rate than it costs GPU prefill rate, a scheduler that
coordinates the two PUs, which is what BetterTogether 2 would be, can beat the limit measured so far. This change
measures that trade-off before any runtime is built. GPU clock caps are not writable from `adb shell` on either phone
(`/sys/kernel/gpu/gpu_max_clock` is `system`-owned on the Samsung; the Mali `devfreq` nodes are root-only on the
Pixel), so all throttling is in software.

## What Changes

A measurement study with a go/no-go decision, building on the `llm-phase-affinity` drivers and data.

- A small paced-execution harness built against the same llama.cpp (`8345f33`) used in `llm-phase-affinity`: it runs
  a prefill loop in ubatches of C tokens with a gap of G after each ubatch, or a decode loop with a gap of G after each
  token, pinned, and logs a timestamp per ubatch or token. With G = 0 it must reproduce `llama-bench` rates.
- Coordination knobs, swept on the Samsung with Llama 3.2 1B (3B at the best settings):
  - **GPU pacing**: gap G as a fraction of the GPU ubatch or token time (duty 1.0, 0.8, 0.67, 0.5, 0.33);
  - **prefill chunk size**: ubatch C = 128, 256, 512;
  - **CPU decode width**: fewer decode cores (decode is bandwidth-bound, so fewer cores may cost it little and
    interfere less).
- The same knobs applied to **both** sides of the comparison: the specialized pairs (GPU prefill + CPU decode) and the
  data-parallel pair (GPU decode + CPU decode), so the baseline is coordinated as well.
- A trade-off curve per pairing (partner rate vs. coordinated rate) and a re-run of the `llm-phase-affinity` D5
  prediction with the best coordinated settings per request mix. Gate: a coordinated specialized pipeline reaches
  >= 1.2x the best coordinated data-parallel throughput on at least one request mix.
- Optional: Pixel ExecuTorch 1B (the best cell of the previous study), with CPU decode width as the only knob, since
  ExecuTorch's runner cannot be paced without modifying it.
- A decision record in `docs/reports-for-human/`.

Out of scope: a pipelining runtime, single-process execution, KV hand-off code, GPU clock control (needs root),
modifying llama.cpp or ExecuTorch sources, and any profiling on rocky-ryzen.

## Capabilities

### New Capabilities
- `studies/coordinated-corun`: the measurements, trade-off curves and gate that establish whether coordinating the
  CPU and GPU (software pacing, chunk size, decode width) reduces co-run interference enough for phase-specialized LLM
  pipelining to beat an equally coordinated data-parallel baseline.

### Modified Capabilities
- None.

## Impact

- **Devices:** Samsung S24+ `R5CY21Y3VEV` (main), Pixel 7a `3A021JEHN02756` (optional ExecuTorch part). Phones only.
- **Reused:** `data/llm-phase-affinity/drivers/` (co-run driver, thermal handling, clock sampling), its alone rates
  as reference, the llama.cpp Android build in `build/lpa/`, and `optimizer/analysis/llm_phase_affinity.py` (D5).
- **New files:** the harness source under `data/coordinated-corun/drivers/`, raw results under
  `data/coordinated-corun/`, an analysis script under `optimizer/analysis/`, the decision record.
- **Depends on** `llm-phase-affinity` being merged first (its drivers, data and analysis module).
