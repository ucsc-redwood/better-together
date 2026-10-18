# Proposal

## Why

The gpu-chunk-cost-model gate (`docs/reports-for-human/2026-10-17-gpu-chunk-cost-model-gate.md`, section 3)
left the CPU chunks as the largest prediction error: median |measured/predicted - 1| of 0.31-0.40 on the
Pixel 7a and 0.63-3.2 on the Samsung for tree. Re-reading the executed schedules (`data/sched_logs`, per-stage
model, 2026-10-17) shows two separate causes:

1. **Tree stage 2 on a CPU is profiled on already-sorted input.** `omp::parallel_sort`
   (`apps/tree/omp/func_sort.hpp`) sorts `u_morton_keys_s1_out` in place before merging into the output. `bm-prof`
   times a stage by running it repeatedly on the same AppData (`profiler/bm_prof_common.hpp`, `time_once`), so
   from the second iteration on it sorts sorted segments. In a pipeline every item arrives with fresh codes.
   - Measured CPU chunks holding stage 2 run 13-31x their price: e.g. Pixel `Medium [2,2]` priced 0.64 ms, measured
     14.6-27.3 ms, slower than the GPU chunk next to it (10.4 ms), so the solver's chosen bottleneck is wrong.
   - The same effect reproduces on x86 with `std::sort` over two 153,600-key segments: 11.8 ms fresh,
     1.46 ms already sorted (8.1x).
2. **Small CPU chunks miss a fixed cost.** Chunks priced under 1 ms run 3.7x (Pixel) and 4.0x (Samsung) their
   price, a median excess of 0.76 and 0.89 ms. No `data/profiling/<serial>/overhead.json` exists for either phone,
   so the solver adds no per-chunk CPU overhead there (`smt/overhead.py`), although `fit_overhead.py` can fit it.

Without these fixed, tree schedules are chosen on wrong prices, and any later cost-model comparison
(including the range model for tree) is dominated by CPU error.

## What Changes

- **Stages are profiled on the input a pipeline gives them.** No OMP stage kernel modifies its own input buffers;
  tree stage 2 sorts a copy. A unit test runs each stage of each app twice on one AppData and checks the inputs
  are unchanged and the outputs identical.
- **Re-profile the cells whose stage kernels changed** (tree OMP stage 2, on every CPU tier of both phones,
  isolated and interference, 2 runs each).
- **Fit and ship the CPU overhead for the phones.** Run `fit_overhead.py` on the phones' executed schedules and
  commit `overhead.json`, with a held-out check: constants fitted on one set of executions must improve the
  error of a different set.
- **Acceptance gate:** re-solve and execute tree (and re-check cifar-dense, cifar-sparse) on both phones; CPU
  chunk error must fall for tree and not rise for the cifar apps, and the best measured period must not get worse.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `profiling/cost-table`: adds "A stage is profiled on unmodified input".
- `optimizer/cost-model`: adds the CPU-chunk accuracy gate. (A shipped per-device CPU overhead was planned too; it failed
  its held-out check on both phones and the requirement was dropped by the owner on 2026-10-18.)

## Impact

- **Code:** `apps/tree/omp/{func_sort.hpp,dispatchers.cpp}` (stage 2 input left intact), any other stage the
  audit finds mutating its input; a new input-purity test in the OMP test suites; no solver code is expected to
  change (`fit_overhead.py` and `smt/overhead.py` exist).
- **Data:** `data/profiling/<serial>/tree/vulkan/{isolated,interference}/run-*.jsonl` (CPU records refreshed),
  `data/profiling/<serial>/overhead.json` for both phones, new `data/sched_logs` executions.
- **Measurement:** phones only. Nothing is profiled on rocky-ryzen.
- **Depends on** `tree-app-improvements` landing first (it changes tree stages 1 and 7; the tree tables are
  re-profiled once, after it).
