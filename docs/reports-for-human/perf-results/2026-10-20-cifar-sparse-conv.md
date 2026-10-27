# cifar-sparse Vulkan conv: one output channel per workgroup row (2026-10-20)

**Change:** OpenSpec `cifar-gpu-kernels`, task group 1 (commits c8457b7..a986a39 on `feat/cifar-sparse-conv`) ·
**Targets:** Samsung Galaxy (Xclipse 940, subgroup 64), Pixel 7a (Mali-G710, subgroup 16) · **Tool:**
`bm-prof-cifar-sparse-vk`, per-stage GPU timestamps · rocky-ryzen ran correctness tests and ACO dumps only, no timing.

> **One-line finding:** on the Samsung the four large pruned convs are 23-48 % faster (conv2 34.0 → 17.8 ms, conv3
> 51.7 → 33.8, conv4 68.5 → 43.2, conv5 45.8 → 35.4) and the cifar-sparse stage sum drops from 232.3 to 161.7 ms
> (-30 %). The Pixel 7a runs the unchanged Mali path and stays within noise. The CSR loads were already scalar; the
> gain comes from issuing the input loads of 8 nonzeros together instead of one at a time.

## What changed

- **S1, one output channel per workgroup row.** `new_cifar_sparse_conv2d_k3s1p1_oc.comp`, grid
  `(ceil(N·H·W/256), K, 1)`, `out_c = gl_WorkGroupID.y`. cifar-sparse's dispatcher runs it when the GPU's subgroup is
  32 or more (Samsung, rocky-ryzen). Below 32 (the Pixel) it runs `new_cifar_sparse_conv2d_k3s1p1.comp`, which is now
  that Mali path alone, its SPIR-V instruction-identical to before at subgroup 16.
- **S2, host-side tap decode.** `AppData::conv{1..5}_taps` hold per nonzero `offset·16 + tap`, the input offset
  relative to the output pixel plus the tap id, packed from `col_idx` after the weights load
  (`cifar_sparse::pack_k3s1p1_taps`). Each invocation builds a 9-bit mask of its in-image taps once; a nonzero is
  then a tap load, a weight load, a mask test and an input load from a clamped index, with no division and no branch.
  The arrays cost 9.0 MB per `AppData` on every backend (sized like `col_idx`), about 3 %.
- **S3, the CSR row in shared memory:** built (`STAGE_ROW` spec constant, `BT_VK_SPARSE_STAGE_ROW=1`), measured and
  left **off**.
- **S4, the sparsity format:** unchanged, unstructured CSR (below).

## Results (final A/B, 2026-10-20 01:00-01:06 UTC)

Method (cifar-gpu-kernels task 1.1): the ee4c68d and a986a39 binaries (`android` preset, NDK r30) in their own
directories on each phone, real weights (`BT_WEIGHTS_DIR`, `saved_params/export`), `BT_PROF_SCENARIO=isolated
BT_PROF_BUDGET_S=1.0 BT_PROF_ABANDON_S=5 BT_PROF_MAX_CELL_S=3 --benchmark_filter='vulkan/'`, alternating, 3 rounds,
20 s idle before every run, no other benchmark or test process on the phone. Per stage: the median of the 3 runs'
p50, with the min-max of the three p50s, `n = runs x samples`. Stages 1, 3, 5, 6, 7 are conv1-conv5.

Samsung (subgroup 64, the new kernel):

| stage | before (ee4c68d) median (min-max) ms | after (a986a39) median (min-max) ms | after vs before |
|---|---|---|---|
| 1 | 4.102 (4.102-4.103, n=3x238-243) | 3.259 (2.824-3.651, n=3x269-284) | -20.6 % |
| 2 | 2.419 (2.400-2.427, n=3x661-703) | 2.416 (2.376-2.419, n=3x689-725) | -0.1 % |
| 3 | 34.025 (34.023-34.027, n=3x29) | 17.818 (17.581-18.248, n=3x53-59) | -47.6 % |
| 4 | 1.808 (1.807-1.812, n=3x1296-1335) | 1.808 (1.806-1.810, n=3x1309-1331) | +0.0 % |
| 5 | 51.655 (51.645-51.660, n=3x19) | 33.752 (32.851-36.277, n=3x30-37) | -34.7 % |
| 6 | 68.537 (68.536-68.578, n=3x14) | 43.219 (41.998-50.626, n=3x23-25) | -36.9 % |
| 7 | 45.767 (45.764-45.769, n=3x21) | 35.399 (32.729-35.878, n=3x28-31) | -22.7 % |
| 8 | 2.702 (2.701-2.706, n=3x1194-1243) | 2.697 (2.675-2.703, n=3x1192-1226) | -0.2 % |
| 9 | 10.036 (10.035-10.074, n=3x99-100) | 10.048 (10.047-10.051, n=3x99-100) | +0.1 % |
| 10 | 10.043 (10.039-10.058, n=3x99) | 10.059 (10.055-10.059, n=3x99) | +0.2 % |
| 11 | 1.173 (1.159-1.180, n=3x2000) | 1.175 (1.173-1.175, n=3x2000) | +0.2 % |
| sum | 232.267 | 161.650 | -30.4 % |

Pixel 7a (Mali path, not changed by this work):

| stage | before (ee4c68d) median (min-max) ms | after (a986a39) median (min-max) ms | after vs before |
|---|---|---|---|
| 1 | 17.007 (17.005-17.037, n=3x68) | 17.028 (17.006-17.120, n=3x59-68) | +0.1 % |
| 2 | 4.465 (4.463-4.465, n=3x226-339) | 4.468 (4.463-4.472, n=3x226-338) | +0.1 % |
| 3 | 80.362 (80.266-80.427, n=3x12) | 80.447 (80.385-80.507, n=3x12) | +0.1 % |
| 4 | 3.366 (3.366-3.369, n=3x643-698) | 3.366 (3.361-3.372, n=3x640-698) | +0.0 % |
| 5 | 128.466 (128.367-128.631, n=3x7) | 128.621 (128.589-128.778, n=3x7) | +0.1 % |
| 6 | 183.461 (182.153-183.513, n=3x5) | 181.864 (181.710-182.862, n=3x5) | -0.9 % |
| 7 | 110.658 (110.455-110.697, n=3x9) | 110.553 (108.485-110.784, n=3x8-9) | -0.1 % |
| 8 | 1.179 (1.178-1.182, n=3x1944-2000) | 1.181 (1.179-1.181, n=3x1917-2000) | +0.1 % |
| 9 | 26.891 (26.877-26.901, n=3x30-37) | 26.955 (26.945-27.009, n=3x36-37) | +0.2 % |
| 10 | 27.252 (27.190-27.282, n=3x36) | 27.144 (27.133-27.225, n=3x36) | -0.4 % |
| 11 | 1.377 (1.375-1.380, n=3x1502-1568) | 1.381 (1.376-1.385, n=3x1441-1534) | +0.3 % |
| sum | 584.487 | 583.010 | -0.3 % |

Identical-binary noise (task 1.1): conv stages within 0.1 % on the Samsung, 0.3 % on the Pixel. The new kernel is
noisier on the Samsung than the old one: conv1 follows the GPU clock the profiler records (600-900 MHz on that short
stage), and conv2-conv5 run at 1095 MHz yet still move by up to 6 % between runs (conv4 above: 42.0-50.6 ms). The
spec gate (Samsung conv2-conv5 faster with non-overlapping spreads; no conv slower beyond noise on either phone) was
checked in task 1.9 and holds again here. In one 1.9 session the Pixel's conv4 read +0.7 % with spreads apart; a
repeat read +0.1 %, and across the sessions of this group the conv4 medians of the old and new binaries interleave
(task 1.9 in tasks.md).

## Why: what ACO showed (rocky-ryzen, Radeon 780M, Mesa 25.2.7)

- **Before** (task 1.2): S1's premise was that the CSR metadata loads were per-lane vector loads. They were not. The
  model's maps (32², 16², 8²) are subgroup multiples, so the old kernel took its subgroup-uniform fast path, and there
  `col_idx` and the weight were already `s_buffer_load_b32`. What it did instead was branch on every tap (the
  `ok ? v * weight : 0` of a border test) and `s_waitcnt vmcnt(0)` before each multiply: one input load in flight per
  wave, plus 15 SALU ops of `/9`, `/3` decode per nonzero.
- **After** (task 1.6): the row bounds are one `s_buffer_load_b64`; 8 taps and 8 weights are one `s_buffer_load_b256`
  each; the 8 input `buffer_load_b32` of an unrolled group are issued back to back before the first
  `s_waitcnt vmcnt(7)`, with no branch.
- Inferred, not measured: the gain is memory-level parallelism (8 loads in flight instead of 1), not scalarization.
  The Samsung's Xclipse driver is not RADV and its ISA was not inspected, and no hardware counters were read on either
  phone.

## S3: staging the row in shared memory does not pay (task 1.8)

Same method, three arms on the a324f38 binary: `STAGE_ROW` off / on against the baseline. On the Samsung `on` had a
lower median on conv1-conv3 (3.397 / 17.456 / 32.535 vs 3.689 / 18.362 / 34.370 ms) but inside the spreads, and was
slower beyond noise on conv4 (47.060 vs 45.216 ms) and conv5 (34.958 vs 33.970); stage sums 163.6 vs 163.8 ms.
Inferred from RADV's code for it (build box), not measured: staging swaps scalar loads, which already serve the
uniform row, for `ds_read2_b32` plus two barriers per 1024-nonzero chunk. Default: off on every device; the knob stays
for re-measuring.

## S4: the sparsity format stays unstructured CSR

Kept, as design S4 decided: structured 2:4 or channel-blocked sparsity would mean retraining, would change the
accuracy (dense 90.48 % / sparse 90.58 % today) and would drop the paper's premise that cifar-sparse is the
irregular-memory workload. S4 said to revisit with the paper authors if S1-S3 do not reach "the spec's 2× bound".
The spec has no 2× bound (it requires faster beyond noise); the speed-ups reached are 1.29-1.91× on conv2-conv5. The
question whether to try structured sparsity later stays in design.md's Open Questions.

## Correctness

At a986a39: local `ctest --test-dir build/pc -L omp` 11/11; rocky-ryzen `ctest -L vulkan` 7/7 and
`test-cifar-sparse-vk --device minipc` 21/21; `scripts/run-on-android.sh <serial> test-cifar-sparse-vk` 21/21 on the
Pixel 7a and the Samsung; the same suite with the real weights 21/21 on both phones (and on rocky at the same kernel
code, task 1.10). New tests: `SparseConvPerChannel.*` (empty, right-edge-only, top-edge-only and 1152-nonzero rows,
against the OMP kernel, `STAGE_ROW` off and on), `SparseConvTaps.DecodeToColIdx`.

## Follow-ups

- cifar-sparse × Vulkan stage times changed on the Samsung (and on rocky, untimed), so its profiling stores and
  fitted schedules are stale and must be regenerated before reporting fleet results.
- The new kernel's workgroups for one output channel span every image, so the workgroups in flight read the whole
  input tensor, where the old mapping read about one image's. Ordering the grid with channels fastest would keep that
  locality; untested, and the Samsung's run-to-run spread may come from it (inferred).
