# Design

## Context

See proposal.md (Why) for the per-stage numbers. The current state of the two kernels:

- **`new_cifar_sparse_conv2d_k3s1p1.comp`:** one invocation per output (n, out_c, oh, ow), workgroup
  256, grid `div_ceil(N*K*H*W, 256)` at five dispatch sites in
  `apps/cifar-sparse/vulkan/dispatchers.cpp`.
  - Each invocation walks its output channel's CSR row. Per nonzero it loads `col_idx`, decodes
    `(in_c, ky, kx)` with literal 9/3, loads the weight, and loads one input value.
  - When `subgroupAllEqual(out_c)` holds, the row bounds come from `subgroupBroadcastFirst`, making the
    walk scalar on AMD. On Mali the same instructions run per lane. The fallback is the per-lane walk.
  - No shared memory is used.
- **`new_cifar_dense_conv2d_k3s1p1.comp`:** one invocation per (n, 4-channel block, oh, ow), grid
  `div_ceil(N*(K/4)*H*W, 256)`.
  - It is already `subgroup_size_dependent`. `SUBGROUP_SIZE >= 32` takes the branchless,
    uniform-weight path; below 32 it takes `accumulate_branchy`, which is the previous kernel verbatim.
  - Per input channel, each invocation loads 9 taps and 4 x 9 weights from global memory, so the
    weights are fetched once per lane, not once per workgroup.
- **Where the cost is, as far as is known:** both kernels re-load the same value in many lanes. In the
  dense kernel every lane of a workgroup that shares an output-channel block loads the same 36
  weights. In the sparse kernel every lane loads the same `col_idx` and weight. Each output pixel loads
  its 9-tap window alone, although neighbours overlap. AMD hides the first through scalar loads; Mali
  has no scalar unit (the measured -42 % / 0 % split and the +12 % slowdown of branchless loads on the
  Pixel are consistent with this). Inferred, not measured on Mali: no `malioc` report or Mali counter
  data exists yet.
- **Tests:** `SparseConvK3S1P1.*` and `DenseConvK3S1P1.*` compare against a double-precision reference
  at 6x6 and 8x8. The differential suites run every stage against the OMP oracle. On the Pixel they
  exercise the Mali path.

## Goals / Non-Goals

**Goals:**
- Each value loaded from global memory on the Mali path feeds several outputs, without relying on
  subgroup-uniform (scalar) loads.
- One shader source per kernel, with the Mali path selected at pipeline creation. No new shader names,
  no device lists.
- A measured decision per kernel and per technique, recorded with its numbers.

**Non-Goals:**
- Changing the subgroup >= 32 paths (spec: "The subgroup-32-and-wider path does not change").
- Reduced precision, cooperative matrix, conv+pool fusion, the FC layers.
- Tuning for a specific Mali generation beyond the Pixel 7a's G710. The path is selected by subgroup
  size, so any other subgroup-16 Mali also takes it, and its correctness tests run there.

## Decisions

### D1. Selection: the existing `SUBGROUP_SIZE` specialization constant
Both kernels select the Mali path with `if (SUBGROUP_SIZE < 32)` on spec constant 0. The dense kernel
already does this. The sparse kernel's algorithm becomes `subgroup_size_dependent()` in its dispatcher
and in `SparseConvK3S1P1`'s test.

The branch is resolved at pipeline creation, so the dead path costs nothing at run time.

*Alternatives:* a second shader per kernel chosen by the dispatcher from `compute_limits()` (two
sources to keep in step, and two pipeline objects); vendor-ID lists (rejected by the pipeline-config
spec's "subgroup size comes from the running device").

### D2. Reuse through output blocking first, shared memory second
Techniques, in the order they are tried. Each is measured before the next one is added:

1. **Pixel blocking along W (both kernels).** One invocation computes `PX` adjacent output pixels of a
   row (start with `PX = 4`; all model maps are 8, 16 or 32 wide).
   - Dense: per input channel, the 3 x (PX+2) input window is loaded once (18 loads for 4 outputs
     instead of 36) and each weight is reused PX times.
   - Sparse: each nonzero is decoded once, its weight is loaded once, and it feeds PX inputs that are
     contiguous in memory (vectorizable when aligned).
   - This needs no shared memory and no barrier.
2. **Weights / CSR row in shared memory, per workgroup.** The workgroup is mapped so that it owns one
   output channel (sparse) or one channel block (dense), as `gl_WorkGroupID.y`. It stages that row in
   shared memory in bounded chunks, and the lanes read it from there instead of each fetching it from
   global memory. This is the Mali counterpart of the AMD scalar load.
3. **Input tile with halo in shared memory (dense only).** Only if 1 and 2 leave the dense kernel
   load-bound. The sparse kernel's CSR row touches arbitrary input channels, so an input tile cannot be
   reused across them.

*Alternatives considered:*
- Vectorizing the channel loop with vec4 over `in_c`: the data layout is NCHW, so channels are not
  contiguous.
- An implicit-GEMM rewrite: large, and its payoff on Mali without matrix units is unclear.
- FP16: out of scope (proposal).

### D3. Dispatch geometry from one contract per kernel
Pixel blocking (D2.1) changes the invocation count, and the count depends on the path, so the
dispatcher must know which path the running GPU takes. A small header next to the shaders (the
`linear_bt_contract.hpp` pattern) gives each kernel's outputs per invocation for a given subgroup size.
Both the dispatchers and the tests size the grid from it.

A mismatch must fail a test, not silently leave outputs uncomputed. The existing `-1` output fill in the
conv tests already catches that.

### D4. Ragged shapes are handled in the kernel, not by the dispatcher
A row whose width is not a multiple of `PX` runs a masked tail. The masked lanes still take part in any
barrier (D2.2) and never write. A workgroup's last channel chunk may be partial.

The tests add a 7x5 map (odd width, not a multiple of 4 or of the subgroup size) to the existing 6x6
and 8x8, and a CSR row whose nonzeros all fall in the padded border (spec scenarios).

### D5. Measurement method
- **Device A/B:** `bm-prof-cifar-{dense,sparse}-vk`, isolated scenario, on the Pixel 7a and the
  Samsung. Previous and new binaries are run 3 rounds interleaved with a 20 s cooldown, using the same
  scripts as the 2026-10-15/16 sessions (build both binaries from two worktrees with the Android
  preset, push both, and alternate). The gate is the spec's "A Mali path ships only if it is measured
  faster".
- **Static:** `malioc --format json` on the compiled Mali path, if Arm Performance Studio is installed.
  It gives register pressure (spills, and work registers against the occupancy limit), the
  arithmetic / load-store / texture cycle split, and the bound unit. It is used to pick D2 steps and to
  explain results, not as a gate.
- **Not on rocky:** rocky-ryzen runs only the correctness suites.

## Risks / Trade-offs

- **[A gain alone does not survive the pipeline]** (thermals, interference) → The gate is per stage,
  and the end-to-end effect is reported separately with a `bm-gen-logs` schedule run. It is not
  required to pass.
- **[Pixel blocking raises register use and lowers occupancy]** → Keep `PX` and `OC_TILE` as
  compile-time constants, try `PX = 2` and `4`, and check spills with `malioc`.
- **[Shared-memory staging adds barriers]**; with only 16-wide subgroups, small workgroups pay for them
  → Stage in chunks sized to the workgroup, and measure against D2.1 alone. Keep D2.2 only if it adds
  beyond D2.1.
- **[Real-weight CSR rows have very different lengths]** (25 % density per layer, uneven per channel),
  so workgroup-per-channel loads are unbalanced → Acceptable for the gate. The differential suite with
  real weights checks correctness. The imbalance is recorded if it shows.
- **[No `malioc`]** → Device A/B only. The design does not depend on it.
- **[A second subgroup-16 Mali in the future behaves differently]** → It takes the same path and its
  correctness tests run. Speed is only claimed for the Pixel 7a.

## Migration Plan

The Mali path ships per kernel, behind the spec's gate. If a technique fails the gate, the shader keeps
the previous kernel for `SUBGROUP_SIZE < 32`, as the dense kernel does today, so rollback means "do not
enable the new branch". No data, schedule or API format changes.

## Baseline

Measured 2026-10-16 (task 1.2) at 9a10ea0, before any kernel change.

**Method (used by every phone A/B of this change).**
- Binaries: `bm-prof-cifar-{dense,sparse}-vk` from `cmake --preset android` (NDK r30), pushed with
  `libc++_shared.so` to `/data/local/tmp/bt-mali` on each phone, with the real weights
  (`saved_params/export`) in `bt-mali/weights`.
- Run: `BT_WEIGHTS_DIR=<dir>/weights BT_PROF_SCENARIO=isolated BT_PROF_BUDGET_S=1.0 BT_PROF_ABANDON_S=5
  BT_PROF_MAX_CELL_S=3 BT_PROF_RUN=<round> ./bm-prof-cifar-<app>-vk.<arm> --device <serial>
  --benchmark_filter='vulkan/'`. The budget knobs are raised from their defaults (0.3 / 0.25 / 2.0 s)
  because, with the defaults, a stage above 0.25 s is sampled once (the Pixel's sparse stage 6, ~0.4 s).
  With these values that stage gets 2 samples and the others 3 to 2000.
- Order: arms alternated A, B (, C) per round, 3 rounds, 20 s sleep before every run, one app at a
  time per phone. The two phones run at the same time, each on its own adb serial.
- Per stage: the median over the 3 rounds of each run's p50 (GPU timestamps), and the spread
  (min-max of the three p50s). `n = runs x samples per run`.
- Here both arms are the **same binary**, so the B-vs-A column is the run-to-run noise.

**Noise.**
- Pixel 7a: dense conv stages within 0.1 %. Sparse convs 3, 5, 6, 7 within 0.5 %. Sparse stage 1
  within 5 %.
- Samsung: bimodal from run to run. The same binary's dense stage 7 reads 2.14 or 2.75 ms and stage 3
  2.35 or 3.09 ms. Sparse convs spread 10-20 %, and medians differ by up to 22 % between identical
  arms. A Samsung "no slowdown" check can therefore only rule out changes larger than that.

Pixel 7a, cifar-dense (ms):

| stage | A median (min-max) ms | B median (min-max) ms | B vs A |
|---|---|---|---|
| 1 | 7.087 (7.085-7.088, n=3x140) | 7.086 (7.085-7.087, n=3x141) | -0.0 % |
| 2 | 0.497 (0.494-0.501, n=3x2000) | 0.496 (0.496-0.497, n=3x2000) | -0.1 % |
| 3 | 24.451 (24.450-24.465, n=3x40) | 24.462 (24.456-24.465, n=3x36) | +0.0 % |
| 4 | 0.385 (0.382-0.385, n=3x2000) | 0.383 (0.381-0.385, n=3x2000) | -0.6 % |
| 5 | 36.351 (36.346-36.353, n=3x27) | 36.352 (36.348-36.358, n=3x27) | +0.0 % |
| 6 | 48.810 (48.809-48.814, n=3x20) | 48.815 (48.811-48.824, n=3x20) | +0.0 % |
| 7 | 32.557 (32.555-32.562, n=3x30) | 32.555 (32.551-32.567, n=3x30) | -0.0 % |
| 8 | 0.195 (0.183-0.199, n=3x2000) | 0.196 (0.193-0.196, n=3x2000) | +0.5 % |
| 9 | 6.854 (6.284-8.313, n=3x78) | 7.901 (6.753-8.846, n=3x104) | +15.3 % |
| 10 | 6.978 (6.717-7.273, n=3x129) | 6.916 (6.766-7.957, n=3x126) | -0.9 % |
| 11 | 0.990 (0.989-0.991, n=3x1278) | 0.879 (0.870-0.989, n=3x1282) | -11.3 % |
| sum | 165.156 | 166.041 | +0.5 % |

Pixel 7a, cifar-sparse (ms):

| stage | A median (min-max) ms | B median (min-max) ms | B vs A |
|---|---|---|---|
| 1 | 21.382 (20.311-21.973, n=3x40) | 20.322 (19.318-21.297, n=3x43) | -5.0 % |
| 2 | 4.483 (4.476-4.526, n=3x391) | 4.534 (4.476-4.534, n=3x390) | +1.1 % |
| 3 | 181.314 (181.159-181.435, n=3x5) | 181.390 (181.320-181.442, n=3x5) | +0.0 % |
| 4 | 3.471 (3.383-3.569, n=3x640) | 3.766 (3.389-3.770, n=3x643) | +8.5 % |
| 5 | 283.459 (283.052-283.591, n=3x3) | 282.810 (282.077-283.380, n=3x3) | -0.2 % |
| 6 | 398.295 (397.601-400.708, n=3x2) | 399.341 (396.474-400.082, n=3x2) | +0.3 % |
| 7 | 242.435 (242.204-242.626, n=3x4) | 242.208 (241.921-242.462, n=3x4) | -0.1 % |
| 8 | 1.201 (1.197-1.218, n=3x1942) | 1.205 (1.192-1.211, n=3x1914) | +0.3 % |
| 9 | 26.882 (26.879-26.917, n=3x37) | 26.885 (26.870-26.917, n=3x37) | +0.0 % |
| 10 | 27.222 (27.171-27.248, n=3x36) | 27.224 (27.175-27.249, n=3x36) | +0.0 % |
| 11 | 1.180 (1.167-1.227, n=3x1161) | 1.201 (1.171-1.381, n=3x1134) | +1.8 % |
| sum | 1191.323 | 1190.886 | -0.0 % |

Samsung, cifar-dense (ms):

| stage | A median (min-max) ms | B median (min-max) ms | B vs A |
|---|---|---|---|
| 1 | 0.911 (0.911-0.916, n=3x1124) | 0.909 (0.908-0.913, n=3x1103) | -0.2 % |
| 2 | 1.300 (1.268-1.304, n=3x776) | 1.300 (1.282-1.303, n=3x833) | -0.0 % |
| 3 | 3.088 (2.348-3.089, n=3x511) | 3.084 (3.083-3.393, n=3x514) | -0.1 % |
| 4 | 0.959 (0.953-0.982, n=3x1020) | 0.970 (0.966-0.980, n=3x1133) | +1.2 % |
| 5 | 2.446 (2.445-2.453, n=3x451) | 2.703 (2.448-3.136, n=3x319) | +10.5 % |
| 6 | 3.377 (3.373-3.377, n=3x327) | 3.375 (3.374-3.375, n=3x327) | -0.0 % |
| 7 | 2.747 (2.144-2.748, n=3x417) | 2.143 (2.142-2.752, n=3x369) | -22.0 % |
| 8 | 0.321 (0.317-0.324, n=3x2000) | 0.325 (0.322-0.326, n=3x2000) | +1.3 % |
| 9 | 2.827 (2.019-2.896, n=3x345) | 2.848 (2.047-2.866, n=3x355) | +0.7 % |
| 10 | 2.047 (2.046-2.990, n=3x337) | 2.731 (2.048-2.846, n=3x349) | +33.4 % |
| 11 | 0.680 (0.624-0.681, n=3x1411) | 0.663 (0.643-0.668, n=3x1404) | -2.5 % |
| sum | 20.703 | 21.052 | +1.7 % |

Samsung, cifar-sparse (ms):

| stage | A median (min-max) ms | B median (min-max) ms | B vs A |
|---|---|---|---|
| 1 | 4.619 (4.104-4.620, n=3x175) | 4.103 (4.103-4.619, n=3x207) | -11.2 % |
| 2 | 2.421 (2.411-2.440, n=3x695) | 2.433 (2.432-2.434, n=3x698) | +0.5 % |
| 3 | 36.972 (34.033-43.383, n=3x23) | 39.050 (34.979-40.928, n=3x28) | +5.6 % |
| 4 | 1.844 (1.830-1.865, n=3x1342) | 1.830 (1.827-1.838, n=3x1316) | -0.8 % |
| 5 | 66.814 (66.076-74.688, n=3x15) | 69.960 (61.361-78.313, n=3x12) | +4.7 % |
| 6 | 100.042 (100.020-105.674, n=3x8) | 109.362 (87.067-111.814, n=3x10) | +9.3 % |
| 7 | 62.804 (55.044-62.852, n=3x17) | 56.848 (53.507-66.717, n=3x16) | -9.5 % |
| 8 | 2.697 (2.695-2.698, n=3x1181) | 2.695 (2.688-2.696, n=3x1208) | -0.1 % |
| 9 | 10.033 (10.024-10.073, n=3x98) | 10.038 (10.022-10.041, n=3x95) | +0.0 % |
| 10 | 10.043 (10.039-10.044, n=3x99) | 10.040 (10.024-10.043, n=3x99) | -0.0 % |
| 11 | 1.155 (1.154-1.156, n=3x2000) | 1.155 (1.152-1.155, n=3x2000) | +0.1 % |
| sum | 299.442 | 307.513 | +2.7 % |

The Pixel's sparse convs are slower than the proposal's table (181 vs 166 ms for stage 3, and so on)
because this method uses the real pruned weights, whose CSR rows are longer for some channels than the
synthetic ~25 % pattern. Every A/B of this change uses the real weights in both arms.

**`malioc` (Mali-G710, driver r56p1, `-S 0=16` where the shader has spec constant 0), current kernels:**

| shader path | work registers | occupancy | stack (spills) | A / LS cycles (static total) | bound |
|---|---|---|---|---|---|
| dense, `SUBGROUP_SIZE < 32` (`accumulate_branchy`) | 64 | 50 % | none | 4.21 / 26.00 | LS |
| dense, `SUBGROUP_SIZE >= 32` (for comparison, `-S 0=64`) | 64 | 50 % | none | 6.20 / 47.00 | LS |
| sparse (one path for every subgroup size today) | 26 | 100 % | none | 7.49 / 29.00 | LS |

Both Mali paths are load/store-bound by a factor of 4-6 over arithmetic. This matches the
"each lane re-loads the same weights" reading of the Context section. The figures are static totals
over the program, with the loop body counted once, so they compare variants of one shader and are not
cycles per output. The dense path already runs at half occupancy (64 work registers).

## Measured decisions

### Dense conv, output blocking (D2.1, tasks 3.1-3.2): **shipped, PX = 4**

The Mali path computes PX = 4 adjacent output pixels x OC_TILE = 4 channels per invocation, with a masked
tail (`main_mali` in the shader, `kDenseConvMaliPixels = 4` in the contract). It replaces the previous
`accumulate_branchy` kernel on subgroup < 32.

`malioc` (Mali-G710, `-S 0=16`), static totals as in Baseline:

| variant | work registers | occupancy | stack | A / LS |
|---|---|---|---|---|
| previous (PX 1) | 64 | 50 % | none | 4.21 / 26.00 |
| PX 2 | 62 | 50 % | none | 6.11 / 36.00 |
| PX 4 | 64 | 50 % | none | 9.22 / 50.00 |
| PX 8 (not built) | 64 | 50 % | 148 B spilled | 15.59 / 104.00 |

Per output that is LS 6.5 → 4.5 → 3.1 cycles. PX 8 spills, so it was not tried on the device.

Phone A/B (method of Baseline, 3 rounds interleaved), arms built from two worktrees: A = previous kernel
(368a884), B = PX 2, C = PX 4 (this tree). Pixel 7a, cifar-dense (ms):

| stage | A median (min-max) ms | B median (min-max) ms | C median (min-max) ms | B vs A | C vs A |
|---|---|---|---|---|---|
| 1 | 7.087 (7.087-7.087, n=3x140) | 4.003 (4.001-4.003, n=3x250) | 2.571 (2.560-2.572, n=3x392) | -43.5 % | -63.7 % |
| 2 | 0.496 (0.495-0.499, n=3x1994) | 0.494 (0.493-0.496, n=3x2000) | 0.493 (0.491-0.494, n=3x2000) | -0.3 % | -0.6 % |
| 3 | 24.464 (24.459-24.465, n=3x40) | 15.802 (15.802-15.812, n=3x51) | 14.564 (12.360-15.769, n=3x55) | -35.4 % | -40.5 % |
| 4 | 0.386 (0.384-0.386, n=3x2000) | 0.384 (0.382-0.384, n=3x2000) | 0.381 (0.380-0.385, n=3x2000) | -0.5 % | -1.3 % |
| 5 | 36.357 (36.355-36.361, n=3x27) | 19.658 (19.657-19.661, n=3x45) | 18.270 (16.529-18.289, n=3x49) | -45.9 % | -49.7 % |
| 6 | 48.812 (48.807-48.818, n=3x20) | 27.907 (27.892-27.918, n=3x35) | 20.324 (20.293-20.328, n=3x49) | -42.8 % | -58.4 % |
| 7 | 32.559 (32.553-32.561, n=3x30) | 17.986 (17.985-17.989, n=3x55) | 15.771 (15.746-15.805, n=3x68) | -44.8 % | -51.6 % |
| 8 | 0.194 (0.193-0.198, n=3x2000) | 0.196 (0.193-0.196, n=3x2000) | 0.191 (0.179-0.192, n=3x2000) | +1.0 % | -1.6 % |
| 9 | 6.886 (6.837-8.423, n=3x107) | 6.908 (6.834-7.805, n=3x78) | 6.852 (6.779-8.379, n=3x118) | +0.3 % | -0.5 % |
| 10 | 6.812 (6.677-7.906, n=3x122) | 6.727 (6.705-6.729, n=3x148) | 6.762 (6.719-7.299, n=3x122) | -1.3 % | -0.7 % |
| 11 | 0.992 (0.988-0.996, n=3x1279) | 0.980 (0.881-0.990, n=3x1354) | 0.990 (0.876-0.995, n=3x1333) | -1.2 % | -0.2 % |
| sum | 165.045 | 101.045 | 87.169 | -38.8 % | -47.2 % |

Samsung, cifar-dense (ms), A = 368a884, B = PX 4. The Samsung takes the unchanged subgroup >= 32 path:

| stage | A median (min-max) ms | B median (min-max) ms | B vs A |
|---|---|---|---|
| 1 | 0.909 (0.908-0.914, n=3x1110) | 0.906 (0.902-0.913, n=3x1116) | -0.3 % |
| 2 | 1.299 (1.285-1.301, n=3x792) | 1.269 (1.269-1.284, n=3x848) | -2.3 % |
| 3 | 3.394 (3.081-3.401, n=3x472) | 3.396 (3.081-3.400, n=3x473) | +0.1 % |
| 4 | 0.967 (0.965-0.969, n=3x1145) | 0.969 (0.943-0.971, n=3x1136) | +0.3 % |
| 5 | 2.702 (2.444-3.135, n=3x319) | 2.705 (2.704-3.136, n=3x334) | +0.1 % |
| 6 | 3.375 (3.373-3.376, n=3x327) | 3.370 (3.370-3.373, n=3x327) | -0.1 % |
| 7 | 2.744 (2.143-2.769, n=3x420) | 2.145 (2.143-2.746, n=3x418) | -21.9 % |
| 8 | 0.325 (0.325-0.326, n=3x2000) | 0.320 (0.318-0.321, n=3x2000) | -1.7 % |
| 9 | 2.788 (1.981-2.834, n=3x355) | 2.781 (2.745-2.847, n=3x352) | -0.2 % |
| 10 | 2.012 (1.980-2.020, n=3x371) | 2.724 (2.723-2.944, n=3x341) | +35.4 % |
| 11 | 0.656 (0.644-0.670, n=3x2000) | 0.649 (0.645-0.670, n=3x1391) | -1.0 % |
| sum | 21.170 | 21.235 | +0.3 % |

**Gate (spec "A Mali path ships only if it is measured faster"): passed for PX 4.**
- On the Pixel, stages 3 / 5 / 6 / 7 are -40.5 / -49.7 / -58.4 / -51.6 %. Each new spread (12.36-15.77 /
  16.53-18.29 / 20.29-20.33 / 15.75-15.81 ms) lies entirely below the previous one (24.46 / 36.36 / 48.81 /
  32.55-32.56 ms).
- Stage 1 is -63.7 %, so no conv stage is slower. The app sum is 165.0 → 87.2 ms (-47 %).
- PX 2 also passes but gains less on every conv, so PX 4 ships.
- The Samsung's convs 1, 3, 5, 6 are within 0.3 %. Stage 7 reads -21.9 %, which is the bimodal 2.14 / 2.75 ms
  noise of the Baseline (both arms show both values).

**The wide path is unchanged.**
- glslc output of the previous and the new shader, specialized at 64 and at 32, then folded, optimized and
  canonicalized (`spirv-opt --set-spec-const-default-value --freeze-spec-const --fold-spec-const-op-composite
  -O --canonicalize-ids`), has the same instructions (0 lines of sorted diff).
- The conv-output hashes of both apps (5 convs x synthetic / real weights) on the Samsung are identical between
  368a884 and this tree (20 of 20).

## Open Questions

- ~~Whether Arm Performance Studio can be installed on the build box (licence acceptance, offline
  install). It does not change the approach (D5).~~ **Answered (task 1.1):** nothing has to be installed.
  Arm Performance Studio 2026.5 is already on the build box, extracted under
  `~/tools/arm-performance-studio/Arm_Performance_Studio_2026.5/` on 2026-09-26, before this change.
  `mali_offline_compiler/malioc --version` prints `Mali Offline Compiler v2026.5.0 (Build 922df8)`. This
  change installs nothing and accepts no licence. It calls that binary by its path, with
  `-c Mali-G710 --vulkan` on the glslc output (the build's flags: `--target-env=vulkan1.3 -O`), and
  `-S 0=16` to specialize the Mali path.
