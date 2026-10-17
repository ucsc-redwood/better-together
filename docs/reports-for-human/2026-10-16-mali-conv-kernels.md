# Mali conv kernels: what shipped, what was rejected, and the numbers (2026-10-16)

OpenSpec change `mali-conv-kernels` (tasks 1-6, commits on `feat/mali-conv-kernels`). The cifar 3x3 convolutions
were the largest cost left on the Pixel 7a (Mali-G710, subgroup 16), and the earlier conv speed-ups did not reach
it: they come from subgroup-uniform (scalar) weight loads, and Mali has no scalar unit. Every number below was
measured on the two phones. rocky-ryzen ran correctness tests only.

| technique (Mali path, `SUBGROUP_SIZE < 32`) | decision | Pixel 7a effect (convs 3 / 5 / 6 / 7, conv 1) |
|---|---|---|
| dense: 4 output pixels x 4 channels per invocation (PX 4) | **shipped** | -40 / -50 / -58 / -52 %, -64 %; cifar-dense stage sum 165.0 → 87.2 ms |
| dense: PX 2 | not used (PX 4 is faster) | -35 / -46 / -43 / -45 %, -44 % |
| sparse: 4 output pixels per invocation, one decode and weight load per nonzero (PX 4) | **shipped** | -56 / -55 / -54 / -54 %, -16 %; cifar-sparse stage sum 1190.7 → 584.8 ms |
| sparse: PX 2 / PX 8 | not used (PX 4 is faster) | PX 2 -36..-39 %, PX 8 -44..-50 % |
| 5.1 dense: weights staged in shared memory, workgroup per channel block | rejected | -6 (noise) / -1 / -5 / -9 %, conv 1 **+5 %** |
| 5.2 sparse: decoded CSR row in shared memory, workgroup per output channel | rejected | -32 / **+49 / +65 / +60 %**, conv 1 **+11 %** |
| 5.3 dense: input rows plus halo staged in shared memory | rejected | **+17** / -2 / -3 / -2 %, conv 1 **+27 %** |

The 5.x rows are measured against the shipped PX 4 kernels. The subgroup >= 32 paths (Samsung Xclipse 940, Radeon
780M) are unchanged: after specialization at 64 and at 32 the optimized SPIR-V of both shaders is
instruction-identical to the previous shaders', and the conv outputs of both apps hash identically on the Samsung
before and after every group (20 of 20 hashes each time).

## Method

- **Per-stage A/B** (design D5). `bm-prof-cifar-{dense,sparse}-vk`, isolated scenario, real weights
  (`saved_params/export`), `--benchmark_filter='vulkan/'`, `BT_PROF_BUDGET_S=1.0 BT_PROF_ABANDON_S=5
  BT_PROF_MAX_CELL_S=3`. The budget knobs are raised from their defaults because with the defaults a stage above
  0.25 s is sampled once, and the Pixel's sparse stage 6 takes ~0.4 s.
- Arms are built from two (or more) git worktrees with the Android preset (NDK r30) and pushed to
  `/data/local/tmp/bt-mali`. They run alternately, 3 rounds, with a 20 s sleep before every run; each run also
  checks that no other benchmark or test process is on the phone.
- Reported per stage: the median over the 3 runs of each run's p50 (GPU timestamps), with the min-max of the three
  p50s, as `n = runs x samples per run`.
- **Ship gate** (spec): stages 3, 5, 6 and 7 must each have a lower median with non-overlapping spreads, and no
  conv stage may have a higher median beyond its spread.
- **Static:** `malioc` v2026.5.0 (Arm Performance Studio 2026.5, already on the build box), `-c Mali-G710 --vulkan
  -S 0=16` on the glslc output. Totals count each loop body once, so they compare variants of one shader and are
  not cycles per output.
- **Correctness**, at every step: `DenseConvK3S1P1` / `SparseConvK3S1P1` (double-precision reference; 6x6, 8x8, 7x5
  maps; the sparse CSR has an empty row and a row that reads only padding at the right edge), and the full
  `test-cifar-{dense,sparse}-vk` differential suites with synthetic and with real weights. These ran on the Pixel,
  the Samsung, rocky-ryzen and the build box's RADV iGPU.

## Baseline and noise (same binary in both arms, 9a10ea0)

Pixel 7a: dense convs within 0.1 %; sparse convs 3/5/6/7 within 0.5 %; sparse conv 1 within 5 %. The Samsung is
bimodal from run to run (dense conv 7 reads 2.14 or 2.75 ms in one binary; sparse conv medians differ by up to 22 %
between identical arms), so its "no slowdown" check can only rule out larger changes.

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

`malioc`, the kernels before this change: dense Mali path 64 work registers (50 % occupancy), A 4.21 / LS 26.00;
sparse 26 registers (100 %), A 7.49 / LS 29.00. Both were load/store-bound.

## Dense: output blocking, shipped with PX 4

A = previous kernel (368a884), B = PX 2, C = PX 4. Pixel 7a, cifar-dense (ms):

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

Samsung (subgroup-64 path, unchanged), A = 368a884, B = PX 4 (ms):

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

`malioc`: PX 1 (previous) / 2 / 4 use 64 / 62 / 64 registers, all at 50 % occupancy and without spills, with LS 26 /
36 / 50, which is 6.5 / 4.5 / 3.1 per output. PX 8 spills 148 B to the stack and was not built.

## Sparse: output blocking, shipped with PX 4

A = previous kernel (8bd33e4), B / C / D = PX 2 / 4 / 8. Pixel 7a, cifar-sparse (ms):

| stage | A median (min-max) ms | B median (min-max) ms | C median (min-max) ms | D median (min-max) ms | B vs A | C vs A | D vs A |
|---|---|---|---|---|---|---|---|
| 1 | 20.298 (19.447-20.305, n=3x45) | 18.187 (18.181-18.189, n=3x47) | 17.065 (17.037-17.073, n=3x58) | 18.267 (18.239-18.302, n=3x45) | -10.4 % | -15.9 % | -10.0 % |
| 2 | 4.479 (4.467-4.532, n=3x447) | 4.470 (4.462-4.518, n=3x225) | 4.464 (4.462-4.537, n=3x226) | 4.484 (4.463-4.542, n=3x281) | -0.2 % | -0.3 % | +0.1 % |
| 3 | 181.350 (181.335-181.359, n=3x5) | 115.658 (115.644-115.695, n=3x8) | 80.330 (80.309-80.398, n=3x12) | 90.106 (89.987-92.638, n=3x11) | -36.2 % | -55.7 % | -50.3 % |
| 4 | 3.435 (3.416-3.504, n=3x642) | 3.519 (3.368-3.589, n=3x640) | 3.545 (3.401-3.764, n=3x643) | 3.415 (3.387-3.428, n=3x642) | +2.4 % | +3.2 % | -0.6 % |
| 5 | 283.558 (283.323-284.577, n=3x3) | 175.887 (175.421-175.948, n=3x5) | 128.505 (128.432-128.926, n=3x7) | 148.073 (147.683-148.121, n=3x6) | -38.0 % | -54.7 % | -47.8 % |
| 6 | 398.662 (398.244-398.689, n=3x2) | 249.452 (249.083-249.760, n=3x3) | 183.689 (183.291-183.960, n=3x5) | 216.609 (216.527-218.126, n=3x4) | -37.4 % | -53.9 % | -45.7 % |
| 7 | 242.437 (242.277-242.502, n=3x4) | 148.925 (148.753-149.148, n=3x6) | 110.735 (110.531-110.841, n=3x9) | 134.761 (134.341-135.721, n=3x7) | -38.6 % | -54.3 % | -44.4 % |
| 8 | 1.201 (1.172-1.208, n=3x1949) | 1.187 (1.185-1.204, n=3x1869) | 1.209 (1.193-1.269, n=3x1881) | 1.205 (1.200-1.264, n=3x1896) | -1.1 % | +0.7 % | +0.4 % |
| 9 | 26.902 (26.890-26.923, n=3x37) | 26.878 (26.851-26.904, n=3x33) | 26.883 (26.843-26.892, n=3x33) | 26.902 (26.896-26.918, n=3x34) | -0.1 % | -0.1 % | -0.0 % |
| 10 | 27.216 (27.189-27.253, n=3x36) | 27.158 (27.153-27.219, n=3x36) | 27.199 (27.180-27.258, n=3x36) | 27.234 (27.197-27.242, n=3x36) | -0.2 % | -0.1 % | +0.1 % |
| 11 | 1.180 (1.159-1.244, n=3x1179) | 1.198 (1.015-1.228, n=3x1035) | 1.185 (1.182-1.244, n=3x1068) | 1.228 (1.179-1.234, n=3x1287) | +1.5 % | +0.4 % | +4.1 % |
| sum | 1190.719 | 772.520 | 584.808 | 672.284 | -35.1 % | -50.9 % | -43.5 % |

Samsung (unchanged path), A = 8bd33e4, B = PX 4 (ms). Every conv spread overlaps the previous one. Conv 1's 4.10 /
4.62 ms are the two values one binary showed in the baseline:

| stage | A median (min-max) ms | B median (min-max) ms | B vs A |
|---|---|---|---|
| 1 | 4.102 (4.102-4.103, n=3x243) | 4.617 (4.102-4.620, n=3x243) | +12.6 % |
| 2 | 2.397 (2.389-2.412, n=3x699) | 2.397 (2.369-2.400, n=3x690) | -0.0 % |
| 3 | 39.268 (35.340-40.740, n=3x26) | 39.773 (38.727-43.201, n=3x26) | +1.3 % |
| 4 | 1.841 (1.828-1.865, n=3x1308) | 1.836 (1.828-1.843, n=3x1345) | -0.3 % |
| 5 | 70.452 (60.190-82.499, n=3x14) | 67.871 (65.209-78.342, n=3x14) | -3.7 % |
| 6 | 94.720 (84.315-97.294, n=3x10) | 108.510 (93.673-113.011, n=3x6) | +14.6 % |
| 7 | 57.100 (55.069-62.909, n=3x15) | 58.232 (54.313-65.648, n=3x15) | +2.0 % |
| 8 | 2.677 (2.667-2.694, n=3x1176) | 2.647 (2.645-2.702, n=3x1204) | -1.1 % |
| 9 | 10.051 (10.044-10.052, n=3x99) | 10.053 (10.052-10.054, n=3x98) | +0.0 % |
| 10 | 10.046 (10.038-10.058, n=3x98) | 10.035 (10.034-10.037, n=3x98) | -0.1 % |
| 11 | 1.153 (1.152-1.179, n=3x2000) | 1.156 (1.153-1.159, n=3x2000) | +0.3 % |
| sum | 293.808 | 307.127 | +4.5 % |

`malioc`: previous 26 registers, A 7.49 / LS 29 (two walks, one unrolled four times); PX 2 / 4 / 8 use 22 / 24 /
30 registers, all at 100 % occupancy, with LS 9 / 13 / 21.

## Shared-memory staging: three variants, all rejected

All three were built on the shipped PX 4 kernels (69954b0) in a scratch worktree and passed the Pixel suites; none
was committed. On Valhall a shared-memory load is issued by the same load/store unit as a global load, so staging
does not lower the static LS count (dense 50 → 52 for 5.1 and 5.3, sparse 13 → 14 for 5.2).

**5.1, dense weights in shared memory.** `gl_WorkGroupID.y` is the channel block, and its weights are staged 32 input
channels at a time. A = 69954b0, B = 5.1, Pixel 7a (ms):

| stage | A median (min-max) ms | B median (min-max) ms | B vs A |
|---|---|---|---|
| 1 | 2.569 (2.553-2.570, n=3x391) | 2.691 (2.690-2.691, n=3x373) | +4.7 % |
| 2 | 0.496 (0.492-0.497, n=3x2000) | 0.497 (0.496-0.498, n=3x2000) | +0.1 % |
| 3 | 15.788 (14.552-17.995, n=3x55) | 14.835 (13.669-16.083, n=3x62) | -6.0 % |
| 4 | 0.385 (0.384-0.385, n=3x2000) | 0.385 (0.385-0.386, n=3x2000) | +0.1 % |
| 5 | 16.504 (16.456-16.523, n=3x63) | 16.287 (14.976-16.293, n=3x44) | -1.3 % |
| 6 | 20.296 (20.285-20.348, n=3x49) | 19.199 (19.191-19.203, n=3x46) | -5.4 % |
| 7 | 15.799 (15.771-15.802, n=3x68) | 14.308 (14.293-15.534, n=3x69) | -9.4 % |
| 8 | 0.193 (0.193-0.196, n=3x2000) | 0.192 (0.190-0.194, n=3x2000) | -1.0 % |
| 9 | 6.873 (6.831-7.860, n=3x125) | 6.736 (6.309-7.388, n=3x127) | -2.0 % |
| 10 | 6.818 (6.720-7.269, n=3x126) | 6.732 (6.669-7.411, n=3x126) | -1.3 % |
| 11 | 0.894 (0.876-0.993, n=3x1140) | 0.885 (0.878-0.992, n=3x1367) | -0.9 % |
| sum | 86.616 | 82.746 | -4.5 % |

**5.2, sparse CSR row in shared memory.** `gl_WorkGroupID.y` is the output channel. Each chunk of 256 nonzeros is
decoded once per workgroup into one `ivec2` (input offset with the tap, and the weight's bits). A = 69954b0,
B = 5.2, Pixel 7a (ms):

| stage | A median (min-max) ms | B median (min-max) ms | B vs A |
|---|---|---|---|
| 1 | 17.042 (16.975-17.096, n=3x58) | 18.933 (18.921-19.072, n=3x65) | +11.1 % |
| 2 | 4.461 (4.461-4.464, n=3x225) | 4.464 (4.462-4.466, n=3x224) | +0.1 % |
| 3 | 80.287 (80.232-80.333, n=3x12) | 54.377 (54.352-54.482, n=3x19) | -32.3 % |
| 4 | 3.556 (3.397-3.577, n=3x642) | 3.407 (3.394-3.407, n=3x642) | -4.2 % |
| 5 | 128.754 (128.752-128.922, n=3x7) | 192.347 (192.099-192.565, n=3x5) | +49.4 % |
| 6 | 183.390 (183.285-183.836, n=3x5) | 302.140 (301.196-303.999, n=3x3) | +64.8 % |
| 7 | 110.555 (110.357-110.561, n=3x8) | 176.586 (175.908-176.706, n=3x5) | +59.7 % |
| 8 | 1.209 (1.206-1.241, n=3x1958) | 1.195 (1.193-1.209, n=3x1936) | -1.2 % |
| 9 | 26.900 (26.887-26.913, n=3x36) | 26.891 (26.878-26.897, n=3x35) | -0.0 % |
| 10 | 27.215 (27.182-27.219, n=3x36) | 27.229 (27.217-27.274, n=3x36) | +0.1 % |
| 11 | 1.218 (1.163-1.220, n=3x1265) | 1.172 (1.162-1.174, n=3x1102) | -3.8 % |
| sum | 584.588 | 808.742 | +38.3 % |

The 16x16 conv gains 32 %, but the 8x8 convs lose 49-65 %. Inferred, not measured: at N = 128 an 8x8 map gives
only 8 workgroups per output channel, and the long rows of those layers (432-864 nonzeros) put each workgroup
through 2-4 chunk barriers.

**5.3, dense input rows with halo in shared memory.** This was tried because `malioc` still reports the shipped
dense path LS-bound. A workgroup is one image, a band of rows and several channel blocks; it stages the band's rows
plus the halo for as many channels as fit in 16 KiB. A = 69954b0, B = 5.3, Pixel 7a (ms):

| stage | A median (min-max) ms | B median (min-max) ms | B vs A |
|---|---|---|---|
| 1 | 2.553 (2.551-2.570, n=3x391) | 3.234 (3.232-3.247, n=3x310) | +26.7 % |
| 2 | 0.496 (0.496-0.497, n=3x2000) | 0.498 (0.495-0.498, n=3x2000) | +0.3 % |
| 3 | 14.554 (14.553-15.773, n=3x55) | 16.968 (15.648-16.972, n=3x58) | +16.6 % |
| 4 | 0.385 (0.384-0.386, n=3x2000) | 0.386 (0.385-0.386, n=3x2000) | +0.4 % |
| 5 | 16.465 (15.220-16.492, n=3x49) | 16.076 (16.030-16.118, n=3x45) | -2.4 % |
| 6 | 20.301 (20.242-20.302, n=3x49) | 19.781 (19.719-19.788, n=3x45) | -2.6 % |
| 7 | 15.783 (15.756-15.796, n=3x68) | 15.400 (15.381-15.428, n=3x63) | -2.4 % |
| 8 | 0.193 (0.191-0.196, n=3x2000) | 0.194 (0.183-0.195, n=3x2000) | +0.3 % |
| 9 | 6.794 (6.773-7.245, n=3x131) | 6.884 (6.759-9.636, n=3x88) | +1.3 % |
| 10 | 6.862 (6.665-7.219, n=3x160) | 7.181 (6.737-7.807, n=3x127) | +4.6 % |
| 11 | 0.988 (0.946-0.992, n=3x1346) | 0.985 (0.982-0.986, n=3x1192) | -0.2 % |
| sum | 85.376 | 87.588 | +2.6 % |

## End to end (one schedule run per app on the Pixel 7a; reported, not gated)

- Profile: `bm-prof-cifar-{dense,sparse}-vk` with the final kernels on the Pixel 7a. Isolated scenario, all PUs
  (Vulkan, big, medium, little), default sampling knobs, real weights, one run.
- Schedule: `02_gen_schedule_merged.py --table_type isolated --minimize_mode tmax -n 5` (no `overhead.json` in this
  store, so the plain cost model). The best-predicted solution is the one that ran.
- Run: `bm-gen-logs-cifar-<app>-vk --schedule-file`, 32 tasks, once with the final binary and once with the binary
  from before this change (9a10ea0), on the **same** schedule, 20 s apart. The schedule was solved from the new
  kernels' profile, so the "before" column is that schedule run on the old kernels, not the old kernels' own best
  schedule.

| app | schedule (chunks) | predicted max chunk | before: ms / task (GPU chunk median) | final: ms / task (GPU chunk median) |
|---|---|---|---|---|
| cifar-dense | Little 1-2, Big 3-5, **GPU 6-8**, Medium 9-11 | 33.5 ms | 86.35 (83.25) | **42.41** (35.82) |
| cifar-sparse | Little 1, Big 2-4, **GPU 5-7**, Medium 8-11 | 419.8 ms | 1148.42 (1115.01) | **436.72** (418.90) |

The CPU chunks are within a few percent between the two runs (dense Big 3-5 38.35 / 38.92 ms; sparse Big 2-4 267.8 /
259.5 ms). The GPU chunk holds the convs this change rewrote, and it sets the period. One run per arm, so this is
reported, not gated.

## Measured vs inferred

- Measured: every table above, the SPIR-V identity of the wide paths, the output hashes, the mutation checks of the
  grid contract, and the `malioc` figures.
- Inferred: why PX 8 is slower for sparse (more masked loads per tap when a tile spans a whole 8-wide row), and
  why 5.2 loses on the 8x8 maps (barriers and too few workgroups per channel). No Mali hardware counters were read
  (no Streamline capture).
- Only the Pixel 7a's G710 was measured. Another subgroup-16 Mali takes the same path and its correctness tests run
  there, but speed is only claimed for the G710.
- The Samsung "unchanged" claim rests on the code identity and the hashes. Its timings are too noisy to show
  anything smaller than about 20 %.

## Follow-ups

- The Mali paths now spend most of their time in the input loads (dense LS 50 vs A 9 static). A layout with
  channels innermost (NHWC, or NCHW4) would allow `vec4` loads over input channels, which D2 ruled out for NCHW.
  That is a cross-backend data-layout change, outside this change.
- FP16 on the Mali paths belongs to `cifar-gpu-kernels` (this change keeps FP32).
