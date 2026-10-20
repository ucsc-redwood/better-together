# cifar-dense Vulkan conv: channel blocks per workgroup and per-conv register tiles (2026-10-20)

**Change:** OpenSpec `cifar-gpu-kernels`, task group 2 (commits 327d6aa..334f576 on `feat/cifar-dense-conv`; the last
code commit is e34ca8d) · **Targets:** Samsung Galaxy (Xclipse 940, subgroup 64), Pixel 7a (Mali-G710, subgroup 16) ·
**Tool:** `bm-prof-cifar-dense-vk`, per-stage GPU timestamps, with the GPU clock it records · rocky-ryzen ran
correctness tests and ACO dumps only, no timing.

> **One-line finding:** on the Samsung, conv2-conv5 take 9.56 instead of 11.40 ms (-16 %; conv2 and conv4 -26 %),
> and 27 % fewer GPU cycles. The cycles fall more than the time because the Samsung's GPU clock settles lower when a
> kernel needs fewer cycles. That is also why each conv has its own tile: the tiles that save the most cycles are
> not the fastest in wall time. The Pixel 7a runs the unchanged Mali path and stays within noise.

## What changed

- **C1, one block of output channels per workgroup.** `new_cifar_dense_conv2d_k3s1p1_oc.comp` takes its channel
  block from `gl_WorkGroupID.x`. The weights were already scalar loads on the model's shapes: the previous kernel's
  `subgroupBroadcastFirst` base did that (ACO on the 780M). C1 makes the uniformity structural, with no subgroup test
  and no second loop. The design put the block on `.y`. Workgroups launch x first, so on `.y` consecutive
  workgroups read different images, and the Samsung's conv2 was 21 % slower (conv1 9 %). On `.x` they share the
  input pixels.
- **C3, a register tile per invocation.** `OC_TILE` output channels x `PX_TILE` adjacent pixels of a row (spec
  constants 1, 2), the 3 x (`PX_TILE`+2) input window loaded once per input channel. Each conv has its own tile
  (one pipeline per distinct tile): conv2 8 x 4, conv3 and conv4 16 x 4, conv5 8 x 1. Conv1 (C = 3) keeps the
  previous kernel, which no tile beat. `BT_VK_DENSE_CONV_TILE` overrides them (one tile or five, `flat` = the previous
  kernel).
- **C2, input rows plus halo in shared memory:** built (`C_TILE`, spec constant 3), measured, and left **off**: every
  staged tile took 2-3.5x the GPU cycles of the previous kernel on the Samsung.
- **C4 (FP16) and C5 (cooperative matrix):** not built. Neither phone has `VK_KHR_cooperative_matrix` (only the 780M
  does). FP16 was optional and the FP32 tiles met the requirement (below).
- cifar-dense below subgroup 32 (the Pixel) runs `new_cifar_dense_conv2d_k3s1p1.comp`'s Mali path as before; that
  shader file is unchanged.

## Variant per device

| device | subgroup | conv1 | conv2 | conv3 | conv4 | conv5 | how chosen |
|---|---|---|---|---|---|---|---|
| Samsung (Xclipse 940) | 64 | previous kernel | `_oc` 8x4 | `_oc` 16x4 | `_oc` 16x4 | `_oc` 8x1 | timed (task 2.6) |
| rocky-ryzen (780M) | 64 | previous kernel | `_oc` 8x4 | `_oc` 16x4 | `_oc` 16x4 | `_oc` 8x1 | Samsung's defaults (not timed) |
| Pixel 7a (Mali-G710) | 16 | Mali path | Mali path | Mali path | Mali path | Mali path | unchanged (`gpu-kernels/mali-conv`) |

No device runs a reduced-precision or cooperative-matrix variant: none exists.

## Results (final A/B, 2026-10-20 07:43-07:48 UTC)

Method (cifar-gpu-kernels task 2.1, as group 1's): the d563dc5 and e34ca8d binaries (`android` preset, NDK r30) in
their own directories on each phone, real weights (`BT_WEIGHTS_DIR`, `saved_params/export`), `BT_PROF_SCENARIO=isolated
BT_PROF_BUDGET_S=1.0 BT_PROF_ABANDON_S=5 BT_PROF_MAX_CELL_S=3 --benchmark_filter='vulkan/'`, alternating, 3 rounds,
20 s idle before every run, no other benchmark or test process on the phone. Per stage: the median of the 3 runs'
p50, with the min-max of the three p50s, `n = runs x samples`. Stages 1, 3, 5, 6, 7 are conv1-conv5. The last row
is each run's conv2-conv5 sum, the spec's measure.

Samsung (subgroup 64):

| stage | before (d563dc5) median (min-max) ms | after (e34ca8d) median (min-max) ms | after vs before |
|---|---|---|---|
| 1 | 0.911 (0.908-0.916, n=3x1074-1124) | 0.915 (0.907-0.916, n=3x1066-1108) | +0.4 % |
| 2 | 1.306 (1.257-1.308, n=3x807-1934) | 1.293 (1.266-1.301, n=3x876-1141) | -1.0 % |
| 3 | 3.402 (3.097-3.405, n=3x472-475) | 2.487 (2.486-2.490, n=3x959-966) | -26.9 % |
| 4 | 0.974 (0.971-0.977, n=3x1189-1325) | 0.973 (0.967-0.976, n=3x1139-1293) | -0.1 % |
| 5 | 2.453 (2.444-3.144, n=3x370-448) | 2.451 (2.447-2.463, n=3x473-606) | -0.1 % |
| 6 | 3.376 (3.371-3.390, n=3x326-328) | 2.514 (2.507-2.908, n=3x352-448) | -25.5 % |
| 7 | 2.155 (2.142-2.758, n=3x360-525) | 2.104 (2.103-2.847, n=3x349-528) | -2.4 % |
| 8 | 0.328 (0.326-0.328, n=3x2000) | 0.324 (0.324-0.329, n=3x2000) | -1.0 % |
| 9 | 2.123 (2.094-2.867, n=3x344-552) | 2.957 (2.091-2.978, n=3x338-522) | +39.3 % |
| 10 | 2.822 (1.976-2.837, n=3x362-543) | 1.974 (1.973-2.726, n=3x370-540) | -30.1 % |
| 11 | 0.635 (0.626-0.641, n=3x1463-2000) | 0.652 (0.639-0.661, n=3x1369-2000) | +2.7 % |
| sum | 20.486 | 18.644 | -9.0 % |
| conv2-conv5 (3+5+6+7), per-run sum | 11.403 (11.059-12.676, 3 runs) | 9.560 (9.550-10.696, 3 runs) | -16.2 % |

The same runs per conv: the median p50, the GPU clock of each run's cell, and the median of p50 x clock in millions of
GPU cycles:

| arm | stage 1: ms / MHz / Mcyc | stage 3: ms / MHz / Mcyc | stage 5: ms / MHz / Mcyc | stage 6: ms / MHz / Mcyc | stage 7: ms / MHz / Mcyc | conv2-5 ms | conv2-5 Mcyc |
|---|---|---|---|---|---|---|---|
| before | 0.911 / 252/252/252 / 0.23 | 3.402 / 315/315/315 / 1.07 | 2.453 / 800/800/650 / 1.96 | 3.376 / 800/800/800 / 2.70 | 2.155 / 800/800/650 / 1.72 | 11.403 (11.059-12.676) | 7.47 (7.35-7.61) |
| after | 0.915 / 252/252/252 / 0.23 | 2.487 / 252/252/252 / 0.63 | 2.451 / 450/450/450 / 1.10 | 2.514 / 800/700/800 / 2.01 | 2.104 / 800/600/800 / 1.68 | 9.560 (9.550-10.696) | 5.42 (5.42-5.47) |

Pixel 7a (Mali path, not changed by this work). Stage 3 reads between 10.0 and 14.5 ms and stage 7 between 14.5 and 15.8
ms from run to run in either binary (task 2.1's identical-binary A/B), so their medians move between
sessions; the spreads overlap:

| stage | before (d563dc5) median (min-max) ms | after (e34ca8d) median (min-max) ms | after vs before |
|---|---|---|---|
| 1 | 2.569 (2.567-2.571, n=3x392-393) | 2.560 (2.557-2.571, n=3x391-487) | -0.3 % |
| 2 | 0.495 (0.495-0.497, n=3x2000) | 0.494 (0.493-0.495, n=3x2000) | -0.0 % |
| 3 | 12.357 (12.357-14.540, n=3x63-80) | 14.545 (13.416-14.546, n=3x55-68) | +17.7 % |
| 4 | 0.383 (0.382-0.385, n=3x2000) | 0.384 (0.383-0.384, n=3x2000) | +0.2 % |
| 5 | 16.378 (15.308-16.425, n=3x54-63) | 15.205 (15.180-16.519, n=3x49-63) | -7.2 % |
| 6 | 20.330 (20.318-20.340, n=3x48-49) | 20.289 (20.273-20.308, n=3x49) | -0.2 % |
| 7 | 14.547 (14.546-15.767, n=3x68) | 15.728 (14.686-15.771, n=3x68-76) | +8.1 % |
| 8 | 0.176 (0.174-0.177, n=3x2000) | 0.176 (0.176-0.177, n=3x2000) | +0.2 % |
| 9 | 6.766 (6.181-7.316, n=3x124-161) | 5.452 (4.870-7.274, n=3x85-205) | -19.4 % |
| 10 | 6.697 (6.623-6.710, n=3x116-149) | 5.601 (4.573-6.700, n=3x129-195) | -16.4 % |
| 11 | 0.986 (0.870-0.989, n=3x1333-1541) | 0.877 (0.865-0.984, n=3x1345-2000) | -11.1 % |
| sum | 81.683 | 81.311 | -0.5 % |
| conv2-conv5 (3+5+6+7), per-run sum | 63.668 (63.599-65.945, 3 runs) | 65.805 (64.724-65.936, 3 runs) | +3.4 % |

## The Samsung's GPU clock follows the load

Each conv cell of a `bm-prof` run settles at one GPU clock between 252 and 1095 MHz. The more cycles a kernel saves,
the lower that clock: in the tile sweep (task 2.6, 2026-10-20 06:27-07:03 UTC; `<oc>x<px>x<c>`, the knob applied to
conv2-conv5, `c` > 0 = staged), 8x4x0 needs 43 % fewer cycles than the previous kernel for conv2-conv5 and is about
as fast in wall time, because it runs conv3 at 252-315 and conv5 at 350 MHz instead of 600-800. Cycles are stable
between runs (the previous kernel: 7.45-7.63 million over every session) while wall time is not (11.06-12.94 ms).
Each cell: median p50 ms / the clock of each run / median Mcycles.

| arm | stage 1: ms / MHz / Mcyc | stage 3: ms / MHz / Mcyc | stage 5: ms / MHz / Mcyc | stage 6: ms / MHz / Mcyc | stage 7: ms / MHz / Mcyc | conv2-5 ms | conv2-5 Mcyc |
|---|---|---|---|---|---|---|---|
| base | 0.913 / 252/252/252 / 0.23 | 3.404 / 315/315/315 / 1.07 | 3.286 / 600/600/800 / 1.97 | 3.371 / 800/800/800 / 2.70 | 2.143 / 800/800/650 / 1.71 | 12.201 (12.138-12.209) | 7.46 (7.45-7.63) |
| 4x1x0 | 0.914 / 252/252/252 / 0.23 | 3.358 / 315/315/350 / 1.06 | 2.955 / 800/600/600 / 1.77 | 3.180 / 800/800/800 / 2.54 | 2.549 / 650/650/800 / 1.66 | 11.288 (11.200-12.074) | 7.02 (7.00-7.06) |
| 8x1x0 | 0.911 / 252/252/252 / 0.23 | 3.290 / 315/315/315 / 1.04 | 2.226 / 800/800/545 / 1.78 | 3.181 / 800/800/800 / 2.55 | 2.113 / 800/800/650 / 1.69 | 10.819 (10.789-12.356) | 7.06 (7.04-7.07) |
| 4x2x0 | 0.913 / 252/252/252 / 0.23 | 2.487 / 252/252/252 / 0.63 | 3.654 / 315/315/315 / 1.15 | 2.633 / 545/500/600 / 1.43 | 4.001 / 350/350/350 / 1.40 | 12.695 (12.657-12.967) | 4.59 (4.58-4.68) |
| 8x2x0 | 0.913 / 252/252/252 / 0.23 | 2.746 / 252/252/252 / 0.69 | 3.899 / 350/350/350 / 1.36 | 1.976 / 600/800/650 / 1.57 | 3.634 / 350/350/350 / 1.27 | 12.281 (12.234-12.872) | 4.89 (4.62-4.89) |
| 8x4x0 | 0.906 / 252/252/252 / 0.23 | 2.497 / 252/252/252 / 0.63 | 3.119 / 252/315/315 / 0.97 | 2.532 / 500/450/500 / 1.27 | 3.854 / 350/350/350 / 1.35 | 12.024 (12.004-12.351) | 4.24 (4.03-4.26) |
| 4x4x0 | 0.910 / 252/252/252 / 0.23 | 2.619 / 252/252/252 / 0.66 | 2.952 / 252/315/252 / 0.80 | 2.599 / 545/545/500 / 1.41 | 3.342 / 315/315/315 / 1.05 | 11.517 (11.499-11.525) | 3.87 (3.81-4.06) |
| 16x4x0 | 0.915 / 252/252/252 / 0.23 | 3.135 / 315/252/315 / 0.97 | 2.474 / 450/450/400 / 1.10 | 2.503 / 800/800/800 / 2.00 | 2.570 / 500/500/545 / 1.29 | 10.635 (10.607-10.990) | 5.34 (5.27-5.37) |
| 8x2x2 | 0.912 / 252/252/252 / 0.23 | 3.131 / 600/650/800 / 2.00 | 4.338 / 800/900/800 / 3.48 | 5.186 / 900/1000/1000 / 5.19 | 3.989 / 800/800/900 / 3.26 | 16.336 (15.898-17.481) | 14.04 (13.91-14.14) |
| 8x4x2 | 0.908 / 252/252/252 / 0.23 | 2.720 / 800/800/800 / 2.18 | 4.434 / 900/900/900 / 3.99 | 6.232 / 1095/1095/1095 / 6.82 | 5.033 / 900/900/900 / 4.53 | 18.407 (18.402-18.544) | 17.51 (17.50-17.63) |

So the defaults are the fastest tile per conv in wall time (tasks.md 2.6 has the per-conv sweeps and a confirmation
run), and a cycle-saving change can look neutral on this phone in `bm-prof`. Conv2 runs at the 252 MHz floor in every
run, so its cycle savings show in full (-27 %).

## ACO (Radeon 780M, Mesa 25.2.7)

- Previous kernel, model shapes: per input channel, 9 vector input loads and 8 scalar weight loads
  (`s_buffer_load_b256` x 4 + `s_buffer_load_b32` x 4) feeding 36 FMAs, and each channel waits for its own loads.
  Only the per-lane branch, which the model never takes, loaded weights as vectors.
- `_oc` 4x1: the same loads with one loop and no subgroup test. Larger tiles raise the VGPRs from 32 (4x1) to 96
  (16x4); none spills. The staged tiles add 8.7-16.9 KB of shared memory and the staging index arithmetic.

## Correctness

`test-cifar-dense-vk` (19 tests, the OMP oracle at rtol 1e-3 / atol 1e-4 per stage, plus `DenseConvTiled`: seven tiles
on 6x6, 8x8 and 7x5 maps against a double-precision reference) passes on rocky-ryzen, the Pixel 7a and the Samsung
with synthetic and real weights; on rocky also with six tile settings and with `flat`. The local OMP gate, rocky's
`ctest -L vulkan` and both phones' suites are green at the final code (tasks.md 2.7, 2.13).

## Not done

FP16 (C4, optional tasks 2.8 and 2.9) and the cooperative-matrix conv (C5, task 2.10). Cooperative matrix exists
only on the 780M, where nothing is timed. FP16 would save cycles, which the Samsung turns mostly into a lower clock.
The VK end-task accuracy harness that group 3's task 3.7 meant to reuse was therefore not built.
