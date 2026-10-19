# Tasks

## Shared definitions

These apply to every group. `<app>` is `cifar-sparse` or `cifar-dense`.

**Gates:**
- **local:** `cmake --build --preset pc && ctest --test-dir build/pc -L omp --output-on-failure`.
- **rocky:** sync the tree, then build and test natively on rocky-ryzen:
  - `git ls-files -co --exclude-standard -z | rsync -a --from0 --files-from=- ./ doremy@100.97.77.80:bt-vk-campaign/`
  - `ssh doremy@100.97.77.80 bash -s` (rocky's login shell is fish), then:
    `cmake --preset vulkan && cmake --build --preset vulkan -j8 && cd build/vulkan && ctest -L vulkan && ./test-<app>-vk --device minipc`
- **phones:** `cmake --build --preset android --target test-<app>-vk`, then
  `scripts/run-on-android.sh 3A021JEHN02756 test-<app>-vk` and the same with `R5CY21Y3VEV`.

**Timing (phones only):** build `bm-prof-<app>-vk` for the baseline and the change from two worktrees
(`android` preset), push both to `/data/local/tmp/<dir>` on each phone, and run them alternately with
`BT_PROF_SCENARIO=isolated` and the real weights, 3 rounds, 20 s apart. Report each stage's median and min-max per
phone. On the Samsung `taskset` takes a hex mask.
**rocky-ryzen is for correctness only:** builds, tests and shader disassembly; no profiler, benchmark or timing run
there (the owner's rule). The rocky numbers quoted below are from before this rule and are context only.
**Which path runs where:** conv changes (S1–S3, C1–C5) touch only the subgroup-32-and-wider path, which runs on the
Samsung (and rocky). The Pixel 7a runs the Mali conv path of `gpu-kernels/mali-conv`, so its conv stages are a
no-regression check. FC changes run on both phones.

## 1. Sparse conv rewrite (gpu-kernels/cifar-sparse-conv; design S1–S4)

- [x] 1.1 Record the pre-change cifar-sparse per-stage GPU times on the Pixel 7a and the Samsung, with an identical-binary A/B to give the noise. Verify: an 11-stage table per phone with spreads in the evidence note.
  - *Method* (used for every timing of this group): `bm-prof-cifar-sparse-vk` built with the `android` preset (NDK
    r30) at ee4c68d, pushed twice (arms A and B, the same binary) to `/data/local/tmp/bt-sparse/{base,base2}` on each
    phone, real weights (`BT_WEIGHTS_DIR`, `saved_params/export` pushed beside them), `BT_PROF_SCENARIO=isolated
    BT_PROF_BUDGET_S=1.0 BT_PROF_ABANDON_S=5 BT_PROF_MAX_CELL_S=3 --benchmark_filter='vulkan/'` (the knobs of the
    mali-conv-kernels A/B). Arms alternate, 3 rounds, 20 s idle before every run, and each run first checks that no
    other `bm-`/`test-` process is on the phone; both phones ran at once (2026-10-19 23:07-23:13 UTC). Per stage:
    the median of the 3 runs' p50 (GPU timestamps), with the min-max of the three p50s, `n = runs x samples`. All 12
    runs exit 0 with 11 records.
  - *Noise:* every conv stage (1, 3, 5, 6, 7) of the two identical arms is within 0.3 % on the Pixel 7a and within
    0.1 % on the Samsung, and the spreads overlap. This is quieter than the Samsung of 2026-10-16 (medians 22 %
    apart then); the stage sum is 232.3 ms here vs 293.8-307.5 ms then. Non-conv stages move by up to 7 % (Pixel
    stage 11).
  - Pixel 7a (Mali path), ms:

    | stage | A median (min-max) ms | B median (min-max) ms | B vs A |
    |---|---|---|---|
    | 1 | 17.050 (17.037-17.413, n=3x58-68) | 17.007 (17.003-17.008, n=3x64-68) | -0.3 % |
    | 2 | 4.461 (4.460-4.467, n=3x336-338) | 4.461 (4.458-4.463, n=3x224-338) | +0.0 % |
    | 3 | 80.287 (80.271-80.300, n=3x12) | 80.290 (80.255-80.405, n=3x12) | +0.0 % |
    | 4 | 3.364 (3.358-3.365, n=3x642-698) | 3.363 (3.363-3.364, n=3x642-696) | -0.0 % |
    | 5 | 128.526 (128.484-128.915, n=3x7) | 128.700 (128.695-128.786, n=3x7) | +0.1 % |
    | 6 | 182.517 (181.577-183.294, n=3x5) | 183.086 (182.553-183.303, n=3x5) | +0.3 % |
    | 7 | 110.689 (109.953-110.874, n=3x9) | 110.621 (110.581-110.642, n=3x9) | -0.1 % |
    | 8 | 1.177 (1.172-1.179, n=3x1936-2000) | 1.181 (1.179-1.183, n=3x1900-1993) | +0.4 % |
    | 9 | 26.906 (26.864-26.922, n=3x34-36) | 26.897 (26.890-26.907, n=3x35-37) | -0.0 % |
    | 10 | 27.165 (27.119-27.170, n=3x36) | 27.183 (27.128-27.194, n=3x36) | +0.1 % |
    | 11 | 1.262 (1.236-1.383, n=3x1404-1531) | 1.174 (1.162-1.244, n=3x1378-1458) | -7.0 % |
    | sum | 583.406 | 583.963 | +0.1 % |

  - Samsung (subgroup-64 path), ms:

    | stage | A median (min-max) ms | B median (min-max) ms | B vs A |
    |---|---|---|---|
    | 1 | 4.102 (4.102-4.102, n=3x222-243) | 4.102 (4.102-4.102, n=3x189-243) | +0.0 % |
    | 2 | 2.424 (2.422-2.447, n=3x702-718) | 2.434 (2.417-2.458, n=3x668-712) | +0.4 % |
    | 3 | 34.028 (34.025-34.028, n=3x29) | 34.026 (34.026-34.027, n=3x29) | -0.0 % |
    | 4 | 1.848 (1.839-1.851, n=3x1326-1338) | 1.847 (1.831-1.848, n=3x1294-1321) | -0.0 % |
    | 5 | 51.650 (51.645-51.654, n=3x19) | 51.643 (51.641-51.646, n=3x19) | -0.0 % |
    | 6 | 68.572 (68.538-68.600, n=3x14) | 68.560 (68.526-68.596, n=3x14) | -0.0 % |
    | 7 | 45.764 (45.764-45.767, n=3x21) | 45.765 (45.758-45.773, n=3x21) | +0.0 % |
    | 8 | 2.696 (2.684-2.700, n=3x1184-1246) | 2.682 (2.681-2.688, n=3x1216-1242) | -0.5 % |
    | 9 | 10.058 (10.046-10.066, n=3x99) | 10.057 (10.054-10.063, n=3x99) | -0.0 % |
    | 10 | 10.052 (10.040-10.058, n=3x99) | 10.039 (10.032-10.041, n=3x98-99) | -0.1 % |
    | 11 | 1.155 (1.151-1.176, n=3x2000) | 1.178 (1.153-1.178, n=3x2000) | +2.0 % |
    | sum | 232.347 | 232.332 | -0.0 % |
- [ ] 1.2 Dump the current kernel's ACO disassembly on rocky (`RADV_DEBUG=shaders ./test-cifar-sparse-vk --device minipc --gtest_filter='*Stage6*'`). Verify: the evidence note quotes the CSR `buffer_load`/`s_load` lines, confirming or refuting S1's premise.
- [ ] 1.3 Add the VK-only packed `(offset, tap)` array (S2), filled per CSR matrix after `build_from_dense()` and `load_real_weights()` with that layer's H and W. Verify: a `test-cifar-sparse-vk` unit test decodes every entry back to its `col_idx` for all five layers, on synthetic and real weights, and prints each layer's maximum offset within the reserved bit width.
- [ ] 1.4 Write the uniform-channel kernel (S1): `out_c = gl_WorkGroupID.y`, grid `(ceil(N·P·Q/256), K, 1)`, a per-invocation 9-bit tap-valid mask; old kernel kept behind a dispatcher flag. Verify: the rocky gate passes with the new kernel and with the old one.
- [ ] 1.5 Add a test with an empty CSR row and a row whose nonzeros only hit the padded border. Verify: it passes against the OMP oracle on rocky (spec "Empty and border-only rows").
- [ ] 1.6 Re-dump ACO for the new kernel. Verify: the CSR metadata loads are scalar, quoted; or the note explains why not and what changed.
- [ ] 1.7 Stage the CSR row in shared memory in ≤ 1024-nonzero chunks behind a `STAGE_ROW` spec constant (S3). Verify: the rocky gate passes with `STAGE_ROW` on and off.
- [ ] 1.8 Time `STAGE_ROW` on/off on the Samsung (and check the Pixel is unchanged), and pick the default. Verify: per-device numbers and the chosen defaults are in the evidence note.
- [ ] 1.9 Check the performance requirement. Verify: on the Samsung each of conv2–conv5 is faster than 1.1 beyond noise; on both phones no conv stage is slower than 1.1 beyond noise. If it isn't met, stop and report rather than loosening the spec.
- [ ] 1.10 Run the real-weight differential with `BT_WEIGHTS_DIR` on rocky and both phones. Verify: all stages pass rtol 1e-3 / atol 1e-4.
- [ ] 1.11 Remove the old-kernel flag if the new kernel wins on the Samsung and passes everywhere; otherwise document why it stays. Verify: clean `vulkan` and `android` preset builds, with no dead code.
- [ ] 1.12 Run the group gate at the final commit (local, rocky, phones) and add a dated perf-results note with the before/after table and the S4 decision. Verify: all green; the note is linked from `docs/reports-for-human/perf-results/README.md`.

## 2. Dense conv optimization (gpu-kernels/cifar-dense-conv; design C1–C5)

- [ ] 2.1 Record the pre-change cifar-dense per-stage GPU times on the Pixel 7a and the Samsung, with an identical-binary A/B to give the noise. Verify: an 11-stage table per phone with spreads.
- [ ] 2.2 Probe each device's optional capabilities with a debug-logged query added to `test-kiss-vk`: `shaderFloat16`, `storageBuffer16BitAccess`, `cooperativeMatrix` and its shapes, CLUSTERED subgroup support, `maxComputeSharedMemorySize`. Verify: the logged values for all three devices replace the "to verify" marks in design.md Context (this also serves groups 1 and 3).
- [ ] 2.3 Dump the current kernel's ACO disassembly on rocky (`--gtest_filter='*Stage5*'`). Verify: the evidence note records whether the weight loads are vector or scalar.
- [ ] 2.4 C1: move `oct` to `gl_WorkGroupID.y`, grid `(ceil(N·H·W/256), K/OC_TILE, 1)`. Verify: the rocky gate passes; ACO shows scalar weight loads (quoted); timing on both phones is recorded against 2.1 with no regression beyond noise.
- [ ] 2.5 C2/C3: input halo tiling in shared memory (≤ 16 KB default) with `C_TILE`, `OC_TILE` and `PX_TILE` spec constants. Verify: `test-cifar-dense-vk` passes on rocky for at least three tile configurations, including 8 oc × 2 px and 8 oc × 4 px.
- [ ] 2.6 Pick the tile defaults by timing on the Samsung. Verify: the per-device sweep and chosen defaults are in the evidence note.
- [ ] 2.7 Check the FP32 performance requirement. Verify: on the Samsung the conv2–conv5 sum is faster than 2.1 beyond noise, no phone conv stage is slower; the fleet gate is green. If it isn't met, record the gap; the optional tasks 2.8–2.12 don't count toward it.
- [ ] 2.8 C4: optional-feature enablement for `shaderFloat16` and `storageBuffer16BitAccess` (enabled if supported, never required) plus a precision mode (`BT_VK_PRECISION=fp16`). Verify: `test-kiss-vk` logs the enabled features on all devices, and default runs are unchanged.
- [ ] 2.9 C4: the FP16 conv variant (FP32 accumulate), with a per-stage tolerance, and a new VK end-task accuracy test on the real weights mirroring the OMP `CifarDenseAppData.RealWeights_EndTaskAccuracy`. Verify: `test-cifar-dense-vk` with `BT_VK_PRECISION=fp16` passes on supporting devices; FP16 accuracy is within 0.5 points of FP32 (printed); timing is recorded.
- [ ] 2.10 C5: query cooperative-matrix shapes and add the implicit-GEMM conv (FP16 in, FP32 accumulate), selected only when supported. Verify: it passes on rocky with the FP16 tolerance and the 2.9 accuracy bound. Timing is on the Samsung only if 2.2 shows it supports cooperative matrix; if no phone does, C5 is not built (it could not be measured) and that is recorded. If it doesn't beat 2.7/2.9, it stays disabled by default and that's documented.
- 2.11-2.12 (C6, Mali variant): removed 2026-10-18; owned and shipped by `mali-conv-kernels` (archived 2026-10-17).
- [ ] 2.13 Run the group gate at the final commit (local, rocky, phones), in default FP32 and in each opt-in mode on supporting devices. Add a dated perf-results note with before/after tables and per-device variant choices. Verify: all green; the note is linked from the perf-results README.

## 3. FC weight bandwidth (gpu-kernels/fully-connected; design F1–F4)

- [ ] 3.1 Record the pre-change fc1/fc2/fc3 GPU times for both apps on the Pixel 7a and the Samsung. Verify: per-phone tables with spreads (2.1 and 1.1 may be reused).
- [ ] 3.2 F3: replace the `sh_red` and lane-0 serial reduction with `subgroupClusteredAdd(…, 16)`, keeping the shared-memory reduction behind a spec constant for devices without CLUSTERED support (per 2.2). Verify: both apps' `test-*-vk` pass on rocky and both phones; timing on both phones is recorded.
- [ ] 3.3 F1: prototype layouts (a) K-outer/tiles-inner and (b) batch-split so weights stream once per batch. Verify: both pass `test-cifar-sparse-vk` and `test-cifar-dense-vk` on rocky; sparse fc1/fc2 timings on both phones for each layout are in the evidence note.
- [ ] 3.4 F1: pick the layout by timing on both phones and remove the other. Verify: fc1 and fc2 are faster than 3.1 beyond noise on both phones, and each phone's sparse-fc1 / dense-fc1 ratio is recorded (spec "Weights are streamed once per batch"); the fleet gate is green for both apps.
- [ ] 3.5 F2: host-side weight quantization at load time (FP16; INT8 with per-row scales) and the precision mode `BT_VK_FC_WEIGHTS=fp16|int8` with FP32 fallback and a warning on unsupported devices. Verify: a unit test round-trips quantize→dequantize within the stated per-element error; default runs are unchanged.
- [ ] 3.6 F2: FP16-weight and INT8-weight `linear_bt` variants, dequantizing to FP32 in registers. Verify: both apps' `test-*-vk` pass their documented FP16/INT8 tolerances on supporting devices.
- [ ] 3.7 F2: VK end-task accuracy on the real weights for both apps in FP16 and INT8 (reusing the 2.9 test harness for cifar-dense, plus a sparse twin). Verify: each is within 0.5 points of FP32 (dense 90.48%, sparse 90.58%), printed. If INT8 fails, record it and ship FP16 only.
- [ ] 3.8 Check the reduced-precision requirement on the phones. Verify: on each phone that supports it, fc1/fc2 with FP16 and with INT8 weights are faster than FP32 beyond noise, with the ratios recorded (spec "Reduced-precision weights cut weight time").
- [ ] 3.9 F4 (optional): if `new_cifar_{dense,sparse}_linear.comp` are removed, first add a test forcing `in_features % 4 != 0` to prove the remaining path covers it. Verify: that test passes, then clean builds. If they're kept, record why.
- [ ] 3.10 Run the group gate at the final commit (local, rocky, phones) for both apps, default and opt-in modes. Add a dated perf-results note with before/after tables. Verify: all green; the note is linked from the perf-results README.
