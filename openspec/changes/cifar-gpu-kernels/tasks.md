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

- [ ] 1.1 Record the pre-change cifar-sparse per-stage GPU times on the Pixel 7a and the Samsung, with an identical-binary A/B to give the noise. Verify: an 11-stage table per phone with spreads in the evidence note.
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
