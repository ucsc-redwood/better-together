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
- [x] 1.2 Dump the current kernel's ACO disassembly on rocky (`RADV_DEBUG=shaders ./test-cifar-sparse-vk --device minipc --gtest_filter='*Stage6*'`). Verify: the evidence note quotes the CSR `buffer_load`/`s_load` lines, confirming or refuting S1's premise.
  - rocky-ryzen (Radeon 780M, Mesa 25.2.7), ee4c68d, `BT_VK_PIPELINE_CACHE_DIR=<empty dir>
    MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaders ./test-cifar-sparse-vk --device minipc
    --gtest_filter='*Stage6*'` (1 test passed). The k3s1p1 shader is the second of the five compute shaders in the
    dump (it has the `v_readfirstlane_b32` of `subgroupBroadcastFirst` and the `0x38e38e39` divide-by-9). The
    descriptor of binding k is at `s[0:1] + 16·k`: input 0x0 (`null`), `weight_vals` 0x20, `weight_row_ptr` 0x30,
    `weight_col_idx` 0x40.
  - *S1's premise is refuted for the model's shapes.* They all take the fast path (a 32², 16² or 8² map is a
    multiple of the subgroup), and there the CSR metadata is already scalar. Per nonzero, `col_idx` and the weight
    are `s_buffer_load_b32` and only the input is a vector load:

        s_load_b128 s[12:15], s[0:1], 0x40          ; col_idx descriptor
        s_buffer_load_b32 s17, s[12:15], s16        ; col_idx[nz], scalar
        s_mul_hi_i32 s18, s17, 0x38e38e39           ; the /9, /3 decode runs on the SALU (15 ops)
        ...
        s_and_saveexec_b64 s[18:19], vcc            ; tap in the padded border: branch
        s_cbranch_execz BB10
        s_load_b128 s[24:27], s[0:1], 0x20          ; weight_vals descriptor
        buffer_load_b32 v5, v5, s[20:23], 0 offen   ; input, the one vector load
        s_buffer_load_b32 s16, s[24:27], s16        ; weight_vals[nz], scalar
        s_waitcnt vmcnt(0) lgkmcnt(0)
        v_mul_f32_e32 v5, s16, v5

    The row bounds are one vector load per invocation, made uniform after it: `buffer_load_b64 v[0:1], v0,
    s[12:15], 0 offen` (row_ptr, 0x30), `v_readfirstlane_b32 s3, v0`, `v_readfirstlane_b32 s4, v1`. Only the
    fallback path (a subgroup spanning two channels, never taken by the model) loads the CSR per lane:
    `buffer_load_b32 v5, v4, s[16:19], 0 offen` (col_idx, 0x40) and `buffer_load_b32 v6, v4, s[24:27], 0 offen`
    (weight_vals, 0x20).
  - *What the dump shows instead:* the `ok ? v * weight_vals[nz] : 0.0` of every tap compiles to an exec-mask
    branch with `s_waitcnt vmcnt(0) lgkmcnt(0)` before the multiply, so each of the four unrolled taps waits for
    its own input load: one input load in flight per wave. S1 and S2 still remove the decode and the
    `subgroupAllEqual` test, and the new kernel must load unconditionally (clamped address, weight selected to 0)
    so that loads overlap.
- [x] 1.3 Add the VK-only packed `(offset, tap)` array (S2), filled per CSR matrix after `build_from_dense()` and `load_real_weights()` with that layer's H and W. Verify: a `test-cifar-sparse-vk` unit test decodes every entry back to its `col_idx` for all five layers, on synthetic and real weights, and prints each layer's maximum offset within the reserved bit width.
  - `cifar_sparse::pack_k3s1p1_taps` stores `offset·2^4 + tap` with `offset = in_c·H·W + (ky−1)·W + (kx−1)` (signed;
    decode `tap = v & 15`, `offset = v >> 4`); `kTapOffsetBits = 27` is the reserved width (|offset| < 2^27).
    `AppData::conv{1..5}_taps` (placed as weights of stages 1, 3, 5, 6, 7) are filled by `pack_conv_taps()` after
    `build_from_dense()` and after `load_real_weights()`, with maps 32, 16, 8, 8, 8. Where they live and what they
    cost: design.md S2, "As built". `InputPurityOmp` lists them as regions, and
    `Placement.DeclaredStagesMatchBindings` now requires a Vulkan conv to bind exactly one of its `col_idx` and taps
    (the old kernel binds `col_idx`).
  - *Test:* `SparseConvTaps.DecodeToColIdx` (`test-cifar-sparse-vk`, host only) decodes every tap of the five convs
    back to `in_c·9 + tap == col_idx` and prints the largest |offset|. Build box, `build/vulkan`: passes without
    `BT_WEIGHTS_DIR` (synthetic) and with `BT_WEIGHTS_DIR=saved_params/export` (real), with the same output for
    both (each layer is pruned to exactly 25 %, and both have a nonzero at tap 8 of the last input channel):

        conv1: nnz 432, max |offset| 2081 < 2^27
        conv2: nnz 27648, max |offset| 16145 < 2^27
        conv3: nnz 165888, max |offset| 12233 < 2^27
        conv4: nnz 221184, max |offset| 24521 < 2^27
        conv5: nnz 147456, max |offset| 16329 < 2^27

    The real export's rows differ from the synthetic pattern (conv1 row 0 is `[3]` vs `[0, 4, …, 24]`; its rows 4-6
    are empty). Same build: `Placement.DeclaredStagesMatchBindings` passes; local gate `ctest --test-dir build/pc -L
    omp` 11/11 passed.
- [x] 1.4 Write the uniform-channel kernel (S1): `out_c = gl_WorkGroupID.y`, grid `(ceil(N·P·Q/256), K, 1)`, a per-invocation 9-bit tap-valid mask; old kernel kept behind a dispatcher flag. Verify: the rocky gate passes with the new kernel and with the old one.
  - *Kernel:* `new_cifar_sparse_conv2d_k3s1p1_oc.comp`. One invocation per output pixel of channel
    `gl_WorkGroupID.y`; it builds its 9-bit tap-valid mask once, then walks the row 8 nonzeros per iteration (8
    accumulators). A nonzero is a tap load, a weight load, a mask test and an input load from `pixel + offset`, or
    from the pixel itself when the tap is in the border, with the weight selected to 0: no division and no branch.
    Binding 4 is the layer's taps instead of `col_idx`.
  - *Dispatcher:* the five conv stages now share `record_conv` (the five copies differed only in their buffers). It
    takes the new kernel, grid `(ceil(N·H·W/256), K, 1)`, when the subgroup is 32 or more, and the previous
    `new_cifar_sparse_conv2d_k3s1p1` otherwise (so the Pixel's Mali path is selected exactly as before) or with
    `BT_VK_SPARSE_CONV=flat` (the old-kernel flag, read once per dispatcher).
  - *rocky gate*, the tree at c4a8032 plus this change, synced to `~/bt-sparse-gate`: `cmake --preset vulkan`
    (glslc of NDK r29), build exit 0, `ctest -L vulkan` 7/7 passed and `./test-cifar-sparse-vk --device minipc` 18
    passed, both with the new kernel and with `BT_VK_SPARSE_CONV=flat`. `just check-shaders`: OK.
  - *The flag routes:* with the new kernel's output scaled by 0.5 (a local mutation, reverted),
    `test-cifar-sparse-vk --gtest_filter='*Stage6*:*Stage1_*'` fails both stages, and passes both with
    `BT_VK_SPARSE_CONV=flat` (build box RADV iGPU).
  - *Exploratory, not a gate* (Samsung, 1 round, base vs this binary, otherwise the 1.1 method; its
    `test-cifar-sparse-vk` passes 18/18): conv1 4.102 → 3.079 ms (-24.9 %), conv2 34.023 → 19.065 (-44.0 %), conv3
    51.653 → 36.715 (-28.9 %), conv4 68.541 → 48.666 (-29.0 %), conv5 45.762 → 30.386 (-33.6 %).
- [x] 1.5 Add a test with an empty CSR row and a row whose nonzeros only hit the padded border. Verify: it passes against the OMP oracle on rocky (spec "Empty and border-only rows").
  - `SparseConvPerChannel.EmptyAndBorderRows{8x8,OddWidth,TwoWorkgroups}` (`test-cifar-sparse-vk`): the new kernel on
    N 3, C 4, K 9 with 8x8, 7x5 and 13x11 maps (13x11 needs two workgroups per channel, the last ragged) against
    `cifar_sparse::omp::conv2d_omp_batched` on the same CSR, rtol 1e-3 / atol 1e-4. Row 4 is empty, row 6 keeps only
    the kernel's right column (all its taps read the border at the right edge) and row 7 only its top row (the top
    edge); there the output must equal `max(bias, 0)` exactly.
  - rocky gate at 6453f58 plus this test: build exit 0, `ctest -L vulkan` 7/7 passed, `./test-cifar-sparse-vk
    --device minipc` 21 passed, the three new tests among them.
  - *It can fail:* with the right-edge bit of the tap mask forced on (a local mutation of the shader, reverted), all
    three fail on the build box's RADV iGPU.
- [x] 1.6 Re-dump ACO for the new kernel. Verify: the CSR metadata loads are scalar, quoted; or the note explains why not and what changed.
  - rocky-ryzen (780M, Mesa 25.2.7), d3022eb's tree, the 1.2 command. `new_cifar_sparse_conv2d_k3s1p1_oc` is the
    third of the six compute shaders in the dump (descriptors: input 0x0, weights 0x20, row_ptr 0x30, taps 0x40).
    All CSR metadata is scalar. The row bounds are one `s_buffer_load_b64` (no vector load and no
    `v_readfirstlane` any more), and the unrolled loop loads 8 taps and 8 weights with one scalar load each:

        s_load_b128 s[16:19], s[2:3], 0x30
        s_buffer_load_b64 s[16:17], s[16:19], s15           ; row_ptr[out_c], row_ptr[out_c+1]
        ...
        s_buffer_load_b256 s[32:39], s[20:23], s16          ; weight_taps[nz .. nz+7]
        buffer_load_b32 v10, v10, s[24:27], 0 offen         ; input, tap 0
        s_clause 0x6
        buffer_load_b32 v12, v12, s[24:27], 0 offen         ; input, taps 1-7 back to back
        ...
        buffer_load_b32 v26, v26, s[24:27], 0 offen
        s_load_b128 s[24:27], s[2:3], 0x20
        s_buffer_load_b256 s[32:39], s[24:27], s16          ; weight_vals[nz .. nz+7]
        s_waitcnt vmcnt(7) lgkmcnt(0)                       ; first wait: all 8 input loads in flight

    The remainder loop (row length mod 8) loads one tap and one weight per nonzero with `s_buffer_load_b32`. There
    is no exec-mask branch in either loop (the 1.2 kernel had one per tap).
- [x] 1.7 Stage the CSR row in shared memory in ≤ 1024-nonzero chunks behind a `STAGE_ROW` spec constant (S3). Verify: the rocky gate passes with `STAGE_ROW` on and off.
  - `constant_id = 1` `STAGE_ROW` (bool, default off) in `new_cifar_sparse_conv2d_k3s1p1_oc.comp`: the workgroup
    copies the row's taps and weights into shared memory 1024 nonzeros (8 KB) at a time, with a barrier before and
    after each copy, and walks the chunk from there. With it, an invocation past the last pixel stays for the
    barriers and stores nothing. The engine had no way to set a spec constant other than the subgroup and workgroup
    sizes, so `kiss_vk::Algorithm::spec_constant(id, value)` is new (`build()` throws if the shader declares no
    such id). The dispatcher sets it from `BT_VK_SPARSE_STAGE_ROW=1` (task 1.8 picks the default).
  - `SparseConvPerChannel.*` now run every map with `STAGE_ROW` off and on, with C 128 and a row that keeps every
    tap (1152 nonzeros, two chunks; the synthetic model rows have 864, one chunk). With the chunk loop cut after the
    first chunk (a local mutation, reverted) they fail only with `STAGE_ROW` on, at that row.
  - rocky gate (b2a8b0b plus this change): build exit 0, `ctest -L vulkan` 7/7 passed and `./test-cifar-sparse-vk
    --device minipc` 21 passed, with `BT_VK_SPARSE_STAGE_ROW=0` and with `=1`. `just check-shaders`: OK.
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
