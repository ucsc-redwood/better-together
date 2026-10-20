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
- [x] 1.8 Time `STAGE_ROW` on/off on the Samsung (and check the Pixel is unchanged), and pick the default. Verify: per-device numbers and the chosen defaults are in the evidence note.
  - The 1.1 method with three arms: `base` (ee4c68d), and the a324f38 binary with `BT_VK_SPARSE_STAGE_ROW=0` (`off`)
    and `=1` (`on`), alternating, 3 rounds, both phones at once (2026-10-20 00:15-00:25 UTC). All 18 runs exit 0
    with 11 records.
  - *Samsung:* `on` has the lower median on conv1-conv3 (3.397 / 17.456 / 32.535 vs 3.689 / 18.362 / 34.370 ms) but
    every spread overlaps `off`'s, and it is slower beyond noise on conv4 (47.060, 46.960-47.224, vs 45.216,
    44.412-45.823) and conv5 (34.958, 34.757-35.178, vs 33.970, 32.314-34.638). The stage sums are equal (163.6 vs
    163.8 ms). Both are far below `base` on every conv.
  - *Pixel 7a:* its convs run the Mali path, which neither arm changes; all three arms agree within 0.7 % on every
    conv, with overlapping spreads.
  - *Noise of the new kernel on the Samsung:* larger than `base`'s. Conv1 follows the GPU clock the profiler records
    (`base` 900 MHz in every run; the new kernel 600-800 MHz, 2.74-3.75 ms). Conv2-conv5 ran at 1095 MHz in every
    run and still differ by up to 6 % between runs (e.g. conv3 `off` 31.646 / 34.370 / 35.769 ms).
  - *Default:* `STAGE_ROW` off on every device: on the Samsung, the only phone that runs this kernel, it is never
    slower beyond noise than `on`, and `on` loses on conv4 and conv5. rocky-ryzen (the other subgroup-64 GPU) is not
    timed, so it gets the same default. No code change: the spec constant already defaults to off, and
    `BT_VK_SPARSE_STAGE_ROW=1` still turns it on.
  - Pixel 7a, ms:

    | stage | base median (min-max) ms | off median (min-max) ms | on median (min-max) ms | off vs base | on vs base |
    |---|---|---|---|---|---|
    | 1 | 17.037 (17.031-17.075, n=3x59-68) | 17.004 (16.974-17.055, n=3x59-68) | 17.032 (17.013-17.056, n=3x58-59) | -0.2 % | -0.0 % |
    | 2 | 4.464 (4.463-4.465, n=3x225-338) | 4.464 (4.460-4.469, n=3x225-337) | 4.463 (4.462-4.467, n=3x226-337) | +0.0 % | -0.0 % |
    | 3 | 80.344 (80.296-80.431, n=3x12) | 80.304 (80.191-80.356, n=3x12) | 80.466 (80.391-80.474, n=3x12) | -0.0 % | +0.2 % |
    | 4 | 3.366 (3.365-3.366, n=3x638-645) | 3.364 (3.364-3.365, n=3x643-696) | 3.366 (3.366-3.368, n=3x641-696) | -0.1 % | +0.0 % |
    | 5 | 128.336 (126.585-128.830, n=3x7) | 128.655 (128.539-128.667, n=3x7) | 128.491 (128.334-128.828, n=3x7) | +0.2 % | +0.1 % |
    | 6 | 182.617 (182.280-183.577, n=3x5) | 183.599 (181.345-183.729, n=3x5) | 181.319 (180.882-182.927, n=3x5) | +0.5 % | -0.7 % |
    | 7 | 110.528 (110.505-110.697, n=3x9) | 110.585 (110.528-110.629, n=3x9) | 110.482 (110.425-110.633, n=3x9) | +0.1 % | -0.0 % |
    | 8 | 1.182 (1.178-1.184, n=3x2000) | 1.181 (1.179-1.185, n=3x1985-1999) | 1.180 (1.177-1.181, n=3x1912-2000) | -0.1 % | -0.2 % |
    | 9 | 26.875 (26.873-26.877, n=3x36-37) | 26.951 (26.925-27.067, n=3x36-37) | 26.951 (26.942-26.967, n=3x36-37) | +0.3 % | +0.3 % |
    | 10 | 27.170 (27.168-27.182, n=3x36) | 27.193 (27.171-27.228, n=3x36) | 27.160 (27.113-27.162, n=3x36) | +0.1 % | -0.0 % |
    | 11 | 1.369 (1.146-1.372, n=3x1068-1522) | 1.178 (1.178-1.388, n=3x1451-1595) | 1.386 (1.247-1.388, n=3x1379-1493) | -14.0 % | +1.2 % |
    | sum | 583.288 | 584.479 | 582.296 | +0.2 % | -0.2 % |

  - Samsung, ms:

    | stage | base median (min-max) ms | off median (min-max) ms | on median (min-max) ms | off vs base | on vs base |
    |---|---|---|---|---|---|
    | 1 | 4.102 (4.102-4.103, n=3x175-216) | 3.689 (2.741-3.746, n=3x276-319) | 3.397 (3.025-3.929, n=3x246-285) | -10.1 % | -17.2 % |
    | 2 | 2.436 (2.427-2.438, n=3x697-708) | 2.447 (2.426-2.458, n=3x705-907) | 2.447 (2.411-2.469, n=3x694-706) | +0.5 % | +0.4 % |
    | 3 | 34.027 (34.023-34.029, n=3x29) | 18.362 (17.081-19.162, n=3x53-55) | 17.456 (16.978-17.951, n=3x56-64) | -46.0 % | -48.7 % |
    | 4 | 1.838 (1.837-1.841, n=3x1296-1302) | 1.842 (1.821-1.866, n=3x1268-1333) | 1.836 (1.812-1.839, n=3x1295-1328) | +0.2 % | -0.1 % |
    | 5 | 51.654 (51.653-51.660, n=3x19) | 34.370 (31.646-35.769, n=3x31-32) | 32.535 (31.405-34.392, n=3x34-36) | -33.5 % | -37.0 % |
    | 6 | 68.553 (68.540-68.602, n=3x14) | 45.216 (44.412-45.823, n=3x23) | 47.060 (46.960-47.224, n=3x21-23) | -34.0 % | -31.4 % |
    | 7 | 45.770 (45.767-45.772, n=3x21) | 33.970 (32.314-34.638, n=3x29-33) | 34.958 (34.757-35.178, n=3x29) | -25.8 % | -23.6 % |
    | 8 | 2.688 (2.687-2.697, n=3x1204-1242) | 2.672 (2.654-2.711, n=3x1189-1241) | 2.689 (2.672-2.705, n=3x1214-1229) | -0.6 % | +0.0 % |
    | 9 | 10.055 (10.035-10.074, n=3x99-100) | 10.062 (10.031-10.069, n=3x99-100) | 10.057 (10.050-10.066, n=3x90-99) | +0.1 % | +0.0 % |
    | 10 | 10.058 (10.035-10.058, n=3x99) | 10.040 (10.037-10.062, n=3x99) | 10.053 (10.035-10.058, n=3x91-100) | -0.2 % | -0.1 % |
    | 11 | 1.174 (1.157-1.178, n=3x2000) | 1.155 (1.154-1.177, n=3x2000) | 1.158 (1.154-1.179, n=3x2000) | -1.6 % | -1.4 % |
    | sum | 232.356 | 163.825 | 163.646 | -29.5 % | -29.6 % |
- [x] 1.9 Check the performance requirement. Verify: on the Samsung each of conv2–conv5 is faster than 1.1 beyond noise; on both phones no conv stage is slower than 1.1 beyond noise. If it isn't met, stop and report rather than loosening the spec.
  - The 1.1 method: `base` (ee4c68d) against `new`, the a324f38 binary (the code of 902c24b) with no knob set
    (`STAGE_ROW` off), 3 rounds alternating, both phones at once (2026-10-20 00:27-00:33 UTC); then the Pixel alone
    again (00:34-00:40). All runs exit 0 with 11 records.
  - *Samsung, conv2-conv5 faster beyond noise: met.* Stage 3 18.169 (17.378-18.448) vs 34.024 (34.023-34.029) ms,
    -46.6 %; stage 5 33.697 (31.676-33.862) vs 51.652 (51.651-51.660), -34.8 %; stage 6 45.300 (44.487-45.551) vs
    68.531 (68.519-68.556), -33.9 %; stage 7 35.168 (35.144-36.060) vs 45.760 (45.759-45.775), -23.1 %. No spread
    overlaps. Conv1 is faster too (3.200 vs 4.102 ms, -22.0 %).
  - *No conv slower beyond noise: met on the Samsung; on the Pixel 7a, met after a repeat.* The Pixel runs the
    unchanged Mali shader with the same bindings and grid. In the first session its stage 6 (conv4) read 183.575
    (183.451-183.981) vs 182.349 (181.609-183.060) ms, +0.7 % with spreads that do not overlap. The repeat gave
    183.486 (182.150-183.543) vs 183.377 (183.269-183.567), +0.1 %, overlapping. Over all sessions the two
    interleave: the ee4c68d binary's stage-6 medians are 182.517 / 183.086 (1.1, two arms), 182.617 (1.8), 182.349
    (1.9) and 183.377 (repeat); the a324f38 binary's are 183.599 / 181.319 (1.8, its `off` and `on` arms, the same
    Mali path), 183.575 (1.9) and 183.486 (repeat). So the first session's +0.7 % is session-to-session noise, not
    this change. Every other Pixel conv is within 0.3 % with overlapping spreads in both sessions.
  - Samsung, ms:

    | stage | base median (min-max) ms | new median (min-max) ms | new vs base |
    |---|---|---|---|
    | 1 | 4.102 (3.696-4.102, n=3x243-270) | 3.200 (2.813-3.693, n=3x264-300) | -22.0 % |
    | 2 | 2.398 (2.388-2.431, n=3x694-706) | 2.418 (2.402-2.447, n=3x697-718) | +0.8 % |
    | 3 | 34.024 (34.023-34.029, n=3x29) | 18.169 (17.378-18.448, n=3x57-60) | -46.6 % |
    | 4 | 1.827 (1.826-1.843, n=3x1293-1303) | 1.821 (1.815-1.832, n=3x1300-1317) | -0.4 % |
    | 5 | 51.652 (51.651-51.660, n=3x19) | 33.697 (31.676-33.862, n=3x33-36) | -34.8 % |
    | 6 | 68.531 (68.519-68.556, n=3x14) | 45.300 (44.487-45.551, n=3x22-24) | -33.9 % |
    | 7 | 45.760 (45.759-45.775, n=3x21) | 35.168 (35.144-36.060, n=3x29) | -23.1 % |
    | 8 | 2.693 (2.687-2.701, n=3x1223-1231) | 2.685 (2.672-2.687, n=3x1195-1210) | -0.3 % |
    | 9 | 10.049 (10.038-10.090, n=3x98-100) | 10.043 (10.041-10.063, n=3x99-100) | -0.1 % |
    | 10 | 10.040 (10.038-10.047, n=3x99) | 10.049 (10.036-10.050, n=3x97-99) | +0.1 % |
    | 11 | 1.176 (1.174-1.178, n=3x2000) | 1.158 (1.156-1.178, n=3x2000) | -1.6 % |
    | sum | 232.253 | 163.707 | -29.5 % |

  - Pixel 7a, first session, ms:

    | stage | base median (min-max) ms | new median (min-max) ms | new vs base |
    |---|---|---|---|
    | 1 | 17.040 (16.980-17.123, n=3x59-68) | 17.012 (16.998-17.047, n=3x59-68) | -0.2 % |
    | 2 | 4.467 (4.465-4.469, n=3x225-226) | 4.464 (4.463-4.469, n=3x226-447) | -0.1 % |
    | 3 | 80.337 (80.330-80.382, n=3x12) | 80.287 (80.279-80.394, n=3x12) | -0.1 % |
    | 4 | 3.366 (3.365-3.370, n=3x643-698) | 3.365 (3.365-3.366, n=3x644-698) | -0.0 % |
    | 5 | 128.636 (126.923-128.678, n=3x7) | 128.712 (128.657-128.805, n=3x7) | +0.1 % |
    | 6 | 182.349 (181.609-183.060, n=3x5) | 183.575 (183.451-183.981, n=3x5) | +0.7 % |
    | 7 | 110.475 (110.308-110.478, n=3x9) | 110.139 (108.606-110.471, n=3x9) | -0.3 % |
    | 8 | 1.180 (1.179-1.183, n=3x1969-2000) | 1.180 (1.179-1.183, n=3x1927-1993) | +0.1 % |
    | 9 | 26.887 (26.877-26.900, n=3x32-34) | 26.939 (26.909-26.981, n=3x36-37) | +0.2 % |
    | 10 | 27.177 (27.163-27.182, n=3x36) | 27.153 (27.134-27.156, n=3x36) | -0.1 % |
    | 11 | 1.378 (1.156-1.383, n=3x1449-1604) | 1.378 (1.179-1.386, n=3x1400-1588) | -0.0 % |
    | sum | 583.292 | 584.204 | +0.2 % |

  - Pixel 7a, repeat, ms:

    | stage | base median (min-max) ms | new median (min-max) ms | new vs base |
    |---|---|---|---|
    | 1 | 17.057 (16.987-17.297, n=3x58-68) | 17.013 (16.898-17.031, n=3x59-68) | -0.3 % |
    | 2 | 4.465 (4.463-4.469, n=3x226-338) | 4.466 (4.437-4.466, n=3x225-334) | +0.0 % |
    | 3 | 80.296 (80.266-80.386, n=3x12) | 80.244 (80.006-80.297, n=3x12) | -0.1 % |
    | 4 | 3.368 (3.364-3.369, n=3x643-700) | 3.365 (3.339-3.366, n=3x586-699) | -0.1 % |
    | 5 | 128.545 (128.503-128.582, n=3x7) | 128.618 (126.641-128.709, n=3x7) | +0.1 % |
    | 6 | 183.377 (183.269-183.567, n=3x5) | 183.486 (182.150-183.543, n=3x5) | +0.1 % |
    | 7 | 110.541 (108.598-110.723, n=3x9) | 110.218 (108.633-110.866, n=3x9) | -0.3 % |
    | 8 | 1.179 (1.174-1.183, n=3x1854-2000) | 1.182 (1.157-1.185, n=3x1988-2000) | +0.2 % |
    | 9 | 26.891 (26.856-26.907, n=3x34-35) | 26.970 (26.957-27.089, n=3x35-37) | +0.3 % |
    | 10 | 27.223 (27.190-27.320, n=3x36) | 27.188 (27.175-27.342, n=3x36) | -0.1 % |
    | 11 | 1.166 (1.165-1.375, n=3x1492-1680) | 1.381 (0.980-1.387, n=3x1348-1553) | +18.5 % |
    | sum | 584.106 | 584.131 | +0.0 % |
- [x] 1.10 Run the real-weight differential with `BT_WEIGHTS_DIR` on rocky and both phones. Verify: all stages pass rtol 1e-3 / atol 1e-4.
  - `test-cifar-sparse-vk` (Vulkan tolerance rtol 1e-3 / atol 1e-4 per stage), real weights from
    `saved_params/export` (`sparse/`, `test_batch.npy`) copied next to the binary:
    - rocky-ryzen, 902c24b's tree built in `~/bt-sparse-gate`, `BT_WEIGHTS_DIR=~/bt-sparse-gate/weights
      ./test-cifar-sparse-vk --device minipc`: 21 passed (`SparseConvTaps` printed "taps of the real weights"); also
      21 passed with `BT_VK_SPARSE_STAGE_ROW=1`.
    - Samsung and Pixel 7a, the a324f38 binary (same code), `BT_WEIGHTS_DIR=/data/local/tmp/bt-sparse/weights`: 21
      passed on each. On the Pixel the convs take the Mali path; the per-channel kernel still runs there in
      `SparseConvPerChannel.*`.
  - The real export reaches cases the synthetic weights do not: rows of up to 1926 nonzeros (conv4, two
    `STAGE_ROW` chunks) and empty rows (27 in conv1, 1 in conv2, 13 in conv5); per layer the row lengths are conv1
    0-25, conv2 0-277, conv3 23-791, conv4 20-1926, conv5 0-854.
- [x] 1.11 Remove the old-kernel flag if the new kernel wins on the Samsung and passes everywhere; otherwise document why it stays. Verify: clean `vulkan` and `android` preset builds, with no dead code.
  - The new kernel wins on the Samsung (1.9) and passes on rocky and both phones (1.4, 1.10), so
    `BT_VK_SPARSE_CONV=flat` is gone: cifar-sparse runs `new_cifar_sparse_conv2d_k3s1p1_oc` whenever the subgroup is
    32 or more. That left the subgroup >= 32 code of `new_cifar_sparse_conv2d_k3s1p1.comp` (`row_sum`, `tap_fast`,
    `tap_generic`) unreachable, so the shader is now the Mali path alone: `main_mali` became `main`, and the
    `SUBGROUP_SIZE` spec constant and the subgroup extensions went with the dead code. The dispatcher picks the shader
    by `engine.subgroup_size()` and no longer builds it `subgroup_size_dependent()`; the contract's
    `sparse_conv_k3s1p1_invocations(…, subgroup_size)` became `sparse_conv_k3s1p1_mali_invocations(n, k, h, w)`.
    `SparseConvK3S1P1.{FallbackWhenSubgroupSpansChannels,FastPathWhenMapIsSubgroupMultiple}` tested the removed path
    by name; as `Map6x6` / `Map8x8` (with `RaggedOddWidthMap`) they now test the Mali shader on every GPU. Docs:
    `03-unit-testing.md` (conv paths, contract, the `BT_VK_SPARSE_STAGE_ROW` knob), `01-hardware.md` (which shaders
    are subgroup-specialized).
  - *The Mali path is unchanged:* the old shader specialized at subgroup 16 (`spirv-opt
    --set-spec-const-default-value 0:16 --freeze-spec-const --fold-spec-const-op-composite -O --strip-debug`) and
    the new Mali-only shader (`-O --strip-debug`) are instruction-identical once ids and declaration order are
    canonicalized (267 lines each; a scratch script names every global by its definition and decorations, bindings
    included, and every local id by first use; with `PX = 2` it reports 314 differing lines). The dispatcher gives
    it the same buffers and the same `grid.x` as before at subgroup 16; the only difference is that the pipeline
    no longer requests `requiredSubgroupSize` 16 on a GPU whose only subgroup size is 16.
  - *Clean builds* on the build box: `cmake --build --preset vulkan --clean-first` and `cmake --build --preset
    android --clean-first` exit 0; their only warnings are 4 pre-existing ones in `apps/tree/omp/func_sort.hpp`.
    `just check-shaders`: OK. `grep` finds no `BT_VK_SPARSE_CONV`, `conv_per_channel` or
    `sparse_conv_k3s1p1_invocations` left in the code, and no `main_mali` in the sparse shader (cifar-dense's shader
    keeps its own).
  - rocky gate (4fab24d plus this change): build exit 0, `ctest -L vulkan` 7/7 passed, `./test-cifar-sparse-vk
    --device minipc` 21 passed.
- [x] 1.12 Run the group gate at the final commit (local, rocky, phones) and add a dated perf-results note with the before/after table and the S4 decision. Verify: all green; the note is linked from `docs/reports-for-human/perf-results/README.md`.
  - Gate at a986a39, the last code commit (the commit after it adds this note and changes only docs):
    - *local:* `cmake --build --preset pc && ctest --test-dir build/pc -L omp`: 11/11 passed.
    - *rocky:* the tree synced to `~/bt-sparse-gate`, `cmake --preset vulkan` (glslc of NDK r29), build exit 0,
      `ctest -L vulkan` 7/7 passed, `./test-cifar-sparse-vk --device minipc` 21 passed.
    - *phones:* `cmake --build --preset android --target test-cifar-sparse-vk`, then
      `BT_ANDROID_DEST=/data/local/tmp/bt-sparse/gate scripts/run-on-android.sh <serial> test-cifar-sparse-vk`:
      21 passed and exit 0 on 3A021JEHN02756 and on R5CY21Y3VEV. The same binary with `BT_WEIGHTS_DIR` (real
      weights): 21 passed on each phone.
  - *Final A/B* (the 1.1 method, ee4c68d vs a986a39, both phones at once, 2026-10-20 01:00-01:06 UTC, all 12 runs
    exit 0 with 11 records): Samsung conv2-conv5 -47.6 / -34.7 / -36.9 / -22.7 %, conv1 -20.6 %, no spread
    overlapping, stage sum 232.3 → 161.7 ms; Pixel 7a every conv within 0.9 % with overlapping spreads.
  - *Note:* `docs/reports-for-human/perf-results/2026-10-20-cifar-sparse-conv.md` (before/after tables, ACO
    findings, the S3 result, the S4 decision: unstructured CSR stays), linked from the perf-results README.
  - Samsung, ms:

    | stage | base median (min-max) ms | final median (min-max) ms | final vs base |
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

  - Pixel 7a, ms:

    | stage | base median (min-max) ms | final median (min-max) ms | final vs base |
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

## 2. Dense conv optimization (gpu-kernels/cifar-dense-conv; design C1–C5)

- [x] 2.1 Record the pre-change cifar-dense per-stage GPU times on the Pixel 7a and the Samsung, with an identical-binary A/B to give the noise. Verify: an 11-stage table per phone with spreads.
  - *Method* (used for every timing of this group): the 1.1 method with cifar-dense. `bm-prof-cifar-dense-vk` built
    with the `android` preset (NDK r30) at d563dc5, pushed twice (arms A and B, the same binary) to
    `/data/local/tmp/bt-dense/{base,base2}` on each phone, real weights (`BT_WEIGHTS_DIR`, `saved_params/export`'s
    `dense/` and test batch pushed beside them), `BT_PROF_SCENARIO=isolated BT_PROF_BUDGET_S=1.0
    BT_PROF_ABANDON_S=5 BT_PROF_MAX_CELL_S=3 --benchmark_filter='vulkan/'`. Arms alternate, 3 rounds, 20 s idle
    before every run, and each run first checks that no other `bm-`/`test-` process is on the phone; both phones ran
    at once (2026-10-20 05:14-05:19 UTC). Per stage: the median of the 3 runs' p50 (GPU timestamps), with the min-max
    of the three p50s, `n = runs x samples`. The last row is the spec's measure: each run's conv2-conv5 sum (stages
    3, 5, 6, 7), median and min-max over the 3 runs. All 12 runs exit 0 with 11 records.
  - *Noise, Samsung:* every conv of the two identical arms is within 2.7 % and the spreads overlap; the conv2-conv5
    sum is 11.408 (11.124-12.216) vs 11.439 (11.408-11.459) ms, +0.3 %. Single runs jump between two levels, as in
    the mali-conv-kernels baseline: conv2 (stage 3) 3.09 or 3.40 ms, conv5 (stage 7) 2.16 or 2.92 ms.
  - *Noise, Pixel 7a* (Mali path): conv1, conv4 and conv5 are within 0.4 %, but conv3 (stage 5) reads 16.451
    (16.436-16.506) vs 15.212 (15.144-15.721) ms between identical arms, -7.5 % with spreads that do not overlap, and
    conv2 (stage 3) spans 9.98-14.54 ms within one binary. A Pixel conv that moves by less than this between arms is
    noise.
  - Pixel 7a (Mali path), ms:

    | stage | A median (min-max) ms | B median (min-max) ms | B vs A |
    |---|---|---|---|
    | 1 | 2.568 (2.558-2.572, n=3x391-392) | 2.568 (2.566-2.572, n=3x392-488) | -0.0 % |
    | 2 | 0.491 (0.491-0.491, n=3x2000) | 0.494 (0.493-0.495, n=3x2000) | +0.5 % |
    | 3 | 12.359 (9.980-13.405, n=3x74-100) | 12.361 (12.342-14.541, n=3x68-100) | +0.0 % |
    | 4 | 0.383 (0.382-0.383, n=3x2000) | 0.384 (0.382-0.384, n=3x2000) | +0.2 % |
    | 5 | 16.451 (16.436-16.506, n=3x49-60) | 15.212 (15.144-15.721, n=3x48-73) | -7.5 % |
    | 6 | 20.352 (20.322-20.355, n=3x49) | 20.312 (20.298-20.340, n=3x49) | -0.2 % |
    | 7 | 14.571 (14.533-15.792, n=3x68-76) | 14.512 (14.506-14.561, n=3x68-69) | -0.4 % |
    | 8 | 0.175 (0.175-0.175, n=3x2000) | 0.175 (0.175-0.176, n=3x2000) | +0.4 % |
    | 9 | 6.201 (4.895-6.702, n=3x136-179) | 6.204 (5.435-6.837, n=3x131-220) | +0.0 % |
    | 10 | 6.672 (5.087-6.783, n=3x133-158) | 6.720 (6.693-6.910, n=3x138-189) | +0.7 % |
    | 11 | 0.980 (0.872-0.985, n=3x1360-1498) | 0.984 (0.979-0.984, n=3x1335-1365) | +0.3 % |
    | sum | 81.203 | 79.925 | -1.6 % |
    | conv2-conv5 (3+5+6+7), per-run sum | 64.764 (61.286-65.012, 3 runs) | 62.413 (62.352-65.087, 3 runs) | -3.6 % |

  - Samsung (subgroup-64 path), ms:

    | stage | A median (min-max) ms | B median (min-max) ms | B vs A |
    |---|---|---|---|
    | 1 | 0.913 (0.912-0.917, n=3x1060-1122) | 0.914 (0.908-0.915, n=3x1084-1129) | +0.1 % |
    | 2 | 1.300 (1.297-1.301, n=3x884-1733) | 1.309 (1.308-1.310, n=3x977-2000) | +0.7 % |
    | 3 | 3.397 (3.090-3.406, n=3x514-609) | 3.404 (3.400-3.405, n=3x472-566) | +0.2 % |
    | 4 | 0.975 (0.973-0.977, n=3x1235-1268) | 0.976 (0.965-0.976, n=3x1212-1241) | +0.1 % |
    | 5 | 2.452 (2.443-2.508, n=3x408-457) | 2.453 (2.445-2.507, n=3x410-455) | +0.0 % |
    | 6 | 3.393 (3.375-3.396, n=3x326-328) | 3.396 (3.370-3.397, n=3x325-330) | +0.1 % |
    | 7 | 2.216 (2.157-2.915, n=3x360-507) | 2.156 (2.154-2.219, n=3x503-517) | -2.7 % |
    | 8 | 0.324 (0.323-0.327, n=3x2000) | 0.326 (0.323-0.332, n=3x2000) | +0.6 % |
    | 9 | 3.014 (2.127-3.075, n=3x334-521) | 3.025 (2.166-3.069, n=3x450-512) | +0.4 % |
    | 10 | 2.770 (2.015-2.831, n=3x355-536) | 2.717 (1.977-2.828, n=3x365-544) | -1.9 % |
    | 11 | 0.645 (0.632-0.651, n=3x1393-2000) | 0.649 (0.622-0.650, n=3x1413-2000) | +0.6 % |
    | sum | 21.398 | 21.322 | -0.4 % |
    | conv2-conv5 (3+5+6+7), per-run sum | 11.408 (11.124-12.216, 3 runs) | 11.439 (11.408-11.459, 3 runs) | +0.3 % |
- [x] 2.2 Probe each device's optional capabilities with a debug-logged query added to `test-kiss-vk`: `shaderFloat16`, `storageBuffer16BitAccess`, `cooperativeMatrix` and its shapes, CLUSTERED subgroup support, `maxComputeSharedMemorySize`. Verify: the logged values for all three devices replace the "to verify" marks in design.md Context (this also serves groups 1 and 3).
  - `VulkanDeviceFeatures.OptionalCapabilitiesAreLogged` (`test-kiss-vk`) queries the five and logs them at debug
    level (it fails only if `maxComputeSharedMemorySize` is below the Vulkan minimum of 16 KB). `SPDLOG_LEVEL=debug
    ./test-kiss-vk --gtest_filter='*OptionalCapabilities*'`, 1 test passed on each:

        AMD Radeon 780M Graphics (RADV PHOENIX): shaderFloat16 true, storageBuffer16BitAccess true,
          VK_KHR_cooperative_matrix true (cooperativeMatrix true; shapes: M16N16K16 A Float16 B Float16 C Float16
          R Float16 Subgroup; M16N16K16 A Float16 B Float16 C Float32 R Float32 Subgroup; 12 M16N16K16 shapes
          with 8-bit integer A/B and 32-bit integer C), subgroup CLUSTERED true, maxComputeSharedMemorySize 65536
        Mali-G710: shaderFloat16 true, storageBuffer16BitAccess true, VK_KHR_cooperative_matrix false
          (cooperativeMatrix false; shapes: none), subgroup CLUSTERED true, maxComputeSharedMemorySize 32768
        Samsung Xclipse 940: shaderFloat16 true, storageBuffer16BitAccess true, VK_KHR_cooperative_matrix false
          (cooperativeMatrix false; shapes: none), subgroup CLUSTERED true, maxComputeSharedMemorySize 32768

    (rocky-ryzen in `~/bt-dense-gate`; the phones from `build/android`. The build box's RADV iGPU, RDNA2, reports
    the same as the 780M without cooperative matrix.) design.md Context now has the table instead of the "to
    verify" marks, as do the C5 shapes and the F3 fallback. No phone has cooperative matrix, so per 2.10 C5 can
    only be built for rocky, where nothing is timed.
- [x] 2.3 Dump the current kernel's ACO disassembly on rocky (`--gtest_filter='*Stage5*'`). Verify: the evidence note records whether the weight loads are vector or scalar.
  - rocky-ryzen (780M, Mesa 25.2.7), the code of 7bcecab (the shader is d563dc5's) built in
    `~/bt-dense-gate`, `BT_VK_PIPELINE_CACHE_DIR=<empty dir> MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaders
    ./test-cifar-dense-vk --device minipc --gtest_filter='*Stage5*'` (1 test passed). `new_cifar_dense_conv2d_k3s1p1`
    at subgroup 64 is the second of the five compute shaders in the dump (`uses_wide_subgroup_intrinsics`, the
    `v_readfirstlane_b32` of `subgroupBroadcastFirst`). One `s_load_b256` reads the input (binding 0, `s[24:27]`) and
    weight (binding 1, `s[28:31]`) descriptors.
  - *The weight loads are scalar on the path the model takes.* Its maps (32², 16², 8²) are subgroup multiples, so
    `subgroupAllEqual(oc)` holds and the weight base is `subgroupBroadcastFirst(oc)`. Per input channel the loop
    loads the 9 taps as vector loads and the 36 weights (4 channels x 9) as 8 scalar loads:

        s_clause 0x8
        buffer_load_b32 v7, v7, s[24:27], 0 offen              ; input, 9 taps
        ...
        buffer_load_b32 v6, v6, s[24:27], 0 offen
        s_clause 0x7
        s_buffer_load_b256 s[40:47], s[28:31], s24             ; weights, row oc, taps 0-7
        s_buffer_load_b32 s25, s[28:31], s25                   ; row oc, tap 8
        s_buffer_load_b256 s[48:55], s[28:31], s27             ; rows oc+1 .. oc+3 likewise
        ...
        s_waitcnt vmcnt(8)
        ...
        v_fmac_f32_e32 v16, s48, v7                            ; 36 FMAs with a scalar weight operand

    Only the other branch (a subgroup spanning two channel blocks, which the model's shapes never take) loads the
    weights per lane: `buffer_load_b128 v[20:23], v17, s[28:31], 0 offen` (and two more per weight row). So C1's
    premise (vector weight loads) is refuted for the model, as S1's was in 1.2; C1 still makes the uniformity
    structural (no `subgroupAllEqual` test, no second loop) and changes which invocations share a workgroup.
  - *What limits it instead:* each loop iteration is one input channel, and it waits for its own 9 input loads
    (`s_waitcnt vmcnt(8)` .. `vmcnt(0)`) and weight loads (`lgkmcnt(0)`) before its 36 FMAs; nothing of the next
    channel is in flight. 9 vector loads feed 36 FMAs (4 FMAs per load). C2/C3 address this: fewer loads per FMA
    (register tile) and inputs from shared memory.
- [x] 2.4 C1: move `oct` to `gl_WorkGroupID.y`, grid `(ceil(N·H·W/256), K/OC_TILE, 1)`. Verify: the rocky gate passes; ACO shows scalar weight loads (quoted); timing on both phones is recorded against 2.1 with no regression beyond noise.
  - *Kernel:* `new_cifar_dense_conv2d_k3s1p1_oc.comp`, the subgroup >= 32 loop of `new_cifar_dense_conv2d_k3s1p1`
    with the channel block taken from the workgroup id: one invocation per output pixel, 4 channels, the 9 taps
    loaded unconditionally on clamped indices; no `subgroupAllEqual` test and no per-lane loop. Its own file, so
    the Mali shader and its selection do not change. The dispatcher's five conv recorders became one
    `record_conv` (they differed only in their buffers); it takes the new kernel when the subgroup is 32 or more,
    except for conv1, and the previous shader otherwise or with `BT_VK_DENSE_CONV=flat` (the old-kernel flag,
    read once per dispatcher). Grid from `kiss_vk::dense_conv_k3s1p1_oc_grid`. `DenseConvPerBlock.{Map6x6,
    Map8x8,RaggedOddWidthMap}` check it against the double-precision reference (N 8, so the pixels span two
    workgroups).
  - *As built, the channel block is `gl_WorkGroupID.x`, not `.y`* (grid `(K/4, ceil(N·H·W/256), 1)`). As written
    (`.y`), consecutive workgroups take the same channel block of different images, and the Samsung slowed down
    beyond noise on conv1 and conv2: the first A/B (`base` d563dc5 vs the `.y` kernel, both phones, 2026-10-20
    05:47-05:52 UTC) read conv1 0.996 (0.976-0.999) vs 0.913 (0.907-0.917) ms, +9.1 %, conv2 4.121
    (4.031-4.165) vs 3.406 (3.397-3.406), +21.0 %, while conv4 (-16.4 %) and conv5 (-10.6 %) gained. A three-arm
    run on the Samsung (05:54-06:02 UTC, `base`, `.y`, `.x`, same method) confirmed it: with `.x` consecutive
    workgroups share the input pixels and conv2 is -3.0 %. Conv1 was still +1.0 % with `.x` (0.921 vs 0.912 ms,
    spreads apart), so conv1 keeps the previous kernel, as design C's non-goals allow. design.md C1, "As built".
    The `.y` kernel was never committed. Samsung, ms:

    | stage | base median (min-max) ms | y median (min-max) ms | x median (min-max) ms | y vs base | x vs base |
    |---|---|---|---|---|---|
    | 1 | 0.912 (0.911-0.914, n=3x1075-1120) | 0.969 (0.968-0.998, n=3x1016-1071) | 0.921 (0.921-0.923, n=3x1091-1120) | +6.2 % | +1.0 % |
    | 2 | 1.302 (1.299-1.309, n=3x1262-1911) | 1.287 (1.281-1.289, n=3x904-1767) | 1.301 (1.289-1.303, n=3x875-1658) | -1.2 % | -0.0 % |
    | 3 | 3.405 (3.405-3.407, n=3x514-661) | 4.066 (3.950-4.118, n=3x449-542) | 3.303 (3.301-3.304, n=3x527-678) | +19.4 % | -3.0 % |
    | 4 | 0.971 (0.962-0.978, n=3x1155-1252) | 0.976 (0.972-0.977, n=3x1234-1285) | 0.974 (0.969-0.976, n=3x1066-1272) | +0.5 % | +0.3 % |
    | 5 | 2.453 (2.451-2.725, n=3x369-452) | 2.906 (1.979-2.915, n=3x344-569) | 2.197 (2.195-2.765, n=3x361-508) | +18.5 % | -10.4 % |
    | 6 | 3.395 (3.393-3.396, n=3x328-335) | 2.840 (2.835-2.846, n=3x397-400) | 3.223 (3.223-3.227, n=3x344-348) | -16.3 % | -5.1 % |
    | 7 | 2.217 (2.216-2.881, n=3x391-505) | 2.701 (1.867-2.704, n=3x436-608) | 2.012 (2.010-2.021, n=3x551-556) | +21.8 % | -9.2 % |
    | 8 | 0.327 (0.326-0.330, n=3x2000) | 0.331 (0.327-0.332, n=3x2000) | 0.330 (0.325-0.330, n=3x2000) | +1.0 % | +0.7 % |
    | 9 | 2.908 (2.891-3.003, n=3x344-347) | 2.903 (2.048-2.906, n=3x458-530) | 2.083 (2.048-2.849, n=3x483-528) | -0.2 % | -28.4 % |
    | 10 | 1.977 (1.976-2.712, n=3x368-538) | 1.975 (1.975-2.809, n=3x480-536) | 2.717 (1.978-2.762, n=3x476-555) | -0.1 % | +37.4 % |
    | 11 | 0.638 (0.624-0.640, n=3x1443-2000) | 0.652 (0.637-0.659, n=3x1391-2000) | 0.640 (0.633-0.649, n=3x1422-2000) | +2.2 % | +0.4 % |
    | sum | 20.504 | 21.605 | 19.701 | +5.4 % | -3.9 % |
    | conv2-conv5 (3+5+6+7), per-run sum | 11.470 (11.467-12.408, 3 runs) | 12.519 (10.630-12.578, 3 runs) | 10.742 (10.735-11.303, 3 runs) | +9.1 % | -6.3 % |

  - *rocky gate* (the tree of this commit in `~/bt-dense-gate`, glslc of NDK r29): build exit 0, `ctest -L vulkan`
    7/7 passed, `./test-cifar-dense-vk --device minipc` 19 passed with the new kernel and with
    `BT_VK_DENSE_CONV=flat`. `just check-shaders`: OK. On the build box's RADV iGPU, with the new kernel's output
    halved (a local mutation, reverted), `*Stage5*:*Stage1_*` fail stage 5 only by default (conv1 is not on it)
    and pass with `BT_VK_DENSE_CONV=flat`. Both phones: `test-cifar-dense-vk` 19 passed, with synthetic and with
    real weights.
  - *ACO* (the 2.3 command on this tree; the new kernel is the third of six compute shaders): the weights are
    scalar, 8 scalar loads per channel as before, now without a subgroup test or a second loop:

        s_clause 0x8
        buffer_load_b32 v7, v7, s[24:27], 0 offen              ; input, 9 taps
        ...
        s_clause 0x7
        s_buffer_load_b256 s[40:47], s[28:31], s24             ; weights (binding 1, s[28:31])
        s_buffer_load_b32 s25, s[28:31], s25
        ...
        s_buffer_load_b256 s[64:71], s[28:31], s39
        s_buffer_load_b32 s15, s[28:31], s15
        s_waitcnt vmcnt(8)
        ...
        v_fmac_f32_e32 v16, s40, v7

  - *Phone gate* (the 2.1 method, `base` d563dc5 vs this commit's binary, both phones at once, 2026-10-20
    06:05-06:11 UTC; all 12 runs exit 0 with 11 records): met. Samsung: no conv slower; conv2 -2.9 % with spreads
    apart, conv3/conv4/conv5 -6.0 / -5.1 / -9.2 % with overlapping spreads, the conv2-conv5 sum 11.490
    (10.728-12.378) vs 11.815 (11.454-12.988) ms. Pixel 7a (the Mali path, the same shader and grid): every conv
    within its 2.1 noise; stage 3 and 7 move by -7.9 / -6.5 % between their two levels (9.98-15.76 and
    14.53-15.83 ms in 2.1).
  - Pixel 7a, ms:

    | stage | base median (min-max) ms | C1 median (min-max) ms | C1 vs base |
    |---|---|---|---|
    | 1 | 2.570 (2.552-2.571, n=3x391-586) | 2.571 (2.569-2.572, n=3x391-393) | +0.0 % |
    | 2 | 0.496 (0.492-0.496, n=3x2000) | 0.491 (0.491-0.495, n=3x2000) | -1.0 % |
    | 3 | 14.551 (13.403-15.761, n=3x55-63) | 13.395 (12.355-15.763, n=3x63-90) | -7.9 % |
    | 4 | 0.382 (0.381-0.383, n=3x2000) | 0.383 (0.379-0.383, n=3x2000) | +0.2 % |
    | 5 | 16.421 (15.171-16.478, n=3x63-73) | 16.446 (15.189-16.468, n=3x51-73) | +0.1 % |
    | 6 | 20.308 (20.289-20.348, n=3x49) | 20.300 (20.277-20.351, n=3x49) | -0.0 % |
    | 7 | 15.757 (15.746-15.770, n=3x68-69) | 14.733 (14.573-15.831, n=3x68) | -6.5 % |
    | 8 | 0.173 (0.172-0.174, n=3x2000) | 0.174 (0.172-0.175, n=3x2000) | +0.1 % |
    | 9 | 6.790 (6.324-7.869, n=3x127-128) | 6.796 (6.754-6.798, n=3x111-121) | +0.1 % |
    | 10 | 6.766 (6.678-7.844, n=3x118-156) | 6.654 (5.567-6.682, n=3x127-194) | -1.7 % |
    | 11 | 0.983 (0.882-0.988, n=3x1275-1350) | 0.980 (0.978-1.162, n=3x1260-1360) | -0.3 % |
    | sum | 85.197 | 82.922 | -2.7 % |
    | conv2-conv5 (3+5+6+7), per-run sum | 67.094 (64.668-68.242, 3 runs) | 64.691 (63.908-67.083, 3 runs) | -3.6 % |

  - Samsung, ms:

    | stage | base median (min-max) ms | C1 median (min-max) ms | C1 vs base |
    |---|---|---|---|
    | 1 | 0.913 (0.909-0.914, n=3x1033-1095) | 0.908 (0.905-0.909, n=3x1081-1132) | -0.6 % |
    | 2 | 1.292 (1.290-1.300, n=3x950-1083) | 1.304 (1.301-1.309, n=3x859-1430) | +0.9 % |
    | 3 | 3.401 (3.399-3.403, n=3x472-565) | 3.302 (3.300-3.304, n=3x525-677) | -2.9 % |
    | 4 | 0.978 (0.977-0.981, n=3x1244-1281) | 0.980 (0.973-0.981, n=3x1204-2000) | +0.1 % |
    | 5 | 3.137 (2.507-3.387, n=3x320-397) | 2.950 (2.195-2.952, n=3x339-511) | -6.0 % |
    | 6 | 3.393 (3.059-3.394, n=3x326-330) | 3.220 (3.192-3.226, n=3x344-352) | -5.1 % |
    | 7 | 2.216 (2.153-2.807, n=3x359-518) | 2.012 (2.011-2.933, n=3x393-554) | -9.2 % |
    | 8 | 0.327 (0.325-0.330, n=3x2000) | 0.326 (0.324-0.326, n=3x2000) | -0.4 % |
    | 9 | 2.988 (2.957-3.033, n=3x332-465) | 3.017 (2.086-3.022, n=3x456-525) | +1.0 % |
    | 10 | 2.757 (2.011-2.869, n=3x358-531) | 1.977 (1.974-2.780, n=3x361-543) | -28.3 % |
    | 11 | 0.629 (0.619-0.644, n=3x1409-2000) | 0.643 (0.631-0.644, n=3x1406-2000) | +2.3 % |
    | sum | 22.032 | 20.637 | -6.3 % |
    | conv2-conv5 (3+5+6+7), per-run sum | 11.815 (11.454-12.988, 3 runs) | 11.490 (10.728-12.378, 3 runs) | -2.8 % |
- [x] 2.5 C2/C3: input halo tiling in shared memory (≤ 16 KB default) with `C_TILE`, `OC_TILE` and `PX_TILE` spec constants. Verify: `test-cifar-dense-vk` passes on rocky for at least three tile configurations, including 8 oc × 2 px and 8 oc × 4 px.
  - *Kernel:* `new_cifar_dense_conv2d_k3s1p1_oc.comp` now has `OC_TILE` (spec constant 1), `PX_TILE` (2) and
    `C_TILE` (3). An invocation computes `PX_TILE` adjacent pixels of a row for `OC_TILE` channels (the 3 x
    `PX_TILE`+2 window loaded once per input channel, each weight used `PX_TILE` times; a row whose width is not a
    multiple of `PX_TILE` ends in a masked tail), and the workgroup is still one channel block (x) by 256
    invocations (y). `C_TILE` 0 loads the window from global memory on clamped indices; `C_TILE` > 0 stages
    `C_TILE` input channels per pass in shared memory: the workgroup's output rows of the flattened N x H plus a
    halo row above and below, each with a zero column on the left and zeros to `ceil(W/PX_TILE)·PX_TILE + 2`
    (a halo row from the neighbouring image is staged but masked). `LDS_FLOATS` (4) is `C_TILE` times
    `kiss_vk::dense_conv_staged_floats_per_channel(PX_TILE)`, the most one channel needs at any width up to 32
    (777 / 1036 / 1554 floats at 1 / 2 / 4 pixels). The dispatcher reads the tile once from
    `BT_VK_DENSE_CONV_TILE=<oc>x<px>x<c>` (default `4x1x0`, the 2.4 kernel, until 2.6), throws on a malformed
    value or one whose staging exceeds the device's `maxComputeSharedMemorySize` (`8x8x8`: "needs 82880 bytes of
    shared memory, the device has 65536"), and stages only maps up to 32 wide. Grid
    `kiss_vk::dense_conv_k3s1p1_oc_grid(n, k, h, w, tile)`: `(K/OC_TILE, ceil(N·H·ceil(W/PX_TILE)/256), 1)`.
  - *Tests:* `DenseConvTiled.{Map6x6,Map8x8,RaggedOddWidthMap}` run the kernel for seven tiles (`4x1x0`, `8x2x0`,
    `8x4x0`, `4x1x3`, `8x2x2`, `8x4x2`, `16x2x4`) against the double-precision reference, N 24 (every channel
    block spans more than one workgroup), C 5 (a short last pass at 2, 3 and 4 staged channels). They replace
    `DenseConvPerBlock.*` (the `4x1x0` tile). With the top staged row of every workgroup read from one row lower
    (a local mutation, reverted) the four staged tiles fail on 6x6 and 7x5 and the three unstaged ones pass (8x8
    workgroups start at an image, where that row is always masked).
  - *rocky gate* (this commit's tree in `~/bt-dense-gate`): build exit 0, `ctest -L vulkan` 7/7 passed; then
    `./test-cifar-dense-vk --device minipc` 19 passed with `BT_VK_DENSE_CONV_TILE` unset, `4x1x0`, `8x2x0`,
    `8x4x0`, `8x2x2`, `8x4x2` and `16x2x4`, each with synthetic and with real weights (`BT_WEIGHTS_DIR`).
    `just check-shaders`: OK.
- [x] 2.6 Pick the tile defaults by timing on the Samsung. Verify: the per-device sweep and chosen defaults are in the evidence note.
  - *Method:* the 2.1 method on the Samsung, `base` (d563dc5) against the 2.5 binary with
    `BT_VK_DENSE_CONV_TILE` set per arm, arms alternating, 3 rounds, 20 s apart. 72 runs exit 0 with 11 records;
    6 more, the mixed arms of sweep 2, exit 127 with none (the run script split their value at the commas): they
    are not counted, and sweep 3 reran those arms. The tables give per conv the median p50, the GPU clock of each
    run's cell (`gpu_clock_mhz`) and the median of p50 x clock (millions of GPU cycles), then each run's conv2-conv5
    sum in ms and in cycles. The Pixel 7a runs the Mali path, which no tile changes; rocky-ryzen is not timed and
    gets the Samsung's defaults (same subgroup size).
  - *The Samsung's GPU clock follows the load.* Within a run each conv cell settles at one clock between 252 and
    1095 MHz, and the tiles that save the most cycles settle lowest: 8x4x0 needs 4.24 Mcycles for conv2-conv5
    against `base`'s 7.46 (-43 %) but runs conv3 at 252-315 and conv5 at 350 MHz instead of 600-800, so its
    wall-time sum is 12.02 vs 12.20 ms. Cycles per conv are stable between runs (`base` 7.45-7.63 Mcycles while its
    wall time spans 11.71-12.94 ms over the sessions below), wall time is not. The spec measures wall time, so the
    defaults minimize it per conv; cycles are reported to explain them.
  - *Sweep 1* (2026-10-20 06:27-07:03 UTC): every register tile saves cycles, the most at 4 or 8 channels x 4
    pixels; staging (`c` > 0) costs 2-3.5x the cycles of `base` (the staging index math, the barriers; the GPU then
    runs at 800-1095 MHz). No tile spills (ACO on rocky: 32 VGPRs at 4x1x0 up to 96 at 16x4x0, 8.7-16.9 KB of
    shared memory for the staged ones).

    | arm | stage 1: ms / MHz / Mcyc | stage 3: ms / MHz / Mcyc | stage 5: ms / MHz / Mcyc | stage 6: ms / MHz / Mcyc | stage 7: ms / MHz / Mcyc | conv2-5 ms | conv2-5 Mcyc |
    |---|---|---|---|---|---|---|---|
    | base | 0.913 / 252/252/252 / 0.23 | 3.404 / 315/315/315 / 1.07 | 3.286 / 600/600/800 / 1.97 | 3.371 / 800/800/800 / 2.70 | 2.143 / 800/800/650 / 1.71 | 12.201 (12.138-12.209) | 7.46 (7.45-7.63) |
    | t410 | 0.914 / 252/252/252 / 0.23 | 3.358 / 315/315/350 / 1.06 | 2.955 / 800/600/600 / 1.77 | 3.180 / 800/800/800 / 2.54 | 2.549 / 650/650/800 / 1.66 | 11.288 (11.200-12.074) | 7.02 (7.00-7.06) |
    | t420 | 0.913 / 252/252/252 / 0.23 | 2.487 / 252/252/252 / 0.63 | 3.654 / 315/315/315 / 1.15 | 2.633 / 545/500/600 / 1.43 | 4.001 / 350/350/350 / 1.40 | 12.695 (12.657-12.967) | 4.59 (4.58-4.68) |
    | t440 | 0.910 / 252/252/252 / 0.23 | 2.619 / 252/252/252 / 0.66 | 2.952 / 252/315/252 / 0.80 | 2.599 / 545/545/500 / 1.41 | 3.342 / 315/315/315 / 1.05 | 11.517 (11.499-11.525) | 3.87 (3.81-4.06) |
    | t810 | 0.911 / 252/252/252 / 0.23 | 3.290 / 315/315/315 / 1.04 | 2.226 / 800/800/545 / 1.78 | 3.181 / 800/800/800 / 2.55 | 2.113 / 800/800/650 / 1.69 | 10.819 (10.789-12.356) | 7.06 (7.04-7.07) |
    | t820 | 0.913 / 252/252/252 / 0.23 | 2.746 / 252/252/252 / 0.69 | 3.899 / 350/350/350 / 1.36 | 1.976 / 600/800/650 / 1.57 | 3.634 / 350/350/350 / 1.27 | 12.281 (12.234-12.872) | 4.89 (4.62-4.89) |
    | t840 | 0.906 / 252/252/252 / 0.23 | 2.497 / 252/252/252 / 0.63 | 3.119 / 252/315/315 / 0.97 | 2.532 / 500/450/500 / 1.27 | 3.854 / 350/350/350 / 1.35 | 12.024 (12.004-12.351) | 4.24 (4.03-4.26) |
    | t1620 | 0.915 / 252/252/252 / 0.23 | 3.470 / 252/252/252 / 0.87 | 3.775 / 350/350/400 / 1.43 | 2.508 / 650/800/700 / 1.81 | 2.821 / 450/450/500 / 1.27 | 12.643 (12.211-12.908) | 5.37 (5.32-5.39) |
    | t1640 | 0.915 / 252/252/252 / 0.23 | 3.135 / 315/252/315 / 0.97 | 2.474 / 450/450/400 / 1.10 | 2.503 / 800/800/800 / 2.00 | 2.570 / 500/500/545 / 1.29 | 10.635 (10.607-10.990) | 5.34 (5.27-5.37) |
    | t414 | 0.913 / 252/252/252 / 0.23 | 4.590 / 900/900/900 / 4.13 | 6.062 / 1095/1095/1095 / 6.64 | 8.369 / 1095/1095/1095 / 9.16 | 5.570 / 1095/1095/1095 / 6.10 | 24.570 (24.547-24.603) | 26.01 (25.99-26.05) |
    | t822 | 0.912 / 252/252/252 / 0.23 | 3.131 / 600/650/800 / 2.00 | 4.338 / 800/900/800 / 3.48 | 5.186 / 900/1000/1000 / 5.19 | 3.989 / 800/800/900 / 3.26 | 16.336 (15.898-17.481) | 14.04 (13.91-14.14) |
    | t842 | 0.908 / 252/252/252 / 0.23 | 2.720 / 800/800/800 / 2.18 | 4.434 / 900/900/900 / 3.99 | 6.232 / 1095/1095/1095 / 6.82 | 5.033 / 900/900/900 / 4.53 | 18.407 (18.402-18.544) | 17.51 (17.50-17.63) |
    | t824 | 0.916 / 252/252/252 / 0.23 | 4.156 / 800/800/900 / 3.36 | 5.840 / 1095/1095/1095 / 6.39 | 7.810 / 1095/1095/1095 / 8.55 | 5.142 / 1095/1095/1095 / 5.63 | 22.982 (22.639-22.993) | 23.95 (23.93-24.04) |

  - *Conv1* (sweep 2, 07:07-07:20 UTC, the knob applied to all five convs; its mixed arms are the broken try):
    conv1 runs at 252 MHz in every run, so cycles saved would show, but no tile is faster on it: 8x1x0 0.930,
    8x4x0 0.916, 16x4x0 1.207 vs 0.912 ms. Conv1 stays `flat`.

    | arm | stage 1: ms / MHz / Mcyc | stage 3: ms / MHz / Mcyc | stage 5: ms / MHz / Mcyc | stage 6: ms / MHz / Mcyc | stage 7: ms / MHz / Mcyc | conv2-5 ms | conv2-5 Mcyc |
    |---|---|---|---|---|---|---|---|
    | base | 0.912 / 252/252/252 / 0.23 | 3.400 / 315/315/315 / 1.07 | 2.719 / 800/700/700 / 1.93 | 3.375 / 800/800/800 / 2.70 | 2.418 / 650/800/650 / 1.77 | 11.764 (11.708-12.236) | 7.46 (7.35-7.49) |
    | a810 | 0.930 / 252/252/252 / 0.23 | 3.289 / 315/315/315 / 1.04 | 2.818 / 650/650/800 / 1.83 | 3.178 / 800/800/800 / 2.54 | 2.104 / 800/800/800 / 1.68 | 11.389 (10.792-11.432) | 7.09 (7.04-7.12) |
    | a840 | 0.916 / 252/252/252 / 0.23 | 2.486 / 252/252/252 / 0.63 | 3.097 / 315/315/252 / 0.97 | 2.538 / 500/500/450 / 1.27 | 3.875 / 350/350/350 / 1.36 | 11.994 (11.963-12.438) | 4.21 (4.09-4.23) |
    | a1640 | 1.207 / 252/252/252 / 0.30 | 3.400 / 252/252/315 / 0.86 | 2.477 / 450/450/450 / 1.11 | 3.261 / 600/800/650 / 2.11 | 2.579 / 500/450/545 / 1.33 | 11.353 (11.351-12.000) | 5.38 (5.31-5.58) |

  - *Per-conv tiles* (sweep 3, 07:20-07:34 UTC): `mixA` = `flat,8x4x0,8x1x0,16x4x0,8x1x0` and `mixB` the same with
    8x4x0 on conv1. Per conv, over all three sweeps, the fastest tiles in wall time are 8x4x0 on conv2 (2.48-2.50 ms
    at 252 MHz in every run, vs 3.40), 16x4x0 on conv3 (2.44-2.48, at 450 MHz; 8x1x0 reads 2.23 at 800 but 2.8-3.0
    at 600-650), 16x4x0 on conv4 (2.49-2.52 in four of five arms, vs 3.37-3.39) and 8x1x0 on conv5 (2.10-2.11 at
    800 MHz in all three sweeps, vs 2.14-2.89).

    | arm | stage 1: ms / MHz / Mcyc | stage 3: ms / MHz / Mcyc | stage 5: ms / MHz / Mcyc | stage 6: ms / MHz / Mcyc | stage 7: ms / MHz / Mcyc | conv2-5 ms | conv2-5 Mcyc |
    |---|---|---|---|---|---|---|---|
    | base | 0.912 / 252/252/252 / 0.23 | 3.403 / 315/315/315 / 1.07 | 2.448 / 800/800/650 / 1.96 | 3.375 / 800/800/800 / 2.70 | 2.893 / 545/600/650 / 1.74 | 12.286 (12.115-12.665) | 7.46 (7.40-7.60) |
    | a810 | 0.931 / 252/252/252 / 0.23 | 3.296 / 315/315/315 / 1.04 | 2.994 / 800/600/600 / 1.80 | 3.187 / 800/800/800 / 2.55 | 2.107 / 800/800/800 / 1.69 | 11.571 (10.818-11.589) | 7.07 (7.06-7.07) |
    | a1640 | 1.211 / 252/252/252 / 0.31 | 3.154 / 315/252/252 / 0.87 | 2.443 / 450/450/450 / 1.10 | 2.518 / 800/800/800 / 2.01 | 2.537 / 545/545/545 / 1.38 | 10.669 (10.552-10.937) | 5.36 (5.30-5.46) |
    | mixA | 0.910 / 252/252/252 / 0.23 | 2.500 / 252/252/252 / 0.63 | 2.999 / 600/600/800 / 1.80 | 2.494 / 700/800/800 / 1.99 | 2.105 / 545/800/800 / 1.68 | 10.098 (9.322-11.430) | 6.10 (6.09-6.11) |
    | mixB | 0.920 / 252/252/252 / 0.23 | 2.484 / 252/252/252 / 0.63 | 2.994 / 545/600/800 / 1.78 | 2.509 / 800/800/800 / 2.01 | 3.055 / 500/545/600 / 1.66 | 11.044 (10.066-11.490) | 6.09 (6.03-6.12) |

  - *Chosen defaults:* `flat,8x4x0,16x4x0,16x4x0,8x1x0` (conv1..conv5) on every subgroup >= 32 GPU, in
    `kDefaultConvTiles`. To allow a tile per conv, `BT_VK_DENSE_CONV_TILE` takes one tile or five, and `flat` (the
    old kernel) replaces 2.4's `BT_VK_DENSE_CONV=flat`; the dispatcher builds one pipeline per distinct tile.
    Documented in `03-unit-testing.md` (conv paths). A last check of the chosen tiles (07:35-07:40 UTC, `base` vs
    `chosen`): conv2-conv5 sum 9.573 (8.597-9.897) vs 12.271 (11.710-12.941) ms, -22.0 %; conv2/conv3/conv4
    -27.1 / -26.2 / -25.7 %, conv5 -1.8 % with overlapping spreads, conv1 -0.2 %. Task 2.7 repeats this on both
    phones.

    | arm | stage 1: ms / MHz / Mcyc | stage 3: ms / MHz / Mcyc | stage 5: ms / MHz / Mcyc | stage 6: ms / MHz / Mcyc | stage 7: ms / MHz / Mcyc | conv2-5 ms | conv2-5 Mcyc |
    |---|---|---|---|---|---|---|---|
    | base | 0.912 / 252/252/252 / 0.23 | 3.404 / 315/315/315 / 1.07 | 3.321 / 600/700/600 / 1.99 | 3.390 / 800/800/800 / 2.71 | 2.154 / 650/800/800 / 1.72 | 12.271 (11.710-12.941) | 7.50 (7.44-7.61) |
    | chosen | 0.911 / 252/252/252 / 0.23 | 2.481 / 252/252/252 / 0.63 | 2.451 / 450/450/600 / 1.10 | 2.518 / 700/800/800 / 2.00 | 2.115 / 800/800/650 / 1.69 | 9.573 (8.597-9.897) | 5.41 (5.03-5.44) |

    | stage | base median (min-max) ms | chosen median (min-max) ms | chosen vs base |
    |---|---|---|---|
    | 1 | 0.912 (0.909-0.918, n=3x1088-1673) | 0.911 (0.908-0.913, n=3x1116-1141) | -0.2 % |
    | 2 | 1.309 (1.288-1.312, n=3x792-1379) | 1.300 (1.245-1.306, n=3x968-1668) | -0.7 % |
    | 3 | 3.404 (3.400-3.404, n=3x473-566) | 2.481 (2.170-2.489, n=3x469-965) | -27.1 % |
    | 4 | 0.973 (0.968-0.975, n=3x1146-1250) | 0.977 (0.225-0.977, n=3x1070-2000) | +0.4 % |
    | 5 | 3.321 (2.768-3.381, n=3x300-375) | 2.451 (1.303-2.451, n=3x534-858) | -26.2 % |
    | 6 | 3.390 (3.389-3.391, n=3x326-328) | 2.518 (2.459-2.860, n=3x316-447) | -25.7 % |
    | 7 | 2.154 (2.153-2.767, n=3x419-517) | 2.115 (2.106-2.665, n=3x375-531) | -1.8 % |
    | 8 | 0.328 (0.326-0.329, n=3x2000) | 0.325 (0.324-0.328, n=3x2000) | -1.0 % |
    | 9 | 2.961 (2.050-2.987, n=3x454-532) | 2.932 (2.100-3.031, n=3x331-524) | -1.0 % |
    | 10 | 2.759 (2.009-2.825, n=3x406-541) | 1.978 (1.976-2.008, n=3x537-542) | -28.3 % |
    | 11 | 0.647 (0.624-0.663, n=3x1414-2000) | 0.626 (0.625-0.660, n=3x2000) | -3.2 % |
    | sum | 22.158 | 18.613 | -16.0 % |
    | conv2-conv5 (3+5+6+7), per-run sum | 12.271 (11.710-12.941, 3 runs) | 9.573 (8.597-9.897, 3 runs) | -22.0 % |
- [x] 2.7 Check the FP32 performance requirement. Verify: on the Samsung the conv2–conv5 sum is faster than 2.1 beyond noise, no phone conv stage is slower; the fleet gate is green. If it isn't met, record the gap; the optional tasks 2.8–2.12 don't count toward it.
  - *A/B* (the 2.1 method, `base` d563dc5 vs `final`, the e34ca8d binary with no knob set, both phones at once,
    2026-10-20 07:43-07:48 UTC; all 12 runs exit 0 with 11 records).
  - *Samsung, conv2-conv5 faster beyond noise: met.* Per-run sums 9.560 (9.550-10.696) vs 11.403 (11.059-12.676)
    ms, -16.2 %, spreads apart; also below both 2.1 arms (11.124-12.216 and 11.408-11.459). Conv2 -26.9 % and
    conv4 -25.5 % with spreads apart; conv3 -0.1 % and conv5 -2.4 % with overlapping spreads. In GPU cycles the four
    convs take 5.42 vs 7.47 million (-27 %); conv3 runs at 450 MHz instead of 650-800, which is why its time does
    not move (2.6).
  - *No conv slower beyond noise: met.* Samsung: conv1 (the same kernel) +0.4 %, overlapping. Pixel 7a (the Mali
    path, the same shader and grid as `base`): stage 3 reads 14.545 (13.416-14.546) vs 12.357 (12.357-14.540) and
    stage 7 15.728 (14.686-15.771) vs 14.547 (14.546-15.767), the two levels each of them showed within one binary
    in 2.1 (9.98-14.54 and 14.53-15.79 ms); spreads overlap. Every other Pixel conv is within 0.3 % or faster.
  - *Fleet gate, green* (the code of e34ca8d; this commit adds only notes):
    - *local:* `cmake --build --preset pc && ctest --test-dir build/pc -L omp`: 11/11 passed.
    - *rocky:* `~/bt-dense-gate`, `cmake --preset vulkan` (glslc of NDK r29), build exit 0, `ctest -L vulkan` 7/7
      passed, `./test-cifar-dense-vk --device minipc` 19 passed with the default tiles and with
      `BT_VK_DENSE_CONV_TILE=flat`, each with synthetic and with real weights. (A first real-weight run failed 13
      tests with "cannot open file": the tree sync had deleted the weights copied into the synced directory; they
      now live beside it.)
    - *phones:* `BT_ANDROID_DEST=/data/local/tmp/bt-dense/gate scripts/run-on-android.sh <serial>
      test-cifar-dense-vk`: 19 passed, exit 0, on 3A021JEHN02756 and on R5CY21Y3VEV; the same binary with real
      weights: 19 passed on each.
  - Samsung, ms:

    | stage | base median (min-max) ms | final median (min-max) ms | final vs base |
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

    | arm | stage 1: ms / MHz / Mcyc | stage 3: ms / MHz / Mcyc | stage 5: ms / MHz / Mcyc | stage 6: ms / MHz / Mcyc | stage 7: ms / MHz / Mcyc | conv2-5 ms | conv2-5 Mcyc |
    |---|---|---|---|---|---|---|---|
    | base | 0.911 / 252/252/252 / 0.23 | 3.402 / 315/315/315 / 1.07 | 2.453 / 800/800/650 / 1.96 | 3.376 / 800/800/800 / 2.70 | 2.155 / 800/800/650 / 1.72 | 11.403 (11.059-12.676) | 7.47 (7.35-7.61) |
    | final | 0.915 / 252/252/252 / 0.23 | 2.487 / 252/252/252 / 0.63 | 2.451 / 450/450/450 / 1.10 | 2.514 / 800/700/800 / 2.01 | 2.104 / 800/600/800 / 1.68 | 9.560 (9.550-10.696) | 5.42 (5.42-5.47) |

  - Pixel 7a, ms:

    | stage | base median (min-max) ms | final median (min-max) ms | final vs base |
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
- [ ] 2.8 C4: optional-feature enablement for `shaderFloat16` and `storageBuffer16BitAccess` (enabled if supported, never required) plus a precision mode (`BT_VK_PRECISION=fp16`). Verify: `test-kiss-vk` logs the enabled features on all devices, and default runs are unchanged.
  - *Not done* (optional; 2.7 is met without it). FP16 math would save GPU cycles, and on the Samsung, the only
    phone that runs this path, cycles saved mostly lower the GPU clock rather than the wall time the spec measures:
    the 8x4x0 tile took 43 % fewer cycles than `base` for conv2-conv5 at about the same time (2.6). Enabling 16-bit
    features changes device creation for every app, so it needs the whole fleet gate for a gain that the 2.6
    measurements do not promise. Group 3's FP16 FC weights (F2) need the same features and can add them.
- [ ] 2.9 C4: the FP16 conv variant (FP32 accumulate), with a per-stage tolerance, and a new VK end-task accuracy test on the real weights mirroring the OMP `CifarDenseAppData.RealWeights_EndTaskAccuracy`. Verify: `test-cifar-dense-vk` with `BT_VK_PRECISION=fp16` passes on supporting devices; FP16 accuracy is within 0.5 points of FP32 (printed); timing is recorded.
  - *Not done*, with 2.8 (it needs 2.8's features and precision mode). The VK end-task accuracy test is not built
    either, so 3.7 cannot reuse it and builds its own.
- [x] 2.10 C5: query cooperative-matrix shapes and add the implicit-GEMM conv (FP16 in, FP32 accumulate), selected only when supported. Verify: it passes on rocky with the FP16 tolerance and the 2.9 accuracy bound. Timing is on the Samsung only if 2.2 shows it supports cooperative matrix; if no phone does, C5 is not built (it could not be measured) and that is recorded. If it doesn't beat 2.7/2.9, it stays disabled by default and that's documented.
  - *Not built, as this task directs when no phone supports cooperative matrix:* 2.2 queried the shapes;
    `VK_KHR_cooperative_matrix` is absent on the Pixel 7a's Mali-G710 and on the Samsung's Xclipse 940, and only the
    780M (rocky-ryzen, where nothing is timed) lists 16x16x16 FP16 x FP16 + FP32. It also depends on 2.9, not done.
- 2.11-2.12 (C6, Mali variant): removed 2026-10-18; owned and shipped by `mali-conv-kernels` (archived 2026-10-17).
- [x] 2.13 Run the group gate at the final commit (local, rocky, phones), in default FP32 and in each opt-in mode on supporting devices. Add a dated perf-results note with before/after tables and per-device variant choices. Verify: all green; the note is linked from the perf-results README.
  - Gate at 334f576 (its code is e34ca8d's; this commit adds only the note and these lines). Default FP32 only:
    no opt-in precision mode exists (2.8-2.10).
    - *local:* `cmake --build --preset pc && ctest --test-dir build/pc -L omp`: 11/11 passed.
    - *rocky:* the tree synced to `~/bt-dense-gate`, `cmake --preset vulkan` (glslc of NDK r29), build exit 0,
      `ctest -L vulkan` 7/7 passed, `./test-cifar-dense-vk --device minipc` 19 passed with synthetic and with real
      weights. `just check-shaders`: OK.
    - *phones:* `cmake --build --preset android --target test-cifar-dense-vk`, then
      `BT_ANDROID_DEST=/data/local/tmp/bt-dense/gate scripts/run-on-android.sh <serial> test-cifar-dense-vk`: 19
      passed and exit 0 on 3A021JEHN02756 and on R5CY21Y3VEV; the same binary with real weights: 19 passed on each.
  - *Note:* `docs/reports-for-human/perf-results/2026-10-20-cifar-dense-conv.md` (the 2.7 before/after tables per
    phone, the GPU cycles, the tile sweep, the variant per device, ACO findings), linked from the perf-results
    README.

## 3. FC weight bandwidth (gpu-kernels/fully-connected; design F1–F4)

- [x] 3.1 Record the pre-change fc1/fc2/fc3 GPU times for both apps on the Pixel 7a and the Samsung. Verify: per-phone tables with spreads (2.1 and 1.1 may be reused).
  - *Method* (used for every timing of this group): the 1.1 method. `bm-prof-cifar-{sparse,dense}-vk` built with the
    `android` preset (NDK r30) at 5454f67, pushed to `/data/local/tmp/bt-fc/base` on each phone, real weights
    (`saved_params/export` pushed to `/data/local/tmp/bt-fc/weights`), `BT_PROF_SCENARIO=isolated
    BT_PROF_BUDGET_S=1.0 BT_PROF_ABANDON_S=5 BT_PROF_MAX_CELL_S=3 --benchmark_filter='vulkan/'`. Arms alternate,
    3 rounds, 20 s idle before every run, and each run first checks that no other `bm-`/`test-` process is on the
    phone; both phones ran at once. Here arms A and B are the same binary, for the noise (2026-10-20 09:24-09:36
    UTC; all 24 runs exit 0 with 11 records). Per stage: the median of the 3 runs' p50 (GPU timestamps), with the
    min-max of the three p50s, `n = runs x samples`; the last row is each run's fc1+fc2 sum, median and min-max.
  - *Noise:* cifar-sparse fc1 and fc2 are within 0.1 % between the identical arms on both phones, with overlapping
    spreads under 0.6 %; every run held the GPU at its top clock (`gpu_clock_mhz` 848 on the Pixel, 1095 on the
    Samsung). cifar-dense fc1 and fc2 follow the governor instead: the Samsung ran them at 800 MHz in 2.0-2.2 ms or
    at 500-650 MHz in 2.8-3.1 ms, the Pixel at 471 MHz in 6.7-6.9 ms or at 510-633 MHz in 5.0-6.2 ms, so single
    dense runs differ by up to 40 % (the Samsung's fc2 medians) in one binary. fc3 moves by up to 15 % (Pixel
    sparse, 1.16 or 1.38 ms).
  - *Ratio* sparse fc1 (N=128) / dense fc1 (N=16), arm A medians: Pixel 7a 26.936 / 6.839 = 3.9; Samsung
    10.051 / 3.066 = 3.3 (4.7 against a dense run at 800 MHz).
  - Pixel 7a, cifar-sparse (N=128), ms:

    | stage | A median (min-max) ms | B median (min-max) ms | B vs A |
    |---|---|---|---|
    | 9 (fc1) | 26.936 (26.934-26.968, n=3x35-37) | 26.942 (26.934-27.073, n=3x32-37) | +0.0 % |
    | 10 (fc2) | 27.173 (27.169-27.182, n=3x36) | 27.174 (27.139-27.201, n=3x36-37) | +0.0 % |
    | 11 (fc3) | 1.373 (1.163-1.386, n=3x1431-1547) | 1.173 (1.172-1.383, n=3x1435-1452) | -14.6 % |
    | fc1+fc2, per-run sum | 54.115 (54.105-54.140, 3 runs) | 54.108 (54.081-54.274, 3 runs) | -0.0 % |

  - Samsung, cifar-sparse (N=128), ms:

    | stage | A median (min-max) ms | B median (min-max) ms | B vs A |
    |---|---|---|---|
    | 9 (fc1) | 10.051 (10.050-10.106, n=3x99-100) | 10.042 (10.030-10.049, n=3x99-100) | -0.1 % |
    | 10 (fc2) | 10.047 (10.045-10.049, n=3x99) | 10.041 (10.036-10.052, n=3x99) | -0.1 % |
    | 11 (fc3) | 1.172 (1.158-1.177, n=3x2000) | 1.155 (1.153-1.159, n=3x2000) | -1.4 % |
    | fc1+fc2, per-run sum | 20.098 (20.094-20.155, 3 runs) | 20.090 (20.066-20.094, 3 runs) | -0.0 % |

  - Pixel 7a, cifar-dense (N=16), ms:

    | stage | A median (min-max) ms | B median (min-max) ms | B vs A |
    |---|---|---|---|
    | 9 (fc1) | 6.839 (6.190-6.849, n=3x97-120) | 6.777 (6.208-6.865, n=3x72-158) | -0.9 % |
    | 10 (fc2) | 6.681 (4.976-6.700, n=3x145-160) | 6.686 (6.675-6.689, n=3x138-159) | +0.1 % |
    | 11 (fc3) | 0.984 (0.982-0.986, n=3x1337-1670) | 0.983 (0.982-0.986, n=3x1330-1353) | -0.1 % |
    | fc1+fc2, per-run sum | 12.870 (11.814-13.549, 3 runs) | 13.463 (12.897-13.540, 3 runs) | +4.6 % |

  - Samsung, cifar-dense (N=16), ms:

    | stage | A median (min-max) ms | B median (min-max) ms | B vs A |
    |---|---|---|---|
    | 9 (fc1) | 3.066 (2.161-3.115, n=3x330-476) | 2.151 (2.092-3.030, n=3x446-527) | -29.8 % |
    | 10 (fc2) | 2.013 (2.013-2.042, n=3x492-537) | 2.828 (2.045-3.012, n=3x342-488) | +40.5 % |
    | 11 (fc3) | 0.634 (0.630-0.643, n=3x2000) | 0.638 (0.636-0.664, n=3x1386-2000) | +0.5 % |
    | fc1+fc2, per-run sum | 5.079 (4.204-5.128, 3 runs) | 4.920 (4.197-6.043, 3 runs) | -3.1 % |

    The other stages of these runs agree with 1.1 and 2.1 (Pixel sparse sum 584.1 ms, dense 84.1 ms).
- [x] 3.2 F3: replace the `sh_red` and lane-0 serial reduction with `subgroupClusteredAdd(…, 16)`, keeping the shared-memory reduction behind a spec constant for devices without CLUSTERED support (per 2.2). Verify: both apps' `test-*-vk` pass on rocky and both phones; timing on both phones is recorded.
  - *Kernel:* `new_cifar_dense_linear_bt.comp` is now subgroup-size dependent (the engine pins the size, >= 16
    on every fleet GPU, with full subgroups, so a row's 16 lanes are one aligned cluster). With `CLUSTERED`
    (spec constant 1, set from the new `ComputeLimits::subgroup_clustered`: `VK_SUBGROUP_FEATURE_CLUSTERED_BIT`
    for compute) and a subgroup of at least 16, each row sums its partials with `subgroupClusteredAdd(vec4, 16)`
    (8 per row, 4 images x 1 feature each) and lane n writes image n of both features: no `sh_red`, no barrier
    after the K loop. `CLUSTERED = 0` or a narrower subgroup keeps the old shared-memory reduction.
    `LinearBt.RaggedFeatureBatchAndChunkEdges` (the clustered path, skipped where a device lacks it) and its new
    `...SharedMemoryReduction` twin cover both against a double-precision reference.
  - *Static (Mali):* `malioc -c Mali-G710 --vulkan` on the glslc output: the old reduction indexed `acc[]` with
    the pass's loop variable, so both accumulator arrays lived on the stack (alloca 128 B, spill 80 B, LS 59.2
    cycles total); clustered, there is no alloca (spill 20 B, LS 28.8). Both use 64 work registers (50 %
    occupancy).
  - *Tests:* rocky (`~/bt-fc-gate`, glslc of NDK r29): `ctest -L vulkan` 7/7 passed; `test-cifar-sparse-vk` 22 and
    `test-cifar-dense-vk` 19 passed with synthetic and with real weights. Pixel 7a and Samsung (`android` preset,
    `/data/local/tmp/bt-fc/f3`): the same counts, synthetic and real, exit 0 on both.
  - *Timing* (the 3.1 method, `--benchmark_filter='vulkan/stage(9|10|11)/'`, arms base = 5454f67 and f3,
    2026-10-20 09:50-09:56 UTC), ms:

    | phone, app | stage | base median (min-max) | f3 median (min-max) | f3 vs base |
    |---|---|---|---|---|
    | Pixel, sparse | 9 (fc1) | 26.941 (26.929-26.943, n=3x35-37) | 21.085 (21.065-21.143, n=3x35-48) | -21.7 % |
    | Pixel, sparse | 10 (fc2) | 27.186 (27.183-27.204, n=3x36) | 22.254 (22.189-22.491, n=3x44-45) | -18.1 % |
    | Pixel, sparse | 11 (fc3) | 1.292 (1.255-1.384, n=3x1398-1451) | 1.097 (1.094-1.100, n=3x1614-1700) | -15.1 % |
    | Samsung, sparse | 9 (fc1) | 10.036 (10.033-10.042, n=3x99-100) | 9.302 (9.261-9.303, n=3x107-109) | -7.3 % |
    | Samsung, sparse | 10 (fc2) | 10.042 (10.041-10.043, n=3x99) | 9.322 (9.310-9.326, n=3x106-108) | -7.2 % |
    | Samsung, sparse | 11 (fc3) | 1.156 (1.156-1.157, n=3x2000) | 1.131 (1.131-1.136, n=3x2000) | -2.2 % |
    | Pixel, dense | 9 (fc1) | 6.202 (5.506-6.837, n=3x119-177) | 6.866 (6.844-7.208, n=3x78-155) | +10.7 % |
    | Pixel, dense | 10 (fc2) | 6.666 (6.340-6.682, n=3x122-141) | 6.703 (6.683-7.221, n=3x123-177) | +0.6 % |
    | Pixel, dense | 11 (fc3) | 0.978 (0.890-0.985, n=3x1365-1525) | 0.868 (0.865-1.017, n=3x1434-1830) | -11.3 % |
    | Samsung, dense | 9 (fc1) | 2.899 (2.854-3.001, n=3x336-358) | 2.734 (2.704-2.736, n=3x374-507) | -5.7 % |
    | Samsung, dense | 10 (fc2) | 1.977 (1.975-2.762, n=3x358-538) | 2.650 (1.875-2.705, n=3x371-569) | +34.0 % |
    | Samsung, dense | 11 (fc3) | 0.644 (0.633-0.663, n=3x1413-2000) | 0.640 (0.634-0.642, n=3x1592-2000) | -0.6 % |

    cifar-sparse (top clock in every run) is faster on both phones beyond noise. cifar-dense moves with the
    governor as in 3.1: at the same clock the arms agree (Pixel fc1 at 471 MHz: 6.837 base, 6.844 f3; Samsung
    fc2 at 800 MHz: 1.975-1.977 base, 1.875 f3), and the Pixel base runs that were faster ran fc1 at 572-633 MHz,
    the slowest f3 run at 400 MHz.
- [ ] 3.3 F1: prototype layouts (a) K-outer/tiles-inner and (b) batch-split so weights stream once per batch. Verify: both pass `test-cifar-sparse-vk` and `test-cifar-dense-vk` on rocky; sparse fc1/fc2 timings on both phones for each layout are in the evidence note.
- [ ] 3.4 F1: pick the layout by timing on both phones and remove the other. Verify: fc1 and fc2 are faster than 3.1 beyond noise on both phones, and each phone's sparse-fc1 / dense-fc1 ratio is recorded (spec "Weights are streamed once per batch"); the fleet gate is green for both apps.
- [ ] 3.5 F2: host-side weight quantization at load time (FP16; INT8 with per-row scales) and the precision mode `BT_VK_FC_WEIGHTS=fp16|int8` with FP32 fallback and a warning on unsupported devices. Verify: a unit test round-trips quantize→dequantize within the stated per-element error; default runs are unchanged.
- [ ] 3.6 F2: FP16-weight and INT8-weight `linear_bt` variants, dequantizing to FP32 in registers. Verify: both apps' `test-*-vk` pass their documented FP16/INT8 tolerances on supporting devices.
- [ ] 3.7 F2: VK end-task accuracy on the real weights for both apps in FP16 and INT8 (reusing the 2.9 test harness for cifar-dense, plus a sparse twin). Verify: each is within 0.5 points of FP32 (dense 90.48%, sparse 90.58%), printed. If INT8 fails, record it and ship FP16 only.
- [ ] 3.8 Check the reduced-precision requirement on the phones. Verify: on each phone that supports it, fc1/fc2 with FP16 and with INT8 weights are faster than FP32 beyond noise, with the ratios recorded (spec "Reduced-precision weights cut weight time").
- [ ] 3.9 F4 (optional): if `new_cifar_{dense,sparse}_linear.comp` are removed, first add a test forcing `in_features % 4 != 0` to prove the remaining path covers it. Verify: that test passes, then clean builds. If they're kept, record why.
- [ ] 3.10 Run the group gate at the final commit (local, rocky, phones) for both apps, default and opt-in modes. Add a dated perf-results note with before/after tables. Verify: all green; the note is linked from the perf-results README.
