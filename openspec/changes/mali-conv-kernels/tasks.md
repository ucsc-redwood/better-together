# Tasks

Gates used below:
- **"local":** `ctest --test-dir build/pc -L omp`, plus `test-cifar-{dense,sparse}-vk` on the build box's
  integrated GPU (subgroup 64, so it covers the wide path only).
- **"rocky correctness":** a native build on rocky-ryzen (`ssh doremy@100.97.77.80`, fish login shell,
  feed `bash -s`; glslc from `~/android-ndk-r29/shader-tools/linux-x86_64/glslc`), then
  `test-{kiss,cifar-dense,cifar-sparse}-vk --device minipc`. **Correctness only: no profiler,
  benchmark or timing run on rocky.**
- **"phone gate":** `cmake --preset android`, build the four VK test targets, then
  `scripts/run-on-android.sh 3A021JEHN02756 <targets>` and `... R5CY21Y3VEV <targets>`.
- **"phone A/B":** design D5. Build the previous and the new `bm-prof-cifar-<app>-vk` from two
  worktrees with the Android preset, push both to each phone, and run them alternately (isolated
  scenario, 3 rounds, 20 s between runs). Report each conv stage's median and spread.

## 1. Baseline and tests first

- [x] 1.1 Install `malioc` (Arm Performance Studio) on the build box, or record why it cannot be installed. Verify: `malioc --version` prints a version, or design.md's open question is answered with the reason.
  - *Evidence:* nothing installed and no licence accepted by this run. `malioc` was already on the build box: Arm
    Performance Studio 2026.5 extracted under `~/tools/arm-performance-studio/Arm_Performance_Studio_2026.5/` on
    2026-09-26. `.../mali_offline_compiler/malioc --version` → `Mali Offline Compiler v2026.5.0 (Build 922df8)`;
    `--list` includes `Mali-G710 (OpenGL ES, Vulkan, OpenCL)`. design.md's open question is answered with how it is
    called (`-c Mali-G710 --vulkan -S 0=16` on the glslc `--target-env=vulkan1.3 -O` output).
- [x] 1.2 Record the Mali baseline: per-stage medians of both apps on the Pixel and the Samsung (phone A/B with the same binary in both arms, to measure the noise), plus `malioc` reports of the two current Mali paths if 1.1 succeeded. Verify: the tables and the noise spread are in design.md under a new "Baseline" section.
  - *Evidence:* design.md "Baseline" has the method, the four per-stage tables (Pixel / Samsung × dense / sparse, all
    11 stages, median and min-max of 3 rounds, same 9a10ea0 binary in both arms) and the `malioc` table.
    - Pixel noise: dense convs within 0.1 % (e.g. stage 6 48.810 vs 48.815 ms); sparse convs 3/5/6/7 within 0.5 %
      (181.3 / 283.5 / 398.3 / 242.4 ms); sparse stage 1 within 5 %.
    - Samsung noise: bimodal. Dense stage 7 is 2.14 or 2.75 ms in the same binary, and sparse conv medians differ by
      up to 22 % between identical arms.
    - `malioc`: dense Mali path 64 work registers (50 % occupancy), A 4.21 / LS 26.00, LS-bound. Sparse 26 registers
      (100 %), A 7.49 / LS 29.00, LS-bound. No stack use in either.
    - Bench knobs: `BT_PROF_BUDGET_S=1.0 BT_PROF_ABANDON_S=5 BT_PROF_MAX_CELL_S=3`, because with the defaults the
      Pixel's ~0.4 s sparse stage 6 is sampled once.
- [x] 1.3 Extend `DenseConvK3S1P1` and `SparseConvK3S1P1` with a 7x5 map, and `SparseConvK3S1P1` with a CSR row whose nonzeros all read the padded border (spec "The Mali path matches the reference at any shape"). Verify: the new cases pass on the current kernels in local, rocky correctness and the phone gate.
  - *Evidence:* both helpers take `(h, w)`, and both suites gain `RaggedOddWidthMap` (H 7, W 5: an odd width, 35
    outputs per channel or block). The sparse CSR gains a row (channel 6) whose nonzeros are the kernel's right
    column only (`kx == 2`, every input channel). At the right edge every one of its taps reads the padded border, and the
    test asserts the reference there is exactly the bias, as it does for the empty row (channel 4), before
    comparing with the shader. Tests run on the current kernels:
    - local (build box, RADV Raphael iGPU, subgroup 64): `ctest --test-dir build/pc -L omp` 10/10. `test-cifar-dense-vk`
      16/16 and `test-cifar-sparse-vk` 17/17 (`--gtest_filter='*ConvK3S1P1*'`: 3/3 each).
    - rocky correctness (`~/bt-mali-gate`, native `vulkan` preset, NDK r29 glslc, `--device minipc`,
      `VK_LOADER_LAYERS_DISABLE='~all~'`): kiss 30, cifar-dense 16 + 16 (real weights), cifar-sparse 17 + 17 (real
      weights), all passed.
    - phone gate (`/data/local/tmp/bt-mali`, `--device <serial>`): Pixel 7a and Samsung each kiss 29 + 1 skip
      (`PipelineCacheTest.FifoAtCachePathIsIgnoredWithoutBlocking`), tree 18, cifar-dense 16, cifar-sparse 17, plus
      cifar-dense 16 and cifar-sparse 17 with `BT_WEIGHTS_DIR`; exit 0. These ran the run-on-android.sh steps but
      staged to this run's own `/data/local/tmp/bt-mali` instead of the shared `/data/local/tmp/bt`.

## 2. Path selection and dispatch geometry (D1, D3)

- [x] 2.1 Make the sparse conv `subgroup_size_dependent` (shader spec constant 0, its dispatcher, its test) with a `SUBGROUP_SIZE < 32` branch that still runs today's code. Verify: the phone gate is green, and the Samsung's sparse suite outputs are unchanged.
  - *Evidence:* `new_cifar_sparse_conv2d_k3s1p1.comp` declares `constant_id = 0 SUBGROUP_SIZE`. Today's per-output
    walk moved unchanged into `row_sum()`, which `main()` calls from both the `SUBGROUP_SIZE < 32` and the other branch.
    The dispatcher and `SparseConvK3S1P1` build it with `->subgroup_size_dependent()` (the engine throws if the two
    disagree).
    - Phone gate (both phones): kiss 29 + 1 skip, tree 18, cifar-dense 16, cifar-sparse 17, plus cifar-dense 16 and
      cifar-sparse 17 with real weights; exit 0.
    - "Outputs unchanged": scratch-only test (never committed, `/tmp/mali-work/scripts/add-hash-test.py`). It runs
      stages 1-11 on Vulkan from the seeded AppData and prints an FNV-1a hash of each conv output buffer. It was added to
      62dfb81 and to this group's tree, built in two worktrees and run on each phone with synthetic and real weights.
      On the Samsung all 20 hashes match (2 apps × 5 convs × 2 weight sets; e.g. sparse real conv3
      `6fee95125cbcdac2` both), and on the Pixel 7a all 20 match too.
- [x] 2.2 Add one conv contract header giving each kernel's outputs per invocation for a subgroup size, and size the grid from it in both apps' dispatchers and in the conv tests. Verify: a mutation that makes the header disagree with the shader fails `DenseConvK3S1P1` / `SparseConvK3S1P1` (the outputs stay at their `-1` fill), and the unmutated tree passes the phone gate.
  - *Evidence:* `platform/engine/vulkan/shaders/conv_k3s1p1_contract.hpp`: `kDenseConvChannelsPerInvocation` (4),
    `kDenseConvMaliPixels` / `kSparseConvMaliPixels` (1 for now), and `{dense,sparse}_conv_k3s1p1_invocations(n, k, h,
    w, subgroup_size)`. The five conv sites of each dispatcher and both conv tests size `grid.x` from it with
    `engine.subgroup_size()`, which is the value the engine specializes constant 0 with.
    - Mutation, wide path (build box, RADV, subgroup 64): the contract's non-Mali pixel count 1 → 2. Every
      `DenseConvK3S1P1` and `SparseConvK3S1P1` case fails with `got evaluates to -1`.
    - Mutation, Mali path (Pixel 7a): `kDenseConvMaliPixels = kSparseConvMaliPixels = 2`, built in a scratch worktree.
      `DenseConvK3S1P1` 3/3 FAILED and `SparseConvK3S1P1` 3/3 FAILED, each `got evaluates to -1`.
    - The opposite mismatch (the header counts fewer outputs than the shader computes) only launches surplus invocations,
      which exit on the shader's `idx >= total` guard. That costs time but computes every output.
    - Unmutated tree: phone gate green on both phones (counts in 2.1). Rocky correctness: kiss 30, cifar-dense 16 + 16,
      cifar-sparse 17 + 17 (synthetic + real weights). Local: `test-cifar-dense-vk` 16/16, `test-cifar-sparse-vk`
      17/17. `just check-shaders`: OK.

## 3. Dense conv: output blocking (D2.1)

- [x] 3.1 On the dense Mali path, compute `PX` adjacent output pixels per invocation (try `PX = 4` and `2`), loading each input window and weight once for all of them, with a masked tail for ragged widths. Verify: `DenseConvK3S1P1` (6x6, 7x5, 8x8) and `test-cifar-dense-vk` pass on the Pixel, and local and rocky correctness are green.
  - *Evidence:* `new_cifar_dense_conv2d_k3s1p1.comp` `main_mali()`, entered when `SUBGROUP_SIZE < 32`. One invocation
    is (n, k/4, oh, ow/PX rounded up). Per input channel it makes guarded loads of the 3 x (PX+2) window (no load
    outside the map) and loads 4 x 9 weights once, each reused PX times. The writes are masked for `ow0 + j >= W`.
    `[[unroll]]` keeps the arrays in registers. `accumulate_branchy` (the previous Mali kernel) is removed, and the
    subgroup >= 32 code is untouched. `kDenseConvMaliPixels = 4` in the contract.
    - PX 4 (this tree) and PX 2 (a scratch worktree with PX and the contract both set to 2) each pass
      `test-cifar-dense-vk` 16/16 on the Pixel 7a, `DenseConvK3S1P1` 6x6 / 8x8 / 7x5 included. PX 4 also passes 16/16
      with real weights.
    - Local: `test-cifar-dense-vk` 16/16 (RADV, wide path); `ctest --test-dir build/pc -L omp` 10/10.
    - Rocky correctness: kiss 30, cifar-dense 16 + 16, cifar-sparse 17 + 17, all passed.
    - `malioc`: PX 2 and 4 use 62 and 64 registers (50 % occupancy, as before) with no stack; PX 8 spills 148 B and
      was not built (design.md "Measured decisions").
- [x] 3.2 Run the phone A/B for cifar-dense (previous kernel vs the best `PX`) and apply the spec's ship gate. Verify: design.md records the per-stage table for both phones and the decision, the Samsung's conv stages are unchanged within noise, and the Mali branch stays enabled only if the gate passed.
  - *Evidence:* the Pixel ran 3 arms interleaved (368a884 / PX 2 / PX 4, 3 rounds, 20 s cooldown) and the Samsung
    368a884 vs PX 4. The tables and the decision are in design.md "Measured decisions".
    - Pixel 7a medians, previous → PX 4: stage 1 7.09 → 2.57, 3 24.46 → 14.56, 5 36.36 → 18.27, 6 48.81 → 20.32,
      7 32.56 → 15.77 ms (-64 / -40 / -50 / -58 / -52 %), with non-overlapping spreads. App sum 165.0 → 87.2 ms (-47 %).
      PX 2 gains less on every conv (-35..-46 %).
    - **Gate passed → the PX 4 Mali branch is enabled.**
    - Samsung (wide path): convs 1 / 3 / 5 / 6 within ±0.3 %. Stage 7 is -21.9 %, the Baseline's bimodal 2.14 / 2.75 ms,
      seen in both arms. So unchanged within noise.
    - Same outputs: on the Samsung the conv-output hashes (scratch test, see 2.1) of 368a884 and this tree match, 20/20.
      Statically, the specialized and optimized SPIR-V at subgroup 64 and 32 is instruction-identical to the previous
      shader.
    - Phone gate (both phones): kiss 29 + 1 skip, tree 18, cifar-dense 16, cifar-sparse 17, plus the cifar suites with
      real weights; exit 0.

## 4. Sparse conv: output blocking (D2.1)

- [x] 4.1 On the sparse Mali path, compute `PX` adjacent output pixels per invocation, decoding each nonzero and loading its weight once for all `PX` inputs, with a masked tail. Verify: `SparseConvK3S1P1` (6x6, 7x5, 8x8, empty and border-only rows) and `test-cifar-sparse-vk` (synthetic and real weights) pass on the Pixel, and local and rocky correctness are green.
  - *Evidence:* `new_cifar_sparse_conv2d_k3s1p1.comp` `main_mali()`, entered when `SUBGROUP_SIZE < 32`. One
    invocation is (b, out_c, oh, ow/PX rounded up). Per nonzero: one `col_idx` and one weight load and one decode, then
    `PX` guarded input loads. A tap row in the padded border is skipped, and the writes are masked for `ow0 + j >= W`.
    `kSparseConvMaliPixels = 4`. The subgroup >= 32 code (`row_sum`) is unchanged.
    - Pixel 7a: `test-cifar-sparse-vk` 17/17 synthetic and 17/17 real weights, `SparseConvK3S1P1` 6x6 / 8x8 / 7x5 with
      the empty and the border-only rows included. The PX 2 and PX 8 builds (scratch worktree, PX and contract both
      changed) pass `SparseConvK3S1P1.*:*Diff*` 15/15 on the Pixel.
    - Local: `test-cifar-sparse-vk` 17/17, `ctest --test-dir build/pc -L omp` 10/10.
    - Rocky correctness: kiss 30, cifar-dense 16 + 16, cifar-sparse 17 + 17.
- [x] 4.2 Run the phone A/B for cifar-sparse and apply the ship gate. Verify: design.md records the per-stage table for both phones and the decision.
  - *Evidence:* the Pixel ran 4 arms interleaved (8bd33e4 / PX 2 / PX 4 / PX 8, 3 rounds, 20 s cooldown, no foreign
    process on the phone before any run) and the Samsung 8bd33e4 vs PX 4. The tables are in design.md
    "Measured decisions".
    - Pixel 7a medians, previous → PX 4: stage 1 20.30 → 17.07, 3 181.35 → 80.33, 5 283.56 → 128.51, 6 398.66 → 183.69,
      7 242.44 → 110.74 ms (-16 / -56 / -55 / -54 / -54 %), with non-overlapping spreads. App sum 1190.7 → 584.8 ms
      (-51 %). PX 2: -36..-39 %, PX 8: -44..-50 % on the big four.
    - **Gate passed → the PX 4 Mali branch is enabled.**
    - Samsung: every conv spread overlaps the previous one, within the Baseline's identical-binary noise (stage 1
      bimodal 4.10 / 4.62 ms; stage 6 +14.6 % against a Baseline identical-arm gap of +9.3 %).
    - Outputs: hashes 20/20 identical to 8bd33e4. The specialized wide-path SPIR-V is identical to 8bd33e4's at 64 and 32.
    - Phone gate on both phones green (kiss 29 + 1 skip, tree 18, cifar-dense 16, cifar-sparse 17, plus real weights).

## 5. Shared-memory staging (D2.2, D2.3), each kept only if it adds beyond group 3 / 4

- [x] 5.1 Dense: map each workgroup to one output-channel block and stage its weights in shared memory in chunks, on top of the best 3.x variant. Verify: the dense tests pass on the Pixel, and the phone A/B against 3.x is recorded in design.md, keeping the change only if the four large convs are faster beyond noise.
  - *Evidence (rejected, not committed):* scratch worktree on 69954b0. `gl_WorkGroupID.y` is the channel block,
    the weights are staged 32 input channels at a time (4.5 KiB), and the contract gains a `dense_conv_k3s1p1_grid`
    that the dispatcher and test use. `test-cifar-dense-vk` 16/16 on the Pixel, synthetic and real weights.
    - Pixel A/B against 69954b0 (PX 4), 3 rounds: stage 1 2.569 → 2.691 (+4.7 %, beyond spread); 3 15.79 → 14.84
      (-6.0 %, spreads 14.55-18.00 vs 13.67-16.08 overlap); 5 / 6 / 7 -1.3 / -5.4 / -9.4 %, beyond spread.
    - Gate not met (stage 3 not beyond noise, stage 1 slower) → **not kept**. Table in design.md "Shared-memory staging".
- [x] 5.2 Sparse: map each workgroup to one output channel and stage its CSR row in shared memory in chunks, on top of the best 4.x variant. Verify: as 5.1, for cifar-sparse (including real weights).
  - *Evidence (rejected, not committed):* scratch worktree on 69954b0. `gl_WorkGroupID.y` is the output channel, and
    chunks of 256 nonzeros are decoded once per workgroup into a shared `ivec2` (offset with tap, weight bits). Plus
    `sparse_conv_k3s1p1_grid` in the contract. `test-cifar-sparse-vk` 17/17 on the Pixel with synthetic and with
    real weights.
    - Pixel A/B against 69954b0: stage 3 80.29 → 54.38 ms (-32 %), but 5 / 6 / 7 +49 / +65 / +60 % and stage 1 +11 %,
      all beyond spread. App sum 584.6 → 808.7 ms.
    - **Not kept.**
- [x] 5.3 Only if `malioc` or the 5.1 numbers show the dense Mali path still load-bound: stage an input tile with its halo in shared memory. Verify: as 5.1, or design.md notes why this step was skipped.
  - *Evidence (rejected, not committed):* the condition held. `malioc` puts the shipped dense Mali path at LS 50.00
    vs A 9.22, LS-bound, and 5.1 did not change that.
    - Scratch worktree on 69954b0: a workgroup is (image, band of rows, channel blocks); the band's rows plus halo
      are staged per channel chunk (16 KiB) and the window is read from shared memory. `test-cifar-dense-vk` 16/16
      on the Pixel, synthetic and real weights.
    - Pixel A/B against 69954b0: stage 1 +26.7 % and stage 3 +16.6 % (beyond spread), 5 / 6 / 7 -2.4 / -2.6 / -2.4 %.
    - **Not kept.** The 3.x / 4.x kernels stay as shipped.

## 6. Documentation and final gate

- [ ] 6.1 Document the Mali paths:
  - the header comments of both shaders;
  - the knob or contract notes in `docs/instruction-for-ai/03-unit-testing.md`;
  - a decision record with every measurement table under `docs/reports-for-human/`.

  Verify: each shipped and each rejected technique appears with its numbers.
- [ ] 6.2 Final gate at the last commit: local, rocky correctness, phone gate, and one `bm-gen-logs` schedule run per cifar app on the Pixel (reported, not gated). Verify: all green, with the results listed in the PR.
