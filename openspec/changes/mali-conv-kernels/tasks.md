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

- [ ] 2.1 Make the sparse conv `subgroup_size_dependent` (shader spec constant 0, its dispatcher, its test) with a `SUBGROUP_SIZE < 32` branch that still runs today's code. Verify: the phone gate is green, and the Samsung's sparse suite outputs are unchanged.
- [ ] 2.2 Add one conv contract header giving each kernel's outputs per invocation for a subgroup size, and size the grid from it in both apps' dispatchers and in the conv tests. Verify: a mutation that makes the header disagree with the shader fails `DenseConvK3S1P1` / `SparseConvK3S1P1` (the outputs stay at their `-1` fill), and the unmutated tree passes the phone gate.

## 3. Dense conv: output blocking (D2.1)

- [ ] 3.1 On the dense Mali path, compute `PX` adjacent output pixels per invocation (try `PX = 4` and `2`), loading each input window and weight once for all of them, with a masked tail for ragged widths. Verify: `DenseConvK3S1P1` (6x6, 7x5, 8x8) and `test-cifar-dense-vk` pass on the Pixel, and local and rocky correctness are green.
- [ ] 3.2 Run the phone A/B for cifar-dense (previous kernel vs the best `PX`) and apply the spec's ship gate. Verify: design.md records the per-stage table for both phones and the decision, the Samsung's conv stages are unchanged within noise, and the Mali branch stays enabled only if the gate passed.

## 4. Sparse conv: output blocking (D2.1)

- [ ] 4.1 On the sparse Mali path, compute `PX` adjacent output pixels per invocation, decoding each nonzero and loading its weight once for all `PX` inputs, with a masked tail. Verify: `SparseConvK3S1P1` (6x6, 7x5, 8x8, empty and border-only rows) and `test-cifar-sparse-vk` (synthetic and real weights) pass on the Pixel, and local and rocky correctness are green.
- [ ] 4.2 Run the phone A/B for cifar-sparse and apply the ship gate. Verify: design.md records the per-stage table for both phones and the decision.

## 5. Shared-memory staging (D2.2, D2.3), each kept only if it adds beyond group 3 / 4

- [ ] 5.1 Dense: map each workgroup to one output-channel block and stage its weights in shared memory in chunks, on top of the best 3.x variant. Verify: the dense tests pass on the Pixel, and the phone A/B against 3.x is recorded in design.md, keeping the change only if the four large convs are faster beyond noise.
- [ ] 5.2 Sparse: map each workgroup to one output channel and stage its CSR row in shared memory in chunks, on top of the best 4.x variant. Verify: as 5.1, for cifar-sparse (including real weights).
- [ ] 5.3 Only if `malioc` or the 5.1 numbers show the dense Mali path still load-bound: stage an input tile with its halo in shared memory. Verify: as 5.1, or design.md notes why this step was skipped.

## 6. Documentation and final gate

- [ ] 6.1 Document the Mali paths:
  - the header comments of both shaders;
  - the knob or contract notes in `docs/instruction-for-ai/03-unit-testing.md`;
  - a decision record with every measurement table under `docs/reports-for-human/`.

  Verify: each shipped and each rejected technique appears with its numbers.
- [ ] 6.2 Final gate at the last commit: local, rocky correctness, phone gate, and one `bm-gen-logs` schedule run per cifar app on the Pixel (reported, not gated). Verify: all green, with the results listed in the PR.
