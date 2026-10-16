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

- [ ] 1.1 Install `malioc` (Arm Performance Studio) on the build box, or record why it cannot be installed. Verify: `malioc --version` prints a version, or design.md's open question is answered with the reason.
- [ ] 1.2 Record the Mali baseline: per-stage medians of both apps on the Pixel and the Samsung (phone A/B with the same binary in both arms, to measure the noise), plus `malioc` reports of the two current Mali paths if 1.1 succeeded. Verify: the tables and the noise spread are in design.md under a new "Baseline" section.
- [ ] 1.3 Extend `DenseConvK3S1P1` and `SparseConvK3S1P1` with a 7x5 map, and `SparseConvK3S1P1` with a CSR row whose nonzeros all read the padded border (spec "The Mali path matches the reference at any shape"). Verify: the new cases pass on the current kernels in local, rocky correctness and the phone gate.

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
