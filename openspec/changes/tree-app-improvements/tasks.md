# Tasks

Gates used below:
- **OMP gate:** `ctest --test-dir build/pc -L omp` passes locally.
- **VK gate:**
  - **rocky-ryzen:** rsync tracked files to `doremy@100.97.77.80:bt-vk-campaign/`, then on rocky `bash -s` → `cmake --preset vulkan && cmake --build --preset vulkan`. There `ctest -L vulkan` (NOT `-L vk`) and `test-tree-vk --device minipc` pass, also with `BT_VK_SUBGROUP_SIZE=32`.
  - **Phones:** `scripts/run-on-android.sh <serial> test-tree-vk test-kiss-vk` passes on the Pixel 7a (`3A021JEHN02756`) and the Samsung (`R5CY21Y3VEV`), and on the Samsung also with `BT_VK_SUBGROUP_SIZE=32`.
- **CUDA gate:** `just build-jetson` (`bt-cross:7.2`) then `scripts/run-on-jetson.sh` pass on duck-stable (all `test-*-cu`).
- **Phone A/B:** build `bm-prof-tree-vk` for the baseline and the change from two worktrees (Android preset), push
  both to each phone, and run them alternately with `BT_PROF_SCENARIO=isolated`, 3 rounds, 20 s apart. Report each
  stage's median and min-max per phone. The Samsung's `taskset` takes a hex mask.
- **rocky-ryzen is for correctness only:** no profiler, benchmark or timing run there (the owner's rule).

Shaders are compiled at build time; nothing generated is committed.

## 1. Correctness reproducers (topic A, design A1)

- [x] 1.1 Add a Morton boundary test: points at `kMinCoord + kRange` and at a value whose scaled product rounds to 1024.0. Assert each axis quantizes to 1023 on OMP, and compare the CUDA/VK codes with OMP. Verify: it FAILS on the current code (record the failing values), and lives in the tree test files.
  - *How the reproducers are committed (all of group 1):* each test that fails on the code before its fix is
    committed with a `DISABLED_` prefix, so every commit's gate stays green, and the group that fixes it removes the
    prefix (2.1, 3.1, 3.2). The failing runs below are `--gtest_also_run_disabled_tests` (or the same tests before
    the prefix was added) on 583080c plus the group 1 tests.
  - *Tests:* `RunAndCheckMortonBoundary<Runner>` (`apps/tree/tree_diff_oracle.hpp`) runs stage 1 on a 4096-point input
    whose first points are (1024, 1024, 1024), (0, 0, 0), (1023.99994, ...), (1025, -1, 512) and (512, 1024, 0); it
    asserts the OMP codes 0x3fffffff, 0, 0x3fffffff, 0x29249249, 0x1a492492 and compares the backend's codes with
    OMP's as a multiset. It runs as `TreeDiffOmp.Stage1_MortonBoundary`, `TreeDiffCuda.Stage1_MortonBoundary` and
    `TreeDiffVulkanChained.Stage1_MortonBoundary`. `TreeMortonOmp.UpperBoundQuantizesTo1023` decodes the code of
    (1024, 1024, 1024). `TreeMortonOmp.ScaledValueRoundingTo1024QuantizesTo1023` uses min -512 / range 1024 and
    x = 511.99997f < 512, for which `(x - min) / range * 1024 == 1024.0f`. With the app's own constants (min 0, range
    1024, a power of two) no coordinate below the top rounds to 1024.0f, so that case is OMP-only.
  - *Fails on the current code:*
    - OMP (build box): both `TreeMortonOmp` tests decode 0 on every axis instead of 1023; `Stage1_MortonBoundary`
      gets 0 for point 0 (want 1073741823), 843654291 for point 3 (want 690262601), 134217728 for point 4 (want
      441001106).
    - Vulkan (rocky-ryzen, `test-tree-vk --device minipc`): the same three OMP codes, and the multiset differs from OMP
      (`element mismatch at index 2033 of 4096 (ref=537029823, out=536870913)`): x86 OMP turns the -1 coordinate into
      1023, the 780M into 0 (a negative float to `uint` is undefined in both C++ and GLSL).
    - CUDA (duck-stable): point 0 0, point 3 536870913 (aarch64 OMP turns -1 into 0, unlike x86), point 4 134217728;
      CUDA and OMP agree with each other there.
- [x] 1.2 Add a leaf-race test: a constructed input where many BRT nodes link leaves into the same octree node. Run OMP and CUDA stage 7 repeatedly (≥ 100×) and compare the leaf masks with a serial reference. Verify: a lost bit is observed at least once on the current code, or record that the race couldn't be triggered, with the iteration count.
  - *Test:* `CountLostLeafBits<Runner>(runner, reps)` (`tree_diff_oracle.hpp`). Input: one point per finest-level
    cell of a 32^3 block, i.e. 4096 clusters of 8 points, one per octant of a 2x2x2 cell. Stages 1-6 run on OMP:
    32768 unique codes, 4681 octree nodes, 4096 of which get 8 leaf links each from 4 BRT nodes with adjacent indices
    (32768 cleared bits in the single-threaded reference). Each repetition resets the masks to 0xFF, runs the backend's
    stage 7 and counts bits the reference cleared but the backend did not. (Spreading the clusters out instead
    overflowed the octree buffers, which hold 0.5 x n_input nodes.)
  - *CUDA* (`TreeLeafRaceCuda.ConcurrentLeafLinksKeepEveryBit`, duck-stable, 100 runs): **2,457,600 of 3,276,800 bits
    lost**, i.e. 6 of 8 bits per node in every run.
  - *OMP, `run_stage_7` as shipped* (`TreeLeafRaceOmp.ConcurrentLeafLinksKeepEveryBit`, 12 threads on the build box):
    0 lost bits in 100 runs, and 0 in 30 x 100 = 3000 runs. Its static schedule gives each thread one contiguous range,
    so links into one node meet only at the 11 range ends: the race could not be triggered through `run_stage_7`.
  - *OMP, interleaved* (`TreeLeafRaceOmp.InterleavedLeafLinksKeepEveryBit`: `process_link_leaf` under
    `schedule(static, 1)`, which hands adjacent BRT nodes to different threads): 2,323,527 / 2,338,911 / 2,336,529 of
    3,276,800 bits lost in three 100-run invocations. A scratch `schedule(static, 1)` on `run_stage_7` itself (not
    committed) lost 373,945 / 198,153 / 183,819.
- [x] 1.3 Add an oracle self-test: give `CheckStage7Topology` an output whose leaf masks are reset to "no leaves". Verify: on the current code it either fails (good, the check works) or passes (the check is vacuous). Record which.
  - *Tests* (`apps/tree/omp/test_main.cpp`): two OMP outputs of the same input must pass `CheckStage7Topology`; then
    one output's leaf masks are reset and the check must fail. `TreeOracleSelfTest.MissingLeafLinksFail` resets them
    to 0xFF ("no leaves", a leaf slot is a zero bit). `TreeOracleSelfTest.SpuriousLeafLinksFail` resets them to 0
    (every free octant a leaf).
  - *Result on the current code:* **MissingLeafLinksFail passes** (the check fails, as it should), but
    **SpuriousLeafLinksFail fails**: the check accepts an output that marks every free octant a leaf. The leaf check
    is vacuous, because the golden's masks are themselves all zero (see 1.4).
- [x] 1.4 Explain why `TreeDiffVulkan.Stage7_Octree` passes though Vulkan never runs `k_LinkLeafNodes`: inspect `u_oct_child_leaf_mask_s7_out`'s initial value and what Vulkan writes to it. Verify: the explanation is written under this task and in design A3, and the test from 1.3 or 3.2 now fails for Vulkan on the current code.
  - *Explanation:* no backend ever sets the leaf masks to all-ones. Every mask buffer is a value-initialized
    `std::pmr::vector<int32_t>` (`tree::AppData`'s `u_oct_child_leaf_mask_s7`, which the golden is built in, and
    `SafeAppData`'s `u_oct_child_leaf_mask_s7_out`), so it starts at 0, and `set_leaf` only clears bits. On OMP and CUDA
    the leaf links therefore change nothing in the mask; on Vulkan `main` never calls `k_LinkLeafNodes`, and the
    optimized SPIR-V (`glslc -O`) has no variable for `ChildLeafMasks` (binding 4, nor for 10-12), so the `_out` mask also stays 0. Both sides are
    all zeros, and `OctreeEdgeSet` reads a zero bit as a leaf, so ref and out both list every octant without a child
    node as a leaf slot, and they agree. Measured on the default input (scratch probe, not committed): 146,708 octree
    nodes, **0 non-zero leaf masks** in the golden, 307,151 leaf links that clear nothing, 1,026,957 octants listed
    as leaf slots on each side.
  - *Now fails for Vulkan:* with the masks initialized to 0xFF (a scratch edit of the two constructors, the 3.2 fix, not
    committed in this group), on rocky `TreeDiffVulkan.Stage7_Octree` fails with `LEAF child slots differ (307151
    only-in-ref, 0 only-in-out; ref_leaf_slots=307151 out_leaf_slots=0)` and `TreeDiffVulkanChained.Stage7_Octree`
    with `307144 only-in-ref, 0 only-in-out`. The explanation is also in design A3.

## 2. Morton clamp (topic A, design A2)

- [x] 2.1 Clamp the quantized coordinate to `[0, 1023]` in OMP `func_morton.hpp`, CUDA `func_morton.cuh` and `tree_morton.comp`, identically. Verify: 1.1 passes on all three backends, and stage-1 output on the default input is byte-identical to before (diff recorded).
  - *Change:* OMP `quantize_coord` = `static_cast<uint32_t>(std::clamp(v_scaled, 0.0f, 1023.0f))`, CUDA the same with
    `fminf(fmaxf(...))`, the shader `uint(clamp(a, 0.0, 1023.0))`; the test reference `MortonRef` (`tree_ref.hpp`)
    clamps too. The clamp is on the float, before the cast, rather than design A2's `min(uint(v), 1023u)` after it:
    the two agree for every v >= 0, and the float clamp also gives 0 below the bottom, as the spec requires, where the
    cast of a negative float is undefined (1.1 saw x86 give 1023 and the 780M 0). Recorded in design A2.
  - *1.1 passes on all three backends* (its five tests enabled): OMP on the build box (`test-tree-omp` 19 passed),
    `TreeDiffVulkanChained.Stage1_MortonBoundary` on rocky (64 and pinned 32), the Pixel 7a and the Samsung (64 and
    pinned 32), `TreeDiffCuda.Stage1_MortonBoundary` on duck-stable.
  - *Default input unchanged:* a scratch probe (not committed) ran OMP stage 1 on the default input (307,200 points, the
    first `tree::AppData` of the process, max coordinate 1023.99866) and hashed `u_morton_keys_s1` in order:
    FNV-1a `2adb067305d8595a` before (4619d05) and after; min/max code 12019 / 1073741425 both times.
- [x] 2.2 Run the OMP gate, the VK gate and the CUDA gate. Verify: all green.
  - *OMP:* `ctest --test-dir build/pc -L omp` 10/10 (`test-tree-omp` 19 passed, 2 disabled: 3.1 and 3.2).
  - *VK, rocky-ryzen* (`~/bt-tree-gate`, native `vulkan` preset, `VK_LOADER_LAYERS_DISABLE='~all~'`): `ctest -L vulkan`
    7/7; `test-tree-vk --device minipc` 19/19, and 19/19 with `BT_VK_SUBGROUP_SIZE=32`.
  - *VK, phones* (`android` preset, NDK r30; pushed to `/data/local/tmp/bt-tree` with `libc++_shared.so`, run with
    `--device <serial>`, the same steps as `scripts/run-on-android.sh`): Pixel 7a `test-tree-vk` 19/19 and
    `test-kiss-vk` 29/29; Samsung 19/19 and 29/29, and 19/19 and 29/29 with `BT_VK_SUBGROUP_SIZE=32`.
  - *CUDA, duck-stable* (`bt-cross:7.2` on rocky with `-DBT_ENABLE_VULKAN=OFF`, `scripts/run-on-jetson.sh`):
    `test-tree-cu` 8 passed (1 disabled: 3.1), `test-cifar-dense-cu` 12, `test-cifar-sparse-cu` 12.

## 3. Atomic leaf masks and Vulkan leaf linking (topic A, design A3)

- [x] 3.1 Make `set_leaf`'s clear atomic: `#pragma omp atomic` in `func_octree.hpp`, `atomicAnd` in `07_octree.cu`. Verify: 1.2 shows no lost bits over ≥ 100 repetitions on OMP and CUDA; the OMP gate and CUDA gate are green.
  - *Change:* `set_leaf`'s clear is `#pragma omp atomic` (`func_octree.hpp`) and `atomicAnd` (`07_octree.cu`); the
    child slot store stays plain. `TreeLeafRaceOmp.InterleavedLeafLinksKeepEveryBit` and
    `TreeLeafRaceCuda.ConcurrentLeafLinksKeepEveryBit` are enabled.
  - *1.2 now:* 0 lost bits in 100 runs each: OMP through `run_stage_7` and interleaved (12 threads, build box), CUDA
    `k_LinkLeafNodes` (duck-stable). Before: CUDA 2,457,600, OMP interleaved ~2.3 million (task 1.2).
  - *Gates:* OMP `ctest -L omp` 10/10 (`test-tree-omp` 21/21, nothing disabled); CUDA on duck-stable `test-tree-cu`
    9/9, `test-cifar-dense-cu` 12/12, `test-cifar-sparse-cu` 12/12.
- [x] 3.2 If 1.3/1.4 showed the leaf-slot check is vacuous, fix `CheckStage7Topology` so missing leaf links fail. Verify: the oracle self-test fails on a no-leaves output, and OMP and CUDA still pass.
  - *Change:* 1.3/1.4 showed the leaf check vacuous because the masks were never initialized, so the fix starts there:
    `tree::AppData::u_oct_child_leaf_mask_s7` and `SafeAppData::u_oct_child_leaf_mask_s7_out` are allocated filled with
    0xFF. `CheckStage7Topology` (both overloads) now also names up to 5 missing and 5 spurious leaf slots (parent
    corner, cell, octant) in its failure. `TreeOracleSelfTest.SpuriousLeafLinksFail` is enabled, and
    `MissingLeafLinksFail` also requires `Missing leaf slots: (node corner` in the message.
  - *Self-tests:* both pass on OMP: removing the leaf links fails the check, and so does marking every free octant a
    leaf.
  - *Vulkan before 3.3* (rocky, this oracle, the old shader): 5 tests fail, `TreeDiffVulkan.Stage7_Octree`,
    `TreeDiffVulkanChained.Stage7_Octree`, and the three tests that run `CheckStage7` (lifetime, concurrent sequences,
    prerecord), e.g. `LEAF child slots differ (307151 only-in-ref, 0 only-in-out; ref_leaf_slots=307151
    out_leaf_slots=0) ... Missing leaf slots: (node corner 0,0,0 cell 32, octant 2) (node corner 0,0,0 cell 32,
    octant 3) ...`.
  - *OMP and CUDA still pass:* `test-tree-omp` 21/21, `test-tree-cu` 9/9 (gates of 3.1).
- [x] 3.3 Wire leaf linking into Vulkan stage 7: a second dispatch after a compute→compute barrier, with `SetLeaf` using `atomicAnd` (design A3). Verify: `TreeDiffVulkan.Stage7_Octree` and `TreeDiffVulkanChained.Stage7_Octree` pass with the fixed oracle on rocky and both phones.
  - *Change:* `tree_build_octree.comp` gets a push-constant `phase`; `main` runs `k_MakeOctNodes` in phase 0 and
    `k_LinkLeafNodes` in phase 1. `SetLeaf` uses `atomicAnd`; the radix-tree walk is shared by both leaves and capped
    like OMP's; bindings 10-12 are now `readonly` (`just check-shaders`: OK). Both `record_stage_7` overloads record
    the pipeline twice (bind, push constants with the phase, dispatch). Phase 1 writes the children buffer phase 0 wrote,
    so the `Sequence`'s tracking records the compute→compute barrier between them. New
    `TreeLeafRaceVulkan.ConcurrentLeafLinksKeepEveryBit` (the 1.2 input, 100 runs) covers the spec's Vulkan case.
  - *rocky-ryzen:* `test-tree-vk --device minipc` 20/20 at 64 and with `BT_VK_SUBGROUP_SIZE=32`; `ctest -L vulkan`
    7/7; `test-pipeline-chained-vk` (compares the leaf masks element-wise) 1/1. Sync validation
    (`BT_VK_VALIDATION=1`, synchronization validation with `VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true`,
    layer loaded per `VK_LOADER_DEBUG=layer`): 20/20, 0 `Validation Error`, 0 `SYNC-HAZARD`.
  - *Phones:* Pixel 7a `test-tree-vk` 20/20, `test-kiss-vk` 29/29; Samsung 20/20 and 29/29, also with
    `BT_VK_SUBGROUP_SIZE=32`.
  - *Mutation (scratch, not committed):* a plain `&=` in `SetLeaf` loses 2,457,600 bits in `TreeLeafRaceVulkan` on rocky
    (3 of 3 invocations), and both Vulkan stage-7 differential tests fail.
- [x] 3.4 Measure stage-7 time before and after with the phone A/B (both phones) and on duck-stable CUDA. Verify: the numbers are recorded; any increase comes from the added leaf work.
  - *Method:* `bm-prof-tree-vk`, `BT_PROF_SCENARIO=isolated`, `--benchmark_filter='vulkan/'`, Android preset (NDK r30),
    arms built from worktrees and pushed to `/data/local/tmp/bt-tree`; process pinned to the big cores (`taskset -c 6,7`
    Pixel, `taskset 180` Samsung); arms rotated, 3 rounds, 20 s sleep before every run, no other `bm-`/`test-` process
    on the phone. Reported per stage: median over runs of each run's p50 (GPU timestamps), min-max of the p50s,
    n = runs x samples. A = 3d6d919 (before group 3), B = group 3, C = group 3 without the phase-1 dispatch (scratch,
    to attribute the change). Governor `sched_pixel` (Pixel) / `energy_aware` (Samsung); the GPU clock is not readable
    (`gpu_clock_mhz` -1) on either.
  - *Phones, stage 7 (ms):*

    | phone | A median (min-max) | B median (min-max) | C median (min-max) | B vs A | C vs A |
    |---|---|---|---|---|---|
    | Pixel 7a | 1.294 (1.292-1.294, n=3x230-232) | 3.014 (3.011-3.014, n=3x99) | 1.304 (1.304-1.304, n=3x224-233) | +133.0 % | +0.8 % |
    | Samsung | 0.822 (0.340-2.919, n=3x392-727) | 2.049 (1.073-2.939, n=3x145-315) | 0.834 (0.475-2.908, n=3x202-627) | +149.3 % | +1.4 % |

    The increase is the leaf pass: without it (C) stage 7 is A's time. The Samsung is bimodal as in earlier
    reports, so its spreads overlap. An earlier A/B session (A and B only) gave the Pixel 1.296 → 2.429 ms and the
    Samsung 1.985 → 1.602 ms (spreads 0.552-2.765 / 1.226-1.783); in that session the Pixel's unchanged stages 1 and 3-6
    also differed by 16-55 % between arms, while in the three-arm session above they agree within 5 %: between-session
    noise on the Pixel, which 4.1 quantifies.
  - *CUDA, duck-stable* (`bm-prof-tree-cu`, cross-built with `-DBT_ENABLE_VULKAN=OFF`, isolated, `--benchmark_filter='cuda/'`,
    3 rounds rotated, 20 s apart, governor `schedutil`; C = group 3 with a plain `&=` in `set_leaf`): stage 7
    A 4.434 (4.433-4.458, n=3x64-67), B 6.801 (6.766-6.809, n=3x43-44), C 4.440 (4.429-4.476, n=3x66-67) ms; B +53.4 %,
    C +0.1 %; stages 1-6 within 1.1 %. CUDA links no extra leaves: the whole increase is the `atomicAnd` on the zero-copy
    pinned buffers, i.e. the race fix itself. The OMP stage 7 on the Jetson's six cores (`little` tier, same runs without
    the filter): A 2.146 (2.021-2.390), B 2.268 (2.263-2.703) ms, +5.7 %, spreads overlapping.
- [x] 3.5 Add entries to `docs/reports-for-human/bugs-found.md` for the Morton wrap, the leaf race and the missing Vulkan leaf links. Verify: each entry links its reproducer and fix commit.
  - *Evidence:* `docs/reports-for-human/bugs-found.md` §13 (Morton wrap: reproducers 4619d05, fix 3d6d919), §14 (leaf
    race and uninitialized masks: reproducers 4619d05, fix the group 3 commit ("tree: atomic leaf-mask clears, initialized leaf masks and Vulkan leaf links")) and §15 (missing Vulkan leaf links: reproducer run
    under 3.2, fix the same commit). The group 3 commit cannot name its own hash; §14/§15 cite it by subject.

## 4. Sort baseline and workgroup-count knob (topic B, design B1)

- [ ] 4.1 Record the baseline on the phones: the phone A/B with the same binary in both arms (this change's tree after group 3, `kRadixNumWorkgroups` 64), recording each phone's governor and GPU clock. Verify: the stage-2/3/7 medians and the identical-binary spread of both phones are written here.
- [x] 4.2 Sweep `kRadixNumWorkgroups` and keep the best value in both `VkAppData_Safe` and `VkAppData`.
  - *Evidence (done before this change was applied, PR #26):* 256 → 64. The 780M sweep (32 / 64 / 128 / 256: stage 2
    0.408 / 0.404 / 0.421 / 0.773 ms) is in `docs/reports-for-human/2026-10-15-vk-shader-profile.md` 4.4; on the phones
    stage 2 measured -20 % (Pixel 7a) and -42 % (Samsung), same report section 9. `test-tree-vk` passed on rocky and
    both phones at 64.

## 5. Global prefix once per pass (topic B, design B2/B3)

- [ ] 5.1 Add the per-pass histogram-table scan, or the upfront all-digit histogram (whichever 4.x favours; record the choice in design B2), and make `multi_radixsort.comp` read only its own offsets. Verify: `test-tree-vk` is exact on rocky at 64 and forced 32.
- [ ] 5.2 Run the full VK gate. Verify: green, including the Pixel 7a at subgroup 16.
- [ ] 5.3 Measure stage 2 with the phone A/B against 4.1. Verify: before/after recorded for both phones, with no regression beyond the spread.

## 6. Scatter and histogram efficiency (topic B, design B4)

- [ ] 6.1 Add a local LDS sort by digit before the scatter, sized for the smallest `maxComputeSharedMemorySize` across the three GPUs (log each value). Verify: the VK gate is green, and the sizes are recorded.
- [ ] 6.2 Add subgroup-aggregated histogram atomics in `multi_radixsort_histograms.comp`. Verify: the VK gate is green.
- [ ] 6.3 Measure stage 2 with the phone A/B against 4.1. Verify: the median is lower beyond noise on both phones (spec "The sort is faster on the phones"); if not, the change that did not help is reverted and the numbers are recorded.

## 7. Full subgroups (topic B, design B5)

- [ ] 7.1 Confirm that the sort pipelines (`multi_radixsort`, plus any new subgroup-dependent sort/scan kernel from groups 4–8) are created as subgroup-size-dependent, so they get `REQUIRE_FULL_SUBGROUPS` from the engine-wide mechanism that `vk-engine-runtime` lands first (that change owns the feature enablement and the `Algorithm` flag; do not re-implement it here). Verify: the debug log on each VK target shows the flag on the sort pipelines, and `BT_VK_VALIDATION=1 test-tree-vk` on rocky reports 0 validation errors.
- [ ] 7.2 Check that the tree suites pass exactly on a device path without `computeFullSubgroups`, using the engine change's fallback. Verify: on rocky, force the fallback with the engine's test hook (or note the device coverage if no such hook exists), then run `test-tree-vk` 15/15.

## 8. Stage-3 compaction (topic B, design B6)

- [ ] 8.1 Implement compaction in fewer than 5 dispatches. Keep `record_device_scan` unchanged for stage 6 unless both stages re-verify. Verify: `test-tree-vk` stage 3 and stage 6 are exact on all VK targets.
- [ ] 8.2 Measure stage 3 with the phone A/B against 4.1. Verify: the after median is not higher beyond the spread on either phone, and the dispatch count is recorded (spec "Compaction is not dispatch-bound").

## 9. Final gate and profile refresh

- [ ] 9.1 Run the OMP, VK and CUDA gates at the final commit, plus `BT_VK_VALIDATION=1` runs of `test-tree-vk` on rocky. Verify: all green, with 0 validation errors.
- [ ] 9.2 Refresh tree × VK profiling on the phones (`bm-prof-tree-vk`, isolated and interference, 2 runs each, into `data/profiling/<serial>/tree/vulkan/`). Verify: the new per-stage times are recorded, with the run noted here.
