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
    race and uninitialized masks: reproducers 4619d05, fix 5477c56) and §15 (missing Vulkan leaf links: reproducer run
    under 3.2, fix 5477c56). The group 3 commit could not name its own hash, so it cited itself by subject; the group 4
    commit put in 5477c56.

## 4. Sort baseline and workgroup-count knob (topic B, design B1)

- [x] 4.1 Record the baseline on the phones: the phone A/B with the same binary in both arms (this change's tree after group 3, `kRadixNumWorkgroups` 64), recording each phone's governor and GPU clock. Verify: the stage-2/3/7 medians and the identical-binary spread of both phones are written here.
  - *Binary:* `bm-prof-tree-vk` at 5477c56 (group 3, `kRadixNumWorkgroups` 64) in both arms; same method as 3.4 (isolated,
    `--benchmark_filter='vulkan/'`, big cores pinned, 3 rounds, 20 s before every run, no other `bm-`/`test-` process).
    The GPU clock was sampled every 0.2 s during each run (Pixel `/sys/class/misc/mali0/device/cur_freq`, Samsung
    `/sys/kernel/gpu/gpu_clock`; bm-prof's own `gpu_clock_mhz` is -1 on both).
  - *Session 1, default sampling knobs:* identical arms differed by up to 50 % in a stage's median (Pixel stage 5
    0.194 vs 0.291 ms, stage 7 2.026 vs 2.998; Samsung stage 5 0.194 vs 0.127). Stage 2: Pixel 4.945 (3.425-4.998) vs
    5.002 (4.985-6.317) ms, Samsung 1.765 (1.732-1.933) vs 1.954 (1.285-2.296). GPU clock Pixel 202-762 MHz (median
    302 / 251), Samsung 252-900 MHz (median 252). With the defaults (0.3 s per cell, 150 ms GPU warm-up) a short stage
    gets few samples while the GPU clock moves, so each run's p50 moves with it.
  - *Session 2, the knobs used from here on:* `BT_PROF_BUDGET_S=1.0 BT_PROF_MAX_CELL_S=3 BT_PROF_GPU_WARM_MS=500` (the
    budget knobs as in the Mali conv report, and a longer GPU warm-up), so a stage gets 3-7x the samples. This is the
    4.1 baseline the later A/Bs compare with; every later A/B runs a 5477c56 arm in the same session with these knobs.

    | phone | stage | A median (min-max) ms | B median (min-max) ms | B vs A |
    |---|---|---|---|---|
    | Pixel 7a | 2 | 7.200 (6.323-7.214, n=3x138-158) | 6.326 (6.310-7.226, n=3x138-158) | -12.1 % |
    | Pixel 7a | 3 | 1.651 (1.645-1.659, n=3x601-712) | 1.744 (1.674-2.013, n=3x592-603) | +5.6 % |
    | Pixel 7a | 7 | 3.015 (3.014-3.017, n=3x330-333) | 3.014 (3.010-3.017, n=3x331-333) | -0.0 % |
    | Samsung | 2 | 2.528 (2.415-3.068, n=3x328-414) | 2.495 (1.327-2.535, n=3x391-743) | -1.3 % |
    | Samsung | 3 | 1.131 (0.990-3.254, n=3x308-902) | 2.158 (1.296-3.201, n=3x306-1213) | +90.8 % |
    | Samsung | 7 | 1.914 (1.840-5.152, n=3x197-544) | 1.925 (1.915-2.111, n=3x472-929) | +0.6 % |

    Other stages: Pixel 1, 4, 5, 6 within 0.5 %; Samsung 1 +90 %, 5 -35 %, 4 and 6 within 2 %.
  - *Identical-binary spread (session 2, all six runs):* Pixel stage 2 6.310-7.226 ms, bimodal (each run's p50 is near
    6.32 or near 7.21, 14 % apart), stage 3 1.645-2.013, stage 7 3.010-3.017. Samsung stage 2 1.327-3.068, stage 3
    0.990-3.254, stage 7 1.840-5.152: bimodal by up to 2x, as in earlier reports. So the Pixel can show a stage-2 change
    beyond noise only if it is larger than the 14 % mode gap, and the Samsung only one larger than about 2x.
  - *Governor and GPU clock:* Pixel CPU governor `sched_pixel`, GPU clock 202-400 MHz (median 202, 367 samples); Samsung
    `energy_aware`, GPU governor `conservative`, GPU clock 252-900 MHz (median 252, 380 samples). Neither phone lets the
    shell user lock clocks, so the GPU runs these short stages mostly at its lowest clock.
- [x] 4.2 Sweep `kRadixNumWorkgroups` and keep the best value in both `VkAppData_Safe` and `VkAppData`.
  - *Evidence (done before this change was applied, PR #26):* 256 → 64. The 780M sweep (32 / 64 / 128 / 256: stage 2
    0.408 / 0.404 / 0.421 / 0.773 ms) is in `docs/reports-for-human/2026-10-15-vk-shader-profile.md` 4.4; on the phones
    stage 2 measured -20 % (Pixel 7a) and -42 % (Samsung), same report section 9. `test-tree-vk` passed on rocky and
    both phones at 64.

## 5. Global prefix once per pass (topic B, design B2/B3)

- [x] 5.1 Add the per-pass histogram-table scan, or the upfront all-digit histogram (whichever 4.x favours; record the choice in design B2), and make `multi_radixsort.comp` read only its own offsets. Verify: `test-tree-vk` is exact on rocky at 64 and forced 32.
  - *Choice (design B2):* the per-pass scan. An upfront all-digit histogram only gives each digit's global bin totals;
    a workgroup's offset inside a bin still needs the counts of the workgroups before it, for every pass, so it would
    not remove the per-pass table read. New `multi_radixsort_scan.comp` (one workgroup, one invocation per bin) turns
    the 256 x 64 histogram table into every workgroup's global offsets in place; it carries the subgroup scan over the
    bins that each scatter workgroup used to run, so it is the subgroup-size-dependent sort kernel now (spec constant 0,
    `subgroup_size_dependent()`). `multi_radixsort.comp` reads only its own 256 offsets and no longer uses subgroups.
    Stage 2 is 12 dispatches (histogram, scan, scatter per digit). `test-kiss-vk`'s reflection and subgroup tests that
    used `multi_radixsort` as the subgroup-dependent example use `multi_radixsort_scan` now; `01-hardware.md` names it.
  - *rocky-ryzen:* `test-tree-vk --device minipc` 20/20 at 64 and with `BT_VK_SUBGROUP_SIZE=32`; `test-kiss-vk` 30/30
    at both.
- [x] 5.2 Run the full VK gate. Verify: green, including the Pixel 7a at subgroup 16.
  - rocky-ryzen: `ctest -L vulkan` 7/7, `test-tree-vk` 20/20 at 64 and pinned 32. Pixel 7a (subgroup 16) `test-tree-vk`
    20/20, `test-kiss-vk` 29/29; Samsung 20/20 and 29/29, also with `BT_VK_SUBGROUP_SIZE=32`.
- [x] 5.3 Measure stage 2 with the phone A/B against 4.1. Verify: before/after recorded for both phones, with no regression beyond the spread.
  - *Method:* as 4.1 session 2 (A = 5477c56, B = group 5, same session, knobs `BT_PROF_BUDGET_S=1.0
    BT_PROF_MAX_CELL_S=3 BT_PROF_GPU_WARM_MS=500`). Stage 2 (ms):

    | phone | A median (min-max) | B median (min-max) | B vs A |
    |---|---|---|---|
    | Pixel 7a, session 1 | 7.212 (6.330-7.304, n=3x135-158) | 8.359 (6.414-8.477, n=3x117-151) | +15.9 % |
    | Pixel 7a, session 2 | 7.214 (7.209-8.344, n=3x119-139) | 8.363 (7.231-8.366, n=3x119-138) | +15.9 % |
    | Samsung | 2.557 (2.548-2.562, n=3x389-730) | 2.510 (1.900-2.523, n=3x416-549) | -1.9 % |

  - *Reading the Pixel:* each run's samples are tight (cv 0.2-1.3 %), but a run lands on one of three levels. Over both
    sessions A's six p50s are 6.33 / 7.21 / 7.21 / 7.21 / 7.30 / 8.34 and B's 6.41 / 7.23 / 8.36 / 8.36 / 8.36 / 8.48: the
    same levels, B more often on the top one. Level for level B is 0.2-1.3 % slower. The GPU clock read 202 MHz (median)
    in both arms; the memory-interface clock is not readable by the shell user, so what picks the level is not known.
    The Samsung was quiet in this session (all stages within 1 %, stage 2 A spread 2.548-2.562).
  - *Verdict:* no regression beyond the spread on either phone (the spreads overlap), so group 5 stays. It also gives
    no measurable gain: at 64 workgroups the table re-read is too small to show on the phones. Other stages unchanged
    (Pixel stage 4 and 6 moved between levels in session 1 only, within 0.3 % in session 2).

## 6. Scatter and histogram efficiency (topic B, design B4)

- [x] 6.1 Add a local LDS sort by digit before the scatter, sized for the smallest `maxComputeSharedMemorySize` across the three GPUs (log each value). Verify: the VK gate is green, and the sizes are recorded.
  - *Change (reverted in 6.3, see there):* `multi_radixsort.comp` ranked each 256-key block by digit with the existing
    `bin_flags`, scanned the block's 256 bin counts (subgroup scan, so the scatter was subgroup-size dependent again),
    placed the keys in shared memory in digit order and wrote them out so that invocations with consecutive slots of a
    bin write consecutive addresses. Shared memory ~11 KiB (bin flags 8 KiB, offsets, block offsets and keys 1 KiB
    each).
  - *maxComputeSharedMemorySize* (now in the engine's `Subgroup size ...` debug line, `SPDLOG_LEVEL=debug test-kiss-vk
    --gtest_filter=VulkanEngineTest.EngineInitialization`): Radeon 780M (RADV) 65536, Mali-G710 32768, Xclipse 940
    32768. Sized for 32 KiB; it used ~11 KiB.
  - *VK gate (with 6.1):* rocky `test-tree-vk` 20/20 and `test-kiss-vk` 30/30 at 64 and pinned 32; Pixel 7a 20/20 and
    29/29; Samsung 20/20 and 29/29, also pinned 32.
- [x] 6.2 Add subgroup-aggregated histogram atomics in `multi_radixsort_histograms.comp`. Verify: the VK gate is green.
  - *Change (reverted in 6.3):* in `multi_radixsort_histograms.comp`, each invocation looped: take the bin of the first
    invocation still looping (`subgroupBroadcastFirst`), ballot the invocations with that bin, one `atomicAdd` of the
    ballot's bit count by an elected invocation, and the counted invocations leave. Correct for any subgroup size and
    any grouping of the loop's invocations, so the kernel needed no specialization.
  - *VK gate (6.1 + 6.2):* rocky `ctest -L vulkan` 7/7, `test-tree-vk` 20/20 at 64 and pinned 32; Pixel 7a 20/20 and
    29/29; Samsung 20/20 and 29/29, also pinned 32.
- [x] 6.3 Measure stage 2 with the phone A/B against 4.1. Verify: the median is lower beyond noise on both phones (spec "The sort is faster on the phones"); if not, the change that did not help is reverted and the numbers are recorded.
  - *Method:* as 4.1 session 2, five arms rotated in one session: A = 5477c56 (the 4.1 baseline), G5 = fce8948
    (group 5), S1 = G5 + 6.1, S2 = G5 + 6.2, S12 = G5 + 6.1 + 6.2. Stage 2 (ms):

    | phone | A | G5 | S1 (6.1) | S2 (6.2) | S12 (both) |
    |---|---|---|---|---|---|
    | Pixel 7a | 5.469 (4.990-5.902) | 5.529 (5.474-5.927), +1.1 % | 6.759 (6.199-7.287), +23.6 % | 9.676 (8.242-9.692), +76.9 % | 7.140 (6.476-8.801), +30.6 % |
    | Samsung | 2.562 (2.556-2.574) | 2.473 (1.814-2.623), -3.5 % | 2.621 (2.486-2.771), +2.3 % | 2.848 (2.846-3.104), +11.2 % | 2.870 (2.869-2.871), +12.0 % |

    n = 3 x 101-198 (Pixel), 3 x 318-548 (Samsung). Other stages within 1.3 % except the Pixel's stage 6, which sat on
    its 1.25 ms level in the three S arms and on its 1.04 ms level in A and G5 (its code is the same in all five).
    GPU clock median 251 MHz (Pixel) / 252 MHz (Samsung) in every arm.
  - *Verdict:* neither change makes stage 2 faster; both make it slower on both phones (6.2 beyond the spread on both).
    Reverted: `multi_radixsort.comp`, `multi_radixsort_histograms.comp` and the dispatcher are back at fce8948. Kept: the
    `maxComputeSharedMemorySize` value in the engine's debug line (6.1's sizing record). Inferred, not measured: with
    256 keys per block and 256 bins a bin run in a block is about one key long, so the local sort buys no coalescing
    and costs two more barriers per block; the Morton digits are close to uniform, so a subgroup rarely holds equal
    bins and the ballot loop runs about once per invocation.
  - *Workgroup count re-swept on top of group 5 (design B1 "keep the best"; scratch builds of fce8948 with
    `kRadixNumWorkgroups` 32 / 128 / 256 in both appdata classes; `test-tree-vk` stages 2, 3 and 7 pass with each on both
    phones).* The 64 was chosen on the 780M while every scatter workgroup re-read the whole table; group 5 removed that
    cost, so more workgroups might have paid now. Stage 2 (ms), same method, one session:

    | phone | A (5477c56, 64) | G5 (64) | 32 | 128 | 256 |
    |---|---|---|---|---|---|
    | Pixel 7a | 5.425 (5.417-5.893) | 5.487 (4.532-5.494), +1.1 % | 7.425 (6.858-8.484), +36.9 % | 5.525 (5.120-5.957), +1.9 % | 6.048 (4.590-7.090), +11.5 % |
    | Samsung | 2.621 (2.620-2.622) | 2.521 (1.891-2.616), -3.8 % | 2.342 (1.697-2.350), -10.7 % | 1.939 (1.939-2.607), -26.0 % | 3.000 (2.941-3.006), +14.4 % |

    No count is lower beyond the spread on both phones (on the Pixel none is lower at all); 64 stays. On the Samsung,
    128's and even G5's spreads end just below A's, but A's three runs happened to sit on one of the Samsung's two
    levels (4.1), so that is not read as a gain.
  - *Outcome for the spec:* the requirement "The sort is faster on the phones" is **not met** by this change. B1 (64,
    PR #26) is the baseline, B2 is neutral, B4 is slower, B3 (Onesweep) was rejected. Recorded in design (Open
    Questions) for the owner to settle before the change is archived.

## 7. Full subgroups (topic B, design B5)

- [x] 7.1 Confirm that the sort pipelines (`multi_radixsort`, plus any new subgroup-dependent sort/scan kernel from groups 4–8) are created as subgroup-size-dependent, so they get `REQUIRE_FULL_SUBGROUPS` from the engine-wide mechanism that `vk-engine-runtime` lands first (that change owns the feature enablement and the `Algorithm` flag; do not re-implement it here). Verify: the debug log on each VK target shows the flag on the sort pipelines, and `BT_VK_VALIDATION=1 test-tree-vk` on rocky reports 0 validation errors.
  - *Which sort pipelines are subgroup dependent:* after groups 5 and 6, only `multi_radixsort_scan` (spec constant 0,
    a `gl_SubgroupID`-indexed shared array) is; `multi_radixsort` and `multi_radixsort_histograms` use no subgroup
    operations, and the engine refuses to build a pipeline whose `subgroup_size_dependent()` flag disagrees with the
    shader's constant 0. Group 8 adds no subgroup-dependent kernel. Nothing in the engine was changed for the flag;
    `test-tree-vk` now applies `SPDLOG_LEVEL` (`spdlog::cfg::load_env_levels()` after the default `off`), so its debug
    log can be read.
  - *Debug log* (`SPDLOG_LEVEL=debug test-tree-vk --gtest_filter=TreeDiffVulkan.Stage2_Sort`): `Pipeline
    [multi_radixsort_scan] created successfully (subgroup size N, pinned, full subgroups; ...)` with N = 64 and 32
    (`BT_VK_SUBGROUP_SIZE=32`) on rocky-ryzen, 16 on the Pixel 7a, 64 and 32 on the Samsung; each also logs
    `computeFullSubgroups supported true enabled true`.
  - *Validation:* `BT_VK_VALIDATION=1 ./test-tree-vk --device minipc` on rocky: 20/20, 0 `Validation Error` lines (layer
    loaded, per `VK_LOADER_DEBUG=layer`).
- [x] 7.2 Check that the tree suites pass exactly on a device path without `computeFullSubgroups`, using the engine change's fallback. Verify: on rocky, force the fallback with the engine's test hook (or note the device coverage if no such hook exists), then run `test-tree-vk` 15/15.
  - *Hook:* the engine has no switch that turns full subgroups off for a whole suite (kiss-vk's
    `NonMultipleWorkgroupFailsWithoutFullSubgroups` builds an `Algorithm` with edited `ComputeLimits`), and all three
    fleet GPUs support `computeFullSubgroups`. So the fallback was forced with a scratch edit of `base_engine.cpp` (not
    committed): `full_subgroups = false`.
  - *Result on rocky:* `test-tree-vk --device minipc` 20/20 at 64 and 20/20 with `BT_VK_SUBGROUP_SIZE=32` (the task's
    15/15 predates the tests added here). The engine logs `computeFullSubgroups supported true enabled false`, and every
    `multi_radixsort_scan` pipeline `(subgroup size 64|32, pinned, full subgroups not guaranteed: computeFullSubgroups
    unsupported; ...)`. That wording is new: the pipeline line said "full subgroups not required", and the tree-sort
    spec asks for a line saying full subgroups are not guaranteed (`platform/engine/vulkan/algorithm.cpp`).

## 8. Stage-3 compaction (topic B, design B6)

- [x] 8.1 Implement compaction in fewer than 5 dispatches. Keep `record_device_scan` unchanged for stage 6 unless both stages re-verify. Verify: `test-tree-vk` stage 3 and stage 6 are exact on all VK targets.
  - *Change (design B6), 2 dispatches instead of 5:* `tree_unique_local.comp` flags the run heads of a 2048-key tile
    straight from the sorted keys and scans them (the Blelloch scan of `tree_scan_local`), writing each key's in-tile
    count to `u_out_idx` and the tile total to `u_scan_block_sums`; `tree_unique_scatter.comp` sums the totals of the
    tiles before its own (at most a few hundred: 150 for 307,200 keys), writes each key's global count back to
    `u_out_idx` (so `u_out_idx[n-1]` is still n_unique for the chained read-back) and writes each run head to its slot.
    Both `record_stage_3` overloads call one helper, `record_unique`. Removed as orphans of this change:
    `tree_find_dups.comp`, `tree_move_dups.comp`, their pipelines and push-constant structs, and the `u_contributes`
    buffer of both appdata classes. `record_device_scan` is unchanged and now serves stage 6 only.
  - *Exact on every VK target:* rocky `test-tree-vk` 20/20 at 64 and pinned 32 (`Stage3_Unique`, `Stage6_EdgeOffset`
    of both suites and `Placement.DeclaredStagesMatchBindings` among them), `ctest -L vulkan` 7/7, sync validation
    (shader-access heuristic on) 20/20 with 0 errors and 0 hazards; Pixel 7a 20/20 and `test-kiss-vk` 29/29; Samsung
    20/20 and 29/29, also with `BT_VK_SUBGROUP_SIZE=32`.
- [x] 8.2 Measure stage 3 with the phone A/B against 4.1. Verify: the after median is not higher beyond the spread on either phone, and the dispatch count is recorded (spec "Compaction is not dispatch-bound").
  - *Method:* as 4.1 session 2 (A = 5477c56, B = group 8, same session). Stage 3 (ms):

    | phone | A median (min-max) | B median (min-max) | B vs A |
    |---|---|---|---|
    | Pixel 7a | 1.666 (1.659-1.678, n=3x607-801) | 1.396 (1.395-1.692, n=3x723-1016) | -16.3 % |
    | Samsung | 3.185 (3.179-3.197, n=3x312-486) | 1.843 (1.843-1.844, n=3x535-792) | -42.2 % |

    Not higher on either phone; on the Samsung lower with disjoint spreads, on the Pixel two of three runs lower (the
    third sat on the upper level). Stage 6 (same `record_device_scan` as before) -1.4 % / -0.3 %, the other stages within
    4 %. GPU clock median 251 / 252 MHz in both arms.
  - *Dispatch count:* 2 (`tree_unique_local`, `tree_unique_scatter`), was 5 (find_dups, 3-pass scan, move_dups).

## 9. Final gate and profile refresh

- [ ] 9.1 Run the OMP, VK and CUDA gates at the final commit, plus `BT_VK_VALIDATION=1` runs of `test-tree-vk` on rocky. Verify: all green, with 0 validation errors.
- [ ] 9.2 Refresh tree × VK profiling on the phones (`bm-prof-tree-vk`, isolated and interference, 2 runs each, into `data/profiling/<serial>/tree/vulkan/`). Verify: the new per-stage times are recorded, with the run noted here.
