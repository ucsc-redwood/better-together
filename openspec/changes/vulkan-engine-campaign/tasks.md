# Tasks

Each phase (A = groups 1–4, B = groups 5–6, C = groups 7–8) is one PR into `dev`. "VK fleet gate"
means:
- `ctest -L vk` passes on `rocky-ryzen`, the Pixel 7a and the Samsung (via `scripts/run-on-rocky.sh`
  and `scripts/run-on-android.sh`);
  - *Note (Phase A):* the repo's CTest label is `vulkan`, and `-L` is a regex, so a literal
    `ctest -L vk` selects **0 tests** and "passes" vacuously (checked on rocky: `ctest -L vk -N` →
    `Total Tests: 0`, `ctest -L vulkan -N` → 7). Throughout this file, "`ctest -L vk`" means
    `ctest -L vulkan` on rocky-ryzen plus the `test-{tree,cifar-dense,cifar-sparse}-vk` binaries
    (and `test-kiss-vk`) run directly, as `scripts/run-on-*.sh` do.
- plus `ctest -L omp` locally;
- plus the shader bake / clean-tree check.

## 1. Phase A — host visibility (execution-sync)

- [x] 1.1 Record a COMPUTE_SHADER/SHADER_WRITE → HOST/HOST_READ memory barrier in `Sequence::cmd_end()`, before the bottom-of-pipe timestamp (design D1). Verify: `ctest -L vk` is green on `rocky-ryzen`.
  - *Evidence (82b3c56, rocky-ryzen):* `cmake --build --preset vulkan`, then
    `VK_LOADER_LAYERS_DISABLE='~all~' ./test-{tree,cifar-dense,cifar-sparse}-vk --device minipc` → 14 / 12 / 12
    PASSED. `ctest -L vulkan` → 7/7 passed.
- [x] 1.2 Run the tree, cifar-dense and cifar-sparse VK suites with `BT_VK_VALIDATION=1` and `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT` on `rocky-ryzen`. Verify: no shader-write/host-read hazard is reported.
  - *Evidence (82b3c56, rocky-ryzen, layer 1.4.328.1; confirmed active, since `VK_LAYER_REPORT_FLAGS=info` prints
    "Current Enables: VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT"):* the three suites print 0 `SYNC-HAZARD`
    lines. The only errors are the `VUID-vkDestroyDevice-device-05137` leak that group 2 fixes; at d65faf3 they print 0
    validation errors of any kind.
  - **Caveat:** a control run at the parent 4d8b376 (no barrier) also prints 0 `SYNC-HAZARD`. Sync validation does not model
    host reads of mapped memory, so this check can't catch the defect. The fix rests on the spec text (bugs-found §11)
    and on the phone oracle runs (Pixel 7a, non-coherent Mali: 15/12/12 green at de389af and b7ac417).
- [x] 1.3 Add a `bugs-found.md` entry for the missing host-visibility barrier (root cause, spec citation, fix commit). Verify: the entry exists and links the commit.
  - *Evidence:* `docs/reports-for-human/bugs-found.md` §11 links fix commit 82b3c56.

## 2. Phase A — object lifetime

- [x] 2.1 Implement `~Algorithm()` to destroy the pipeline, pipeline layout, descriptor pool, descriptor-set layout, shader module and the per-Algorithm pipeline cache, guarding null handles (design D2). Verify: `ctest -L vk` is green on `rocky-ryzen`.
  - *Evidence (d65faf3, rocky-ryzen):* tree / cifar-dense / cifar-sparse → 15 / 12 / 12 PASSED. tree has 15 = 14 + the
    new 2.2 test. `ctest -L vulkan` → 7/7 passed.
- [x] 2.2 Add a VK test that constructs, destroys and reconstructs a dispatcher in one process. Verify: it passes, and with `BT_VK_VALIDATION=1` it reports no `VUID-vkDestroyDevice-device-05137`.
  - *Evidence (d65faf3, rocky-ryzen):* `VulkanDispatcherLifetime.ConstructDestroyReconstruct` in `test-tree-vk` runs two full
    construct → stages 1..7 → check → destroy rounds. `BT_VK_VALIDATION=1 ./test-tree-vk --device minipc
    --gtest_filter='VulkanDispatcherLifetime.*'` → PASSED, 0 `Validation Error` lines, 0 `05137`.
  - *Control:* at 4d8b376 a single tree test prints 20 `05137` lines.
- [x] 2.3 Run every VK test and `bm-*-vk` binary once with `BT_VK_VALIDATION=1` on `rocky-ryzen`. Verify: the exit logs contain no undestroyed-object errors.
  - *Evidence (d65faf3, rocky-ryzen):* each binary was run once from `build/vulkan` with `BT_VK_VALIDATION=1
    BT_WEIGHTS_DIR=/tmp/bt/weights`:
    - all 8 VK tests (`test-kiss-vk`, `test-{tree,cifar-dense,cifar-sparse}-vk`, `test-pipeline-e2e{,-cifar-dense,-cifar-sparse}-vk`,
      `test-pipeline-chained-vk`);
    - all 15 `bm-*-vk`: `bm-<app>-vk --benchmark_min_time=1x`, `bm-baseline-<app>-vk --benchmark_min_time=1x`,
      `bm-fully-<app>-vk -t 1`, `bm-gen-logs-<app>-vk` on one btpm schedule, and `bm-prof-<app>-vk` with short
      `BT_PROF_*` budgets;
    - plus `stressor-vk`.
    That is 8 + 15 + 1 = **24** binaries. Every log has **0** `Validation Error` lines, and 23/24 exit 0.
    (Corrected from an earlier miscount of "25/26", found in review.)
  - **Caveat:** `bm-fully-tree-vk` aborts with `n_unique is not set` (`apps/tree/tree_appdata.hpp:154`) before its last
    teardown. This is **pre-existing and unrelated**: identical abort at 4d8b376 with `VK_LOADER_LAYERS_DISABLE='~all~'`.
    Its earlier engine lifetimes in the same log tear down with 0 errors. Not fixed here (out of Phase A scope).
  - *Re-run at b7ac417* (final Phase A code, with the minimal feature set from group 3): identical. All 24 logs have 0
    `Validation Error` lines; the only non-zero exit is the same pre-existing `bm-fully-tree-vk` abort.
  - *Normal-exit coverage for `bm-fully-tree-vk`* (rocky-ryzen, at the BaseEngine failure-path commit that follows
    fc6b3e6): it runs with a stage range that avoids the abort, and the log is checked to prove validation was active.
    - `BT_VK_VALIDATION=1 VK_LOADER_DEBUG=layer ./bm-fully-tree-vk --device minipc -s 1 -e 1 -t 1` → exit 0,
      0 `Validation Error` lines. The loader logs `Insert instance layer "VK_LAYER_KHRONOS_validation"` for all 3
      instances the binary creates; a control run without `BT_VK_VALIDATION` logs it 0 times.
    - The full range `-s 1 -e 7 -t 1` still aborts with the pre-existing `n_unique is not set` (exit 134).
    - With this run, every one of the 24 binaries has a validated normal teardown.
- [x] 2.4 Add a `bugs-found.md` entry for the Algorithm leak, as a follow-up to §9. Verify: the entry exists.
  - *Evidence:* `docs/reports-for-human/bugs-found.md` §12 ("follow-up to §9") links fix commit d65faf3.

## 3. Phase A — feature enablement (device-init)

- [x] 3.1 Replace `check_vulkan_12_features` with a single required-feature table, using the query → check → minimal-enable flow from design D3. A missing feature throws an error naming it. Verify: `ctest -L vk` is green on all three VK targets.
  - *Evidence (de389af):*
    - rocky-ryzen: tree / cifar-dense / cifar-sparse → 15 / 12 / 12 PASSED, `ctest -L vulkan` 7/7. Under
      `BT_VK_VALIDATION=1` the three suites print 0 `Validation Error` lines, so no shader uses a capability whose
      feature is no longer enabled.
    - `scripts/run-on-android.sh 3A021JEHN02756 …` (Pixel 7a) and `R5CY21Y3VEV` (Samsung) → 15 / 12 / 12, plus
      `test-kiss-vk` 3/3, with 0 skips on both phones.
  - *Spec correction:* the required set is `storageBuffer8BitAccess`, `shaderInt8` and `bufferDeviceAddress`. That is
    exactly what the shaders' SPIR-V capabilities and VMA's BDA flag need. `uniformAndStorageBuffer8BitAccess` was
    dropped from proposal.md because no shader needs it.
- [x] 3.2 Add a unit test for the feature check: given a supported-features struct missing `shaderInt8`, the check reports `shaderInt8`. Verify: the test is labelled `vk` and passes, and it skips on hosts without an iGPU.
  - *Evidence:* `VulkanDeviceFeatures.MissingShaderInt8IsReported` lives in `test-kiss-vk` (CTest labels `vulkan;engine`).
    It takes the real iGPU's supported set, expects nothing missing, then clears `shaderInt8` and expects exactly
    `{"shaderInt8"}`. Results:
    - OK on rocky-ryzen, the Pixel 7a and the Samsung (de389af);
    - SKIPPED ("no integrated GPU", gtest exit 0) locally with only llvmpipe exposed
      (`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json ./test-kiss-vk`).
- [x] 3.3 Confirm that `has_integrated_gpu()` returns false (and the suites skip) when device creation throws on missing features. Verify: covered by the same unit test, or by code inspection noted in the PR.
  - *Evidence (code inspection, de389af):* `has_integrated_gpu()` builds `BaseEngine probe(false)` inside
    `try { … } catch (...) { return false; }`. The missing-feature `std::runtime_error` is thrown from
    `BaseEngine::create_device()`, which that constructor calls, so the probe returns false, and every suite's
    `Runner::Available()` (and the new 3.2 test) `GTEST_SKIP`s. The same path was exercised by the llvmpipe-only run
    above, where construction throws "No integrated GPU found" and the suite skips. A real missing-feature device
    was not available to run.

## 4. Phase A — validation opt-in (device-init) + phase gate

- [x] 4.1 Make `Engine()` request validation only when `BT_VK_VALIDATION=1`, and demote the layer-list dump to `debug` (design D4). Verify: on `rocky-ryzen`, a run without the variable shows no validation layer in the debug log, and a run with it shows the layer enabled.
  - *Evidence (b7ac417, rocky-ryzen, `SPDLOG_LEVEL=debug ./test-kiss-vk`; test-kiss-vk now honours `SPDLOG_LEVEL`):*
    - no variable, or `BT_VK_VALIDATION=0` → every engine logs `Vulkan instance layers enabled: none`, and the layer
      list is not dumped;
    - `BT_VK_VALIDATION=1` → `Vulkan instance layers enabled: VK_LAYER_KHRONOS_validation`, with the
      "Available Vulkan Layers" dump at `[debug]`. The `has_integrated_gpu()` probe stays `none` by design;
    - `BT_VK_VALIDATION=1 VK_LOADER_LAYERS_DISABLE='~all~'` (layer hidden) → `[warning] Validation layer requested
      but not available, continuing without it...`, layers `none`, 3/3 PASSED.
- [x] 4.2 Document `BT_VK_VALIDATION` in `docs/instruction-for-ai/03-unit-testing.md`. Note there that fleet scripts also set `VK_LOADER_LAYERS_DISABLE`. Verify: the documented command runs as written on `rocky-ryzen`.
  - *Evidence (b7ac417, rocky-ryzen):* the three documented commands were extracted verbatim from
    `03-unit-testing.md` and run from `build/vulkan`:
    - under `bash` → rc=0, tree 15 PASSED twice, `test-kiss-vk` 3 PASSED, the `…layers enabled:
      VK_LAYER_KHRONOS_validation` lines present, 0 `Validation Error`;
    - under rocky's login shell `fish` 4.2.0 → rc=0, same results.
- [x] 4.3 Phase A gate: VK fleet gate green. Then do one `fit_overhead` refit on `rocky-ryzen` and record the per-chunk overhead delta in the PR. Verify: the delta is within run-to-run noise, or the new constants are committed with an explanation.
  - *VK fleet gate (code at b7ac417; later commits are docs-only):*
    - local `ctest -L omp` → 9/9 passed;
    - rocky-ryzen → 15/12/12, `ctest -L vulkan` 7/7;
    - Pixel 7a and Samsung (`scripts/run-on-android.sh`) → 15/12/12, `test-kiss-vk` 3/3, 0 skips;
    - `cmake --build build/vulkan --target bake-shaders` → no-op, `git status` clean (Phase A changes no shader).
  - *Overhead refit (rocky-ryzen only).* Each run was a full `00_run_fleet.py --only minipc --fresh` (profile 5 runs/scenario →
    z3 → run the top-4 btpm + isolated schedules), then `fit_overhead.py --dry-run`:
    - "before" = parent 4d8b376 (worktree);
    - "after" = 06bad92 (code = b7ac417).
    - Order: before-1, after-1, before-2, after-2.
    - Conditions were identical in all four: governor `powersave`, clocks unlocked (no passwordless sudo for
      `lock-gpu-clocks.sh`), AMD DPM `auto`, rocky otherwise idle, same deployed weights.

    | class | before-1 | before-2 | after-1 | after-2 | mean before → after | Δ |
    |---|---|---|---|---|---|---|
    | `gpu_vulkan` per_chunk_ms | 2.034 | 1.562 | 1.658 | 1.529 | 1.798 → 1.594 | **−0.20 ms** |
    | `cpu` per_chunk_ms | 1.628 | 2.427 | 2.587 | 1.546 | 2.028 → 2.067 | +0.04 ms |

    Notes on the fits:
    - per_stage_ms = 0 in all eight (model `median`, n = 24 chunks each).
    - **Verdict: within run-to-run noise.** The same 4d8b376 binaries alone spread 0.47 ms (GPU) and 0.80 ms (CPU)
      between two runs, and both deltas are well inside that. No new constants need committing (`overhead.json`
      lives under the git-ignored `data/` anyway).
  - *Per-stage cross-check (isolated GPU-timestamp p50, median over 5 runs):*
    - The barrier's own cost **is** visible and reproducible on µs-scale VK stages: +4–15 µs per submit, e.g.
      tree s5 0.034 → 0.041 ms, tree s7 0.216 → 0.231 ms, cifar-dense s8 0.0253 → 0.0297 ms. Both runs on each side
      agree to ≤ 1 µs.
    - That is +6–21 % on tree stages 1 and 3–7, and the design's "tiny" holds only in absolute terms.
    - Stages ≥ 0.5 ms are unchanged within noise.
    - The cost sits inside the profiled stage times (D1 records it before the closing timestamp), so z3 already
      prices it through the profiling table, not through the overhead constants. Phase C's tree × VK re-measurement
      (7.7) should start from these post-barrier tables.

## 5. Phase B — device-derived subgroup size (pipeline-config)

- [x] 5.1 In `BaseEngine`, query `subgroupSize` and the subgroup-size-control properties. Enable `subgroupSizeControl` when it is supported and COMPUTE is in `requiredSubgroupSizeStages`. Expose `Engine::subgroup_size()` (design D5). Verify: on each VK target, a debug log line shows the chosen size and whether it is pinned.
  - *Evidence (bd59780; `SPDLOG_LEVEL=debug ./test-kiss-vk --gtest_filter=VulkanEngineTest.EngineInitialization`):*
    one debug line per engine shows the chosen size, whether it is pinned, and what the device reports:

    | target | logged line (after `[debug] `) |
    |---|---|
    | rocky-ryzen (Radeon 780M, RADV PHOENIX) | `Subgroup size 64 (pinned): device default 64, min 32, max 64, subgroupSizeControl true, requiredSubgroupSizeStages { Compute \| TaskEXT \| MeshEXT }, maxComputeWorkgroupSubgroups 4294967295` |
    | Pixel 7a (Mali-G710) | `Subgroup size 16 (pinned): device default 16, min 16, max 16, subgroupSizeControl true, requiredSubgroupSizeStages { Fragment \| Compute }, maxComputeWorkgroupSubgroups 64` |
    | Samsung (Xclipse 940) | `Subgroup size 64 (pinned): device default 64, min 32, max 64, subgroupSizeControl true, requiredSubgroupSizeStages { Compute }, maxComputeWorkgroupSubgroups 4294967295` |

  - **Finding:** the Samsung reports a default of **64**, not the 32 that `devices/R5CY21Y3VEV.json`, the docs and the
    pipeline-config spec gave. No fleet GPU defaults to 32. The spec scenario, proposal.md and the JSON were corrected
    (5e7d320, a9e57d2), and 32 is now reached by pinning (see 5.4).
- [x] 5.2 Add the SPIR-V scan for `SpecId` and `LocalSize` and the `Algorithm` support for specialization constants. Chain `PipelineShaderStageRequiredSubgroupSizeCreateInfo` when pinning is available, and check the workgroup size against `requiredSubgroupSize × maxComputeWorkgroupSubgroups`. Verify: a unit test covers the scan on a committed `*_spv.h`.
  - *Evidence:* `reflect_spirv()` (`platform/engine/vulkan/algorithm.cpp`) reads `OpDecorate … SpecId`,
    `OpExecutionMode LocalSize` and `OpExecutionModeId LocalSizeId` (the SPIR-V 1.6 form glslc emits for
    `local_size_*_id`, resolved through `OpConstant`/`OpSpecConstant`). In `test-kiss-vk`:
    - `SpirvReflection.CommittedShaders`: the committed `multi_radixsort_spv.h` has spec ids `{0}` and LocalSize
      (256, 1, 1) with no size spec ids; `tree_edge_count_spv.h` has no spec ids and LocalSize (512, 1, 1); a non-SPIR-V
      buffer throws.
    - `SpirvReflection.LocalSizeId`: `engine_test_workgroup_size_spv.h` (added in 6.1) maps x/y/z to spec ids 100/101/102.
    - Both pass on all four targets (test-kiss-vk 10/10 at bd59780).
  - *Pinning:* subgroup-dependent pipelines chain `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` when pinning is
    available. With `-l debug`, `bm-fully-tree-vk -s 1 -e 1 -t 1` logs `Pipeline [multi_radixsort] created successfully
    (subgroup size N, pinned)`: N = 64 on rocky-ryzen (3 lines, one per engine), 16 on the Pixel (5 lines) and 64 on the
    Samsung (5 lines). `build()` checks `x·y·z ≤ requiredSubgroupSize × maxComputeWorkgroupSubgroups`
    (VUID-…-pNext-02756) before chaining. With `BT_VK_VALIDATION=1` on rocky-ryzen (layer inserted 16 times), the tree
    suite prints 0 `Validation Error` lines, both pinned at 64 and pinned at 32.
- [x] 5.3 Merge `multi_radixsort_warp{16,32,64}.comp` into `multi_radixsort.comp` with `constant_id = 0` for `SUBGROUP_SIZE`. Re-bake, delete the three old `.comp` and `*_spv.h` files, and update `all_shaders.hpp`. Verify: the bake / clean-tree check passes.
  - *Evidence (6a004ff):* `multi_radixsort.comp` declares `layout(constant_id = 0) const uint SUBGROUP_SIZE = 32;`.
    glslang emits `OpDecorate %78 SpecId 0` and sizes `sums[]` with `OpSpecConstantOp UDiv %uint_256 %78`.
    - The spec-constant-sized shared array compiles and runs on Mali (Pixel tree 15/15, see 5.4), so the D5 fallback
      (fixed array sized `RADIX_SORT_BINS / 16`) is not needed.
    - 9 files deleted: `multi_radixsort_warp{16,32,64}` as `.comp`, `_spv.h` and `.spv`. `all_shaders.hpp` now lists
      `multi_radixsort`. `tmp_single_radixsort_warp*` is untouched.
    - Bake / clean-tree check, at bd59780:
      - `cmake --build build/vulkan --target bake-shaders` → no-op, `git status` clean;
      - after `touch`ing both shaders this phase added (`multi_radixsort.comp`, `engine_test_workgroup_size.comp`), the
        bake rebuilds exactly those 2 and `git status` stays clean, so the committed headers are what glslc produces.
- [x] 5.4 Switch `apps/tree/vulkan/dispatchers.cpp` (and any `profiler/tree-vk` user) to the single shader with `subgroup_size_dependent()`. Verify: `ctest -L vk` tree suites match the oracle exactly on `rocky-ryzen`, the Pixel 7a (sg16) and the Samsung (sg32).
  - *Evidence (bd59780):* the tree dispatcher builds `multi_radixsort` with `->subgroup_size_dependent()`.
    `profiler/tree-vk` has no shader reference of its own (it uses the dispatcher), so nothing changed there.
    `test-tree-vk` runs 15/15 PASSED, exact oracle match, at every width:

    | width | rocky-ryzen | Pixel 7a | Samsung |
    |---|---|---|---|
    | device default (pinned) | 64: 15/15 | 16: 15/15 | 64: 15/15 |
    | `BT_VK_SUBGROUP_SIZE=32` (pinned) | 32: 15/15 | refused: "Subgroup size 32 cannot be pinned on this device (pinning available, supported 16..16)", 15 failures, exit 1 (by design) | 32: 15/15 |

    - All of these use one radix-sort shader source.
    - The pinned width is confirmed by the `-l debug` line `Pipeline [multi_radixsort] created successfully (subgroup
      size 32, pinned)` on rocky-ryzen (3×) and the Samsung (5×).
  - *Spec correction (5e7d320):* `BT_VK_SUBGROUP_SIZE` was added because no fleet GPU defaults to 32 (see 5.1). It is a test
    hook that pins another size the device supports and throws otherwise. It is recorded in design D5 and as a new
    pipeline-config scenario. The "Every fleet subgroup width" scenario now reads 16 / 64 / pinned 32 instead of
    "the Samsung (32)".
- [x] 5.5 Run the tree VK suite on `rocky-ryzen` with the `gpu.subgroup_size` field temporarily removed from `minipc.json`. Verify: the suite passes, which demonstrates the registry is no longer required.
  - *Evidence (rocky-ryzen, at 6a004ff and again at bd59780):* `, "subgroup_size": 64` was deleted from rocky's copy of
    `devices/minipc.json` (never committed). The rebuild re-embedded it: `device_specs_embedded.hpp` shows
    `"gpu": { "backend": "vulkan", "name": "AMD Radeon 780M" }`.
    - `test-tree-vk --device minipc` → 15/15 PASSED;
    - `bm-fully-tree-vk -l debug` → `Pipeline [multi_radixsort] created successfully (subgroup size 64, pinned)`, with no
      warning (no registry value to compare).
    - Restored and rebuilt afterwards: `rsync -n -c` shows rocky's `minipc.json` and embedded header identical to the
      committed ones.
  - *Control (471001d, same edit):* all 15 tree tests fail with `device [minipc] has no gpu.subgroup_size in its
    devices/*.json spec`.
  - `scripts/validate_devices.py` and the schema now accept a `gpu` without `subgroup_size` (a9e57d2). The same temporary
    edit locally → `OK: 6 device spec(s) valid`.
- [x] 5.6 Downgrade `get_vulkan_warp_size()` to a cross-check that warns on a mismatch (spec "Registry subgroup value is a cross-check only"). Verify: a unit test with a mismatching registry value produces the warning.
  - *Evidence:* `get_vulkan_warp_size()` is replaced by `check_vulkan_subgroup_size()` (`platform/registry/device_registry.cpp`),
    which `Engine()` calls. No `--device`, an unknown device, or no declared size is not a mismatch; a mismatch warns.
    - `RegistrySubgroupCrossCheck.MismatchWarns` (`test-kiss-vk`, hardware-free) sets `g_device_id = "minipc"` (registry
      64) and passes 16. It expects exactly one `[warning]` line naming `gpu.subgroup_size 64` and `engine uses 16`, a
      `false` return, and no log line for a matching 64. It passes on all four targets.
    - *On real hardware (6a004ff, Samsung, registry then 32):* 5 engines each logged `devices/R5CY21Y3VEV.json declares
      gpu.subgroup_size 32 but the Vulkan device reports 64; using 64`, and the run completed (rc=0) at 64.
    - The wording was later changed to "…but the Vulkan engine uses N (the registry value is informational)", because
      with `BT_VK_SUBGROUP_SIZE` the size isn't what the device reports. At bd59780, Samsung with
      `BT_VK_SUBGROUP_SIZE=32`: `declares gpu.subgroup_size 64 but the Vulkan engine uses 32`.
- [x] 5.7 Compare tree × VK stage-2 time before and after on `rocky-ryzen`, to cover the wave32/wave64 risk. Verify: the numbers are recorded in the PR; if it regressed beyond noise, pin to `minSubgroupSize` and re-measure.
  - *Evidence (rocky-ryzen; "before" = 471001d, "after" = 5491c54, whose subgroup code is the same as bd59780's apart
    from enabling `maintenance4`):*
    - Method: `bm-tree-vk --device minipc --benchmark_filter='Tree/Stage2/' --benchmark_repetitions=10`. Stage-2 time
      is the GPU timestamp of each stage-2 submit (`run_vk_stage_timed`, manual time).
    - Order, interleaved: before, after, after-32 (`BT_VK_SUBGROUP_SIZE=32`), repeated 3 times, with a 10 s settle
      before each run.
    - Conditions, the same for all 9 runs:
      - governor `powersave` on all 16 CPUs, AMD DPM `auto`, clocks unlocked (no passwordless sudo);
      - GPU busy 0% before every run (`gpu_busy_percent`);
      - load average 0.05–0.52, no other process holding `/dev/dri` (`nvtop -s` lists only itself).

    | arm | run medians (ms) | mean of medians | spread of medians |
    |---|---|---|---|
    | before (warp64 shader, unpinned) | 0.7834, 0.7827, 0.7832 | 0.78308 | 0.0007 |
    | after (spec-constant shader, pinned 64) | 0.7831, 0.7826, 0.7836 | 0.78311 | 0.0010 |
    | after, pinned 32 | 0.9695, 0.9706, 0.9705 | 0.97019 | 0.0010 |

    - **Δ after − before = +0.00004 ms (+0.00 %)**, well inside the before-run spread of 0.0007 ms, so there is no
      regression. This is consistent with RADV already compiling the unpinned compute shader at wave64.
    - Pinning to `minSubgroupSize` (32), the D5 mitigation, would cost **+0.18712 ms (+23.89 %)**, so it is not applied.
    - An earlier set without the settle gave the same picture (before 0.7828/0.7827/0.7833, after
      0.7832/0.7834/0.7834, pinned-32 0.9697/0.9705/0.9701), but its pre-run GPU-busy samples (86–90 %) were leftovers
      from the run that had just ended (busy decays to 0 % within ~5 s), so it was redone.
- [x] 5.8 Update `01-hardware.md`, `devices/README.md` (subgroup_size is informational) and `testing-status.md` §F (close the subgroup gap). Verify: the docs mention the runtime query and not the `--device` map.
  - *Evidence (a9e57d2):*
    - `01-hardware.md` gains a "Subgroup size comes from the device, at runtime" section, with the queried table and
      the `BT_VK_SUBGROUP_SIZE` hook, and its matrix and per-device entries now give the queried widths.
      `devices/README.md` gains "GPU subgroup size is informational".
    - `testing-status.md` §F marks the gap CLOSED. The schema description no longer says `subgroup_size` "selects the
      Vulkan compute-shader variant".
    - The `devices/R5CY21Y3VEV.json` `subgroup_size` changed 32 → 64. `grep -i 'warp'` over `01-hardware.md`,
      `devices/README.md` and the schema → 0 hits.
    - Also fixed: the stale "phones are attached to rocky-ryzen" text in `01-hardware.md`, `02-building.md`,
      `03-unit-testing.md` and the `scripts/run-on-android.sh` comments and error text (behaviour unchanged). The phones
      are on the build box's adb.
    - `03-unit-testing.md` also documents `BT_VK_SUBGROUP_SIZE`.

## 6. Phase B — workgroup size via specialization (pipeline-config) + phase gate

- [x] 6.1 Make `Algorithm::work_group_size()` feed `local_size_{x,y,z}_id` spec constants when the shader declares them. Otherwise it fails if the value disagrees with the SPIR-V literal `LocalSize`. Verify: a unit test shows a mismatching declared size throws, and all current dispatchers still build.
  - *Evidence:* `work_group_size()` feeds the `LocalSizeId` spec constants, else must equal the literal `LocalSize`, and
    `create_pipeline()` checks this before creating any Vulkan object. In `test-kiss-vk`:
    - `VulkanWorkGroupSize.DeclaredSizeReachesShader` dispatches the new engine self-test shader
      `engine_test_workgroup_size.comp` (`local_size_{x,y,z}_id = 100/101/102`) once, at a declared (64, 2, 1). It reads
      back 128 invocations and `gl_WorkGroupSize` = (64, 2, 1).
      - Mutation check (local, not committed): with the `specialize()` call for workgroup dims removed, it fails
        (got 1 and 1).
    - `VulkanWorkGroupSize.MismatchWithLiteralLocalSizeThrows`: `tree_morton` (literal 256) declared as 128 throws
      `declared workgroup size (128, 1, 1) but the SPIR-V LocalSize is (256, 1, 1)`.
    - `VulkanSubgroupSize.DependentPipelinesMatchTheShader`: `constant_id = 0` and `subgroup_size_dependent()` must agree
      both ways, so no shader silently runs on the default.
    - All current dispatchers still build: tree / cifar-dense / cifar-sparse suites and `stressor-vk` build and pass on
      every target (6.3).
  - **Found by the gate (a9e57d2 → fixed in bd59780):** with `BT_VK_VALIDATION=1`, rocky-ryzen reported
    `VUID-RuntimeSpirv-LocalSizeId-06434` 3× in `test-kiss-vk`, because `LocalSizeId` needs `maintenance4`.
    `maintenance4` is now a required (1.3) feature (design D3 and proposal.md updated). `MissingShaderInt8IsReported`
    also checks that a missing `maintenance4` is named. At bd59780 the same run prints 0 validation errors.
- [x] 6.2 Reject a declared workgroup size above the device's `maxComputeWorkGroupSize` or invocation limits, with an error naming both. Verify: a unit test with an oversized request.
  - *Evidence:* `VulkanWorkGroupSize.ExceedsDeviceLimitsThrows` (`test-kiss-vk`, passes on all four targets):
    - (`maxComputeWorkGroupSize[0]` + 1, 1, 1) throws `workgroup size (…) exceeds maxComputeWorkGroupSize (…)`;
    - every dimension at its own limit throws `… = N invocations exceeds maxComputeWorkGroupInvocations M`.
    The test asserts that both the size and the limit appear in the message.
- [x] 6.3 Phase B gate: VK fleet gate green. Verify: the PR lists the results for each target.
  - *VK fleet gate at bd59780* (final code; later commits are docs-only):
    - local `ctest -L omp` → 9/9 passed;
    - bake / clean-tree check → clean (5.3);
    - rocky-ryzen (`--device minipc`, `VK_LOADER_LAYERS_DISABLE='~all~'`):
      - `test-kiss-vk` / tree / cifar-dense / cifar-sparse → 10 / 15 / 12 / 12 PASSED, 0 skipped;
      - `ctest -L vulkan` → 7/7;
      - tree with `BT_VK_SUBGROUP_SIZE=32` → 15/15;
      - with `BT_VK_VALIDATION=1` (layer inserted 7 / 16 / 12 / 12 times) → 0 `Validation Error` lines in all four, and
        0 in tree pinned at 32 (16 inserts);
    - Pixel 7a and Samsung (`scripts/run-on-android.sh <serial> test-kiss-vk test-tree-vk test-cifar-dense-vk
      test-cifar-sparse-vk`) → 10 / 15 / 12 / 12 on both, 0 skipped, script exit 0. Samsung tree with
      `BT_VK_SUBGROUP_SIZE=32` → 15/15.
    - test-kiss-vk grew from 3 to 10 tests. The 7 new ones are `SpirvReflection.{CommittedShaders,LocalSizeId}`,
      `VulkanWorkGroupSize.{DeclaredSizeReachesShader,MismatchWithLiteralLocalSizeThrows,ExceedsDeviceLimitsThrows}`,
      `VulkanSubgroupSize.DependentPipelinesMatchTheShader` and `RegistrySubgroupCrossCheck.MismatchWarns`.

## 7. Phase C — per-dispatch binding (resource-binding)

- [ ] 7.1 Enable `VK_KHR_push_descriptor` when it is available. Add `Algorithm::record_bind(cmd, buffers)` using push descriptors, plus the per-`Sequence` transient-pool fallback reset in `cmd_begin()`, and `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1` (design D6). Verify: an engine-level unit test records the same kernel twice with different buffers in one command buffer and checks both outputs, on both paths.
- [ ] 7.2 Migrate the `apps/cifar-dense` VK dispatcher and its profiler runner to `record_bind`, removing its `num_sets` and set indices. Verify: the cifar-dense VK suites pass on all three targets.
- [ ] 7.3 Migrate the `apps/cifar-sparse` VK dispatcher and its profiler runner. Verify: the cifar-sparse VK suites pass on all three targets.
- [ ] 7.4 Migrate the `apps/tree` VK dispatcher (radix-sort passes, scans, dup removal) and its profiler runner. Verify: the tree VK suites match exactly on all three targets.
- [ ] 7.5 Remove `num_sets()`, `update_descriptor_set()`, `record_bind_core()` and the descriptor pool and sets from `Algorithm` once nothing uses them. Verify: a clean build of the `vulkan` and `android` presets.
- [ ] 7.6 Run the full VK suites on `rocky-ryzen` with `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1`. Verify: green, with identical results.
- [ ] 7.7 Re-profile tree × VK on `rocky-ryzen` before and after, with the same governor and clocks, and refit the overhead. Verify: the per-chunk makespan does not increase beyond noise, and the delta is recorded in the PR.

## 8. Phase C — persisted pipeline cache (pipeline-config) + phase gate

- [ ] 8.1 Give `Engine` one pipeline cache shared by all `Algorithm`s. Load it from `$BT_VK_PIPELINE_CACHE_DIR` (default: the current directory) after validating the header against vendorID, deviceID and UUID, and save it atomically on destruction (design D7). Verify: a second run of a VK test on `rocky-ryzen` logs a cache hit at debug level.
- [ ] 8.2 Add tests for a corrupt cache file, a cache from a foreign device or UUID, and a read-only directory. Verify: all three runs complete with unchanged results and a cold compile.
- [ ] 8.3 Ignore `*.bin` pipeline-cache files in `.gitignore` and the fleet cleanup scripts, and document `BT_VK_PIPELINE_CACHE_DIR` in `03-unit-testing.md`. Verify: `git status` is clean after a local VK run.
- [ ] 8.4 Phase C gate: VK fleet gate green, including the Pixel 7a and Samsung with push descriptors active. Verify: the debug log on each phone shows which binding path is in use.

## Workflow follow-up

- Decide whether a standing `BT_VK_VALIDATION=1` CTest job on `rocky-ryzen` is wanted (design, Open Questions).
- Archive the change after all three phase PRs are merged to `dev`, then sync the five `vulkan-engine/*` specs into `openspec/specs/`.
