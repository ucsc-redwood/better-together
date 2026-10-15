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

- [x] 7.1 Enable `VK_KHR_push_descriptor` when it is available. Add `Algorithm::record_bind(cmd, buffers)` using push descriptors, plus the per-`Sequence` transient-pool fallback reset in `cmd_begin()`, and `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1` (design D6). Verify: an engine-level unit test records the same kernel twice with different buffers in one command buffer and checks both outputs, on both paths.
  - *Evidence (43d180f + 7c18fef):*
    - (The call takes the `Sequence`, not a bare command buffer; see the design D6 correction.)
    - `BaseEngine` enables `VK_KHR_push_descriptor` only when the device lists it, outside the required-feature table,
      and fetches `vkCmdPushDescriptorSetKHR` with `vkGetDeviceProcAddr`. `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1` leaves the
      extension disabled even where it exists. `Algorithm::record_bind(seq, {buffers…})` pushes on a PUSH_DESCRIPTOR
      layout, or allocates, writes and binds one set from the `Sequence`'s transient pool. The pool is created on first
      use, grows by whole pools, and is reset by `cmd_begin()`.
    - Path in use, from the `SPDLOG_LEVEL=debug` line `Descriptor binding: …`. It is the same on every target:
      `push descriptors (VK_KHR_push_descriptor)` by default, and `transient descriptor pool (forced by
      BT_VK_FORCE_DESCRIPTOR_FALLBACK=1)` with the variable. Targets: rocky-ryzen (Radeon 780M, RADV), Pixel 7a (Mali-G710),
      Samsung (Xclipse 940), and also this build box's Ryzen 9600X iGPU (RADV RAPHAEL_MENDOCINO). All four have the
      extension, as gpuinfo predicted.
    - Unit test `BothPaths/VulkanBinding.SameKernelManyBuffersInOneCommandBuffer/{PushDescriptors,Fallback}`
      (`test-kiss-vk`). Each variant sets `BT_VK_FORCE_DESCRIPTOR_FALLBACK` explicitly, so both run on the same device
      even under an outer forced-fallback run. The test records one kernel 75 times (25 rounds × 3 buffers, more than one
      64-set pool) into one command buffer, twice (two submits). Each buffer must end at its own start value
      (1000/2000/3000) + 128 per dispatch bound to it. Both variants are OK on rocky-ryzen, the Pixel 7a, the Samsung and
      the build box.
    - *Mutation check (rocky-ryzen, not committed):* with `record_bind` changed to reuse the first buffers it ever saw,
      both variants FAIL (e.g. buffer 0 holds 4200, buffer 1 holds 2000). With the code restored, both pass.
- [x] 7.2 Migrate the `apps/cifar-dense` VK dispatcher and its profiler runner to `record_bind`, removing its `num_sets` and set indices. Verify: the cifar-dense VK suites pass on all three targets.
  - *Evidence (87418bc; suites run at 7c18fef):* the 11 `update_descriptor_set` + `record_bind_core(cmd, i)` pairs became
    `record_bind(*seq, {…})`, and the 5 `num_sets()` calls and the set-index comments were removed.
    - `grep -rn "update_descriptor_set\|record_bind\|num_sets" profiler/` → 0 hits, at 7cc0916 and now. The
      `cifar-dense-vk` profiler runners only call the dispatcher (`bm_prof_vulkan.hpp`: `disp.dispatch_multi_stage(app, s, s)`).
    - `test-cifar-dense-vk` → 12/12 on rocky-ryzen (default and forced fallback), the Pixel 7a and the Samsung.
- [x] 7.3 Migrate the `apps/cifar-sparse` VK dispatcher and its profiler runner. Verify: the cifar-sparse VK suites pass on all three targets.
  - *Evidence (ed8237b; suites run at 7c18fef):* the same migration: 11 call sites, 5 `num_sets()` removed. The profiler
    runner has no binding code of its own. `test-cifar-sparse-vk` → 12/12 on rocky-ryzen (default and forced fallback),
    the Pixel 7a and the Samsung.
- [x] 7.4 Migrate the `apps/tree` VK dispatcher (radix-sort passes, scans, dup removal) and its profiler runner. Verify: the tree VK suites match exactly on all three targets.
  - *Evidence (4eb856b; suites run at 7c18fef):*
    - All 19 call sites moved, in both the `VkAppData_Safe` and the chained `VkAppData` bodies; 13 `num_sets()` removed.
    - The radix sort binds each pass's ping-pong buffers at record time, so the "bind all 4 passes up front" loops are
      gone. `record_device_scan` lost its `descriptor_set` parameter.
    - Barriers, dispatch sizes and the one-submit-per-chunk `dispatch_multi_stage` are unchanged.
    - Across the three dispatchers that is 11 + 11 + 19 = 41 `update_descriptor_set` call sites at 7cc0916, the count
      the brief gave. The only other callers were `stressor-vk` and one `test-kiss-vk` test (1 each), migrated in 7c18fef.
    - `test-tree-vk` → 15/15 with an exact oracle match on rocky-ryzen (default and forced fallback, and pinned at 32 on
      both paths), on the Pixel 7a (sg16) and on the Samsung (default 64, and pinned at 32).
- [x] 7.5 Remove `num_sets()`, `update_descriptor_set()`, `record_bind_core()` and the descriptor pool and sets from `Algorithm` once nothing uses them. Verify: a clean build of the `vulkan` and `android` presets.
  - *Evidence (7c18fef):*
    - Removed: `num_sets()`, `update_descriptor_set()`, `record_bind_core()`, `create_descriptor_pool()`,
      `allocate_descriptor_sets()`, the `descriptor_pool_`, `descriptor_sets_` and `descriptor_set_count_` members, and
      the commented-out single-set and `update_buffer` remnants. Also removed: the unused `buffer_infos_` member
      (pre-existing dead code, part of the same binding state).
    - `~Algorithm()` now destroys the pipeline, pipeline layout, descriptor-set layout and shader module. The pipeline
      cache went to the Engine in 8.1.
    - `grep -rn` for any of those names over `platform apps profiler tools runtime tests` → 0 hits.
    - Clean builds in fresh directories at 1e732ea (`cmake --preset vulkan -B <new dir>` / `--preset android -B <new dir>`,
      full build): both rc=0. Android: 0 warnings. Vulkan: 2 warnings, both pre-existing OMP ones in
      `apps/tree/omp/func_sort.hpp` (none in Vulkan code).
    - Validation-layer teardown on both binding paths (rocky-ryzen, `BT_VK_VALIDATION=1 VK_LOADER_DEBUG=layer`, the four
      suites): 0 `Validation Error` lines and 0 `VUID-vkDestroyDevice-device-05137`. The layer was inserted 9 / 16 / 12 / 12
      times at 7c18fef, and 24 / 16 / 12 / 12 at 1e732ea (the new cache tests add 15 engines).
- [x] 7.6 Run the full VK suites on `rocky-ryzen` with `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1`. Verify: green, with identical results.
  - *Evidence (rocky-ryzen, `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1`; at 7c18fef and again at 1e732ea):*
    - `test-kiss-vk` / tree / cifar-dense / cifar-sparse → 12 / 15 / 12 / 12 at 7c18fef and 16 / 15 / 12 / 12 at 1e732ea.
      `ctest -L vulkan` → 7/7 both times (6 min 11 s at 7c18fef, against 6 min 4 s on the default path).
    - With `BT_VK_VALIDATION=1`: 0 validation errors, also with the tree pinned at 32.
    - The debug log confirms the forced path ("transient descriptor pool (forced by …)").
    - "Identical results" means every suite meets the same oracle bar as on the default path: an exact match for tree,
      and `NearEqual` for cifar.
- [x] 7.7 Re-profile tree × VK on `rocky-ryzen` before and after, with the same governor and clocks, and refit the overhead. Verify: the per-chunk makespan does not increase beyond noise, and the delta is recorded in the PR.
  - *Method (rocky-ryzen; "before" = 7cc0916, "after" = 7c18fef, the final binding code; group 8 changes no binding).*
    - Each arm has its own clean worktree, with benchmark binaries built on rocky (`scripts/build-bench-on-rocky.sh
      vulkan`). Each run is a full `00_run_fleet.py --only minipc --fresh --phases profile,schedule,run,summary`
      (profile 5 runs/scenario → z3 → run the top-4 btpm + isolated schedules), then `fit_overhead.py --dry-run`.
    - The host-side extra comes after each run: `bm-prof-tree-vk` isolated twice, once timed by GPU timestamps and once
      with `BT_PROF_GPU_WALLCLOCK=1`.
    - Order, interleaved: before-1, after-1, before-2, after-2, before-3, after-3. A third pair was added because the
      tree-only GPU constant's first two before/after ranges did not overlap; see below.
    - Conditions, identical and recorded before every run: governor `powersave` on all 16 CPUs, AMD DPM `auto`, clocks
      unlocked (as in 4.3), `gpu_busy_percent` 0, 1-min load ≤ 0.48 (the script waits for < 0.5), no process holding
      `/dev/dri`, no CI runner process. All six fleet runs exit 0.

    | metric (ms) | before 1/2/3 | after 1/2/3 | mean before → after | Δ | before spread |
    |---|---|---|---|---|---|
    | fitted `gpu_vulkan` per_chunk (all apps) | 1.6763 / 1.5503 / 1.5786 | 1.6578 / 1.4114 / 1.4637 | 1.6017 → 1.5110 | −0.0908 | 0.1260 |
    | fitted `cpu` per_chunk (all apps) | 1.5713 / 2.5410 / 1.7782 | 1.2959 / 2.0429 / 1.4899 | 1.9635 → 1.6096 | −0.3539 | 0.9697 |
    | fitted `gpu_vulkan` per_chunk (tree only) | 0.8592 / 0.8680 / 0.8248 | 0.8889 / 0.8752 / 0.8373 | 0.8507 → 0.8671 | +0.0165 | 0.0432 |
    | fitted `cpu` per_chunk (tree only) | 0.1655 / 0.1913 / 0.1605 | 0.1996 / 0.1918 / 0.1207 | 0.1724 → 0.1707 | −0.0017 | 0.0308 |
    | tree best measured makespan (ms/task) | 1.7182 / 1.7255 / 1.9072 | 1.7286 / 1.7257 / 1.7408 | 1.7836 → 1.7317 | −0.0519 | 0.1890 |

    - Every fit is model `median` with per_stage_ms = 0 (n = 24 chunks over all apps, 8 for tree). The tree-only fit is
      the same `fit_overhead` code restricted to the two tree cells.
    - The best makespan depends on which candidates z3 picks. before-3's top-4 had no "GPU stages 1–2" schedule, the
      winner in the other five runs, hence its 1.9072.
    - The per-chunk comparison that does not depend on z3's choice: the measured duration of each GPU chunk shape that
      every run executed (the mean over that run's schedules with that GPU chunk). The GPU chunk is the makespan-setting
      chunk in every tree schedule.

    | tree GPU chunk | before 1/2/3 | after 1/2/3 | mean before → after | Δ | before spread |
    |---|---|---|---|---|---|
    | stages 1–3 | 1.9298 / 1.9112 / 1.9291 | 1.9225 / 1.9309 / 1.9362 | 1.9234 → 1.9299 | +0.0065 (+0.3 %) | 0.0186 |
    | stages 1–4 | 2.1178 / 2.1316 / 2.1069 | 2.1348 / 2.0915 / 2.0967 | 2.1188 → 2.1077 | −0.0111 (−0.5 %) | 0.0247 |
    | stages 1–5 | 2.0567 / 2.0754 / 2.0794 | 2.0803 / 2.0334 / 2.0653 | 2.0705 → 2.0597 | −0.0108 (−0.5 %) | 0.0227 |
    | stages 1–6 | 2.2170 / 2.1945 / 2.2255 | 2.2374 / 2.2023 / 2.2077 | 2.2123 → 2.2158 | +0.0035 (+0.2 %) | 0.0311 |

    - Per-stage tree × VK GPU time, in ms. Each arm's value is the GPU-timestamp p50 per stage, as the median over that
      arm's 5 profiling runs. "isolated" and "interference" are the two profiling scenarios.

    | stage | before 1/2/3 | after 1/2/3 | mean before → after | Δ | before spread |
    |---|---|---|---|---|---|
    | s1 isolated | 0.08084 / 0.08092 / 0.08112 | 0.08096 / 0.08108 / 0.08072 | 0.08096 → 0.08092 | −0.00004 (−0.0 %) | 0.00028 |
    | s2 isolated | 0.78052 / 0.78084 / 0.78084 | 0.78060 / 0.78064 / 0.78100 | 0.78073 → 0.78075 | +0.00001 (+0.0 %) | 0.00032 |
    | s3 isolated | 0.17816 / 0.17832 / 0.17800 | 0.17808 / 0.17732 / 0.17708 | 0.17816 → 0.17749 | −0.00067 (−0.4 %) | 0.00032 |
    | s4 isolated | 0.13336 / 0.13284 / 0.12920 | 0.12932 / 0.12876 / 0.12880 | 0.13180 → 0.12896 | −0.00284 (−2.2 %) | 0.00416 |
    | s5 isolated | 0.04100 / 0.04136 / 0.04108 | 0.04116 / 0.04144 / 0.04112 | 0.04115 → 0.04124 | +0.00009 (+0.2 %) | 0.00036 |
    | s6 isolated | 0.11560 / 0.11548 / 0.11644 | 0.10768 / 0.10728 / 0.10920 | 0.11584 → 0.10805 | −0.00779 (−6.7 %) | 0.00096 |
    | s7 isolated | 0.23088 / 0.23052 / 0.23072 | 0.23108 / 0.23024 / 0.23040 | 0.23071 → 0.23057 | −0.00013 (−0.1 %) | 0.00036 |
    | s1 interference | 0.08120 / 0.08072 / 0.08092 | 0.08052 / 0.08048 / 0.08044 | 0.08095 → 0.08048 | −0.00047 (−0.6 %) | 0.00048 |
    | s2 interference | 0.77984 / 0.78012 / 0.77972 | 0.77976 / 0.77976 / 0.78024 | 0.77989 → 0.77992 | +0.00003 (+0.0 %) | 0.00040 |
    | s3 interference | 0.18024 / 0.18016 / 0.18008 | 0.17984 / 0.17980 / 0.18008 | 0.18016 → 0.17991 | −0.00025 (−0.1 %) | 0.00016 |
    | s4 interference | 0.21748 / 0.21952 / 0.22092 | 0.21400 / 0.21484 / 0.21308 | 0.21931 → 0.21397 | −0.00533 (−2.4 %) | 0.00344 |
    | s5 interference | 0.04264 / 0.04268 / 0.04276 | 0.04268 / 0.04272 / 0.04268 | 0.04269 → 0.04269 | −0.00000 (−0.0 %) | 0.00012 |
    | s6 interference | 0.10652 / 0.10640 / 0.10656 | 0.10648 / 0.10460 / 0.10708 | 0.10649 → 0.10605 | −0.00044 (−0.4 %) | 0.00016 |
    | s7 interference | 0.25412 / 0.25376 / 0.25384 | 0.25476 / 0.25352 / 0.25444 | 0.25391 → 0.25424 | +0.00033 (+0.1 %) | 0.00036 |

      - Deltas beyond the before spread: isolated s3 (−0.00067 vs 0.00032) and s6 (**−0.00779, −6.7 %** vs 0.00096; all
        three after-runs are below all three before-runs); interference s3 (−0.00025 vs 0.00016), s4 (−0.00533 vs 0.00344)
        and s6 (−0.00044 vs 0.00016). All of them are in the faster direction.
      - No stage got slower beyond its before spread. The largest increase is interference s7, +0.00033 vs a spread of
        0.00036.
    - Host side, as the tooling exposes it: the single-stage dispatch round trip, wall-clock p50 minus GPU p50, in ms (the
      same `bm-prof-tree-vk` run in each arm, with and without `BT_PROF_GPU_WALLCLOCK=1`).

    | stage | before 1/2/3 | after 1/2/3 | mean before → after | Δ | before spread |
    |---|---|---|---|---|---|
    | s1 | 0.07014 / 0.07221 / 0.07476 | 0.07653 / 0.07681 / 0.07369 | 0.07237 → 0.07568 | +0.00331 (+4.6 %) | 0.00462 |
    | s2 | 0.07193 / 0.10240 / 1.29970 | 0.07016 / 0.07066 / 0.06990 | 0.49135 → 0.07024 | −0.42111 (−85.7 %) | 1.22777 |
    | s3 | 0.07654 / 0.07315 / 0.07870 | 0.06771 / 0.06882 / 0.06639 | 0.07613 → 0.06764 | −0.00849 (−11.2 %) | 0.00555 |
    | s4 | 0.07342 / 0.06762 / 0.07317 | 0.06586 / 0.07452 / 0.07443 | 0.07140 → 0.07160 | +0.00020 (+0.3 %) | 0.00580 |
    | s5 | 0.06533 / 0.06508 / 0.06499 | 0.06590 / 0.06456 / 0.06525 | 0.06514 → 0.06524 | +0.00010 (+0.2 %) | 0.00034 |
    | s6 | 0.06968 / 0.07097 / 0.07463 | 0.07469 / 0.07217 / 0.07057 | 0.07176 → 0.07248 | +0.00072 (+1.0 %) | 0.00496 |
    | s7 | 0.06641 / 0.07059 / 0.06998 | 0.06810 / 0.06817 / 0.05776 | 0.06899 → 0.06468 | −0.00431 (−6.3 %) | 0.00417 |
    | median over stages | 0.07014 / 0.07097 / 0.07463 | 0.06810 / 0.07066 / 0.06990 | 0.07191 → 0.06955 | −0.00236 (−3.3 %) | 0.00449 |

      - Per stage it is 64.6–78.7 µs on both sides, apart from three outliers (57.8, 102.4 and 1299.7 µs). The 1.3 ms
        outlier is before-3's stage 2, which is why the summary row uses the median over stages.
      - Deltas beyond the before spread are s3 (−0.00849 vs 0.00555) and s7 (−0.00431 vs 0.00417, driven by after-3's
        57.8 µs), both faster. No stage got slower beyond its spread.
      - That round trip is dominated by submit + fence wait + thread wake-up, so it cannot resolve binding cost; hence
        the probe below.
    - *Recording-cost probe (rocky-ryzen, not committed).* It measures what design D6 targets, the CPU time to record
      one bind + dispatch, with no submit:
      - 20 binds per command buffer (one tree stage 1–7 chunk), 1900 measured reps after 100 warm-up, pinned to CPU 2.
      - before: `update_descriptor_set(i, {…})` + `record_bind_core(cmd, i)` with `num_sets(20)`; after: `record_bind`.
      - Interleaved 3×; GPU busy 0, load ≤ 0.28. p50 ns per bind:

      | buffers | before | after, push descriptors | after, forced fallback |
      |---|---|---|---|
      | 2 (radix histograms) | 124.2 / 124.2 / 127.2 | 121.2 / 108.7 / 131.2 | 214.4 / 208.9 / 207.9 |
      | 13 (build octree) | 301.6 / 303.1 / 304.1 | 322.1 / 309.1 / 321.6 | 400.8 / 390.2 / 384.2 |

  - **Verdict on the 7.7 check (makespan): met.** This verdict covers only the check written in 7.7. It does **not**
    satisfy the resource-binding requirement "Binding overhead does not regress", which stays open as 7.8. The per-chunk
    makespan does not increase beyond run-to-run noise. Every matched
    GPU chunk moves by ≤ 0.0111 ms (≤ 0.52 %), inside its before spread. The best makespan, the all-apps constants and the
    tree-only constants move inside their before spreads too. The tree-only GPU constant (+0.0165 ms) needed the third
    pair: after two pairs the ranges did not overlap (before 0.8592–0.8680, after 0.8752–0.8889). before-3 (0.8248) shows
    the before spread is 0.043 ms.
  - **Finding: the resource-binding requirement is not met, and is not tuned around (open: 7.8).** The requirement says
    "Per-dispatch host-side binding cost SHALL be no greater than the descriptor-update path it replaces". On RADV:
    - push descriptors are no cheaper to record at 2 buffers (mean 125.2 → 120.4 ns, inside the after spread of 22.5);
    - they are **+14.7 ns per bind (+4.8 %)** at 13 buffers, beyond the before spread of 2.5 ns;
    - the transient-pool fallback costs **+85 to +89 ns per bind** (+68 % / +29 %).
    - The makespan scenario is unaffected: a whole tree chunk binds 20 times, so binding costs a few µs against a
      ~1.7 ms makespan whose ~0.85 ms per-chunk GPU overhead is clearly not descriptor binding.
    - This also means D6's premise (proposal "Why": per-item descriptor rewrites dominate the µs-scale tree × VK
      overhead) does not hold on rocky-ryzen.
    - Not measured: the recording cost on the phones (Mali, Xclipse), where push descriptors may behave differently.
    - The requirement text is left unchanged. The makespan result does not discharge its SHALL.
- [ ] 7.8 **BLOCKED pending a justified, reviewed spec/design resolution, and Phase C is incomplete until then. Open, unresolved:** the resource-binding requirement "Binding overhead does not regress" ("Per-dispatch host-side binding cost SHALL be no greater than the descriptor-update path it replaces") is not satisfied by the implemented code. Per 7.7, on rocky-ryzen push descriptors cost +14.7 ns per bind (+4.8 %) at 13 buffers, and the fallback +85–89 ns per bind. Close it only with a compliant implementation, or with a justified spec/design change reviewed on its own. Neither the makespan result nor a narrower reading of the requirement closes it.
  - Possible next steps, none taken: measure the recording cost on the Pixel 7a and the Samsung; profile RADV's
    push-descriptor path at 13 bindings; or decide, with a reviewed spec and design change, what the requirement should
    guarantee.

## 8. Phase C — persisted pipeline cache (pipeline-config) + phase gate

- [x] 8.1 Give `Engine` one pipeline cache shared by all `Algorithm`s. Load it from `$BT_VK_PIPELINE_CACHE_DIR` (default: the current directory) after validating the header against vendorID, deviceID and UUID, and save it atomically on destruction (design D7). Verify: a second run of a VK test on `rocky-ryzen` logs a cache hit at debug level.
  - *Evidence (1e732ea):*
    - `Engine` owns one `vk::PipelineCache`, passed to every `Algorithm` it makes; the per-Algorithm cache is gone.
    - It loads `$BT_VK_PIPELINE_CACHE_DIR/bt-vk-pipeline-cache-<vendorID>-<deviceID>-<uuid>.bin`, defaulting to the
      working directory. (The `bt-vk-pipeline-cache-` prefix was added to design D7's name, so a stray file says what it
      is.) It saves on `~Engine` via `<file>.tmp.<pid>` + `rename`.
    - `pipeline_cache_header_mismatch()` checks headerSize = 32, headerVersion = 1, vendorID, deviceID and
      pipelineCacheUUID before the data reaches `vkCreatePipelineCache`.
    - Pipelines are still built only in dispatcher constructors. Each logs at debug level whether the driver reports an
      application-cache hit (`VkPipelineCreationFeedback`, core 1.3, chained with no per-stage array).
    - rocky-ryzen, run twice into an empty cache directory:
      `BT_VK_PIPELINE_CACHE_DIR=<dir> ./bm-fully-tree-vk --device minipc -l debug -s 1 -e 1 -t 1` (3 engines × 11
      pipelines).
      - Run 1: the first engine logs `Pipeline cache: no readable …, compiling cold` (11 misses). Engines 2 and 3 log
        `Pipeline cache: loaded 25816 / 25828 bytes from …` (22 hits).
      - Run 2: all 3 engines load, **33/33 `pipeline cache hit`**.
    - Pixel 7a and Samsung, the same check with `test-kiss-vk --gtest_filter=VulkanWorkGroupSize.DeclaredSizeReachesShader`:
      - run 1: `compiling cold` → `pipeline cache miss` → `saved 697` (Pixel) / `1052` (Samsung) bytes;
      - run 2: `loaded … bytes` → `pipeline cache hit`.
  - *Review fix (0b127b6):* the cache file I/O is now checked end to end; see 8.2 for the two defects and their tests.
    - The header check now runs only on a complete regular-file read: open, fstat (regular files only), and a read
      loop that rejects any error or short read.
    - The save is `write_file_atomically()`: every `write()` and the `close()` are checked, and it renames only after
      both succeed.
    - The same rocky-ryzen check at 0b127b6:
      - run 1: the first engine logs `Pipeline cache: no file …, compiling cold`, then 11 misses + 22 hits;
      - run 2: `Pipeline cache: loaded 25820 bytes from …` → **33/33 `pipeline cache hit`**.
  - *Second review fix (8e8302b + 52b582f):* the read never blocks and never throws out of Engine construction.
    `read_pipeline_cache_file()`:
    - opens with `O_NONBLOCK`, then requires a regular file by `fstat` on the opened descriptor;
    - reads and checks the 32-byte header before allocating the payload;
    - rejects a file over `kMaxPipelineCacheBytes` (64 MiB) unread. The largest fleet cache is 62,664 bytes (Pixel 7a).
    - treats a failed allocation or read as rejected data;
    - closes the descriptor via RAII on every path.
    - `load_pipeline_cache` also turns any other exception from the read into a cold compile.
    - design D7 records these guards and the bound.
    - The same rocky-ryzen check at 52b582f: run 1 `no file …, compiling cold`, 11 misses + 22 hits; run 2 `loaded 25820
      bytes`, **33/33 hits**.
- [x] 8.2 Add tests for a corrupt cache file, a cache from a foreign device or UUID, and a read-only directory. Verify: all three runs complete with unchanged results and a cold compile.
  - *Evidence (1e732ea):*
    - `PipelineCacheTest.*` in `test-kiss-vk`. Each test uses its own `bt-vk-cache-test-<pid>-<name>` directory under the
      working directory and removes it afterwards; 0 were left on any target. Every engine lifetime runs a kernel and
      checks its output ({128, 64, 2, 1}).
      - `SecondRunLoadsTheSavedCache`: kMissing, then kLoaded.
      - `CorruptFileIsIgnored`: 4096 bytes of 0xAB, and a valid file cut to 20 bytes → kRejected.
      - `ForeignDeviceOrDriverIsIgnored`: a valid file with one byte flipped in headerSize, headerVersion, vendorID,
        deviceID, or the first or last UUID byte → kRejected each time. The untouched file → kLoaded.
      - `ReadOnlyDirectoryRunsWithoutPersisting`: the directory is chmod 0555 and asserted unwritable → two runs, both
        kMissing, and the directory stays empty.
    - All four are OK on rocky-ryzen, the Pixel 7a, the Samsung and the build box. kRejected/kMissing means the
      application cache started empty, i.e. a cold compile.
    - The full tree suite under each condition: 15/15 PASSED in every case (exact oracle match) on rocky-ryzen and the
      Pixel 7a. Each condition is a dedicated directory set through `BT_VK_PIPELINE_CACHE_DIR`; the phone deploy location
      is unchanged.
      - Corrupt (4096 random bytes) and foreign (a valid file with one UUID byte changed): the file is ignored and
        replaced by a valid cache. Both conditions end with the same md5 (rocky 69cfe8bd, Pixel 5b854a52).
      - Read-only (chmod 555): 0 files written. rocky logs `Pipeline cache: not saved (cannot write ….tmp.<pid>),
        continuing`.
    - rocky-ryzen, debug log of the first engine (`bm-fully-tree-vk -l debug`, 11 misses then 22 hits):
      - `ignoring … (headerSize 2552050312, expected 32), compiling cold` for the corrupt file;
      - `ignoring … (pipelineCacheUUID differs from the driver's), compiling cold` for the foreign one.
  - *Review fix (0b127b6): two defects in the 1e732ea cache I/O, reproduced with the reviewer's
    `/tmp/phase-c-review-cache-repro.py` on the build box before the fix.*
    - **Read:** a cache path that cannot be read escaped Engine construction. With a directory at the path,
      `VulkanEngineTest.EngineInitialization` failed with `std::__ios_failure` "… Is a directory", because the
      `istreambuf_iterator` read was outside any handler. Now it logs `ignoring … (not a regular file), compiling
      cold`, the status is kRejected, and the following save fails with `cannot rename to …: Is a directory`, previous
      file kept.
    - **Save:** a truncated write replaced a valid cache and reported success. `file.good()` was checked before the
      buffered stream flushed and closed. With SIGXFSZ ignored and `RLIMIT_FSIZE` = 32, a valid 1056-byte cache became
      32 bytes while the log said `saved 1056 bytes`. Now the save logs `not saved to … (write failed after 32 of
      1056 bytes: File too large), previous file kept`, and the file stays 1056 bytes.
    - The repro after the fix: build box "original bytes 1056 current bytes 1056", directory case exit 0. rocky-ryzen
      "original bytes 1480 current bytes 1480"; the attempted write was 1488 bytes, failed after 32, directory case
      exit 0.
    - New tests:
      - `PipelineCacheTest.UnreadableCacheFileIsIgnored`: a directory at the cache path gives kRejected; the kernel
        result is correct, the path is still a directory, and no temporary file is left.
      - `PipelineCacheFile.FailedWriteKeepsThePreviousFile`: hardware-free. It writes a 1056-byte file, lowers the soft
        `RLIMIT_FSIZE` to 32 (SIGXFSZ ignored) around one `write_file_atomically` call and restores both, then expects
        "write failed after 32 of 1056 bytes", the previous bytes intact and no temporary file. Control: without the
        limit the same call replaces the file.
      - Not separately induced: a failure of `close()` itself, which has no portable trigger on a local filesystem.
        The code checks it the same way as `write()`.
    - *Mutation checks (build box, not committed):*
      - with the old `ifstream` read restored, `UnreadableCacheFileIsIgnored` FAILS (only that test);
      - with failed `write()`s ignored, `FailedWriteKeepsThePreviousFile` FAILS (only that test);
      - restored, all 6 pass.
    - All 6 cache tests (the four above plus these two) are OK at 0b127b6 on rocky-ryzen, the Pixel 7a, the Samsung and
      the build box.
    - rocky-ryzen, the full tree suite with `BT_VK_PIPELINE_CACHE_DIR` set to each condition at 0b127b6 → 15/15 PASSED
      in every case. The first engine's debug line for each:
      - corrupt: `ignoring … (headerSize 3333445683, expected 32), compiling cold`;
      - foreign UUID: `ignoring … (pipelineCacheUUID differs from the driver's), compiling cold`;
      - a directory at the path: `ignoring … (not a regular file), compiling cold`, and it is still a directory
        afterwards;
      - read-only: `no file …, compiling cold`, with 0 files written.
  - *Second review fix (8e8302b + 52b582f): two more defects in 0b127b6, reproduced first on the build box.*
    - **FIFO at the cache path** (`/tmp/phase-c-review-fifo-repro.py`): the blocking `O_RDONLY` open waited for a writer
      before `fstat` could reject the file, so `VulkanEngineTest.EngineInitialization` timed out after 5 s. Now: `FIFO
      case exit: 0`, logging `ignoring … (not a regular file), compiling cold`.
    - **Sparse 2 GiB file with a valid header** (`/tmp/phase-c-review-cache-size-repro.py`, `RLIMIT_DATA` 512 MiB): the
      whole payload was allocated from the unchecked size, and Engine construction threw `std::bad_alloc`. Now: exit 0,
      logging `ignoring … (2147483648 bytes exceeds the 67108864-byte limit), compiling cold`.
    - New tests (test-kiss-vk 18 → 21):
      - `PipelineCacheTest.FifoAtCachePathIsIgnoredWithoutBlocking`, bounded: Engine construction runs on a worker
        thread. If it has not finished after 5 s, the test opens the FIFO's write end to release the blocked open, then
        fails; the suite never hangs.
      - `PipelineCacheTest.OversizedCacheFileIsIgnored`: a valid file extended sparsely to 64 MiB + 1 gives kRejected.
      - `PipelineCacheFile.OversizedOrUnallocatableDataIsRejected`, hardware-free:
        - A 2 GiB file is rejected for size, with nothing allocated, even with the soft `RLIMIT_AS` lowered to VmSize +
          16 MiB.
        - A 48 MiB file (under the bound) is read in a fresh child, this binary re-exec'd with `--cache-alloc-child`
          under `RLIMIT_AS` = VmSize + 16 MiB. The child must report `cannot allocate 50331648 bytes` as kRejected.
        - Control: with no limit the 48 MiB file loads in full.
    - *Why a child process and `RLIMIT_AS` (52b582f, found on the phones):*
      - `RLIMIT_DATA` does not cap bionic's allocator; the 48 MiB read loaded on both phones.
      - In-process, memory the test process already holds (glibc keeps up to ~64 MiB at the heap top) could serve the
        allocation: the case passed alone but failed after the earlier tests locally.
      - In a fresh child it fails to allocate deterministically: 3/3 full local runs, both phones and rocky-ryzen.
    - *FIFO on the phones:* Android's SELinux policy denies the shell domain `fifo_file create` in `/data/local/tmp`
      (`avc: denied { create } … scontext=u:r:shell:s0 tcontext=u:object_r:shell_data_file:s0 tclass=fifo_file`; plain
      `mkfifo` and `mknod p` fail too). A FIFO cannot exist at the cache path there.
      - The test therefore **skips** on the two phones, only when `mkfifo` fails with EACCES/EPERM, and says so. Any
        other `mkfifo` error fails the test.
      - The test runs and passes on rocky-ryzen and the build box, where the defect was reproducible. This is the one
        skip in the Phase C gate.
    - *Mutation checks (build box, not committed). Each mutation fails only the tests aimed at it, and the restored code
      passes 21/21:*
      - a blocking open fails `FifoAtCachePathIsIgnoredWithoutBlocking`, in 6 s with no hang;
      - removing the size bound fails `OversizedCacheFileIsIgnored` and `OversizedOrUnallocatableDataIsRejected`;
      - leaving `bad_alloc` unhandled fails `OversizedOrUnallocatableDataIsRejected`.
    - Through real Engines at 52b582f:
      - rocky-ryzen `bm-fully-tree-vk -l debug`, first engine:
        - a FIFO: `ignoring … (not a regular file), compiling cold`;
        - a valid file extended to 2 GiB: `ignoring … (2147483648 bytes exceeds the 67108864-byte limit), compiling
          cold`;
        - corrupt, foreign UUID, directory and read-only behave as above.
      - The full tree suite under each of the six conditions → 15/15.
      - On the Pixel 7a and the Samsung, the 2 GiB file logs the same `exceeds … limit, compiling cold` line.
      - All three review repro scripts (truncated write/directory, FIFO, 2 GiB) pass on the build box and on
        rocky-ryzen.
- [x] 8.3 Ignore `*.bin` pipeline-cache files in `.gitignore` and the fleet cleanup scripts, and document `BT_VK_PIPELINE_CACHE_DIR` in `03-unit-testing.md`. Verify: `git status` is clean after a local VK run.
  - *Evidence (1e732ea + the 03-unit-testing.md location fix):*
    - `.gitignore` gains `bt-vk-pipeline-cache-*.bin`, `bt-vk-pipeline-cache-*.bin.tmp.*` and `bt-vk-cache-test-*/`.
      These patterns are narrower than a blanket `*.bin`, so a future unrelated `.bin` stays visible.
    - `03-unit-testing.md` documents `BT_VK_PIPELINE_CACHE_DIR`: where the file lands for each runner, what is ignored,
      and that the cache is outside timed regions. It also documents `BT_VK_FORCE_DESCRIPTOR_FALLBACK`.
    - What the cleanup scripts do:
      - `just bench-clean` only removed the local `data/` result dirs. The run scripts (`scripts/run-on-{rocky,android,jetson}.sh`,
        the `_test-ssh` recipe, `optimizer/orchestrate/transport.py`) stage into `/tmp/bt` and `/data/local/tmp/bt`, run
        there, and bring back only stdout. They never copy files back, so a cache file cannot leak into results.
      - There it stays as one file per (vendor, device, driver UUID), rewritten on each run. That is the intended warm
        start; a new file appears only after a driver update.
      - A run from the repo root does leave a file there, so `bench-clean` now also deletes
        `bt-vk-pipeline-cache-*.bin` and its temp files from the repo root.
    - Verify: this build box has an iGPU (Ryzen 9600X, RADV), so a local VK run is real. `ctest -L vulkan` → 7/7, and
      `test-kiss-vk` / tree / cifar-dense / cifar-sparse → 16 / 15 / 12 / 12. That left cache files in the repo root and
      in `build/vulkan/{platform,apps/*}`, and `git status --short` was empty.
      - `git check-ignore -v` names the three new patterns for a cache file, its `.tmp.123` copy and a
        `bt-vk-cache-test-*` dir.
      - `just bench-clean` removed the repo-root file.
    - Re-run at 0b127b6 on the build box: `ctest -L vulkan` 7/7 and 18 / 15 / 12 / 12. `git status --short` is empty,
      and 0 `bt-vk-cache-test-*` dirs are left in the repo root or `build/vulkan/{platform,apps/*}`.
      `XDG_RUNTIME_DIR=/tmp BUN_TMPDIR=$TMPDIR just fmt-check` passed all five stages (clang-format, ruff, gersemi,
      shfmt, prettier). In this sandbox Bun needs `BUN_TMPDIR=$TMPDIR`; `/tmp` gives EACCES.
- [x] 8.4 Phase C gate: VK fleet gate green, including the Pixel 7a and Samsung with push descriptors active. Verify: the debug log on each phone shows which binding path is in use.
  - *VK fleet gate, re-run at 52b582f (the final code; later commits are docs-only). Earlier gates at 1e732ea and 0b127b6
    were also green.*
    - local `ctest -L omp` → 9/9 passed.
    - Bake / clean-tree check: `cmake --build build/vulkan --target bake-shaders` → no-op, `git status` clean. Phase C
      changes no shader (`git diff --stat 7cc0916 HEAD -- platform/engine/vulkan/shaders` is empty).
    - rocky-ryzen (`--device minipc`, `VK_LOADER_LAYERS_DISABLE='~all~'`), the same results on the default path (push
      descriptors) and with `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1`:
      - `test-kiss-vk` / tree / cifar-dense / cifar-sparse → 21 / 15 / 12 / 12 PASSED, 0 skipped;
      - `ctest -L vulkan` → 7/7 (6 min 0 s on each path);
      - tree at `BT_VK_SUBGROUP_SIZE=32` → 15/15;
      - with `BT_VK_VALIDATION=1` (layer inserted 30 / 16 / 12 / 12 times; the FIFO and oversized Engine tests add 2
        engines each) → 0 `Validation Error` lines and 0 `05137`, plus 0 in tree pinned at 32;
      - 0 `bt-vk-cache-test-*` dirs left.
    - Pixel 7a and Samsung (`scripts/run-on-android.sh <serial> test-kiss-vk test-tree-vk test-cifar-dense-vk
      test-cifar-sparse-vk`) → script exit 0 on both:
      - `test-kiss-vk` 20 passed + **1 skipped**: `FifoAtCachePathIsIgnoredWithoutBlocking`, because SELinux forbids the
        FIFO (see 8.2);
      - tree / cifar-dense / cifar-sparse → 15 / 12 / 12;
      - Samsung tree at `BT_VK_SUBGROUP_SIZE=32` → 15/15; 0 test dirs left.
    - Binding path, from the debug log (`SPDLOG_LEVEL=debug ./test-kiss-vk --gtest_filter=VulkanEngineTest.EngineInitialization`):
      `Descriptor binding: push descriptors (VK_KHR_push_descriptor)` on rocky-ryzen (Radeon 780M), the Pixel 7a
      (Mali-G710) and the Samsung (Xclipse 940), as expected from gpuinfo. With the forced fallback on rocky-ryzen:
      `transient descriptor pool (forced by BT_VK_FORCE_DESCRIPTOR_FALLBACK=1)`.
    - test-kiss-vk grew from 10 to 21 tests:
      - `BothPaths/VulkanBinding.SameKernelManyBuffersInOneCommandBuffer/{PushDescriptors,Fallback}`;
      - `PipelineCacheTest.{SecondRunLoadsTheSavedCache,CorruptFileIsIgnored,ForeignDeviceOrDriverIsIgnored,UnreadableCacheFileIsIgnored,FifoAtCachePathIsIgnoredWithoutBlocking,OversizedCacheFileIsIgnored,ReadOnlyDirectoryRunsWithoutPersisting}`;
      - `PipelineCacheFile.{FailedWriteKeepsThePreviousFile,OversizedOrUnallocatableDataIsRejected}`.
    - Also green on the build box's own iGPU (RADV RAPHAEL_MENDOCINO): 21 / 15 / 12 / 12 (and `test-kiss-vk` 21/21
      under the forced fallback), `ctest -L vulkan` 7/7.
    - Formatting:
      - `XDG_RUNTIME_DIR=/tmp BUN_TMPDIR=$TMPDIR BUN_INSTALL=$TMPDIR/bun just fmt-check` → exit 0, through the prettier
        stage. Here `$TMPDIR` is a writable per-session directory; Bun fails with EACCES when its temp dir isn't
        writable.
      - The prettier step run directly (`git ls-files '*.json' ':(exclude)dashboard/*' | xargs bunx prettier@3.8.4
        --check`) → "All matched files use Prettier code style!" over 19 files, exit 0.
    - **Phase C is not complete: 7.8 is open.** The gate is green, but the resource-binding requirement "per-dispatch
      host-side binding cost SHALL be no greater than the descriptor-update path it replaces" is not satisfied. On RADV
      it costs +14.7 ns per bind at 13 buffers, and the fallback costs +85–89 ns. The unchanged makespan does not
      discharge it.

## Workflow follow-up

- Decide whether a standing `BT_VK_VALIDATION=1` CTest job on `rocky-ryzen` is wanted (design, Open Questions).
- Archive the change after all three phase PRs are merged to `dev`, then sync the five `vulkan-engine/*` specs into `openspec/specs/`.
