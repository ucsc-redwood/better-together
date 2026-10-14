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
    Every log has **0** `Validation Error` lines, and 25/26 exit 0.
  - **Caveat:** `bm-fully-tree-vk` aborts with `n_unique is not set` (`apps/tree/tree_appdata.hpp:154`) before its last
    teardown. This is **pre-existing and unrelated**: identical abort at 4d8b376 with `VK_LOADER_LAYERS_DISABLE='~all~'`.
    Its earlier engine lifetimes in the same log tear down with 0 errors. Not fixed here (out of Phase A scope).
  - *Re-run at b7ac417* (final Phase A code, with the minimal feature set from group 3): identical. All 26 logs have 0
    `Validation Error` lines; the only non-zero exit is the same pre-existing `bm-fully-tree-vk` abort.
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

- [ ] 5.1 In `BaseEngine`, query `subgroupSize` and the subgroup-size-control properties. Enable `subgroupSizeControl` when it is supported and COMPUTE is in `requiredSubgroupSizeStages`. Expose `Engine::subgroup_size()` (design D5). Verify: on each VK target, a debug log line shows the chosen size and whether it is pinned.
- [ ] 5.2 Add the SPIR-V scan for `SpecId` and `LocalSize` and the `Algorithm` support for specialization constants. Chain `PipelineShaderStageRequiredSubgroupSizeCreateInfo` when pinning is available, and check the workgroup size against `requiredSubgroupSize × maxComputeWorkgroupSubgroups`. Verify: a unit test covers the scan on a committed `*_spv.h`.
- [ ] 5.3 Merge `multi_radixsort_warp{16,32,64}.comp` into `multi_radixsort.comp` with `constant_id = 0` for `SUBGROUP_SIZE`. Re-bake, delete the three old `.comp` and `*_spv.h` files, and update `all_shaders.hpp`. Verify: the bake / clean-tree check passes.
- [ ] 5.4 Switch `apps/tree/vulkan/dispatchers.cpp` (and any `profiler/tree-vk` user) to the single shader with `subgroup_size_dependent()`. Verify: `ctest -L vk` tree suites match the oracle exactly on `rocky-ryzen`, the Pixel 7a (sg16) and the Samsung (sg32).
- [ ] 5.5 Run the tree VK suite on `rocky-ryzen` with the `gpu.subgroup_size` field temporarily removed from `minipc.json`. Verify: the suite passes, which demonstrates the registry is no longer required.
- [ ] 5.6 Downgrade `get_vulkan_warp_size()` to a cross-check that warns on a mismatch (spec "Registry subgroup value is a cross-check only"). Verify: a unit test with a mismatching registry value produces the warning.
- [ ] 5.7 Compare tree × VK stage-2 time before and after on `rocky-ryzen`, to cover the wave32/wave64 risk. Verify: the numbers are recorded in the PR; if it regressed beyond noise, pin to `minSubgroupSize` and re-measure.
- [ ] 5.8 Update `01-hardware.md`, `devices/README.md` (subgroup_size is informational) and `testing-status.md` §F (close the subgroup gap). Verify: the docs mention the runtime query and not the `--device` map.

## 6. Phase B — workgroup size via specialization (pipeline-config) + phase gate

- [ ] 6.1 Make `Algorithm::work_group_size()` feed `local_size_{x,y,z}_id` spec constants when the shader declares them. Otherwise it fails if the value disagrees with the SPIR-V literal `LocalSize`. Verify: a unit test shows a mismatching declared size throws, and all current dispatchers still build.
- [ ] 6.2 Reject a declared workgroup size above the device's `maxComputeWorkGroupSize` or invocation limits, with an error naming both. Verify: a unit test with an oversized request.
- [ ] 6.3 Phase B gate: VK fleet gate green. Verify: the PR lists the results for each target.

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
