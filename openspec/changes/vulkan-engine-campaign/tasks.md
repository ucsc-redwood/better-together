# Tasks

Each phase (A = groups 1–4, B = groups 5–6, C = groups 7–8) is one PR into `dev`. "VK fleet gate"
means:
- `ctest -L vk` passes on `rocky-ryzen`, the Pixel 7a and the Samsung (via `scripts/run-on-rocky.sh`
  and `scripts/run-on-android.sh`);
- plus `ctest -L omp` locally;
- plus the shader bake / clean-tree check.

## 1. Phase A — host visibility (execution-sync)

- [ ] 1.1 Record a COMPUTE_SHADER/SHADER_WRITE → HOST/HOST_READ memory barrier in `Sequence::cmd_end()`, before the bottom-of-pipe timestamp (design D1). Verify: `ctest -L vk` is green on `rocky-ryzen`.
- [ ] 1.2 Run the tree, cifar-dense and cifar-sparse VK suites with `BT_VK_VALIDATION=1` and `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT` on `rocky-ryzen`. Verify: no shader-write/host-read hazard is reported.
- [ ] 1.3 Add a `bugs-found.md` entry for the missing host-visibility barrier (root cause, spec citation, fix commit). Verify: the entry exists and links the commit.

## 2. Phase A — object lifetime

- [ ] 2.1 Implement `~Algorithm()` to destroy the pipeline, pipeline layout, descriptor pool, descriptor-set layout, shader module and the per-Algorithm pipeline cache, guarding null handles (design D2). Verify: `ctest -L vk` is green on `rocky-ryzen`.
- [ ] 2.2 Add a VK test that constructs, destroys and reconstructs a dispatcher in one process. Verify: it passes, and with `BT_VK_VALIDATION=1` it reports no `VUID-vkDestroyDevice-device-05137`.
- [ ] 2.3 Run every VK test and `bm-*-vk` binary once with `BT_VK_VALIDATION=1` on `rocky-ryzen`. Verify: the exit logs contain no undestroyed-object errors.
- [ ] 2.4 Add a `bugs-found.md` entry for the Algorithm leak, as a follow-up to §9. Verify: the entry exists.

## 3. Phase A — feature enablement (device-init)

- [ ] 3.1 Replace `check_vulkan_12_features` with a single required-feature table, using the query → check → minimal-enable flow from design D3. A missing feature throws an error naming it. Verify: `ctest -L vk` is green on all three VK targets.
- [ ] 3.2 Add a unit test for the feature check: given a supported-features struct missing `shaderInt8`, the check reports `shaderInt8`. Verify: the test is labelled `vk` and passes, and it skips on hosts without an iGPU.
- [ ] 3.3 Confirm that `has_integrated_gpu()` returns false (and the suites skip) when device creation throws on missing features. Verify: covered by the same unit test, or by code inspection noted in the PR.

## 4. Phase A — validation opt-in (device-init) + phase gate

- [ ] 4.1 Make `Engine()` request validation only when `BT_VK_VALIDATION=1`, and demote the layer-list dump to `debug` (design D4). Verify: on `rocky-ryzen`, a run without the variable shows no validation layer in the debug log, and a run with it shows the layer enabled.
- [ ] 4.2 Document `BT_VK_VALIDATION` in `docs/instruction-for-ai/03-unit-testing.md`. Note there that fleet scripts also set `VK_LOADER_LAYERS_DISABLE`. Verify: the documented command runs as written on `rocky-ryzen`.
- [ ] 4.3 Phase A gate: VK fleet gate green. Then do one `fit_overhead` refit on `rocky-ryzen` and record the per-chunk overhead delta in the PR. Verify: the delta is within run-to-run noise, or the new constants are committed with an explanation.

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
