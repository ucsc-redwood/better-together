# Design

## Context

The motivation is in proposal.md (Why). The current state that shapes the approach:

- **Engine layering.**
  - `BaseEngine` creates the instance, device, queue and VMA allocator.
  - `Engine` adds the `VulkanMemoryResource`.
  - `Algorithm` is one shader. It owns its pipeline, layout, descriptor pool and sets, and is built
    through a builder (`num_buffers`, `num_sets`, `push_constant`, `work_group_size`, `build`).
  - `Sequence` is one command buffer, fence and timestamp query pair.
- **Dispatcher pattern.** Each app's `VulkanDispatcher` holds `engine`, then `seq`, then
  `cached_algorithms`, in that member order. Destruction therefore runs algorithms → sequence →
  engine, so destroying `Algorithm` objects in its destructor happens before `vkDestroyDevice`
  without reordering anything. Stages are recorded into one command buffer per chunk
  (`dispatch_multi_stage`), with compute→compute barriers between them, and run with one submit and
  one fence wait.
- **Shaders.** Shaders are GLSL baked to SPIR-V with `glslc --target-env=vulkan1.3`
  (`cmake/bt_shaders.cmake`). The committed `*_spv.h` files are the fallback when glslc is
  absent, which is the case for Android builds. Only `multi_radixsort_warp{16,32,64}.comp` depend
  on the subgroup size and are selected at runtime. They are picked through `get_vulkan_warp_size()`,
  which reads `gpu.subgroup_size` from `devices/*.json`. The other `SUBGROUP`-mentioning shaders
  are already subgroup-agnostic.
- **Validation in practice.** Fleet runs already set `VK_LOADER_LAYERS_DISABLE='~all~'`
  (`optimizer/orchestrate/transport.py`, `scripts/run-on-rocky.sh`). The always-on request in
  `Engine()` therefore mostly affects ad-hoc runs and the CTest suites on `rocky-ryzen`.
- **Fleet Vulkan targets.** `rocky-ryzen` (Radeon 780M, RADV), the Pixel 7a (Mali-G710) and the
  Samsung (Xclipse). gpuinfo.org lists `VK_KHR_push_descriptor` on the Pixel 7a (Mali driver 50)
  and the Galaxy S22 (Xclipse), and RADV exposes it.

## Goals / Non-Goals

**Goals:**
- Each phase (A, B, C) is an independently mergeable, independently revertible PR on `dev`, gated
  on real hardware.
- The engine's public surface that dispatchers use shrinks rather than grows.

**Non-Goals:**
- Kernel algorithm or performance tuning inside shaders, beyond replacing the subgroup `#define`
  with a specialization constant.
- Changing the per-chunk submit / fence-wait threading model.
- CUDA or OMP backends.

## Decisions

### D1. Host visibility: barrier inside `Sequence::cmd_end()`
Record a `vkCmdPipelineBarrier` with src = COMPUTE_SHADER/SHADER_WRITE and
dst = HOST/HOST_READ immediately before the bottom-of-pipe timestamp and `end()`.
- Placing it in `Sequence` covers every dispatcher and every profiler runner from one site.
- The existing VMA `invalidate_touched()` after the fence stays, because it handles the separate
  CPU-cache side on HOST_CACHED memory.
- *Alternative:* put it in each dispatcher's `dispatch_multi_stage`. Rejected: that is several
  sites, and the profilers' per-stage paths would be missed.
- *Timing impact:* this is one barrier per chunk, not per stage. It is recorded before the closing
  timestamp, so measured GPU time absorbs its (tiny) cost consistently.

### D2. Object lifetime: explicit `~Algorithm()`
Destroy, in reverse creation order: pipeline, pipeline layout, descriptor pool (which frees its
sets), descriptor-set layout and shader module. The per-Algorithm pipeline cache is removed
entirely by D7. Destruction guards on null handles, so a partially built `Algorithm` is safe.
- *Alternative:* RAII via `vk::raii` or `vk::Unique*`. Rejected for now. It is a broader type change
  across `algorithm.hpp`, and it fights the dynamic-dispatch setup (`VULKAN_HPP_NO_CONSTRUCTORS`)
  for little gain over six explicit destroys.

### D3. Feature enablement: query, check, then enable a minimal struct
1. Query `PhysicalDeviceVulkan12Features` (and later `Vulkan13Features`) into one struct.
2. Check the required fields against it.
3. Throw if any are missing, with the names collected into the message.
4. Pass a **separate**, zero-initialised struct with only the required fields set to
   `createDevice`.

The required list lives in one table in `base_engine.cpp`. Phase B adds `subgroupSizeControl`
(1.3) to it as *optional* (enabled if supported). Phase C adds the `VK_KHR_push_descriptor`
device extension as optional. `has_integrated_gpu()` already catches throws, so an incapable
device makes the suites skip rather than crash.

### D4. Validation opt-in via `BT_VK_VALIDATION`
- `Engine()` reads `BT_VK_VALIDATION` (`"1"` enables it) instead of hard-coding `true`. The
  existing "requested but not installed → warn and continue" path is unchanged.
- The startup "Available Vulkan Layers" dump moves from `info` to `debug`.
- *Alternative:* key it on CMake build type. Rejected: the benchmarking binaries are Release, but we
  still want to switch validation on for a one-off investigation without rebuilding.
- A CI or fleet job may set `BT_VK_VALIDATION=1` on `rocky-ryzen` CTest runs to turn the
  object-lifetime and sync scenarios into enforced checks (task 4.x).

### D5. Subgroup size: device-derived specialization constant, pinned when possible
- **Engine side.** At device creation, query `PhysicalDeviceSubgroupProperties.subgroupSize`
  (1.1) and, when available, `PhysicalDeviceSubgroupSizeControlProperties` (1.3). Expose the result
  as `Engine::subgroup_size()`.
- **Pinning.**
  - If `subgroupSizeControl` is supported and `requiredSubgroupSizeStages` includes COMPUTE (both
    are needed per `VUID-VkPipelineShaderStageCreateInfo-pNext-02755`), subgroup-dependent
    pipelines chain a `PipelineShaderStageRequiredSubgroupSizeCreateInfo{subgroupSize}`.
  - The workgroup must also fit: local size ≤ `requiredSubgroupSize × maxComputeWorkgroupSubgroups`
    (`VUID-…-02756`). This is trivially true for 256 at sizes 16, 32 and 64, but checked anyway.
- **Test hook.** `BT_VK_SUBGROUP_SIZE=<n>` pins `n` instead of the default, when pinning is available
  and `n` is a power of two in `[minSubgroupSize, maxSubgroupSize]`; otherwise `Engine()` throws
  (it is applied in `Engine`, not `BaseEngine`, so the `has_integrated_gpu()` probe cannot turn a
  bad value into a silent skip). It is how the fleet covers width 32, and it is the "pin to
  `minSubgroupSize`" knob from the risks below. *Added in Phase B,* when the queried fleet turned
  out to be (default / min..max): rocky-ryzen 780M 64 / 32..64, Pixel 7a Mali-G710 16 / 16..16,
  Samsung Xclipse 940 64 / 32..64 — all three pin COMPUTE. No fleet GPU defaults to 32.
- **Why pin.** Without a required size, a 1.3 driver may compile the shader at a different width
  than `subgroupSize` reports. RADV on RDNA3 can run wave32 or wave64. That would silently break the
  `RADIX_SORT_BINS / SUBGROUP_SIZE` shared-array arithmetic. Pinning makes the width a contract.
- **Shader side.** `multi_radixsort_warp{16,32,64}.comp` collapse to one `multi_radixsort.comp`
  with `layout(constant_id = 0) const uint SUBGROUP_SIZE = 32;`. The shared array
  `sums[RADIX_SORT_BINS / SUBGROUP_SIZE]` becomes a spec-constant-sized array, which glslang emits
  as `OpSpecConstantOp`. The three old shaders and their `*_spv.h` are removed.
  `tmp_single_radixsort_warp*` is unused dead code and is left alone (see proposal).
- **Algorithm API.**
  - A builder step `->subgroup_size_dependent()` makes `Algorithm` add the spec constant and the
    pinning.
  - `work_group_size(x, y, z)` feeds spec constants 100, 101 and 102 (the `local_size_x_id` path).
    This applies only for shaders that declare them; for others the existing behaviour (validation
    only) is kept, and it fails if the value disagrees with the SPIR-V's literal `LocalSize`.
  - Detection uses the shader's reflected spec-constant ids. This needs a tiny SPIR-V scan (no new
    dependency) of `OpDecorate … SpecId` and `OpExecutionMode … LocalSize`.
- **Registry.** `get_vulkan_warp_size()` stops being used for shader selection. If the registry
  value exists and differs from the engine's value, log a warning (spec: "Registry subgroup value is
  a cross-check only"). The JSON field stays, documented as informational.

### D6. Binding: push descriptors, transient-pool fallback
- **Primary path.** `Algorithm` records bindings with `vkCmdPushDescriptorSetKHR`, using a set
  layout created with the PUSH_DESCRIPTOR flag. This replaces `update_descriptor_set(set_index, …)`
  + `record_bind_core(cmd, set_index)` with `record_bind(cmd, {buffers…})`, and removes
  `num_sets()`, the descriptor pool and the set vector.
- **Fallback.** When the extension is absent, `Sequence` owns a transient descriptor pool.
  `cmd_begin()` resets it, and each `record_bind` allocates and writes one set from it. Behaviour is
  identical: the dispatcher API is the same, and no planning is needed.
- **Why not the fallback alone?** It is simpler and still removes dispatcher bookkeeping, but it
  keeps the alloc+update per dispatch, which is the CPU cost we are trying to shave on tree × Vulkan.
- **Why not buffer device address?** It would need rewriting every shader's buffer declarations.
  It stays a possible future step.
- **Test hook.** `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1` forces the fallback, so CI on `rocky-ryzen`
  covers both paths.

### D7. Pipeline cache: one per engine, persisted
- `Engine` owns one `vk::PipelineCache`, passed to each `Algorithm`.
- **Load.** On construction it loads `$BT_VK_PIPELINE_CACHE_DIR/<vendorID>-<deviceID>-<uuid>.bin`.
  - The directory defaults to the current working directory. That is writable in every fleet
    location, including `/data/local/tmp` on the phones.
  - Before passing the data to the driver, the file's `VkPipelineCacheHeaderVersionOne` (vendor,
    device, `pipelineCacheUUID`) is validated. On a mismatch or a short or corrupt file it is
    ignored. Drivers are supposed to reject bad blobs, but not all mobile drivers are trusted to.
- **Save.** On `~Engine` it writes via `getPipelineCacheData` to a temp file plus a rename. On any
  I/O error it logs at debug level and continues.
- **Timing.** Pipelines are built in dispatcher constructors, outside any timed region, so profiling
  numbers are unaffected.

## Risks / Trade-offs

- **[Risk]** The host barrier or the feature change perturbs the fitted per-chunk overhead
  constants. **→ Mitigation:** Phase A's gate includes one `fit_overhead` refit on `rocky-ryzen`.
  The expected delta is within noise; if it isn't, refit and record it.
- **[Risk]** Spec-constant-sized shared arrays could hit a glslang or driver bug on Mali.
  **→ Mitigation:** Phase B is gated on the Pixel 7a tree suite. If it fails, size the shared array
  at the maximum (`RADIX_SORT_BINS / 16`) and index with the spec constant. That costs a little
  shared memory and needs no spec-sized array.
- **[Risk]** RADV may report `subgroupSize = 64` while compiling unpinned compute shaders at wave32
  (unverified on the 780M), so pinning to 64 could change the occupancy or performance of the tree
  radix sort. **→ Mitigation:** compare the
  tree × Vulkan stage-2 time before and after on `rocky-ryzen`. Pinning to `minSubgroupSize`
  instead is an option with the same code path.
- **[Risk]** Android builds use the committed `*_spv.h`, so a forgotten re-bake ships stale SPIR-V.
  **→ Mitigation:** run the existing bake/clean-tree check (P5b) in each phase's gate.
- **[Risk]** Push-descriptor support differs between the phone drivers on the fleet and gpuinfo's
  reports. **→ Mitigation:** the fallback path plus the forced-fallback CI hook. The phase gate runs
  on the real phones.
- **[Trade-off]** Validation off by default means a developer can miss errors locally.
  **→ Mitigation:** documented in `03-unit-testing.md`. Optionally a `rocky-ryzen` CTest job with
  `BT_VK_VALIDATION=1`.

## Migration Plan

- **Sequencing.** Phase A → B → C, each a separate PR into `dev`. The gate for each phase:
  - `ctest -L omp` locally;
  - `ctest -L vk` on `rocky-ryzen`, the Pixel 7a and the Samsung;
  - `ctest -L cuda` is not affected.
- **Rollback.** Each phase is self-contained: revert its PR.
- **Phase C and the fallback.** Phase C keeps the fallback path, so turning push descriptors off
  needs only `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1`, with no revert.
- **Docs.** Update the docs in the same PR as the behaviour they describe:
  - `03-unit-testing.md` (validation env var);
  - `01-hardware.md` and `devices/README.md` (subgroup source of truth);
  - `testing-status.md` §F (close the subgroup gap);
  - `bugs-found.md` (new entries for the host-visibility and leak defects).

## Open Questions

- Should `BT_VK_VALIDATION=1` become a standing `rocky-ryzen` CTest job, or stay a manual switch?
  Either way the specs and tasks hold. The job is a small CI addition.
