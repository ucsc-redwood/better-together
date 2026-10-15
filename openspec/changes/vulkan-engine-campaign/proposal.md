# Proposal

## Why

A review of the Vulkan engine (`platform/engine/vulkan`), checked against the Vulkan spec, found
latent correctness defects that only hold today because the fleet GPUs are forgiving:
- device→host visibility after the fence wait is missing a barrier;
- every `Algorithm` leaks its Vulkan objects past `vkDestroyDevice`;
- device features are requested in a way that silently enables or drops them.

It also found structural costs. Subgroup size comes from `devices/*.json` instead of the device,
so one shader is duplicated for each subgroup size. Per-item descriptor-set rewrites add CPU
overhead, and that overhead dominates the µs-scale tree × Vulkan cells, which measured 0.78–0.91×
in the 2026-07-03 baseline. This campaign fixes all of these in risk order, one gated phase at a
time.

## What Changes

Phase A — correctness (small, independent fixes):
- Record a compute-shader-write → host-read memory barrier at the end of every `Sequence`
  command buffer, so results read after `wait_for_fence()` are guaranteed host-visible.
- `Algorithm` destroys every Vulkan object it creates. These are the pipeline, the pipeline
  layout, the descriptor-set layout, the descriptor pool, the shader module and the pipeline cache.
  This satisfies `VUID-vkDestroyDevice-device-05137`.
- Device creation enables exactly the features the engine requires. These are `storageBuffer8BitAccess`,
  `shaderInt8` and `bufferDeviceAddress`, plus any added by later phases. A missing required feature
  fails device creation with an error that names it. (Corrected during Phase A: an earlier draft also
  listed `uniformAndStorageBuffer8BitAccess`, but no committed shader declares the
  `UniformAndStorageBuffer8BitAccess` SPIR-V capability — the shaders declare only `Shader`,
  `Int8`, `StorageBuffer8BitAccess` and `GroupNonUniform{,Arithmetic}` — so requiring it would break
  the device-init spec's "exactly the required features".)
- The validation layer becomes opt-in through an environment variable (`BT_VK_VALIDATION=1`) instead of
  always requested by `Engine()`. **BREAKING (behavioral):** direct runs no longer get validation
  by default. Fleet runs are unchanged because they already disable layers.

Phase B — device-derived subgroup size:
- The engine queries the subgroup size at runtime and passes it to shaders as a specialization
  constant, pinned with `requiredSubgroupSize` when the device supports subgroup size control
  (core 1.3). The 16-, 32- and 64-wide copies of `multi_radixsort_warp*.comp` become one shader.
  `get_vulkan_warp_size()` stops selecting the shader. The `gpu.subgroup_size` field in
  `devices/*.json` remains as a cross-check that warns on a mismatch.
- `Algorithm::work_group_size()` actually sets the shader's workgroup size via specialization
  constants, for shaders that declare one. Today the value is only validated, never used.

Phase C — binding and pipeline overhead:
- Stage buffers are bound with push descriptors (`VK_KHR_push_descriptor`) instead of
  per-item `vkUpdateDescriptorSets` on pre-allocated sets. This removes descriptor pools,
  `num_sets()` and the per-stage set-index bookkeeping in the dispatchers. It needs a fallback
  for devices without the extension.
- One pipeline cache per engine, persisted to disk across runs. This only affects startup.

Out of scope (noted, not changed):
- Choosing the compute queue family.
- `VulkanMemoryResource::do_is_equal` semantics.
- Migrating to synchronization2.
- The unused `tmp_single_radixsort_warp*.comp` shaders, which no app or profiler references.
  This is pre-existing dead code; it is mentioned, not deleted.

## Capabilities

### New Capabilities
- `vulkan-engine/device-init`: How the engine selects and creates the Vulkan device. This covers which
  features are required and enabled, failure behaviour, and when validation layers are active.
- `vulkan-engine/execution-sync`: How a command sequence's GPU results become safely readable by
  the host after submission.
- `vulkan-engine/object-lifetime`: Every Vulkan object the engine creates is destroyed before its
  parent device.
- `vulkan-engine/pipeline-config`: How compute pipelines get their subgroup size, workgroup size and
  pipeline cache.
- `vulkan-engine/resource-binding`: How stage buffers are bound to a dispatch.

### Modified Capabilities
<!-- none: openspec/specs/ is empty -->

## Impact

- **Code:**
  - `platform/engine/vulkan/{base_engine,engine,algorithm,sequence}.{hpp,cpp}`
  - `platform/engine/vulkan/shaders/comp/multi_radixsort_warp*.comp` and their committed `*_spv.h`
    plus `all_shaders.hpp`
  - `platform/registry/device_registry.{hpp,cpp}` (`get_vulkan_warp_size`)
  - `apps/{cifar-dense,cifar-sparse,tree}/vulkan/dispatchers.{hpp,cpp}`
  - the Vulkan profiler runners under `profiler/*-vk`
- **Behaviour:** validation is opt-in; on-disk pipeline-cache files appear. Numerical results are
  unchanged, and the OMP oracle enforces this.
- **Gates:** every phase must pass `ctest -L vk` on `rocky-ryzen` (RDNA3, sg64/32) and on both phones
  (Pixel 7a Mali sg16, Samsung Xclipse sg32), plus `ctest -L omp` locally. Phase C is additionally
  gated on a tree × Vulkan re-measurement.
- **Docs:**
  - `docs/reports-for-human/testing-status.md` §F: close the subgroup gap.
  - `bugs-found.md`: new entries for the sync and leak defects.
  - `docs/instruction-for-ai/01-hardware.md`: the subgroup source of truth changes.
