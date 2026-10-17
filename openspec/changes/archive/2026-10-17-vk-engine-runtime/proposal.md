# Proposal

## Why

The Vulkan campaign (archived `2026-10-14-vulkan-engine-campaign`) fixed correctness. A follow-up
review of the engine and runtime found four remaining engine-side limits. Each one either costs
time on every chunk or narrows what the framework can schedule and measure:

1. **Synchronization is coarse and legacy.**
   - Every inter-stage barrier is a global legacy `vkCmdPipelineBarrier` with
     SHADER_WRITE→SHADER_READ (`sequence.cpp:185-194`, and inline in
     `apps/tree/vulkan/dispatchers.cpp:300/368/419`), whether or not the next stage depends on the
     previous one.
   - The dst mask omits SHADER_WRITE, although the spec says "read-after-write and
     write-after-write hazards need appropriate memory dependencies" (synchronization chapter).
   - Timestamps use TOP_OF_PIPE/BOTTOM_OF_PIPE (`sequence.cpp:150/177`).
   - Subgroup-dependent pipelines don't request full subgroups. The radix sort's
     `gl_SubgroupID`-indexed shared reductions assume them, but only
     `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT` guarantees that "the subgroup sizes
     must be launched with all invocations active".
2. **All memory has one placement.** `VulkanMemoryResource::do_allocate` (`vma_pmr.cpp:80-81`)
   makes every buffer HOST_VISIBLE, preferring HOST_CACHED. On the RADV APU that is probably
   snooped GTT (to verify), and fc1/fc2 are DRAM-bound (≈77 GB/s on rocky).
3. **The queue choice is accidental, and a schedule may have only one GPU chunk.**
   - `create_device` takes the first compute-capable family (`base_engine.cpp:268-290`), which on
     AMD is the universal queue, at default priority.
   - One `Sequence` per dispatcher forces `first_concurrent_gpu_chunk`
     (`runtime/schedule.hpp:202`) to reject multi-GPU-chunk schedules.
   - z3's `add_contiguity_constraints` (`optimizer/smt/constraints.py:109`) never emits them.
4. **Per-chunk submission overhead is about as large as the work.**
   - The fitted `gpu_vulkan` per-chunk overhead on rocky is ≈1.5 ms (≈0.85 ms tree-only; archived
     tasks 7.7). Tree's entire GPU time is ≈1.6 ms.
   - Each chunk records a one-time command buffer, submits, blocks on a fence and resets it
     (`sequence.cpp:139-264`). The runtime worker (`runtime/pipeline.hpp:55-90`) handles one item
     at a time, so the GPU idles between items.

## What Changes

Grouped by topic, in dependency order:

- **Sync hygiene:**
  - Migrate barriers and timestamps to synchronization2 (core in 1.3; the `synchronization2`
    feature is enabled).
  - Inter-stage dependencies name SHADER_STORAGE_READ|WRITE on both sides, and are emitted only
    where the next stage depends on the previous one.
  - Timestamps are written at the COMPUTE_SHADER stage.
  - Subgroup-dependent pipelines set REQUIRE_FULL_SUBGROUPS when `computeFullSubgroups` is
    available. Otherwise the workgroup size must be a multiple of the pinned subgroup size, and the
    engine falls back as specified.
- **Memory placement:**
  - Measure fc1 with its weights in each available host-visible memory type (a go/no-go gate).
  - Then add buffer classes: weights, GPU-local and boundary. Every class stays host-mapped
    (AppData buffers are `std::pmr::vector`s); classes differ only in memory type.
- **Queue selection and several GPU chunks:**
  - The queue family is logged and selectable (`BT_VK_QUEUE`), with an optional global priority.
    The default changes only if an A/B justifies it.
  - Each GPU chunk gets its own `Sequence`, submits are externally synchronized, and the
    touched-buffer cache-maintenance set moves per Sequence.
  - The runtime guard is narrowed.
  - An opt-in solver mode emits multi-GPU-chunk schedules under a time-sharing cost model.
- **Submission overhead:**
  - Measure first with `VK_KHR_calibrated_timestamps`: where the ≈1.5 ms goes (recording, submit,
    fence wake-up, DVFS ramp-up after idle on the APU's DPM `auto`).
  - Then pre-record one command buffer per pooled item and chunk, since push descriptors make the
    bindings static per item.
  - Keep at least 2 items in flight per GPU chunk with timeline semaphores. This changes the GPU
    worker model in `runtime/pipeline.hpp` / `pipeline_runner.hpp`.
  - Gate: a `fit_overhead` refit, before vs after.

Out of scope:
- Shader/kernel changes (other changes).
- CUDA streams.
- Sharing weights across pooled items (follow-up).

## Capabilities

### New Capabilities
- `vulkan-engine/memory-placement`: memory properties chosen per buffer usage class, and what each
  class guarantees to host code.
- `vulkan-engine/queue-selection`: which queue family and priority compute work uses, how the choice
  is made and reported, and how unavailable choices fail over.
- `runtime/gpu-chunks`: how a schedule's GPU chunks run alongside CPU chunks: several chunks per
  GPU, submit synchronization, and the optimizer's model for them.
- `vulkan-engine/submission`: how GPU work is recorded, submitted and waited on per chunk: reuse of
  recorded work, items in flight, and how the overhead is measured.

### Modified Capabilities
- `vulkan-engine/execution-sync`: the host-visibility requirement now names synchronization2
  stages and accesses. A new requirement covers inter-stage dependencies (RAW and WAW, emitted
  only where needed).
- `vulkan-engine/pipeline-config`: the subgroup-pinning requirement now also requires full
  subgroups where supported.

## Impact

- **Code:**
  - `platform/engine/vulkan/{base_engine,engine,sequence,algorithm,vma_pmr}.{hpp,cpp}`;
  - `apps/{tree,cifar-dense,cifar-sparse}/vulkan/dispatchers.{hpp,cpp}` and the cifar AppData;
  - `runtime/{pipeline,pipeline_runner,schedule}.hpp`, `runtime/tests/test_schedule.cpp`;
  - `optimizer/smt/{constraints,solver}.py`, `optimizer/tests/*`;
  - possibly `schemas/schedule.schema.json`.
- **Behaviour:** results are unchanged (the OMP oracle enforces it). Defaults stay as they are
  unless a recorded measurement justifies a change. Multi-GPU-chunk schedules and the in-flight
  submission model are opt-in until measured.
- **Gates:**
  - `ctest -L vulkan` on rocky-ryzen;
  - the VK suites on the Pixel 7a and Samsung;
  - local `ctest -L omp` and the optimizer pytest;
  - synchronization validation at 0 hazards;
  - before/after measurements on rocky for memory placement, the queue A/B and `fit_overhead`.
