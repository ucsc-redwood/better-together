# Design

One change covering four topics, in dependency order:
- **A. Sync hygiene**
- **B. Memory placement**
- **C. Queue selection and several GPU chunks**
- **D. Submission overhead**

D builds on C's Sequence-per-chunk refactor and on A's synchronization2 primitives. See proposal.md
for the motivation.

## Context

Confirmed against the code at `4a4e5bf` (dev):

- **Sequence** (`platform/engine/vulkan/sequence.cpp`):
  - `cmd_begin` begins a ONE_TIME_SUBMIT command buffer (`:139-141`), resets the 2-entry query
    pool, and writes a TOP_OF_PIPE timestamp (`:150`).
  - `cmd_end` records the legacy COMPUTE_SHADER/SHADER_WRITE → HOST/HOST_READ barrier (`:165-171`),
    then a BOTTOM_OF_PIPE timestamp (`:177`).
  - `cmd_memory_barrier` is a global legacy SHADER_WRITE→SHADER_READ barrier (`:185-194`).
  - `submit` flushes the touched buffers and `vkQueueSubmit`s with a fence (`:228-241`).
  - `wait_for_fence` blocks on `UINT64_MAX` and then invalidates (`:247`).
  - `reset_fence` (`:263`).
- **Tree dispatcher:** records three more inline legacy barriers (`apps/tree/vulkan/dispatchers.cpp:300/368/419`).
- **Features:** `BaseEngine` enables Vulkan 1.3 features from a required table (`maintenance4`) plus
  an optional `subgroupSizeControl` (`base_engine.cpp:249-348`). `synchronization2` and
  `computeFullSubgroups` are not enabled.
- **Memory:** `VulkanMemoryResource::do_allocate` (`vma_pmr.cpp:80-81`) uses HOST_VISIBLE, prefers
  HOST_CACHED, with `HOST_ACCESS_RANDOM | MAPPED`.
  - The touched-buffer list `touched_` (`:39-42`) lives on the engine-wide resource, and
    `Sequence::cmd_begin` clears it.
  - AppData buffers are `std::pmr::vector`s built through one `memory_resource*`. Each pooled item
    owns its own weights (`apps/cifar-dense/appdata.hpp:44-68`; the profiler pool is 4,
    `profiler/cifar-dense-vk/const.hpp:20`).
- **Queue:** `create_device` uses `find_if` for the first family with eCompute, one queue at
  priority 1.0 (`base_engine.cpp:268-290`), and `getQueue(…, 0)` (`:385`).
- **Runtime:**
  - `first_concurrent_gpu_chunk` (`runtime/schedule.hpp:202-218`) rejects a GPU backend in more
    than one chunk; it's called at `runtime/pipeline_runner.hpp:81`, with tests at
    `runtime/tests/test_schedule.cpp:135+`.
  - `worker` (`runtime/pipeline.hpp:55-90`) does dequeue → `func(app)` (synchronous) → enqueue.
- **Solver:** `add_contiguity_constraints` (`optimizer/smt/constraints.py:109-115`) puts each core
  type in one contiguous block. The chunk cost is the stage sum plus the overhead constants
  (`add_chunk_time_constraint`).
- **Overhead:** the fitted `gpu_vulkan` per-chunk overhead on rocky is ≈1.5 ms over all apps, ≈0.85
  ms tree-only (archived campaign tasks 7.7, model `median`).

## Goals / Non-Goals

**Goals:**
- Spec-precise synchronization with no new hazards (synchronization validation stays at 0).
- Placement, queue choice and submission depth become measured, configurable variables. A default
  changes only on a recorded win beyond noise.
- Several GPU chunks per schedule, correct at runtime and modelled honestly by the solver.
- Explain the ≈1.5 ms per-chunk overhead by phase, then cut the parts that can be cut.

**Non-Goals:**
- Kernel/shader optimization (separate changes).
- More than one queue used concurrently.
- CUDA streams.
- Sharing weights across pooled items (a follow-up; it multiplies with B).

## Decisions

### A. Sync hygiene

**A1. Synchronization2 everywhere.** Enable the `synchronization2` feature in the 1.3 table.
- **Support status (to verify):** Vulkan 1.3 lists it as mandatory, and the fleet's 1.3 drivers
  should all expose it. Check the features chapter's required list, then log the support bit per
  device.
- **Barriers:** `vkCmdPipelineBarrier2` with `VkMemoryBarrier2` replaces every legacy barrier.
- **Timestamps:** `vkCmdWriteTimestamp2` replaces the timestamp writes.
- *Alternative:* keep the legacy API. Rejected: synchronization2 is core, more explicit, and lets
  the barrier helpers below take stage and access as one value.
- **Verified (tasks 1.1):** `synchronization2` is in the required 1.3 table (a missing bit is named
  like any other required feature). The debug line `Vulkan 1.3 features: synchronization2 supported
  true enabled true, …` appears on rocky-ryzen (RADV PHOENIX), the Pixel 7a (Mali-G710) and the
  Samsung (Xclipse 940).

**A2. RAW + WAW inter-stage dependency.** `Sequence::cmd_stage_dependency()` records a
`VkMemoryBarrier2` with src COMPUTE_SHADER / SHADER_STORAGE_WRITE and dst COMPUTE_SHADER /
SHADER_STORAGE_READ|SHADER_STORAGE_WRITE. This follows the spec: WAR needs only an execution
dependency, while RAW and WAW "need appropriate memory dependencies".

**A3. Barriers only where needed.** Each recorded stage declares the buffers it reads and writes.
`record_bind` already sees them; the write set is the outputs, as listed in each dispatcher's
`record_stage_*`.
- **Tracking:** the Sequence keeps a "dirty since last barrier" set. A barrier is emitted only when
  the next stage reads or writes a dirty buffer.
- **Granularity:** stays global (`VkMemoryBarrier2`), since buffer barriers buy nothing on these
  drivers, but conditional.
- **Status:** how much this saves is an open measurement; most cifar stages depend on their
  predecessor, while tree's multi-dispatch stages have some independent pairs.
- **As implemented (tasks 1.4):**
  - *Declarations come from the shaders.* `reflect_spirv` reads each set-0 buffer's
    `NonWritable`/`NonReadable` decorations (GLSL `readonly`/`writeonly`, which glslang puts on the
    block members). A plain `buffer` counts as read and written, so an undeclared binding can only
    add barriers, never drop one. `Algorithm::record_bind` passes the bound buffers and their access
    to `Sequence::track_access`, so the dispatcher's explicit barriers (between stages, and tree's
    inline ones) are gone. No shader changed.
  - *Granularity is the dispatch, not the stage.* Tracking every dispatch subsumes the stage
    boundary and tree's intra-stage barriers with one rule.
  - *WAR is ordered too.* A barrier is recorded when the next dispatch reads or writes a buffer
    written since the last barrier (RAW, WAW) **or writes a buffer read since it (WAR)**. The WAR
    case needs only an execution dependency, which the one barrier type provides; the execution-sync
    spec delta was amended to say so.
  - *Where it saves barriers:* on the production (chained) data every tree and cifar dispatch
    depends on the one before it, so the counts are unchanged (tasks 1.4 table). On tree's
    golden-decoupled `VkAppData_Safe` (the differential tests), stages read construction-time goldens
    and write `_out` fields, so consecutive stages are independent and their boundary barriers go.

**A4. Timestamps at COMPUTE_SHADER.** The start timestamp goes after the first dispatch's stage
scope begins, and the end timestamp covers COMPUTE_SHADER completion. Both use `vkCmdWriteTimestamp2`.
Measured GPU stage times may shift slightly, so the before/after per-stage tables are recorded and
the profiler tables regenerated if needed.
- **As implemented (tasks 1.3):** the start timestamp is written right after the query reset, at
  COMPUTE_SHADER, i.e. once every earlier command has finished its compute stage; placing it after
  the first dispatch would have excluded that dispatch. On rocky every stage moved by at most ±4.4 %
  (most faster; the largest slowdown +1.9 %, cifar-dense s3), table in tasks 1.3.

**A5. Full subgroups.** Enable `computeFullSubgroups` as optional.
- **When supported:** subgroup-dependent pipelines (`subgroup_size_dependent()`) set
  `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`. Valid usage requires the feature
  (`VUID-VkPipelineShaderStageCreateInfo-flags-02785`). With a required subgroup size, the X
  workgroup size must be a multiple of it (`…-pNext-02757`); the radix sort's 256 satisfies this
  for 16, 32 and 64.
- **When unsupported:** the engine checks that the X size is a multiple of the subgroup size in
  use and fails pipeline creation otherwise (the modified pipeline-config requirement).
- **As implemented (tasks 1.5):** the multiple-of check runs whether or not the feature is
  supported, because with it the VUIDs require the same multiple (`…-pNext-02757` when pinned,
  `…-flags-02759` when not). All three fleet GPUs support `computeFullSubgroups`, so the unsupported
  path is covered by a unit test that forces the feature off.

### B. Memory placement

**B1. A go/no-go experiment first.** An experiment-only `BT_VK_WEIGHTS_PLACEMENT={default,device_local,uncached}`
allocates cifar-dense's weights from a second resource with the matching VMA flags.
- **Method:** fc1 (stage 9) and fc2 (stage 10) GPU time, 3 interleaved runs per arm, on rocky, the
  Pixel 7a and the Samsung. Log the memory types first (whether a DEVICE_LOCAL|HOST_VISIBLE
  carve-out exists on the 780M, and its heap size, is to verify).
- **Why RADV may differ from the CUDA mirror's negative result**
  (`docs/reports-for-human/2026-07-03-kernel-wave-and-definitive-baseline.md:23`, "same DRAM"):
  the hypothesis is about the access path, not the DRAM. GPU reads of snooped GTT may run below
  uncached GTT or the carve-out. This is unverified.
- **Memory types (tasks 2.1),** from the `Memory heap/type` debug lines `BaseEngine` now logs:

  | device | heap | size | heap flags | types (index: property flags) |
  |---|---|---|---|---|
  | rocky-ryzen (Radeon 780M, RADV 25.2.7) | 0 | 5609 MiB | — | 2: HostVisible\|HostCoherent; 5, 6: HostVisible\|HostCoherent\|HostCached; 8, 10: same + DeviceCoherentAMD\|DeviceUncachedAMD |
  | | 1 | 11219 MiB | DeviceLocal | 0, 1: DeviceLocal; **3, 4: DeviceLocal\|HostVisible\|HostCoherent**; 7, 9: AMD device-coherent variants |
  | Pixel 7a (Mali-G710) | 0 | 7466 MiB | DeviceLocal | 0: DeviceLocal\|HostVisible\|HostCoherent; 1: DeviceLocal\|HostVisible\|HostCached; 2: DeviceLocal\|LazilyAllocated |
  | | 1 | 100 MiB | DeviceLocal | 3: DeviceLocal\|Protected |
  | Samsung (Xclipse 940) | 0 | 8192 MiB | DeviceLocal\|MultiInstance | 0, 8: DeviceLocal\|HostVisible\|HostCoherent; 4, 12: + HostCached; 1, 5, 9, 13: DeviceLocal |
  | | 1 | 64 MiB | DeviceLocal | 2, 10: DeviceLocal\|HostVisible\|HostCoherent; 3, 6, 11, 14: DeviceLocal; 7, 15: Protected |

  - **The 780M has a DEVICE_LOCAL|HOST_VISIBLE type** (3/4), on the 11219 MiB device-local heap. Today's
    placement is type 5 (snooped GTT), as the proposal suspected.
  - On both phones every type is DEVICE_LOCAL; the only host-visible choice is cached vs uncached.
- **Arms as implemented (tasks 2.2):** `default` = today's placement; `device_local` = DEVICE_LOCAL|HOST_VISIBLE
  types without HOST_CACHED; `uncached` = HOST_VISIBLE without HOST_CACHED and, where the device has one, without
  DEVICE_LOCAL. Without the "not cached" exclusion VMA's random-access preference picks the cached type on the phones,
  i.e. the default. Resolved types: rocky 5 / 3 / 2; Pixel 1 / 0 / 0; Samsung 4 / 0 / 0 (on the phones both
  non-default arms are the same coherent uncached type).
- **Measurement (tasks 2.3).** `bm-cifar-dense-vk --benchmark_filter='Stage(9|10)/' --benchmark_min_time=0.5s`,
  GPU time in ms, arms interleaved per run. rocky: 6 runs per arm (3, then 3 more because the first 3 were
  borderline), idle-gated as in tasks 1.3, `BT_WEIGHTS_DIR=/tmp/bt/weights`. Phones: 3 runs per arm, 10 s apart,
  screen dozing, battery 25.4–26.0 °C; the Samsung with its deployed real weights, the Pixel (no deployed weights)
  with the synthetic ones.
  - rocky-ryzen:

  | stage | default r1–r6 | device_local r1–r6 | uncached r1–r6 | default mean | device_local mean (Δ) | uncached mean (Δ) | default spread |
  |---|---|---|---|---|---|---|---|
  | s9 | 0.8627 / 0.8650 / 0.8739 / 0.9000 / 0.8908 / 0.8882 | 0.8514 / 0.8606 / 0.8569 / 0.8513 / 0.8514 / 0.8573 | 0.8604 / 0.8637 / 0.8632 / 0.8722 / 0.8679 / 0.8692 | 0.8801 (+0.0 %) | 0.8548 (-2.9 %) | 0.8661 (-1.6 %) | 0.0373 |
  | s10 | 0.8708 / 0.8699 / 0.8663 / 0.8884 / 0.8861 / 0.8801 | 0.8532 / 0.8557 / 0.8541 / 0.8510 / 0.8550 / 0.8528 | 0.8675 / 0.8663 / 0.8699 / 0.8863 / 0.8684 / 0.8676 | 0.8769 (+0.0 %) | 0.8536 (-2.7 %) | 0.8710 (-0.7 %) | 0.0221 |

  - Pixel 7a:

  | stage | default r1–r3 | device_local r1–r3 | uncached r1–r3 | default mean | device_local mean (Δ) | uncached mean (Δ) | default spread |
  |---|---|---|---|---|---|---|---|
  | s9 | 17.0777 / 16.6515 / 16.1957 | 11.5836 / 11.6917 / 11.1457 | 11.0977 / 10.6225 / 11.6180 | 16.6416 (+0.0 %) | 11.4737 (-31.1 %) | 11.1127 (-33.2 %) | 0.8820 |
  | s10 | 16.8091 / 16.3957 / 15.7825 | 12.6959 / 11.6096 / 12.1566 | 11.3828 / 10.3103 / 10.0949 | 16.3291 (+0.0 %) | 12.1540 (-25.6 %) | 10.5960 (-35.1 %) | 1.0266 |

  - Samsung:

  | stage | default r1–r3 | device_local r1–r3 | uncached r1–r3 | default mean | device_local mean (Δ) | uncached mean (Δ) | default spread |
  |---|---|---|---|---|---|---|---|
  | s9 | 3.8315 / 3.2398 / 3.2600 | 3.2097 / 3.2629 / 3.2890 | 3.2497 / 3.1963 / 3.2514 | 3.4437 (+0.0 %) | 3.2539 (-5.5 %) | 3.2325 (-6.1 %) | 0.5917 |
  | s10 | 3.2251 / 3.3082 / 3.3414 | 3.3261 / 3.3304 / 3.3109 | 3.1849 / 3.2545 / 3.3074 | 3.2916 (+0.0 %) | 3.3225 (+0.9 %) | 3.2489 (-1.3 %) | 0.1163 |

- **Decision rule:** "a gain beyond run-to-run noise" = the arm's mean is lower than the default's by more than the
  default arm's spread (max − min over its runs), on both fc1 and fc2 (the rule of the archived campaign's
  before/after tables).
  - **Pixel 7a: go, `uncached`.** −31 to −35 % on both stages, about 5× the spread (both non-default arms are the same
    type here). The hypothesis holds on Mali, in the opposite direction from the "same DRAM" CUDA result: the GPU
    reads Mali's host-cached type markedly slower than its coherent uncached one.
  - **rocky-ryzen: no-go.** `device_local` is faster in every run (its slowest run beats the default's fastest on both
    stages), but only by 2.9 % / 2.7 %; fc1's 0.0253 ms gain is inside the default spread of 0.0373 ms, so the rule is
    not met. `uncached` gains nothing beyond noise. The weights class keeps the default placement on rocky; a larger
    sample could revisit the carve-out.
  - **Samsung: no-go.** The −5.5 % on fc1 comes from one slow default run (spread 0.59 ms); fc2 is within ±1.3 %.

**B2. Classes are separate resources owned by Engine.** One `VulkanMemoryResource` per class:
default/boundary, weights, gpu_local.
- **Accessors:** `Engine::get_mr(BufferClass)`; `get_buffer(ptr)` searches the resources in turn.
- **Host mapping:** every class stays host-visible and mapped (the pmr allocator must return a host
  pointer, and containers value-initialize through it). Classes differ only in memory type.
- **Default rule:** a class is promoted on a device only after B1 shows a win there. Otherwise it
  maps to the default flags on that device.
- **As implemented (tasks 2.4):** `Engine` always owns three resources, named `boundary`, `weights` and `gpu_local`.
  - The weights class resolves through `weights_class_placement()`: a table of measured-go devices keyed by
    (vendorID, deviceID), holding only the Pixel 7a's Mali-G710 (0x13B5, 0xa8620000) → `uncached`.
    `BT_VK_WEIGHTS_PLACEMENT` stays as an override for re-measuring.
  - `gpu_local` keeps the default flags everywhere: nothing measured it.
  - A pointer resolves to its `VkBuffer` through whichever resource allocated it. A `Sequence` flushes and invalidates
    its own touched buffers through the owning resource; this moved the touched set out of the shared resource ahead
    of C2, which needed it anyway.
- *Alternative:* one resource with a per-allocation tag. Rejected: pmr carries no per-allocation
  tag.

**B3. Boundary vs GPU-local.**
- Weights are always the weights class. *(Corrected in tasks 2.4: only weights that no CPU chunk reads. The
  boundary requirement keeps anything a CPU chunk reads host-cached, and on the Pixel, the one go device, the weights
  class is uncached for the CPU. A weight of a CPU-run stage is therefore a boundary buffer; the memory-placement
  delta's "Weight buffer" scenario was amended to match.)*
- An activation is GPU-local only if every stage that reads or writes it runs on the GPU under the
  running schedule. The runner computes this from each stage's declared buffers before building the
  pool. That's the same declaration A3 needs (to verify whether one exists; if not, A3 introduces
  it).
- Benchmarks that prime items on the CPU keep activations in the default class.
- **As implemented (tasks 2.5):**
  - *Declarations.* The declaration A3 introduced is per dispatch, inside the Sequence, and is not available before
    the pool is built. So each cifar AppData buffer now declares, at construction, the stages that read or write it
    (`bt::PlaceBuffer`, 0 standing for host code: the input written at construction, the final output read by
    checks).
  - *Placement.* `runtime/placement.hpp` turns a schedule into a placement: a buffer whose every declared accessor is
    a GPU stage of the schedule goes to the weights or GPU-local class by kind; anything else is boundary.
    `run_pipeline`, `bm-gen-logs` and the all-GPU benchmarks (`bm-cifar-*-vk`, the GPU baseline) build their items
    with it.
  - *Profiler.* `bm-prof`'s GPU cell for stage s places as for "GPU stage {s}", so the profiled GPU time matches what
    the runtime will see. Its CPU cells, and the differential tests (whose oracle reads every buffer on the host),
    keep everything boundary.
  - *Tree.* Tree has no weights, and `gpu_local` maps to the default flags, so its AppData is unchanged.

### C. Queue selection and several GPU chunks

**C1. Queue.**
- **Logging:** log every family (flags, count, `timestampValidBits`) at debug level.
- **Family:** `BT_VK_QUEUE=universal|compute`. The default, universal, is today's behaviour;
  compute picks the first family with eCompute and no eGraphics, and warns and falls back if none
  exists.
- **Priority:** `BT_VK_QUEUE_PRIORITY=low|medium|high` chains `VkDeviceQueueGlobalPriorityCreateInfo`
  through the KHR or EXT extension, and retries without it on `VK_ERROR_NOT_PERMITTED`.
- **A/B:** tree × VK and cifar-dense × VK, isolated and under the profiler's interference load, for
  universal/compute × default/high.

**C2. One Sequence per GPU chunk.**
- **API:** `dispatch_multi_stage(data, s, e, Sequence&)` overloads take a runtime-owned Sequence;
  the member `seq` stays for benchmarks.
- **Submit lock:** a `std::mutex` in `Engine` guards `vkQueueSubmit`, because the queue needs
  external synchronization.
- **Touched set:** moves from `VulkanMemoryResource::touched_` into the Sequence. `record_bind`
  already receives the Sequence.
- *Alternative:* one Engine per chunk. Rejected: `g_vma_allocator` is a process global, and two
  engines would duplicate every pipeline.

**C3. Narrow the guard.** `first_concurrent_gpu_chunk` stops rejecting repeated chunks of the same
GPU backend. `TwoVulkanChunksRejected` becomes `TwoVulkanChunksAllowed`, and an end-to-end
two-GPU-chunk test is added.

**C4. Solver.** An opt-in `allow_gpu_multi_chunk` exempts the GPU core type from contiguity.
- **Cost:** its cost is the sum over all its maximal segments, each including the per-chunk
  overhead, constrained ≤ T_max. That models one serially shared device, a lower bound on the
  period.
- **Default:** unchanged.
- **Schema:** whether the schema, `config_reader.hpp`'s coverage check, or the contract tests
  forbid more than one chunk per PU is to verify. The schema itself only requires contiguous
  coverage.

### D. Submission overhead

**D1. Measure by phase first.**
- **Calibrated timestamps:** enable `VK_KHR_calibrated_timestamps`, or EXT where only that exists,
  when available. Sample `(host CLOCK_MONOTONIC, device)` pairs to map GPU timestamps onto the host
  timeline.
- **Per chunk, record:**
  - host time in recording (cmd_begin→cmd_end) and in `vkQueueSubmit`;
  - the GPU start/end (from A4's timestamps) and therefore the submit→GPU-start latency;
  - the GPU end → host wake-up latency (fence wait return minus mapped GPU end).
- **Idle-gap sweep (DVFS check):** insert controlled idle gaps (0, 1, 5, 20 ms) between chunks and
  record GPU execution time against the gap. That exposes the ramp-up after idle under APU DPM
  `auto`; rocky has no sudo to pin clocks, so the sweep stands in.
- **Output:** a phase table that explains the ≈1.5 ms. It decides which of D2/D3 to do.

**D2. Pre-recorded command buffers per (chunk, pooled item).**
- **Why it works:** with push descriptors (Phase C) the bindings are recorded into the command
  buffer, and for a fixed item and chunk the buffers and push constants don't change. So each
  (chunk, item) pair gets one command buffer, recorded on first use and resubmitted afterwards
  without `ONE_TIME_SUBMIT`.
- **Re-record trigger:** the Sequence keeps a key per command buffer (buffer handles plus
  push-constant bytes); a mismatch re-records.
- **Host-side work that stays:** flush/invalidate stay per submit, and they are scoped to the
  item's touched set.
- **Timestamps:** the query pool moves to per command buffer, or uses one query slot pair per
  in-flight item.
- **Fallback path:** on the descriptor fallback, the per-Sequence transient pool must not be reset
  while a recorded command buffer still references its sets. Either keep a per-item set
  allocation alive with the command buffer, or disable pre-recording on that path (to decide when
  implementing; the spec allows either, as long as results hold).

**D3. Items in flight with timeline semaphores.**
- **Fence replacement:** the GPU chunk's worker gets a submit/complete split. A per-Sequence
  timeline semaphore replaces the fence: the signal value is the item's ticket, and the host waits
  on the oldest ticket with `vkWaitSemaphores`.
- **Worker model:**
  - Dequeue and submit while fewer than `BT_VK_INFLIGHT` (default 1, today's behaviour) are
    outstanding.
  - Then wait for the oldest, invalidate its touched set, and enqueue it downstream. FIFO order is
    preserved.
  - CPU chunks keep the synchronous worker.
- **Pool size:** the pool must be large enough for depth × GPU chunks + CPU chunks in flight;
  `run_pipeline` asserts it.
- **Depends on:**
  - C2 (Sequence per chunk, submit lock, per-Sequence touched set);
  - A1's feature-table work: timeline semaphores are core 1.2, and `timelineSemaphore` is not in
    the required 1.2 table today (`base_engine.cpp:243`), so D3 adds it.

**D4. Gate by refit.**
- **Method:** before/after fleet runs on rocky (`00_run_fleet.py --only minipc --fresh --phases profile,schedule,run,summary`),
  at least 2 interleaved runs per arm, recorded conditions, then `fit_overhead.py --dry-run`, as in
  archived tasks 4.3 and 7.7.
- **Acceptance:** the after `gpu_vulkan` per-chunk constant ≤ the before mean + the before spread.
- **Also report:** tree × VK best makespan and the per-GPU-chunk-shape wall time.

## Risks / Trade-offs

- **[Risk]** A4's timestamp move shifts measured stage times, so profiling tables built before and
  after aren't comparable. **→ Mitigation:** regenerate the tables on rocky and record both.
- **[Risk]** A3's dirty tracking could miss a dependency, which would be a correctness bug.
  **→ Mitigation:** synchronization validation on every VK suite at the end of group A, and an
  `BT_VK_BARRIER_ALL=1` debug override that restores unconditional barriers.
- **[Risk]** The 780M may have no usable carve-out, or Mali may give no gain (B).
  **→ Mitigation:** a per-device decision rule; the default stays.
- **[Risk]** The submit lock serializes the GPU chunks' submissions (C). That's inherent with one
  queue and assumed by C4's cost model.
- **[Risk]** In flight (D3), a GPU chunk can hold `depth` items, which changes pipeline dynamics and
  pool requirements, and could hide latency the solver's model doesn't capture.
  **→ Mitigation:** depth 1 by default. Depth >1 is enabled only after D4's refit, and the fitted
  constants are updated with it.
- **[Risk]** On the descriptor fallback path, pre-recording conflicts with the transient pool reset
  (D2). **→ Mitigation:** as D2 says, per-item sets or no pre-recording on that path.
- **[Trade-off]** More engine state (per-item command buffers, timeline values), traded for removing
  host work from the steady state.

## Migration Plan

- **Groups land in order A → B → C → D**, each its own PR into `dev` with its gate. B and the C1
  queue part are independent and may swap.
- **Defaults:** each change keeps default behaviour, using knobs where needed (`BT_VK_QUEUE`,
  `BT_VK_QUEUE_PRIORITY`, `BT_VK_INFLIGHT`, the solver flag, `BT_VK_BARRIER_ALL`). Defaults change
  only with recorded evidence.
- **Rollback:** revert the group's PR. The knobs allow a runtime rollback without a revert.

## Open Questions

- Whether A3's conditional barriers measurably help cifar; most of its stages depend on their
  predecessor. Kept, since the dependency tracking is shared with B3.
- Whether multi-GPU-chunk schedules join the definitive fleet sweep. That depends on C4's results.
