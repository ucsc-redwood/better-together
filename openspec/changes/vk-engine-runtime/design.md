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
- **Families (tasks 3.1),** from the debug log `Queue family i: flags …, queueCount …, timestampValidBits …`:

  | device | family | flags | queues | timestampValidBits |
  |---|---|---|---|---|
  | rocky-ryzen (RADV PHOENIX) | **0** | Graphics\|Compute\|Transfer\|SparseBinding | 1 | 64 |
  | | 1 | Compute\|Transfer\|SparseBinding | 4 | 64 |
  | | 2 | VideoDecodeKHR | 1 | 0 |
  | | 3 | SparseBinding | 1 | 0 |
  | Pixel 7a (Mali-G710) | **0** | Graphics\|Compute\|Transfer\|Protected | 2 | 64 |
  | Samsung (Xclipse 940) | **0** | Graphics\|Compute\|Transfer\|SparseBinding\|Protected | 4 | 64 |
  | | 1 | Compute\|Transfer\|SparseBinding\|Protected | 2 | 64 |

  - Bold = the default (`universal`) selection, unchanged from before. `compute` selects family 1 on rocky and the
    Samsung; the Pixel has no compute-only family, so it warns and stays on family 0.
  - Global priority: rocky's RADV refuses `high` with `VK_ERROR_NOT_PERMITTED` (presumably the amdgpu kernel's
    privilege check for high-priority contexts), so the engine warns and runs at the default; both phones accept `high` and `low`.
  - **A/B and decision (tasks 3.2):** no arm beats universal/default beyond noise, so the default stays `universal`
    at the driver's default priority. On rocky the `high` arms run at the default priority, because the driver
    refuses `high` (VK_ERROR_NOT_PERMITTED); the A/B therefore cannot test a raised priority there. The phones accept
    `high` but were not part of the A/B.

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
- **As implemented (tasks 3.3/3.4):**
  - *Push constants are passed when recorded.* The old `Algorithm::update_push_constant` stored them in the
    Algorithm, which two chunk threads recording the same kernel would race on; `record_push_constants(seq, pc)` now
    takes them with the call. `record_dispatch` and the dispatchers' `record_stage_*` take the `Sequence&` too.
  - *Per-chunk Sequences.* Each dispatcher gains `make_seq()` and a `dispatch_multi_stage(…, Sequence&)` overload. Its
    own Sequence is renamed `default_seq`, so the record functions' `seq` parameter shadows nothing.
    `run_pipeline` and `bm-gen-logs` give every Vulkan GPU chunk its own Sequence.
  - *The touched set* moved into the Sequence in tasks 2.4.
  - *Scope of the narrowed guard (interpretation, for the owner to confirm).* `first_concurrent_gpu_chunk` now allows
    repeated **Vulkan** chunks and still rejects repeated **CUDA** chunks. The CUDA dispatchers share one stream and
    mutable per-dispatcher state across chunk threads (e.g. cifar-dense's lazily filled per-item device-weight map),
    and per-chunk CUDA streams are out of scope (proposal). The runtime/gpu-chunks requirement is written in Vulkan
    terms (command buffer, fence, queue), and this change reads it as the Vulkan runtime's requirement. CUDA serving
    several chunks stays open. `bm-gen-logs` skips such a CUDA schedule with a warning instead of racing it.

**C4. Solver.** An opt-in `allow_gpu_multi_chunk` exempts the GPU core type from contiguity.
- **Cost:** its cost is the sum over all its maximal segments, each including the per-chunk
  overhead, constrained ≤ T_max. That models one serially shared device, a lower bound on the
  period.
- **Default:** unchanged.
- **Schema:** whether the schema, `config_reader.hpp`'s coverage check, or the contract tests
  forbid more than one chunk per PU is to verify. The schema itself only requires contiguous
  coverage.
- **Verified (tasks 3.5):** none of them forbids it. The schema requires only contiguous coverage, and so does
  `validate_schedule_coverage`; `readSchedulesFromJson` reads chunks verbatim; the contract test checks schema
  validity and required fields. The one consumer that assumed one chunk per PU was `finalize_solution` (also used by
  the union sweep's `reprice_solution`): it reported the largest chunk as the predicted period. It now reports the
  largest per-PU sum, which is unchanged when every PU has one chunk.
- **As implemented:** `add_contiguity_constraints` skips the exempted GPU; `add_chunk_time_constraint` adds
  Σ_k [x_k,GPU]·(t_k + per-stage) + per-chunk·#GPU-segments ≤ T_max on top of the per-segment bounds.
  `02_gen_schedule_merged.py --allow-gpu-multi-chunk` turns it on.

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
- **As implemented (tasks 4.1):**
  - *Opt-in.* With `BT_VK_PHASES=1` each Sequence timestamps (CLOCK_MONOTONIC) the host phases of every
    submission: record (cmd_begin → end of cmd_end), between (→ submit), flush, submit (`vkQueueSubmit2` incl. the
    lock), wait (→ semaphore wait returned), invalidate (+ timestamp read-back). The phases are contiguous, so they
    sum to the wall time exactly.
  - *Calibration.* Where the device has `VK_KHR_calibrated_timestamps` (or the EXT) with the device and
    CLOCK_MONOTONIC domains, a calibration pair sampled right after the wait maps the GPU start/end timestamps onto
    the host clock, splitting submit + wait into start latency (from the start of the `vkQueueSubmit2` call: the GPU
    may start before the call returns) + GPU + wake. The driver's own deviation bound for the pair is reported with
    it. Without calibration the split is reported as uncorrelated (`calibrated=0`).
  - *Calibration is verified, and fails on rocky.* With `BT_VK_PHASES=1` the Engine first checks that a queue
    timestamp lies between the calibrated device ticks taken just before and after it. On rocky-ryzen (RADV 25.2.7,
    PHOENIX) it does not: queue timestamps run a constant ≈2.898e9 ticks (28.98 s at 10 ns) ahead of
    `VK_TIME_DOMAIN_DEVICE`, at the same rate (a probe saw the offset stable to ±0.3 ms over three submissions). This
    is a driver defect: Mesa 26.2.4 on the build box's iGPU passes the check. The engine then warns, naming the
    offset, and reports rocky's phases as uncorrelated (the submission spec's "Device without calibrated timestamps"
    path). There, start latency + wake are reported only together, as wait − GPU. The Pixel 7a (KHR) and the
    Samsung (EXT) pass the check.
  - *Tool.* `bm-phases-{tree,cifar-dense}-vk --device <id> -s <start> -e <end> [--gap-ms G]` prints one CSV row per
    chunk and median summaries, and also times each stage alone so the chunk can be set against the solver's
    per-stage table sum.

- **Results (tasks 4.1/4.2), rocky-ryzen.** `bm-phases-<app>-vk`, 200 chunks (tree) / 100 (cifar-dense) after 20
  warm-up chunks, 4 pooled items, production AppData (chained tree, real cifar weights). Two runs per (shape, gap),
  interleaved, idle-gated (load 0.05–0.49). Medians per run, the mean of the two runs shown, in ms. rocky's
  calibration is broken (see 4.1 above), so its start latency and wake are reported together, as wait − GPU. For tree
  1–7 (a range crossing 3|4, i.e. two submissions) the phase columns describe the second submission (4–7) only;
  `dispatch_wall` covers both.

  | shape | dispatch_wall | record | between | flush | submit | wait | invalidate | gpu | wait − GPU | Σ stage GPU (alone) | dispatch − Σ stage | GPU − Σ stage | recorded |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | tree 1–3 | 1.1862 | 0.0039 | 0.0002 | 0.0001 | 0.0188 | 1.1578 | 0.0046 | 1.0886 | 0.0691 | 1.5705 | -0.3844 | -0.4819 | 0 |
  | tree 1–7 | 1.7429 | 0.0027 | 0.0003 | 0.0001 | 0.0183 | 0.5544 | 0.0040 | 0.5110 | 0.0434 | 2.0793 | -0.3364 | -1.5683 | 0 |
  | tree 4–7 | 0.6279 | 0.0027 | 0.0003 | 0.0001 | 0.0182 | 0.6021 | 0.0039 | 0.5522 | 0.0499 | 0.8440 | -0.2161 | -0.2918 | 0 |
  | cifar-dense 1–11 | 9.5740 | 0.0324 | 0.0083 | 0.0005 | 0.0857 | 9.4070 | 0.0278 | 8.9758 | 0.4311 | 9.0319 | +0.5421 | -0.0561 | 0 |
  | cifar-dense 5–11 | 8.0106 | 0.0270 | 0.0071 | 0.0004 | 0.0875 | 7.8487 | 0.0325 | 7.5548 | 0.2939 | 7.7807 | +0.2299 | -0.2259 | 0 |

  GPU execution time of the chunk after an idle gap of G ms before it:

  | shape | gap 0 ms | gap 1 ms | gap 5 ms | gap 20 ms |
  |---|---|---|---|---|
  | tree 1–3 | 1.0938 / 1.0835 → 1.0886 | 1.5173 / 1.4407 → 1.4790 | 2.5471 / 2.5384 → 2.5427 | 2.9841 / 2.9807 → 2.9824 |
  | tree 1–7 | 0.5117 / 0.5103 → 0.5110 | 0.5632 / 0.5836 → 0.5734 | 0.8582 / 0.8630 → 0.8606 | 0.8626 / 0.8634 → 0.8630 |
  | tree 4–7 | 0.5537 / 0.5507 → 0.5522 | 0.8603 / 0.8601 → 0.8602 | 0.8666 / 0.8634 → 0.8650 | 1.3977 / 1.4068 → 1.4022 |
  | cifar-dense 1–11 | 8.9713 / 8.9804 → 8.9758 | 8.9908 / 8.9878 → 8.9893 | 12.0990 / 12.3618 → 12.2304 | 24.1032 / 24.1098 → 24.1065 |
  | cifar-dense 5–11 | 7.5581 / 7.5514 → 7.5548 | 7.5582 / 7.5631 → 7.5607 | 12.4257 / 11.8628 → 12.1442 | 20.7230 / 20.7007 → 20.7118 |

  Dispatch wall time (host, one chunk) after the same gaps:

  | shape | gap 0 ms | gap 1 ms | gap 5 ms | gap 20 ms |
  |---|---|---|---|---|
  | tree 1–3 | 1.1722 / 1.2001 → 1.1862 | 1.7284 / 1.6549 → 1.6917 | 3.1655 / 3.0242 → 3.0949 | 3.6400 / 3.6644 → 3.6522 |
  | tree 1–7 | 1.7290 / 1.7569 → 1.7429 | 2.1124 / 2.8202 → 2.4663 | 4.0239 / 4.5088 → 4.2664 | 4.8734 / 4.9244 → 4.8989 |
  | tree 4–7 | 0.6328 / 0.6229 → 0.6279 | 1.0125 / 1.1111 → 1.0618 | 1.3646 / 1.4212 → 1.3929 | 1.9614 / 1.9922 → 1.9768 |
  | cifar-dense 1–11 | 9.6013 / 9.5467 → 9.5740 | 9.4635 / 9.6665 → 9.5650 | 12.8030 / 13.2992 → 13.0511 | 24.8864 / 24.8895 → 24.8880 |
  | cifar-dense 5–11 | 8.0058 / 8.0154 → 8.0106 | 8.0668 / 8.0795 → 8.0732 | 13.2015 / 12.5237 → 12.8626 | 21.5034 / 21.4722 → 21.4878 |

  - *Phones (calibrated; the split includes the start latency)* — `bm-phases`, gap 0, medians in ms:

    | device | shape | wall | record | flush | submit | start latency | GPU | wake | invalidate | calibration deviation |
    |---|---|---|---|---|---|---|---|---|---|---|
    | Pixel 7a | tree 4–7 | 12.007 | 0.076 | 2.318 | 0.323 | 0.305 | 5.693 | 1.061 | 2.296 | 0.016 (max 0.139) |
    | Pixel 7a | cifar-dense 1–11 | 168.217 | 0.279 | 3.950 | 0.607 | 0.284 | 157.598 | 1.490 | 4.138 | 0.027 (max 0.037) |
    | Samsung | tree 4–7 | 1.971 | 0.082 | 0.002 | 0.273 | 0.151 | 1.155 | 0.299 | 0.057 | 0.049 (max 0.534) |
    | Samsung | cifar-dense 1–11 | 24.051 | 0.185 | 0.004 | 0.438 | 0.387 | 22.298 | 0.936 | 0.114 | 0.092 (max 0.124) |

- **What the ≈1.5 ms (≈0.85 ms tree-only) fitted per-chunk overhead is, by phase:**
  - *Not submission.* On rocky the host's whole share of a chunk is record 3–32 µs, flush < 1 µs, submit 18–88 µs,
    invalidate 4–33 µs, plus 0.04–0.43 ms between submit and the wait's return beyond the GPU's own time (start
    latency + wake). That totals 0.07 ms (tree) to 0.6 ms (cifar-dense), and recording is the smallest part.
  - *Not the GPU work of the chunk either.* A back-to-back chunk runs *faster* on the GPU than the sum of its stages
    timed alone: tree 1–3 −0.48 ms, tree 4–7 −0.29 ms, cifar-dense −0.06/−0.23 ms. So the per-stage table
    over-predicts a hot chunk.
  - *It is the GPU slowing down after idling (DVFS, DPM `auto`).* One idle millisecond before a chunk raises tree 1–3
    from 1.09 to 1.48 ms and tree 4–7 from 0.55 to 0.86 ms; 5 ms raises tree 1–3 to 2.54 ms; 20 ms raises cifar-dense
    1–11 from 8.98 to 24.1 ms (×2.7). In a pipelined schedule the GPU chunk waits for its next item (handoff, a
    slower CPU chunk), so it pays this regularly. A 1–5 ms idle costing +0.3–1.5 ms on a tree chunk is the size of the
    tree-only constant. The rest of the measured chunk time is the SPSC hand-off and thread wake-up that the fit's
    `per_chunk` also absorbs (outside the Sequence, so not in this table).
  - *Consequence for D2/D3.* D2 can only remove recording (a few to tens of µs per chunk; the measured chunks all
    reused their command buffers, `recorded` 0). D3 is the lever that matters: a second submission already queued
    keeps the GPU from idling between items. Both were implemented as planned; 4.5 measures them.
  - *On the Pixel* the picture differs: host cache maintenance of its non-coherent cached buffers costs 2.3 ms flush +
    2.3 ms invalidate per tree chunk (≈ 40 % of the chunk wall), and wake-up ≈ 1.1–1.5 ms. That is a candidate for
    a follow-up (coherent placement for GPU-only buffers, i.e. promoting `gpu_local` there after measuring it).

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

- **As implemented (tasks 4.3):**
  - *Capture, then match.* Between `cmd_begin()` and `cmd_end()` the Algorithms no longer touch a command buffer. They
    describe the work (`Sequence::bind` / `push_constants` / `dispatch`, with A3's barriers decided on the way) as a
    recipe of ops, buffers and push-constant bytes. The recipe is the key: pipelines, buffer handles/ranges,
    push-constant bytes, dispatch sizes and barriers. `cmd_end()` reuses a recorded command buffer whose recipe is
    equal, and otherwise records one (without ONE_TIME_SUBMIT; the query reset and both timestamps are inside it, so
    it can be resubmitted as is).
  - *Cache.* Up to 32 recorded command buffers per Sequence, then least-recently-used eviction among those not in
    flight. A slot in flight is never re-recorded.
  - *Invalidation.* Any VMA deallocation bumps `VulkanMemoryResource::generation()`, and a slot recorded under an
    older generation is not reused, because a new buffer may reuse a freed buffer's handle. Steady-state pipelines
    free nothing (`AppData::reset()` only renumbers).
  - *Fallback path: no pre-recording.* With the transient descriptor pool, every submission re-records into its
    slot, resetting the slot's own pools first. Per-item set allocations kept alive with the command buffer would
    also have worked; this path is not the default on any fleet device, so the simpler choice was taken.
  - `Sequence::recording_count()` counts recordings (the debug counter of tasks 4.3).

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

- **As implemented (tasks 4.4):**
  - *Tickets.* `Sequence::submit()` returns a ticket, the value its timeline semaphore reaches when that submission
    completes (`vkQueueSubmit2` signalling it); `Sequence::wait(ticket)` waits with `vkWaitSemaphores`, then
    invalidates that submission's own touched set and reads its own timestamp pair. The fence is gone.
  - *Dispatchers* split `dispatch_multi_stage` into `submit_multi_stage` (returns the ticket) and
    `complete_multi_stage` (waits, then tree's chained count read-backs).
  - *Runtime.* `inflight_worker` (runtime/pipeline.hpp) runs Vulkan GPU chunks in `run_pipeline` and `bm-gen-logs`,
    at `BT_VK_INFLIGHT` (default 1). `run_pipeline` asserts pool ≥ depth × GPU chunks + CPU chunks. `bm-gen-logs`
    warns instead, because a smaller pool only caps the depth (it cannot deadlock: a chunk with items in flight
    never blocks on its input).
  - *Order.* `run_pipeline` now checks for every schedule that items reach the last chunk in pool order, cyclically.
  - `timelineSemaphore` joins the required 1.2 features.

**D4. Gate by refit.**
- **Method:** before/after fleet runs on rocky (`00_run_fleet.py --only minipc --fresh --phases profile,schedule,run,summary`),
  at least 2 interleaved runs per arm, recorded conditions, then `fit_overhead.py --dry-run`, as in
  archived tasks 4.3 and 7.7.
- **Acceptance:** the after `gpu_vulkan` per-chunk constant ≤ the before mean + the before spread.
- **Also report:** tree × VK best makespan and the per-GPU-chunk-shape wall time.

- **Results (tasks 4.5), rocky-ryzen.** Arms:
  - "before" = 9236dda, the parent of group 4;
  - "depth 1" = 1ffb88f (the group 4 code) with `BT_VK_INFLIGHT=1`, the default;
  - "depth 2" = the same binaries with `BT_VK_INFLIGHT=2`.

  Each arm has its own worktree with binaries built on rocky (`scripts/build-bench-on-rocky.sh vulkan`), and each run
  is a full `00_run_fleet.py --only minipc --fresh --phases profile,schedule,run,summary` (profile 5 runs/scenario →
  z3 → run the top-4 btpm + isolated schedules, 100 items each), then `fit_overhead.py --dry-run`. Four rounds, each
  interleaved as before, depth 1, depth 2: 12 runs, all exit 0. Conditions before every run: 5 s settle, then
  `gpu_busy_percent` 0, 1-min load 0.46–0.49, no other `/dev/dri` holder, no CI job; governor `powersave`, DPM `auto`,
  clocks unlocked. Rounds 3–4 were added because after two rounds the all-apps GPU constant's before and depth-1 ranges
  did not overlap (before 1.33–1.45, depth 1 1.67–1.69).

  | metric (ms) | before 1/2/3/4 | depth 1 1/2/3/4 | depth 2 1/2/3/4 | mean before → depth 1 | Δ | before spread | before mean + spread |
  |---|---|---|---|---|---|---|---|
  | fitted `gpu_vulkan` per_chunk (all apps) | 1.4528 / 1.3276 / 1.6859 / 1.3918 | 1.6704 / 1.6938 / 1.8753 / 1.4183 | *9.0269 / 8.2654 / 8.7848 / 7.8339* | 1.4645 → 1.6644 | +0.1999 | 0.3583 | **1.8228** |
  | fitted `cpu` per_chunk (all apps) | 1.4541 / 2.2717 / 1.3616 / 2.2794 | 1.6002 / 1.9784 / 2.6808 / 1.1742 | 1.5990 / 1.6980 / 1.7912 / 1.5936 | 1.8417 → 1.8584 | +0.0167 | 0.9178 | 2.7595 |
  | fitted `gpu_vulkan` per_chunk (tree only) | 0.8665 / 0.8696 / 0.8796 / 0.8685 | 0.8773 / 0.8486 / 0.8512 / 0.8886 | *1.9549 / 1.8910 / 1.9539 / 1.8381* | 0.8711 → 0.8664 | −0.0046 | 0.0131 | 0.8842 |
  | fitted `cpu` per_chunk (tree only) | 0.2062 / 0.2083 / 0.2026 / 0.2204 | 0.2773 / 0.3119 / 0.2478 / 0.3074 | 0.4031 / 0.3713 / 0.5186 / 0.3459 | 0.2094 → 0.2861 | **+0.0767** | 0.0178 | 0.2272 |
  | tree best measured makespan (max chunk, `speedup_summary.py`) | 1.7426 / 1.7062 / 1.7066 / 1.7339 | 1.7210 / 1.7566 / 1.7435 / 1.7086 | *2.9586 / 2.9697 / 2.9555 / 2.9666* | 1.7223 → 1.7324 | +0.0101 | 0.0364 | 1.7588 |

  - Every fit is model `median` with per_stage_ms = 0 (n = 24 chunks over all apps, 8 for tree), except the tree-only
    GPU fit of depth-1 run 2 (lstsq, 0.0044 ms per stage) and all depth-2 tree GPU fits (lstsq). The tree-only fit is
    the same `fit_overhead` code restricted to the two tree cells.
  - *Per app* (the same fit restricted to one app's cells), before → depth 1 means: tree 0.871 → 0.866, cifar-dense
    2.443 → 2.522 (before 2.24–2.62, depth 1 2.35–2.70), cifar-sparse 8.447 → 8.473 (8.26–8.61 vs 8.28–8.61). The
    all-apps constant is the median over a mix of these three populations, which is why it moves with whichever
    chunks fall in the middle.
  - *Italics: not comparable.* At depth > 1 a chunk's `Start`/`End` ticks bracket submit → complete
    (`inflight_worker`'s `on_submit`/`on_complete`), so they include the time an item waits behind the other one in
    flight. The chunk durations, the `speedup_summary` "max chunk" makespan and the residuals `fit_overhead` fits are
    then latencies, not throughput. Depth 2 is therefore compared by throughput below.

  Throughput, per run the best over its schedules: the per-task makespan `(max End − min Start) / n` (06 step 4) and the
  steady period (the median interval between consecutive completions of the last chunk, tasks ≥ 10):

  | metric (ms) | before 1/2/3/4 | depth 1 1/2/3/4 | depth 2 1/2/3/4 | before mean | depth 1 mean (Δ) | depth 2 mean (Δ) | before spread |
  |---|---|---|---|---|---|---|---|
  | tree per-task makespan | 1.8250 / 1.7761 / 1.7837 / 1.8136 | 1.9294 / 2.0934 / 1.9295 / 1.9184 | 1.8438 / 1.8370 / 1.8437 / 1.8149 | 1.7996 | 1.9677 (+0.1680) | 1.8348 (+0.0352) | 0.0489 |
  | tree steady period | 1.5273 / 1.5167 / 1.5193 / 1.5477 | 1.5146 / 1.5398 / 1.5491 / 1.5304 | 1.5278 / 1.5451 / 1.5376 / 1.4949 | 1.5278 | 1.5335 (+0.0057) | 1.5264 (−0.0014) | 0.0310 |
  | cifar-dense per-task makespan | 8.7489 / 8.7900 / 8.9970 / 8.6202 | 8.7976 / 9.0084 / 8.8703 / 8.9050 | 8.6711 / 8.5429 / 8.5042 / 8.7009 | 8.7890 | 8.8953 (+0.1063) | 8.6048 (−0.1842) | 0.3769 |
  | cifar-dense steady period | 6.9602 / 6.6476 / 6.8903 / 6.9942 | 7.0056 / 7.0140 / 7.0025 / 7.0201 | 6.9673 / 6.9681 / 6.9620 / 6.9401 | 6.8731 | 7.0106 (+0.1375) | 6.9594 (+0.0863) | 0.3466 |
  | cifar-sparse per-task makespan | 112.0273 / 112.2330 / 112.2323 / 112.4607 | 112.1978 / 112.1414 / 112.1841 / 112.5360 | 111.6661 / 111.8216 / 111.8582 / 111.8071 | 112.2383 | 112.2648 (+0.0265) | 111.7883 (−0.4500) | 0.4334 |
  | cifar-sparse steady period | 108.2226 / 108.3594 / 108.5302 / 108.5588 | 108.2243 / 108.2313 / 108.3203 / 108.8930 | 107.9314 / 107.8799 / 107.9067 / 107.7288 | 108.4178 | 108.4172 (−0.0005) | 107.8617 (−0.5561) | 0.3362 |

  Tree GPU chunk shapes that every run executed: the measured GPU chunk duration (the mean over that run's schedules
  with that GPU chunk; depth 2 omitted, see above), then the per-task makespan of those schedules:

  | tree GPU chunk | before 1/2/3/4 | depth 1 1/2/3/4 | mean before → depth 1 | Δ | before spread | depth 2 per-task makespan 1/2/3/4 → mean |
  |---|---|---|---|---|---|---|
  | stages 1–2, chunk | 1.7426 / 1.7062 / 1.7066 / 1.7339 | 1.7210 / 1.7178 / 1.7435 / 1.7086 | 1.7223 → 1.7227 | +0.0004 | 0.0364 | |
  | stages 1–3, chunk | 1.9030 / 1.9530 / 1.9426 / 1.9160 | 1.9367 / 1.9145 / 1.9436 / 1.9363 | 1.9287 → 1.9328 | +0.0041 | 0.0500 | |
  | stages 1–4, chunk | 2.1133 / 2.1159 / 2.1272 / 2.1000 | 2.0936 / 2.0816 / 2.1296 / 2.1665 | 2.1141 → 2.1178 | +0.0037 | 0.0273 | |
  | stages 1–5, chunk | 2.0361 / 2.0388 / 2.0614 / 2.0642 | 2.1017 / 2.0970 / 2.1894 / 2.1235 | 2.0501 → 2.1279 | +0.0778 | 0.0281 | |
  | stages 1–6, chunk | 2.1495 / 2.2205 / 2.1587 / 2.1404 | 2.2094 / 2.2106 / 2.1872 / 2.2148 | 2.1673 → 2.2055 | +0.0382 | 0.0801 | |
  | stages 1–2, per-task makespan | 1.8250 / 1.7761 / 1.7837 / 1.8136 | 1.9294 / 2.1474 / 1.9295 / 1.9184 | 1.7996 → 1.9811 | +0.1815 | 0.0489 | 1.8438 / 1.8370 / 2.0819 / 1.8149 → 1.8944 |
  | stages 1–3, per-task makespan | 1.9331 / 1.9827 / 1.9714 / 1.9478 | 2.1068 / 2.0934 / 1.9864 / 2.1792 | 1.9588 → 2.0914 | +0.1327 | 0.0496 | 1.9422 / 1.8475 / 1.8437 / 2.0059 → 1.9098 |
  | stages 1–4, per-task makespan | 2.1338 / 2.1401 / 2.1515 / 2.1220 | 2.1945 / 2.1104 / 2.2536 / 2.4073 | 2.1369 → 2.2414 | +0.1046 | 0.0295 | 2.0862 / 2.1930 / 2.2116 / 2.0405 → 2.1328 |
  | stages 1–5, per-task makespan | 2.0591 / 2.0619 / 2.0850 / 2.0882 | 2.1500 / 2.1335 / 2.2440 / 2.1928 | 2.0736 → 2.1801 | +0.1065 | 0.0291 | 2.2091 / 2.1889 / 2.1896 / 2.2019 → 2.1973 |
  | stages 1–6, per-task makespan | 2.1730 / 2.2390 / 2.1808 / 2.1516 | 2.2951 / 2.2631 / 2.1990 / 2.2518 | 2.1861 → 2.2522 | +0.0661 | 0.0873 | 2.2581 / 2.8005 / 2.3102 / 2.3555 → 2.4311 |

  - **Verdict on the acceptance: met.** The depth-1 `gpu_vulkan` constant, 1.6644 ms, is ≤ before mean + spread,
    1.8228 ms; the tree-only one moves −0.0046 ms. The tree "max chunk" best makespan (+0.0101) and four of the five
    GPU chunk durations stay inside their before spreads; stages 1–5 is +0.0778 ms (+3.8 %) against a spread of 0.0281.
  - **Found: an end-of-run stall in the CPU chunk.** The tree-only `cpu` constant (+0.0767 ms vs a spread of 0.0178)
    and every tree per-task makespan are worse at depth 1, but the steady period is unchanged. The difference is at
    the end of each run: in the group 4 builds one of the CPU chunk's last items (tasks 94–99) takes up to 24 ms
    instead of ≈ 1.4 ms. Over the fleet runs' 32 tree schedule runs per arm, the longest of the last six items
    exceeds 5 ms in 0 before (max 4.7 ms), 21 at depth 1 and 20 at depth 2. Spread over 100 items, that is the
    +0.1–0.18 ms per task. It coincides with the
    GPU chunk's thread exiting, which destroys its `Sequence`. Since D2 that also tears down up to 32 recorded command
    buffers and their pools, while the OMP chunk still runs on every core.
    - *Experiment (rocky, not committed):* fixed tree schedules (`bm-gen-logs-tree-vk`, GPU 1–2 / 1–3 / 1–4 + Big
      rest), 5 interleaved runs per arm, with a third arm that keeps each chunk's `Sequence` alive until the process
      exits. The table gives the longest of the last 6 CPU-chunk items per run, in ms:

      | schedule | before | group 4 | group 4, Sequence kept alive |
      |---|---|---|---|
      | GPU 1–2 + Big 3–7 | 3.4 / 1.3 / 1.2 / 2.1 / 2.0 | 14.9 / 4.3 / 14.5 / 3.8 / 4.5 | 1.5 / 16.1 / 1.5 / 1.4 / 1.4 |
      | GPU 1–3 + Big 4–7 | 2.9 / 2.2 / 2.1 / 1.2 / 2.0 | 16.8 / 14.4 / 1.8 / 15.0 / 16.8 | 1.4 / 2.3 / 7.2 / 5.4 / 1.4 |
      | GPU 1–4 + Big 5–7 | 2.2 / 2.0 / 1.8 / 2.4 / 1.7 | 9.8 / 14.4 / 5.2 / 11.1 / 3.9 | 0.9 / 0.9 / 0.9 / 1.0 / 0.9 |

      Stalls above 5 ms: 0 of 15 before, 10 of 15 with group 4, 3 of 15 with the Sequence kept alive. So the teardown
      explains most of it, not all; the GPU-only schedule (1–7) has no stall in any arm. The steady state is
      unaffected, so production pipelines are not, but the benchmark's whole-run numbers and the fitted CPU constant
      are. It is left as a follow-up: destroy the chunk Sequences after the timed region in `bm-gen-logs`, and find
      the remainder.
  - **Depth 2: not made the default.** Against depth 1 it wins beyond depth 1's spread on 5 of 6 throughput metrics
    (all four depth-2 runs below all four depth-1 runs):
    - per-task makespan tree −6.7 %, cifar-dense −3.3 %, cifar-sparse −0.4 %;
    - steady period cifar-dense −0.7 %, cifar-sparse −0.5 %;
    - tree's steady period is level (−0.5 %, inside the spread).

    Against the pre-group-4 baseline it is level, apart from cifar-sparse −0.4 % / −0.5 %. The steady-state gain is
    therefore under 1 %, and most of the whole-run gain recovers depth 1's end-of-run stall. The risk mitigation above
    makes depth > 1 conditional on refitting the constants with it. That is not possible with today's tooling: at
    depth 2 `fit_overhead` would hand z3 a `gpu_vulkan` constant of 8.48 ms (latency, not cost), and
    `speedup_summary` would report tree's best makespan as 2.96 ms. **The default stays `BT_VK_INFLIGHT=1`.**
    Revisit once the logger records each chunk's service time (e.g. completion-to-completion) at depth > 1.

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
