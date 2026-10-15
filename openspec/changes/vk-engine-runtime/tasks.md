# Tasks

Gates used below:
- **"rocky gate":**
  - rsync the tree: `git ls-files -co --exclude-standard -z | rsync -a --from0 --files-from=- ./ doremy@100.97.77.80:bt-vk-campaign/`;
  - then `ssh doremy@100.97.77.80 bash -s` (fish login shell);
  - there: `cmake --preset vulkan && cmake --build --preset vulkan -j8`, then `cd build/vulkan && ctest -L vulkan` plus `test-{kiss,tree,cifar-dense,cifar-sparse}-vk --device minipc`.
- **"phone gate":** `cmake --preset android`, build the four VK test targets, then `scripts/run-on-android.sh 3A021JEHN02756 <targets>` and `scripts/run-on-android.sh R5CY21Y3VEV <targets>`.
- **"local":** `ctest --test-dir build/pc -L omp` and `uv run pytest -q optimizer`.
- **"sync-val":** the VK suites on rocky with `BT_VK_VALIDATION=1` and `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`; layer activity is confirmed with `VK_LOADER_DEBUG=layer`.
  - *Correction (found in 1.2/1.4):* with only those two variables, synchronization validation does **not** check
    shader accesses through descriptors at all. The layer (1.4.328.1 on rocky, 1.4.341 on the build box) analyses
    them only with `syncval_shader_accesses_heuristic`, default off. Control on rocky with the dependency tracking
    disabled (a mutation, not committed): `test-tree-vk` fails 9 tests and prints **0** `SYNC-HAZARD` lines as
    specified, but **210** with `VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true`. Every sync-val run
    below therefore also sets `VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true`. (A `dev` control,
    4a4e5bf, with the heuristic on: `test-tree-vk` 15/15, 0 hazards.)
- **"refit":** before/after `optimizer/orchestrate/00_run_fleet.py --only minipc --fresh --phases profile,schedule,run,summary` on rocky, at least 2 interleaved runs per arm, then `optimizer/analysis/fit_overhead.py --dry-run`, as in the archived campaign's tasks 4.3/7.7.

## 1. Sync hygiene (topic A)

- [x] 1.1 Enable `synchronization2` in the 1.3 feature table (A1) and log its support bit per device. Verify: debug logs on rocky, the Pixel 7a and the Samsung show it supported and enabled, and the rocky gate is green.
  - *Evidence (group 1 commit):* `synchronization2` joins `maintenance4` in the required 1.3 table, so a device without it
    fails creation with the feature named (`VulkanDeviceFeatures.MissingShaderInt8IsReported` now also clears
    `synchronization2` and expects exactly `{"synchronization2"}`). `SPDLOG_LEVEL=debug ./test-kiss-vk
    --gtest_filter=VulkanEngineTest.EngineInitialization` logs, on all three devices:
    `Vulkan 1.3 features: synchronization2 supported true enabled true, computeFullSubgroups supported true enabled true`
    (rocky-ryzen Radeon 780M / RADV PHOENIX, Pixel 7a Mali-G710, Samsung Xclipse 940).
  - Rocky gate (`--device minipc`, `VK_LOADER_LAYERS_DISABLE='~all~'`): `test-kiss-vk` / tree / cifar-dense / cifar-sparse →
    26 / 15 / 12 / 12 PASSED (kiss grew 21 → 26, see 1.3–1.5), `ctest -L vulkan` → 7/7.
- [x] 1.2 Replace every legacy barrier (`sequence.cpp` `cmd_end` and `cmd_memory_barrier`, and the tree dispatcher's inline barriers at `:300/:368/:419`) with `vkCmdPipelineBarrier2`. The host barrier is COMPUTE_SHADER/SHADER_STORAGE_WRITE → HOST/HOST_READ; the inter-stage one is SHADER_STORAGE_WRITE → SHADER_STORAGE_READ|WRITE (A2). Verify: `git grep -n "pipelineBarrier(" platform apps` finds no legacy calls, and the rocky gate, phone gate and sync-val (0 hazards) all pass.
  - *Evidence:* `Sequence::cmd_end` records `vkCmdPipelineBarrier2` with one `VkMemoryBarrier2`
    COMPUTE_SHADER/SHADER_STORAGE_WRITE → HOST/HOST_READ; `Sequence::cmd_stage_dependency` (replacing
    `cmd_memory_barrier`) records COMPUTE_SHADER/SHADER_STORAGE_WRITE → COMPUTE_SHADER/SHADER_STORAGE_READ|SHADER_STORAGE_WRITE.
    The five inline legacy barrier lambdas in `apps/tree/vulkan/dispatchers.cpp` (8 call sites) and the three
    dispatchers' between-stage calls are gone; dependencies now come from 1.4's tracking.
  - `git grep -n "pipelineBarrier(" platform apps` → no output (exit 1).
  - Rocky gate and phone gate green (see 1.4 for the counts).
  - sync-val on rocky (with the shader-access heuristic, see the gate correction above): the four suites print 0
    `Validation Error` and 0 `SYNC-HAZARD` lines; `VK_LOADER_DEBUG=layer` shows the layer inserted 35 / 16 / 12 / 12
    times; `VK_LAYER_REPORT_FLAGS=info` prints `Current Enables: VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`.
- [x] 1.3 Move both timestamps to `vkCmdWriteTimestamp2` at COMPUTE_SHADER (A4). Record per-stage GPU times for tree, cifar-dense and cifar-sparse on rocky, before and after. Verify: the before/after table is recorded, and `test-kiss-vk`'s timestamp checks pass.
  - *Evidence:* both timestamps are `vkCmdWriteTimestamp2(COMPUTE_SHADER)`: the start one right after
    `vkResetQueryPool` in `cmd_begin` (written once all earlier commands have finished their compute stage), the end
    one after the host barrier in `cmd_end`. New `VulkanTimestamps.GpuTimeIsPositiveAndWithinWallTime` (`test-kiss-vk`):
    three submits of 16 dispatches each; the GPU time must be > 0 and ≤ the host's submit-to-wait time, and the output
    count exact. OK on rocky (26/26), the Pixel 7a and the Samsung (25 passed + the FIFO skip each).
  - *Before/after per-stage GPU time (rocky-ryzen).* "before" = `dev` 4a4e5bf (`~/bt-vk-before`), "after" = the group 1
    code (`~/bt-vk-campaign`), both built natively on rocky. `bm-<app>-vk --device minipc --benchmark_min_time=0.5s
    --benchmark_format=csv`, `real_time` of `StageN/manual_time` (ms), `BT_WEIGHTS_DIR=/tmp/bt/weights`,
    `VK_LOADER_LAYERS_DISABLE='~all~'`. Order interleaved per run: tree before/after, cifar-dense before/after,
    cifar-sparse before/after, ×3. Before each of the 18 runs: 10 s settle, then waited for `gpu_busy_percent` 0, 1-min
    load < 0.5 (recorded 0.06–0.47) and no other `/dev/dri` holder; governor `powersave`, DPM `auto` throughout.

    - cifar-dense:

      | stage | before r1/r2/r3 | after r1/r2/r3 | mean before → after | Δ | Δ % | before spread |
      |---|---|---|---|---|---|---|
      | s1 | 0.1099 / 0.1150 / 0.1136 | 0.1142 / 0.1089 / 0.1045 | 0.1128 → 0.1092 | -0.0036 | -3.2 % | 0.0051 |
      | s2 | 0.0913 / 0.0902 / 0.0915 | 0.0905 / 0.0885 / 0.0874 | 0.0910 → 0.0888 | -0.0022 | -2.4 % | 0.0012 |
      | s3 | 0.8760 / 0.8618 / 0.8775 | 0.8859 / 0.8973 / 0.8812 | 0.8717 → 0.8881 | +0.0164 | +1.9 % | 0.0157 |
      | s4 | 0.0747 / 0.0748 / 0.0748 | 0.0737 / 0.0738 / 0.0737 | 0.0748 → 0.0737 | -0.0011 | -1.4 % | 0.0001 |
      | s5 | 1.9286 / 1.9293 / 1.9592 | 1.9186 / 1.9147 / 1.9162 | 1.9390 → 1.9165 | -0.0225 | -1.2 % | 0.0306 |
      | s6 | 2.4262 / 2.4248 / 2.4310 | 2.3589 / 2.4266 / 2.4006 | 2.4273 → 2.3954 | -0.0320 | -1.3 % | 0.0063 |
      | s7 | 1.5503 / 1.6251 / 1.5637 | 1.5546 / 1.5610 / 1.5695 | 1.5797 → 1.5617 | -0.0180 | -1.1 % | 0.0748 |
      | s8 | 0.0299 / 0.0300 / 0.0299 | 0.0289 / 0.0289 / 0.0289 | 0.0299 → 0.0289 | -0.0011 | -3.6 % | 0.0000 |
      | s9 | 0.8857 / 0.8829 / 0.8868 | 0.8697 / 0.8661 / 0.8759 | 0.8851 → 0.8705 | -0.0146 | -1.6 % | 0.0039 |
      | s10 | 0.8780 / 0.8668 / 0.8700 | 0.8724 / 0.8723 / 0.8680 | 0.8716 → 0.8709 | -0.0007 | -0.1 % | 0.0112 |
      | s11 | 0.0972 / 0.1019 / 0.1016 | 0.0947 / 0.0990 / 0.0967 | 0.1002 → 0.0968 | -0.0034 | -3.4 % | 0.0046 |

    - cifar-sparse:

      | stage | before r1/r2/r3 | after r1/r2/r3 | mean before → after | Δ | Δ % | before spread |
      |---|---|---|---|---|---|---|
      | s1 | 1.7897 / 1.7085 / 1.7599 | 1.7165 / 1.7554 / 1.7194 | 1.7527 → 1.7304 | -0.0223 | -1.3 % | 0.0812 |
      | s2 | 0.5973 / 0.5921 / 0.5915 | 0.5978 / 0.5911 / 0.5911 | 0.5936 → 0.5933 | -0.0003 | -0.1 % | 0.0058 |
      | s3 | 17.5451 / 17.5490 / 17.5496 | 17.5628 / 17.5517 / 17.5775 | 17.5479 → 17.5640 | +0.0161 | +0.1 % | 0.0045 |
      | s4 | 0.4728 / 0.4576 / 0.4484 | 0.4777 / 0.4598 / 0.4663 | 0.4596 → 0.4679 | +0.0083 | +1.8 % | 0.0244 |
      | s5 | 27.4382 / 27.4566 / 27.4088 | 27.4644 / 27.4213 / 27.4396 | 27.4345 → 27.4418 | +0.0072 | +0.0 % | 0.0478 |
      | s6 | 38.1341 / 38.1511 / 38.0761 | 38.0840 / 38.0603 / 38.1596 | 38.1204 → 38.1013 | -0.0191 | -0.1 % | 0.0750 |
      | s7 | 23.7121 / 23.7114 / 23.7127 | 23.7329 / 23.7216 / 23.7279 | 23.7121 → 23.7275 | +0.0154 | +0.1 % | 0.0013 |
      | s8 | 0.1431 / 0.1431 / 0.1431 | 0.1429 / 0.1428 / 0.1426 | 0.1431 → 0.1428 | -0.0003 | -0.2 % | 0.0001 |
      | s9 | 6.3959 / 6.4981 / 6.4676 | 6.3983 / 6.4988 / 6.4661 | 6.4539 → 6.4544 | +0.0005 | +0.0 % | 0.1021 |
      | s10 | 6.4098 / 6.4929 / 6.4134 | 6.4121 / 6.4152 / 6.4121 | 6.4387 → 6.4132 | -0.0255 | -0.4 % | 0.0831 |
      | s11 | 0.0883 / 0.0889 / 0.0893 | 0.0886 / 0.0893 / 0.0889 | 0.0888 → 0.0889 | +0.0001 | +0.1 % | 0.0010 |

    - tree:

      | stage | before r1/r2/r3 | after r1/r2/r3 | mean before → after | Δ | Δ % | before spread |
      |---|---|---|---|---|---|---|
      | s1 | 0.0806 / 0.0807 / 0.0800 | 0.0794 / 0.0796 / 0.0797 | 0.0804 → 0.0796 | -0.0009 | -1.1 % | 0.0007 |
      | s2 | 0.7873 / 0.7866 / 0.7850 | 0.7846 / 0.7847 / 0.7853 | 0.7863 → 0.7849 | -0.0014 | -0.2 % | 0.0023 |
      | s3 | 0.1789 / 0.1809 / 0.1789 | 0.1788 / 0.1788 / 0.1791 | 0.1796 → 0.1789 | -0.0007 | -0.4 % | 0.0020 |
      | s4 | 0.1312 / 0.1303 / 0.1308 | 0.1313 / 0.1304 / 0.1314 | 0.1308 → 0.1311 | +0.0003 | +0.2 % | 0.0009 |
      | s5 | 0.0436 / 0.0433 / 0.0435 | 0.0433 / 0.0427 / 0.0414 | 0.0434 → 0.0425 | -0.0010 | -2.2 % | 0.0003 |
      | s6 | 0.1090 / 0.1122 / 0.1116 | 0.1103 / 0.1048 / 0.1032 | 0.1109 → 0.1061 | -0.0049 | -4.4 % | 0.0031 |
      | s7 | 0.2363 / 0.2414 / 0.2408 | 0.2355 / 0.2453 / 0.2467 | 0.2395 → 0.2425 | +0.0030 | +1.3 % | 0.0052 |

    - The move is small. Every stage is within ±4.4 %. Deltas beyond the before spread: faster on cifar-dense s2, s4, s6,
      s8, s9, s11, cifar-sparse s8, tree s1, s5, s6; slower on cifar-dense s3 (+0.0164 ms, +1.9 %, spread 0.0157),
      cifar-sparse s3 (+0.0161, +0.1 %) and s7 (+0.0154, +0.1 %). No profiling table needs regenerating for this
      alone; the fleet runs of 4.5 re-profile with the new timestamps anyway.
- [x] 1.4 Have stages declare their read/write buffers, and emit an inter-stage dependency only when the next stage touches a buffer written since the last barrier (A3). Add the `BT_VK_BARRIER_ALL=1` override. Verify: sync-val reports 0 hazards with and without the override; the rocky gate and phone gate are green; the number of barriers recorded per chunk is logged at debug level for each app, before vs after.
  - *Evidence:* access comes from the shaders' `readonly`/`writeonly` qualifiers (`reflect_spirv` →
    `SpirvReflection::binding_access`, design A3 "As implemented"); `Algorithm::record_bind` hands the bound buffers and
    their access to `Sequence::track_access`, which records the dependency only on RAW/WAW/WAR overlap since the last
    one. `BT_VK_BARRIER_ALL=1` restores one before every dispatch but the first, which is exactly where `dev` had them
    (every consecutive dispatch pair in a chunk).
  - New tests in `test-kiss-vk`: `SpirvReflection.BufferAccess` (hardware-free: e.g. `tree_scan_local` {R, W, W},
    `multi_radixsort` all RW) and `VulkanStageDependency.OnlyWhereStagesDepend` (independent dispatches on two buffers →
    0 barriers, a third on the first buffer → 1; with `BT_VK_BARRIER_ALL=1` → 1 then 2; outputs 256 / 128).
  - sync-val on rocky, default and `BT_VK_BARRIER_ALL=1`: all four suites 0 `Validation Error`, 0 `SYNC-HAZARD` (layer
    inserted 34 / 16 / 12 / 12 times each; `test-kiss-vk` re-run after the 1.3 timestamp test was added: 35 inserts, 0/0). *Control:* with the tracking forced off (mutation, not committed),
    `test-tree-vk` on rocky fails 9 tests and prints 210 `SYNC-HAZARD` lines (RAW and WAR) under the same settings.
  - Rocky gate: 26 / 15 / 12 / 12, `ctest -L vulkan` 7/7. Phone gate (`scripts/run-on-android.sh`, script exit 0):
    Pixel 7a and Samsung `test-kiss-vk` 25 passed + 1 skipped (the SELinux FIFO test, as on `dev`), tree / cifar-dense
    / cifar-sparse 15 / 12 / 12.
  - *Barriers per chunk, debug log* (`bm-gen-logs-<app>-vk --device minipc -l debug` over schedules covering these GPU
    chunk shapes; the `Sequence: command buffer with N dispatches, M inter-stage barriers` line, production chained
    AppData). "before" = `BT_VK_BARRIER_ALL=1`, "after" = tracked:

    | app | GPU chunk (submission) | dispatches | barriers before | barriers after |
    |---|---|---|---|---|
    | tree | stages 1–2 | 9 | 8 | 8 |
    | tree | stages 1–3 (first submit of a chunk crossing 3\|4) | 14 | 13 | 13 |
    | tree | stages 4–7 | 6 | 5 | 5 |
    | cifar-dense | stages 1–11 | 11 | 10 | 10 |
    | cifar-dense | stages 5–11 | 7 | 6 | 6 |
    | cifar-sparse | stages 1–11 | 11 | 10 | 10 |
    | cifar-sparse | stages 5–11 | 7 | 6 | 6 |

    - On the production data every dispatch depends on its predecessor, so the counts do not change (the design's
      open question answered: conditional barriers do not help cifar, nor chained tree). They do drop on tree's
      golden-decoupled `VkAppData_Safe` path, whose consecutive stages are independent; the unit test covers that case.
- [x] 1.5 Enable `computeFullSubgroups` as optional, and set REQUIRE_FULL_SUBGROUPS on subgroup-dependent pipelines. Without the feature, check that the X workgroup size is a multiple of the subgroup size (A5). Verify: tree suites pass exactly at 16 (Pixel), 32 (`BT_VK_SUBGROUP_SIZE=32` on rocky and the Samsung) and 64; a unit test with a non-multiple workgroup size and the feature forced off fails with both sizes named.
  - *Evidence:* `computeFullSubgroups` is enabled when supported (`ComputeLimits::full_subgroups`); subgroup-dependent
    pipelines then set `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`. Whether or not it is supported,
    `build()` requires the X size to be a multiple of the subgroup size in use (design A5).
  - All three devices support it; `-l debug` pipeline lines:
    - rocky-ryzen `bm-fully-tree-vk -s 1 -e 1 -t 1`: `Pipeline [multi_radixsort] created successfully (subgroup size 64,
      pinned, full subgroups; …)` ×3, and with `BT_VK_SUBGROUP_SIZE=32` the same line at `subgroup size 32` ×3;
    - Pixel 7a: `subgroup size 16, pinned, full subgroups` ×5; Samsung: `subgroup size 64, pinned, full subgroups` ×5.
  - Tree suites, exact oracle match: Pixel 7a (16) 15/15; rocky default (64) 15/15 and `BT_VK_SUBGROUP_SIZE=32` 15/15;
    Samsung default (64) 15/15 and `BT_VK_SUBGROUP_SIZE=32` 15/15.
  - New engine self-test shader `engine_test_subgroup.comp` (constant_id 0 + `local_size_x_id`), baked with
    `cmake --build build/vulkan --target bake-shaders` (only its own files were produced; no other header changed).
    - `VulkanSubgroupSize.FullSubgroupsDispatch`: X = 4 × subgroup size; reads back `gl_SubgroupSize` = the pinned size
      and the smallest active-invocation count of any subgroup = that size (all subgroups full). OK on rocky, both
      phones, and the Samsung pinned at 32 (`… subgroup size 32, pinned, full subgroups`).
    - `VulkanSubgroupSize.NonMultipleWorkgroupFailsWithoutFullSubgroups`: `ComputeLimits::full_subgroups` forced off,
      X = 1.5 × subgroup size → `build()` throws `workgroup X size 96 is not a multiple of the subgroup size 64
      (computeFullSubgroups unsupported)` on rocky (24/16 on the Pixel), and the test checks both numbers; X = 2 × size
      still builds. OK on all three devices.

## 2. Memory placement (topic B)

- [ ] 2.1 Record each fleet GPU's memory types and heaps (B1). Verify: the table is in design.md B1, including whether a DEVICE_LOCAL|HOST_VISIBLE type exists on the 780M and its heap size.
- [ ] 2.2 Add the experiment-only `BT_VK_WEIGHTS_PLACEMENT` override and log the chosen memory type. Verify: on rocky the debug log shows a different `memoryTypeIndex` per arm, and `test-cifar-dense-vk` passes 12/12 under each arm.
- [ ] 2.3 Measure fc1/fc2 GPU time per arm on rocky, the Pixel 7a and the Samsung: 3 interleaved runs each, with conditions recorded. Verify: tables and per-device go/no-go decisions are recorded in design.md B1.
- [ ] 2.4 If any device is a go: add per-class resources to `Engine` (B2) and route the cifar apps' weights and biases to the weights class; no-go devices map the class to the default flags. Verify: the rocky gate and phone gate are green; a `test-kiss-vk` case allocates from each class and resolves each pointer to its VkBuffer; full cifar stage times on rocky before vs after are recorded, with no stage slower beyond noise.
- [ ] 2.5 If 2.4 landed and 1.4's stage→buffer declarations exist: compute GPU-local activations from the schedule (B3). Verify: a unit test asserts that every buffer a CPU chunk touches stays in the default class; the hybrid `test-pipeline-e2e-*-vk` suites pass on the rocky gate and phone gate.

## 3. Queue selection and several GPU chunks (topic C)

- [ ] 3.1 Log the queue families, the selection and the priority; add `BT_VK_QUEUE` and `BT_VK_QUEUE_PRIORITY` with warn-and-fallback (C1). Verify: the family tables of all three devices are recorded in design.md C1; the rocky gate is green with `BT_VK_QUEUE=compute`; a refused priority is logged as a warning while the gate stays green.
- [ ] 3.2 Run the queue A/B on rocky: tree × VK and cifar-dense × VK, isolated and under `bm-prof-*-vk` interference, for universal/compute × default/high, with 3 interleaved runs. Verify: tables and a decision on the default are recorded.
- [ ] 3.3 Move the touched-buffer set into `Sequence`, add the `Engine` submit mutex and the `dispatch_multi_stage(…, Sequence&)` overloads (C2). Verify: the rocky gate and phone gate are green; a new test runs two threads with their own Sequences against one dispatcher for 1,000 items under `BT_VK_VALIDATION=1`, with 0 validation errors.
- [ ] 3.4 Narrow `first_concurrent_gpu_chunk`, update `runtime/tests/test_schedule.cpp`, and add an end-to-end tree and cifar-dense test with Vulkan 1–2 / CPU 3–5 / Vulkan 6–N (C3). Verify: `ctest -L omp` is green; the end-to-end test matches the oracle on the rocky gate and phone gate (the Pixel covers non-coherent memory) with 0 validation errors on rocky.
- [ ] 3.5 Check whether the schema, `config_reader.hpp`'s coverage check, or `optimizer/tests/test_schedule_contract.py` forbid more than one chunk per PU; then add the opt-in `allow_gpu_multi_chunk` solver mode with the GPU-sum cost (C4). Verify: the findings are recorded; a pytest asserts the default mode never emits two GPU chunks, and that in multi-chunk mode the toy profile's period is ≥ the sum of its GPU segments; `uv run pytest -q optimizer` is green.

## 4. Submission overhead (topic D)

- [ ] 4.1 Add phase timing: record, submit, wait, GPU start/end and start latency. Use calibrated timestamps when `VK_KHR_calibrated_timestamps` or its EXT form is available (D1). Verify: on rocky, a tree × VK run reports phases per chunk that sum to wall time within the timer resolution; the device's calibrated-timestamp support is logged.
- [ ] 4.2 Run the idle-gap sweep (0, 1, 5, 20 ms between chunks) on rocky for tree × VK and cifar-dense × VK (D1). Verify: a phase table and a GPU-time-vs-gap table are recorded in design.md D1, with an explanation of the ≈1.5 ms (≈0.85 ms tree-only) fitted overhead by phase.
- [ ] 4.3 Pre-record command buffers per (chunk, pooled item), with re-record on a key mismatch and a defined behaviour on the descriptor-fallback path (D2). Verify: a test shows no recording on the second pass of the same item (debug counter), and a re-record when push constants change; the rocky gate and phone gate pass with and without `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1`.
- [ ] 4.4 Enable `timelineSemaphore`, then split the GPU worker into submit and complete with `BT_VK_INFLIGHT` (default 1), preserving FIFO hand-off; `run_pipeline` asserts the pool size is sufficient (D3). Verify: the end-to-end suites pass at depth 1 and 2 on the rocky gate and phone gate; a test asserts output order equals input order at depth 2.
- [ ] 4.5 Run the refit gate for 4.3 + 4.4, at depth 1 and depth 2 against the pre-group-4 baseline (D4). Verify: the before/after `gpu_vulkan` per-chunk constants, the spreads, tree × VK best makespan and the per-GPU-chunk-shape wall times are recorded; the after constant is ≤ the before mean + the before spread. Depth 2 becomes the default only if it wins beyond noise.

## 5. Docs and final gate

- [ ] 5.1 Document every new knob (`BT_VK_BARRIER_ALL`, `BT_VK_WEIGHTS_PLACEMENT` if kept, `BT_VK_QUEUE`, `BT_VK_QUEUE_PRIORITY`, `BT_VK_INFLIGHT`, and the solver's `allow_gpu_multi_chunk`) in `docs/instruction-for-ai/03-unit-testing.md` and `06-end-to-end-scheduling.md`. Update `bugs-found.md` §10 to say the single-GPU-chunk guard is lifted, and add a decision record with the measurement tables under `docs/reports-for-human/`. Verify: each knob is documented with its default.
- [ ] 5.2 Final gate at the last commit: local, rocky gate, phone gate and sync-val. Verify: all green, with results listed in the PR.
