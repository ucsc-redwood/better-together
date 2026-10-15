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

- [x] 2.1 Record each fleet GPU's memory types and heaps (B1). Verify: the table is in design.md B1, including whether a DEVICE_LOCAL|HOST_VISIBLE type exists on the 780M and its heap size.
  - *Evidence:* `BaseEngine::create_device` logs every heap and memory type at debug level (`Memory heap i: … MiB,
    flags …` / `Memory type t: heap h, flags …`). `SPDLOG_LEVEL=debug ./test-kiss-vk
    --gtest_filter=VulkanEngineTest.EngineInitialization` on rocky-ryzen, the Pixel 7a and the Samsung; the tables
    are in design.md B1 "Memory types". The 780M **has** DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT types (3, 4) on heap 1,
    11219 MiB, flagged DeviceLocal. Today's placement is type 5 (HostVisible|HostCoherent|HostCached, heap 0, 5609 MiB).
- [x] 2.2 Add the experiment-only `BT_VK_WEIGHTS_PLACEMENT` override and log the chosen memory type. Verify: on rocky the debug log shows a different `memoryTypeIndex` per arm, and `test-cifar-dense-vk` passes 12/12 under each arm.
  - *Evidence:* `BT_VK_WEIGHTS_PLACEMENT=default|device_local|uncached` (arm definitions in design.md B1), resolved
    to a memory-type mask against the device's types. Each allocation logs `VulkanMemoryResource [<name>]: <bytes>
    bytes in memoryTypeIndex <t>` at debug level (`bm-cifar-dense-vk` now honours `SPDLOG_LEVEL`).
  - rocky-ryzen, `SPDLOG_LEVEL=debug bm-cifar-dense-vk --benchmark_filter='Stage9/' --benchmark_min_time=1x`: the 16
    weight and bias buffers land in **memoryTypeIndex 5 / 3 / 2** for default / device_local / uncached, while the
    other 12 buffers stay in 5. The same check at the final group 2 code, where the knob overrides the weights
    class: identical (unset → `Buffer class weights: default placement (no measured gain on this device)`, type 5).
  - `test-cifar-dense-vk --device minipc` under each arm (with the experiment-stage code, where the differential tests
    put the weights in the arm's resource): 12/12 / 12/12 / 12/12, run twice, once at the end of each placement
    measurement batch.
  - Phones (`… --device <serial>`): Pixel 7a 1 / 0 / 0, Samsung 4 / 0 / 0. On both, device_local and uncached are the
    same coherent uncached type, because every phone type is DEVICE_LOCAL.
- [x] 2.3 Measure fc1/fc2 GPU time per arm on rocky, the Pixel 7a and the Samsung: 3 interleaved runs each, with conditions recorded. Verify: tables and per-device go/no-go decisions are recorded in design.md B1.
  - *Evidence:* tables and decisions in design.md B1 "Measurement" / "Decision rule" (`bm-cifar-dense-vk
    --benchmark_filter='Stage(9|10)/' --benchmark_min_time=0.5s`, arms interleaved within each run).
    - rocky: 6 runs per arm (3, then 3 more because the first were borderline). Before every run: 10 s settle,
      `gpu_busy_percent` 0, 1-min load 0.08–0.48, no other `/dev/dri` holder; `powersave`, DPM `auto`.
    - Phones: 3 runs per arm, 10 s apart, screen dozing; battery 25.6–25.7 °C (Pixel) and 25.4–26.0 °C (Samsung).
  - Decisions (rule: Δ mean beyond the default arm's spread on both fc1 and fc2):
    - **Pixel 7a go** (`uncached`): fc1 16.642 → 11.113 ms (−33.2 %), fc2 16.329 → 10.596 ms (−35.1 %); spreads
      0.88 / 1.03 ms.
    - **rocky no-go:** best arm `device_local` −2.9 % / −2.7 %, but fc1's Δ 0.0253 ms is inside its 0.0373 ms spread.
      Its runs are all faster than the default's, so this is a small real effect that the rule does not admit.
    - **Samsung no-go:** −5.5 % fc1 inside a 0.59 ms spread, fc2 +0.9 % / −1.3 %.
  - So 2.4 and 2.5 apply, for the Pixel.
- [x] 2.4 If any device is a go: add per-class resources to `Engine` (B2) and route the cifar apps' weights and biases to the weights class; no-go devices map the class to the default flags. Verify: the rocky gate and phone gate are green; a `test-kiss-vk` case allocates from each class and resolves each pointer to its VkBuffer; full cifar stage times on rocky before vs after are recorded, with no stage slower beyond noise.
  - *Evidence:* `Engine` owns one resource per class (`boundary`, `weights`, `gpu_local`; design B2 "As
    implemented"). `Engine::get_mr(bt::BufferClass)` selects one, and `get_buffer(ptr)` searches all three. The weights
    class is `uncached` only on the measured-go device (vendor 0x13B5, device 0xa8620000, the Pixel's Mali-G710) and
    the default elsewhere; `gpu_local` is the default everywhere. Both cifar apps route their 16 weight/bias buffers
    (cifar-sparse: 5 CSR matrices + 11 dense) through the class chosen by the schedule; see 2.5 for which ones move.
  - Debug log (`SPDLOG_LEVEL=debug bm-cifar-dense-vk`): rocky `Buffer class weights: default placement (no measured
    gain on this device)`, all buffers type 5. Pixel 7a: `Buffer class weights: uncached placement (measured gain on
    this device), memory type bits 0x1`, the 16 weight buffers in type 0 and the other 12 in type 1. Samsung: default
    placement, all in type 4.
  - New `VulkanBufferClass.EachClassAllocatesAndResolves` (`test-kiss-vk`): for the weights placement unset,
    `uncached` and `device_local`, a vector from each class resolves to its own `VkBuffer` (3 distinct, matching
    `find_buffer`), and a value written through the class's mapping (WC for the forced arms) is read and updated by a
    dispatch (1000 → 1128). OK on rocky, the Pixel 7a and the Samsung.
  - Rocky gate: `test-kiss-vk` / tree / cifar-dense / cifar-sparse 27 / 15 / 13 / 13 (+1 each for cifar: 2.5's
    test), `ctest -L vulkan` 7/7; sync-val (with the heuristic) 0 errors / 0 hazards in those four plus
    `test-pipeline-e2e-vk` and `test-pipeline-e2e-cifar-dense-vk` (layer inserted 38 / 16 / 13 / 13 / 2 / 2 times).
    Phone gate: both phones 26 + 1 skipped (FIFO) / 15 / 13 / 13, script exit 0.
  - *Full cifar stage times on rocky, before vs after.* "before" = the group 1 commit 3746427 (`~/bt-vk-g1`), "after"
    = this code; same method and idle gating as 1.3, `BT_WEIGHTS_DIR=/tmp/bt/weights`. cifar-sparse: 3 interleaved
    runs. cifar-dense: 6, because after 3 its s3 (+6.6 %, one 1.0084 ms run) and s5 (+2.2 %) exceeded the before
    spread; with 6, no stage is slower beyond noise. Load 0.04–0.49 across the 18 runs.

    - cifar-dense (rows: run values / means):

      | stage | before r1/r2/r3 | after r1/r2/r3 | mean before → after | Δ | Δ % | before spread |
      |---|---|---|---|---|---|---|
      | s1 | 0.1142 / 0.1130 / 0.1117 / 0.1148 / 0.1121 / 0.1114 | 0.1139 / 0.1137 / 0.1121 / 0.1120 / 0.1129 / 0.1022 | 0.1129 → 0.1111 | -0.0018 | -1.6 % | 0.0034 |
      | s2 | 0.0886 / 0.0869 / 0.0893 / 0.0908 / 0.0880 / 0.0915 | 0.0910 / 0.0883 / 0.0907 / 0.0910 / 0.0907 / 0.0898 | 0.0892 → 0.0903 | +0.0011 | +1.2 % | 0.0045 |
      | s3 | 0.8731 / 0.8954 / 0.8589 / 0.8952 / 0.9609 / 0.8907 | 1.0084 / 0.8866 / 0.9068 / 0.8893 / 0.8938 / 0.8617 | 0.8957 → 0.9078 | +0.0121 | +1.3 % | 0.1020 |
      | s4 | 0.0736 / 0.0726 / 0.0737 / 0.0737 / 0.0737 / 0.0738 | 0.0737 / 0.0737 / 0.0738 / 0.0736 / 0.0729 / 0.0736 | 0.0735 → 0.0736 | +0.0000 | +0.1 % | 0.0012 |
      | s5 | 1.9616 / 1.9337 / 1.9208 / 1.9594 / 2.0116 / 2.0209 | 1.9321 / 1.9896 / 2.0195 / 1.9188 / 2.0098 / 1.9515 | 1.9680 → 1.9702 | +0.0022 | +0.1 % | 0.1001 |
      | s6 | 2.4260 / 2.3452 / 2.4136 / 2.4431 / 2.3216 / 2.3693 | 2.4119 / 2.3588 / 2.4198 / 2.3413 / 2.3304 / 2.3284 | 2.3865 → 2.3651 | -0.0214 | -0.9 % | 0.1215 |
      | s7 | 1.6092 / 1.6285 / 1.5537 / 1.5829 / 1.5748 / 1.5988 | 1.5715 / 1.6208 / 1.5531 / 1.6035 / 1.5636 / 1.5706 | 1.5913 → 1.5805 | -0.0108 | -0.7 % | 0.0748 |
      | s8 | 0.0289 / 0.0289 / 0.0289 / 0.0288 / 0.0289 / 0.0289 | 0.0289 / 0.0289 / 0.0289 / 0.0289 / 0.0289 / 0.0289 | 0.0289 → 0.0289 | +0.0000 | +0.0 % | 0.0000 |
      | s9 | 0.8936 / 0.8796 / 0.8667 / 0.8708 / 0.8666 / 0.8699 | 0.8785 / 0.8660 / 0.8633 / 0.8669 / 0.8676 / 0.8748 | 0.8745 → 0.8695 | -0.0050 | -0.6 % | 0.0270 |
      | s10 | 0.8807 / 0.8819 / 0.8702 / 0.8786 / 0.8737 / 0.8750 | 0.8730 / 0.8710 / 0.8699 / 0.8787 / 0.8719 / 0.8788 | 0.8767 → 0.8739 | -0.0028 | -0.3 % | 0.0117 |
      | s11 | 0.0967 / 0.0955 / 0.0972 / 0.0963 / 0.0962 / 0.0977 | 0.0939 / 0.0987 / 0.0998 / 0.0978 / 0.1013 / 0.0977 | 0.0966 → 0.0982 | +0.0016 | +1.6 % | 0.0022 |

    - cifar-sparse (rows: run values / means):

      | stage | before r1/r2/r3 | after r1/r2/r3 | mean before → after | Δ | Δ % | before spread |
      |---|---|---|---|---|---|---|
      | s1 | 1.7292 / 1.7633 / 1.7676 | 1.7829 / 1.7531 / 1.7320 | 1.7534 → 1.7560 | +0.0026 | +0.2 % | 0.0384 |
      | s2 | 0.5775 / 0.5809 / 0.5839 | 0.5775 / 0.5809 / 0.5852 | 0.5808 → 0.5812 | +0.0005 | +0.1 % | 0.0064 |
      | s3 | 17.5402 / 17.5468 / 17.5202 | 17.5096 / 17.5534 / 17.5502 | 17.5357 → 17.5377 | +0.0020 | +0.0 % | 0.0266 |
      | s4 | 0.4501 / 0.4707 / 0.4749 | 0.4638 / 0.4819 / 0.4989 | 0.4652 → 0.4815 | +0.0163 | +3.5 % | 0.0248 |
      | s5 | 27.4214 / 27.4149 / 27.4489 | 27.4335 / 27.4235 / 27.4450 | 27.4284 → 27.4340 | +0.0056 | +0.0 % | 0.0340 |
      | s6 | 38.1429 / 38.1914 / 38.2026 | 38.1916 / 38.2051 / 38.1042 | 38.1790 → 38.1670 | -0.0120 | -0.0 % | 0.0597 |
      | s7 | 23.6958 / 23.6895 / 23.7181 | 23.7002 / 23.7199 / 23.7313 | 23.7011 → 23.7171 | +0.0160 | +0.1 % | 0.0286 |
      | s8 | 0.1427 / 0.1431 / 0.1429 | 0.1427 / 0.1428 / 0.1427 | 0.1429 → 0.1427 | -0.0001 | -0.1 % | 0.0004 |
      | s9 | 6.3987 / 6.3957 / 6.4413 | 6.3969 / 6.4028 / 6.3913 | 6.4119 → 6.3970 | -0.0149 | -0.2 % | 0.0457 |
      | s10 | 6.3968 / 6.3895 / 6.3914 | 6.3916 / 6.3932 / 6.3919 | 6.3925 → 6.3922 | -0.0003 | -0.0 % | 0.0074 |
      | s11 | 0.0887 / 0.0888 / 0.0888 | 0.0878 / 0.0893 / 0.0876 | 0.0888 → 0.0882 | -0.0005 | -0.6 % | 0.0001 |

    - Nothing is slower beyond its before spread; the largest increases are cifar-dense s11 +0.0016 ms (spread 0.0022)
      and cifar-sparse s4 +0.0163 ms (spread 0.0248). As expected: on rocky every buffer is still in type 5.
  - *Pixel 7a, the go device* (not required, recorded for the record): full `bm-cifar-dense-vk`, weights class forced
    to `default` vs the measured placement, 3 interleaved runs, battery 28.6–28.9 °C. Columns: default runs / measured
    runs / default mean / measured mean (Δ) / default spread:

      | s1 | 5.3474 / 7.0797 / 7.0815 | 6.5410 / 7.0838 / 7.0807 | 6.5028 (+0.0 %) | 6.9018 (+6.1 %) | 1.7341 |
      | s2 | 0.6523 / 0.6571 / 0.6571 | 0.6629 / 0.6461 / 0.6621 | 0.6555 (+0.0 %) | 0.6570 (+0.2 %) | 0.0047 |
      | s3 | 27.2662 / 26.0491 / 27.0231 | 26.2834 / 26.4706 / 26.0619 | 26.7795 (+0.0 %) | 26.2720 (-1.9 %) | 1.2171 |
      | s4 | 0.4874 / 0.4952 / 0.4980 | 0.4819 / 0.5057 / 0.5049 | 0.4935 (+0.0 %) | 0.4975 (+0.8 %) | 0.0106 |
      | s5 | 36.7471 / 37.8139 / 36.3989 | 36.9916 / 36.9651 / 36.9256 | 36.9866 (+0.0 %) | 36.9608 (-0.1 %) | 1.4150 |
      | s6 | 49.6263 / 49.2922 / 48.9664 | 49.1312 / 49.1209 / 48.8687 | 49.2950 (+0.0 %) | 49.0403 (-0.5 %) | 0.6599 |
      | s7 | 34.6950 / 34.1823 / 33.7802 | 32.6534 / 32.8133 / 32.9697 | 34.2192 (+0.0 %) | 32.8121 (-4.1 %) | 0.9148 |
      | s8 | 0.2412 / 0.2477 / 0.2662 | 0.2379 / 0.2393 / 0.2450 | 0.2517 (+0.0 %) | 0.2407 (-4.4 %) | 0.0250 |
      | s9 | 18.0707 / 17.0263 / 14.7424 | 11.3739 / 10.1343 / 12.8399 | 16.6131 (+0.0 %) | 11.4494 (-31.1 %) | 3.3283 |
      | s10 | 17.4427 / 15.9026 / 15.8961 | 12.3026 / 12.7640 / 11.2206 | 16.4138 (+0.0 %) | 12.0957 (-26.3 %) | 1.5466 |
      | s11 | 1.1342 / 1.0427 / 1.1077 | 1.0547 / 1.0810 / 1.1209 | 1.0949 (+0.0 %) | 1.0855 (-0.9 %) | 0.0915 |

    - fc1 −31.1 % and fc2 −26.3 % through the class mechanism; s7 (conv5) −4.1 % and s8 −4.4 % beyond their spreads; no
      stage slower beyond its spread.
  - The committed code differs from the gated build only by clang-tidy's `misc-misplaced-const` fixes in
    `sequence.cpp`/`vma_pmr.cpp` (dropping `const` from a `VkBuffer`/`VmaAllocation` local); group 3's gates re-run
    everything.
- [x] 2.5 If 2.4 landed and 1.4's stage→buffer declarations exist: compute GPU-local activations from the schedule (B3). Verify: a unit test asserts that every buffer a CPU chunk touches stays in the default class; the hybrid `test-pipeline-e2e-*-vk` suites pass on the rocky gate and phone gate.
  - *Evidence:* each cifar AppData buffer declares at construction the stages that touch it (`bt::PlaceBuffer`), and
    `runtime/placement.hpp` places a buffer in the weights/GPU-local class only when every declared accessor is a GPU
    stage of the schedule (design B3 "As implemented"). `run_pipeline`, `bm-gen-logs`, `bm-prof`'s GPU cells, the GPU
    baseline and `bm-cifar-*-vk` build their items this way.
  - `Placement.CpuChunkBuffersStayBoundary` (`test-cifar-{dense,sparse}-omp`, hardware-free, recording resources):
    for all 66 schedules with one GPU stage range of the 11 stages, every buffer a CPU stage or host code touches is
    boundary, and every GPU-only buffer is in the class of its kind. Passes; `ctest -L omp` locally 9/9.
    *Mutation:* with the policy's boundary branch removed it fails.
  - The declarations are checked against reality: `Placement.DeclaredStagesMatchBindings` (`test-cifar-{dense,sparse}-vk`)
    runs stages 1..11 and requires the buffers each dispatch binds (the Sequence's touched set) to equal the buffers
    declaring that stage. OK on rocky, the Pixel 7a and the Samsung. *Mutation:* with `u_fc1_out` declared `{9}`
    instead of `{9, 10}` it fails at stage 10.
  - Hybrid e2e suites (`test-pipeline-e2e{,-cifar-dense,-cifar-sparse}-vk`):
    - rocky: 3/3, 2/2, 2/2 in `ctest -L vulkan`; with `BT_VK_WEIGHTS_PLACEMENT=uncached` (weights actually moved, WC
      writes) cifar-dense 2/2 and cifar-sparse 2/2;
    - Samsung: 3/3, 2/2, 2/2, script exit 0 (run before the oracle fix below; its weights do not move);
    - Pixel 7a (weights moved): 3/3, then cifar-dense 2/2 (42 s and 43 s) and cifar-sparse 2/2 (257 s and 271 s),
      script exit 0.
  - *Found by the Pixel run:* the e2e oracle `CheckFinalPipeline` computed its double-precision reference by
    re-reading every weight many times straight from the AppData. Once the GPU-only weights were uncached for the CPU,
    the first cifar-dense test was still running after ~25 min (killed). The oracle now reads one host-cached copy
    of each weight (same reference, same tolerances).

## 3. Queue selection and several GPU chunks (topic C)

- [x] 3.1 Log the queue families, the selection and the priority; add `BT_VK_QUEUE` and `BT_VK_QUEUE_PRIORITY` with warn-and-fallback (C1). Verify: the family tables of all three devices are recorded in design.md C1; the rocky gate is green with `BT_VK_QUEUE=compute`; a refused priority is logged as a warning while the gate stays green.
  - *Evidence:* `BaseEngine::create_device` logs every queue family (flags, queueCount, timestampValidBits) and the
    selection (`Compute queue: family N (flags, BT_VK_QUEUE=…), priority …`) at debug level. `BT_VK_QUEUE=universal`
    (default) is the old first-compute-family choice; `compute` picks the first compute-without-graphics family, and
    otherwise warns and names the family it keeps. `BT_VK_QUEUE_PRIORITY=low|medium|high` chains
    `VkDeviceQueueGlobalPriorityCreateInfoKHR` through VK_KHR_global_priority (or the EXT), warns and continues
    without it when the device lacks both, and retries without it on `VK_ERROR_NOT_PERMITTED`. Family tables of all
    three devices: design.md C1.
  - rocky, `BT_VK_QUEUE=compute` (debug: `Compute queue: family 1 ({ Compute | Transfer | SparseBinding },
    BT_VK_QUEUE=compute)`): `test-kiss-vk` / tree / cifar-dense / cifar-sparse 27 / 16 / 13 / 13, `ctest -L vulkan` 7/7.
  - rocky, `BT_VK_QUEUE_PRIORITY=high`: `[warning] BT_VK_QUEUE_PRIORITY=high: the driver refused that global queue
    priority (VK_ERROR_NOT_PERMITTED); using the default priority`, then `priority default`; 27 / 16 / 13 / 13,
    `ctest -L vulkan` 7/7.
  - Phones (`SPDLOG_LEVEL=debug ./test-kiss-vk --gtest_filter=VulkanEngineTest.EngineInitialization`): the Pixel with
    `BT_VK_QUEUE=compute` warns `no compute-only queue family; using family 0 (…)` and passes; the Samsung selects
    family 1. Both accept `high` and `low` (`priority high` / `priority low`).
- [x] 3.2 Run the queue A/B on rocky: tree × VK and cifar-dense × VK, isolated and under `bm-prof-*-vk` interference, for universal/compute × default/high, with 3 interleaved runs. Verify: tables and a decision on the default are recorded.
  - *Evidence (rocky-ryzen, code 4321e25):* `bm-prof-{tree,cifar-dense}-vk --device minipc --benchmark_filter='vulkan/'`
    (the GPU cells only), `BT_PROF_SCENARIO` isolated and interference (every CPU tier saturated with the same stage
    while the GPU cell is measured), timed by GPU timestamps (`gpu`) and by host wall clock around each dispatch
    (`wall`, `BT_PROF_GPU_WALLCLOCK=1`). Arms `BT_VK_QUEUE` ∈ {universal, compute} × `BT_VK_QUEUE_PRIORITY` ∈
    {default, high}.
    - 3 runs, interleaved over every app × scenario × timing × arm: 96 runs, all exit 0.
    - Before each run: 5 s settle, then `gpu_busy_percent` 0, 1-min load < 0.5 (recorded 0.19–0.49), no other
      `/dev/dri` holder; `powersave`, DPM `auto`.
  - Metric: the sum over the app's stages of each GPU cell's p50 (ms), per run → mean (Δ vs universal:default):

    | app | scenario | timing | universal:default | universal:high | compute:default | compute:high | universal:default spread |
    |---|---|---|---|---|---|---|---|
    | tree | isolated | gpu | 1.5407 / 1.5456 / 1.5473 → 1.5445 (+0.0 %) | 1.5422 / 1.5323 / 1.5441 → 1.5395 (-0.3 %) | 1.5451 / 1.5446 / 1.5436 → 1.5444 (-0.0 %) | 1.5406 / 1.5438 / 1.5465 → 1.5436 (-0.1 %) | 0.0066 |
    | tree | isolated | wall | 2.0132 / 2.0327 / 2.0117 → 2.0192 (+0.0 %) | 2.0620 / 2.5158 / 2.0100 → 2.1959 (+8.8 %) | 2.0166 / 2.0503 / 3.1861 → 2.4177 (+19.7 %) | 2.0424 / 2.0436 / 2.0251 → 2.0370 (+0.9 %) | 0.0209 |
    | tree | interference | gpu | 1.6560 / 1.6552 / 1.6479 → 1.6530 (+0.0 %) | 1.6490 / 1.6514 / 1.6502 → 1.6502 (-0.2 %) | 1.6473 / 1.6503 / 1.6599 → 1.6525 (-0.0 %) | 1.6562 / 1.6586 / 1.6592 → 1.6580 (+0.3 %) | 0.0081 |
    | tree | interference | wall | 2.2221 / 2.1508 / 2.2369 → 2.2032 (+0.0 %) | 2.2143 / 2.1893 / 2.1621 → 2.1886 (-0.7 %) | 2.1968 / 2.2337 / 2.2247 → 2.2184 (+0.7 %) | 2.2038 / 2.1879 / 2.2025 → 2.1981 (-0.2 %) | 0.0861 |
    | cifar-dense | isolated | gpu | 8.9309 / 8.5441 / 8.5056 → 8.6602 (+0.0 %) | 8.5568 / 8.9164 / 8.5474 → 8.6735 (+0.2 %) | 8.5602 / 8.5316 / 8.4902 → 8.5273 (-1.5 %) | 8.4733 / 8.9001 / 8.4658 → 8.6131 (-0.5 %) | 0.4253 |
    | cifar-dense | isolated | wall | 9.8267 / 10.6761 / 10.6572 → 10.3867 (+0.0 %) | 9.7540 / 11.1755 / 10.7730 → 10.5675 (+1.7 %) | 11.5509 / 10.2915 / 9.8625 → 10.5683 (+1.7 %) | 10.7303 / 9.3370 / 11.5084 → 10.5253 (+1.3 %) | 0.8494 |
    | cifar-dense | interference | gpu | 10.9127 / 10.8760 / 10.8841 → 10.8909 (+0.0 %) | 10.9044 / 10.8344 / 10.8473 → 10.8621 (-0.3 %) | 10.8719 / 10.8949 / 10.9204 → 10.8957 (+0.0 %) | 10.8872 / 10.8859 / 10.8389 → 10.8707 (-0.2 %) | 0.0367 |
    | cifar-dense | interference | wall | 12.4491 / 12.3073 / 12.2820 → 12.3461 (+0.0 %) | 12.1202 / 12.3016 / 12.3051 → 12.2423 (-0.8 %) | 12.2514 / 12.2663 / 12.3133 → 12.2770 (-0.6 %) | 12.2831 / 12.2657 / 12.2874 → 12.2787 (-0.5 %) | 0.1672 |

  - Every arm's mean is within ±1.5 % of universal:default, mostly inside its spread. Two wall-clock means move
    more (tree isolated universal:high +8.8 %, compute:default +19.7 %), each because of one outlier run (2.5158 and
    3.1861 ms) while its other two runs sit with the base.
  - The `high` arms ran at the **default** priority: rocky's driver refuses `high` (`VK_ERROR_NOT_PERMITTED`, see
    3.1; `bm-prof` keeps logging off, so the warning is not in these logs). So the priority half of the A/B could
    not be measured on rocky.
  - **Decision: the defaults stay** (`universal` family, driver-default priority); recorded in design.md C1.
- [x] 3.3 Move the touched-buffer set into `Sequence`, add the `Engine` submit mutex and the `dispatch_multi_stage(…, Sequence&)` overloads (C2). Verify: the rocky gate and phone gate are green; a new test runs two threads with their own Sequences against one dispatcher for 1,000 items under `BT_VK_VALIDATION=1`, with 0 validation errors.
  - *Evidence:* the touched set moved into `Sequence` in 2.4 (`Sequence::touched_buffers()`, flushed and
    invalidated through the owning resource). `Engine` owns a `std::mutex` that every one of its Sequences locks
    around `vkQueueSubmit`. Each dispatcher has `make_seq()` and `dispatch_multi_stage(…, kiss_vk::Sequence&)`; push
    constants are recorded with the call, so recording the same Algorithm from two threads no longer races (design C3
    "As implemented").
  - New `VulkanConcurrentSequences.TwoThreadsOneDispatcher` (`test-tree-vk`): 2 threads × 500 full chunks (stages
    1–7) on one `VulkanDispatcher`, each thread with its own Sequence and two `VkAppData_Safe` items; every stage of
    all 4 items then matches the oracle exactly.
    - rocky with `BT_VK_VALIDATION=1` (plus the sync-val settings): passes, **0** `Validation Error` lines, and the
      whole suite 16/16 with 0 errors and 0 hazards.
    - *Control (rocky, not committed):* with the submit lock removed, the same run reports
      `UNASSIGNED-Threading-MultipleThreads-Write` — `vkQueueSubmit(): THREADING ERROR : object of type VkQueue is
      simultaneously used in current thread … and thread …`. Restored: 0.
  - Rocky gate: 27 / 16 / 13 / 13, `ctest -L vulkan` 7/7. Phone gate: both phones 26 + 1 skipped (FIFO) / 16 / 13 / 13,
    script exit 0.
- [x] 3.4 Narrow `first_concurrent_gpu_chunk`, update `runtime/tests/test_schedule.cpp`, and add an end-to-end tree and cifar-dense test with Vulkan 1–2 / CPU 3–5 / Vulkan 6–N (C3). Verify: `ctest -L omp` is green; the end-to-end test matches the oracle on the rocky gate and phone gate (the Pixel covers non-coherent memory) with 0 validation errors on rocky.
  - *Evidence:* `first_concurrent_gpu_chunk` allows repeated Vulkan chunks and still rejects repeated CUDA chunks
    (scope: design C3). In `runtime/tests/test_schedule.cpp`, `TwoVulkanChunksRejected` became
    `TwoVulkanChunksAllowed`, and `TwoCudaChunksRejected` now also checks the reported chunk. `run_pipeline` and
    `bm-gen-logs` give each Vulkan GPU chunk its own Sequence.
  - End-to-end: `PipelineE2EVk.TwoVulkanChunksAroundCpu` (VK 1–2 / CPU 3–5 / VK 6–7) and
    `PipelineE2ECifarDenseVk.TwoVulkanChunksAroundCpu` (VK 1–2 / CPU 3–5 / VK 6–11). The tree suite's
    `RejectsMultiGpuChunkSchedule` became `ThreeVulkanChunksAlternating`, which now **runs** the VK 1–3 / CPU 4 / VK 5 /
    CPU 6 / VK 7 schedule that used to crash the device, checking every item against the oracle.
  - `ctest -L omp` locally 9/9 (`test-schedule-omp` 23/23).
  - rocky: `test-pipeline-e2e-vk` 4/4 and `test-pipeline-e2e-cifar-dense-vk` 3/3 under `BT_VK_VALIDATION=1` with the
    sync-val settings, 0 `Validation Error` and 0 `SYNC-HAZARD` (layer inserted 4 and 3 times); `ctest -L vulkan` 7/7.
  - Pixel 7a (non-coherent cached memory): 4/4 and 3/3, e.g. `TwoVulkanChunksAroundCpu` 15.9 s (tree) and 33.9 s
    (cifar-dense). Samsung: 4/4 and 3/3. Script exit 0 on both.
- [x] 3.5 Check whether the schema, `config_reader.hpp`'s coverage check, or `optimizer/tests/test_schedule_contract.py` forbid more than one chunk per PU; then add the opt-in `allow_gpu_multi_chunk` solver mode with the GPU-sum cost (C4). Verify: the findings are recorded; a pytest asserts the default mode never emits two GPU chunks, and that in multi-chunk mode the toy profile's period is ≥ the sum of its GPU segments; `uv run pytest -q optimizer` is green.
  - *Findings* (design C4 "Verified"): the schema, `config_reader.hpp` + `validate_schedule_coverage` and
    `test_schedule_contract.py` all allow a PU in several chunks. Only `finalize_solution`'s predicted period (largest
    chunk) assumed otherwise; it is now the largest per-PU sum (unchanged for one chunk per PU).
  - `solve_optimization_problem(…, allow_gpu_multi_chunk=False)` and `02_gen_schedule_merged.py
    --allow-gpu-multi-chunk`: the GPU is exempt from contiguity, and the sum of its segments, each with the per-chunk
    overhead, is ≤ T_max.
  - New `optimizer/tests/test_gpu_multi_chunk.py`, on a toy profile where two GPU chunks around a Big chunk win:
    - default mode, 20 solutions: no PU repeats, ≤ 1 GPU chunk, and the best period is above the 2-chunk one;
    - multi-chunk mode: the best schedule is GPU 1–2 / Big 3–5 / GPU 6–11 at period 8 ms = 2 + 6; for every solution
      the period ≥ the sum of its GPU chunk times; the output is schema-valid;
    - with a 5 ms per-chunk GPU overhead every GPU chunk carries its own 5 ms, and the period covers the sum.
  - `uv run pytest -q optimizer` → 21 passed. *Mutation:* without the time-sharing sum constraint, 2 of the 3 new tests
    fail.

## 4. Submission overhead (topic D)

- [x] 4.1 Add phase timing: record, submit, wait, GPU start/end and start latency. Use calibrated timestamps when `VK_KHR_calibrated_timestamps` or its EXT form is available (D1). Verify: on rocky, a tree × VK run reports phases per chunk that sum to wall time within the timer resolution; the device's calibrated-timestamp support is logged.
  - *Evidence:* `BT_VK_PHASES=1` records per submission record / between / flush / submit / wait / invalidate (host,
    contiguous), the GPU interval, and with calibrated timestamps the split of submit + wait into start latency + GPU +
    wake (design D1 "As implemented"). `VK_KHR_calibrated_timestamps` or the EXT is enabled where the device has the
    device and CLOCK_MONOTONIC domains. Tool: `bm-phases-{tree,cifar-dense}-vk`.
  - Support logged (`SPDLOG_LEVEL=debug ./test-kiss-vk --gtest_filter=VulkanEngineTest.EngineInitialization`):
    - rocky `Calibrated timestamps: VK_KHR_calibrated_timestamps (device + CLOCK_MONOTONIC)`;
    - Pixel 7a `VK_KHR_calibrated_timestamps (…)`;
    - Samsung `VK_EXT_calibrated_timestamps (…)`.
  - *Found: rocky's calibration is wrong.* The first sweep's calibrated split was off by a constant ~28,983 ms. A
    probe showed rocky's queue timestamps 2.898e9 ticks (10 ns/tick) ahead of `VK_TIME_DOMAIN_DEVICE`, at the same
    rate. The Engine now checks the calibration once when phases are on, and treats a failing device as uncorrelated
    (design D1). rocky, `bm-phases-tree-vk --device minipc -s 4 -e 7 -n 50` at b2657ee: `[warning] Calibrated
    timestamps: a queue timestamp (6566338792851) lies outside the calibrated device ticks around it
    (6563440371417..6563440475701, offset 2898421434 ticks); GPU and host phases stay uncorrelated`, `calibrated=0/50`.
    The phases are still reported per chunk, and their sum equals the wall time exactly (`max_abs_residual_ms
    host=0.000000`; the steady clock's resolution is 1 ns).
  - The phones pass the check and show the full split (design D1 "Phones"). Start latency is measured from the start
    of the submit call (3d3393d: the GPU may start before the call returns). Minimum start latency ≥ 0 except one
    Samsung chunk at −0.229 ms, within the 0.534 ms deviation the driver reported for that pair.
- [x] 4.2 Run the idle-gap sweep (0, 1, 5, 20 ms between chunks) on rocky for tree × VK and cifar-dense × VK (D1). Verify: a phase table and a GPU-time-vs-gap table are recorded in design.md D1, with an explanation of the ≈1.5 ms (≈0.85 ms tree-only) fitted overhead by phase.
  - *Evidence:* rocky, `bm-phases-<app>-vk --gap-ms G` for G ∈ {0, 1, 5, 20} on tree 1–3, 4–7, 1–7 and cifar-dense
    1–11, 5–11, 2 runs each, interleaved over shapes and gaps (40 runs, all exit 0).
    - Before each run: 5 s settle, `gpu_busy_percent` 0, load 0.05–0.49, no other `/dev/dri` holder; `powersave`, DPM
      `auto`; `BT_WEIGHTS_DIR=/tmp/bt/weights`.
    - The sweep ran at 0e32da4, before the calibration check, so only its host phases and GPU times are used (start
      latency + wake as wait − GPU).
  - Phase table, GPU-time-vs-gap table, dispatch-wall-vs-gap table and the explanation are in design.md D1 "Results"
    / "What the ≈1.5 ms … is". In short:
    - host submission costs 0.07 ms (tree) to 0.6 ms (cifar-dense) per chunk, recording only µs;
    - a hot chunk runs 0.06–0.48 ms *faster* than its stages timed alone;
    - idle before a chunk slows it sharply: tree 1–3 1.09 → 1.48 / 2.54 / 2.98 ms after 1 / 5 / 20 ms; cifar-dense
      1–11 8.98 → 24.1 ms after 20 ms.
    - So the fitted per-chunk constant is mostly the GPU's DVFS slowdown after idling between pipelined items (plus
      the SPSC hand-off), not submission overhead.
- [x] 4.3 Pre-record command buffers per (chunk, pooled item), with re-record on a key mismatch and a defined behaviour on the descriptor-fallback path (D2). Verify: a test shows no recording on the second pass of the same item (debug counter), and a re-record when push constants change; the rocky gate and phone gate pass with and without `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1`.
  - *Evidence:* recipe capture + cached command buffers (design D2 "As implemented"): an equal recipe (pipelines,
    buffers, push-constant bytes, dispatch sizes, barriers) reuses its recorded command buffer, and any difference
    re-records. Up to 32 per Sequence, LRU among slots not in flight, invalidated by any deallocation. The
    transient-pool fallback re-records every submission (its defined behaviour). `Sequence::recording_count()` is the
    debug counter.
  - `BothPaths/VulkanPrerecord.SecondPassRecordsNothingChangeReRecords/{PushDescriptors,Fallback}` (`test-kiss-vk`,
    tree_morton): passes record 1, 0 (same work), 1 (push constant `n` halved), 1 (other output buffer), 0, 0 (both
    earlier recipes still cached), and the outputs show which `n` ran. On the fallback every pass records 1.
  - `VulkanPrerecord.SameItemSecondPassRecordsNothing` (`test-tree-vk`): stages 1–7 on item a records 1, a again 0, b
    1, a again 0; stage 7 of both items matches the oracle exactly. Under the forced fallback the same test expects
    and gets 1 every pass.
  - rocky (`--device minipc`), default and `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1`: `test-kiss-vk` / tree / cifar-dense /
    cifar-sparse 29 / 17 / 13 / 13 both times; `ctest -L vulkan` 7/7. sync-val with the forced fallback: those four
    plus `test-pipeline-e2e{,-cifar-dense}-vk`, 0 errors / 0 hazards.
  - Phones, default (`scripts/run-on-android.sh`, exit 0) and forced fallback: both 28 + 1 skipped (FIFO) / 17 / 13 /
    13.
- [x] 4.4 Enable `timelineSemaphore`, then split the GPU worker into submit and complete with `BT_VK_INFLIGHT` (default 1), preserving FIFO hand-off; `run_pipeline` asserts the pool size is sufficient (D3). Verify: the end-to-end suites pass at depth 1 and 2 on the rocky gate and phone gate; a test asserts output order equals input order at depth 2.
  - *Evidence:* `timelineSemaphore` is required (1.2 table); `Sequence::submit()` returns a timeline ticket and
    `wait(ticket)` completes it, with no fence. The dispatchers split into `submit_multi_stage`/`complete_multi_stage`.
    `inflight_worker` runs the Vulkan GPU chunks of `run_pipeline` and `bm-gen-logs` at `BT_VK_INFLIGHT` (default
    1). `run_pipeline` asserts pool ≥ depth × GPU chunks + CPU chunks and checks every run's FIFO completion order
    (design D3 "As implemented").
  - Order tests:
    - `InflightWorker.Depth2KeepsTwoInFlightAndPreservesOrder` (`test-schedule-omp`, hardware-free): 20 items reach 2
      outstanding and complete and leave in input order;
    - `Depth1IsSubmitThenWait`: never more than 1 outstanding;
    - `PipelineE2EVk.InflightDepth2KeepsOrderAndResults` (`test-pipeline-e2e-vk`): forces depth 2 over all-VK,
      OMP|VK and VK|OMP|VK schedules, oracle per item plus the order check.
    - *Mutation:* completing the newest instead of the oldest in-flight item fails the hardware-free test, and
      `item 0 left the ring out of order (depth 2)` fails the e2e one.
  - End-to-end suites at depth 1 and 2:
    - rocky: `BT_VK_INFLIGHT=1 ctest -L vulkan` 7/7 and `BT_VK_INFLIGHT=2 ctest -L vulkan` 7/7; sync-val at both depths
      on the four suites plus `test-pipeline-e2e{,-cifar-dense}-vk`, 0 errors / 0 hazards (layer inserted 40 / 18 / 13 /
      13 / 7 / 3 times);
    - Pixel 7a and Samsung: `test-pipeline-e2e-vk` 5/5, `-cifar-dense-vk` 3/3, `-cifar-sparse-vk` 2/2, at
      `BT_VK_INFLIGHT=1` and `=2`.
  - `ctest -L omp` locally 9/9 (`test-schedule-omp` 26/26); `uv run pytest -q optimizer` 21 passed.
- [x] 4.5 Run the refit gate for 4.3 + 4.4, at depth 1 and depth 2 against the pre-group-4 baseline (D4). Verify: the before/after `gpu_vulkan` per-chunk constants, the spreads, tree × VK best makespan and the per-GPU-chunk-shape wall times are recorded; the after constant is ≤ the before mean + the before spread. Depth 2 becomes the default only if it wins beyond noise.
  - *Method (rocky-ryzen):* "before" = 9236dda (parent of group 4), "depth 1" / "depth 2" = 1ffb88f with
    `BT_VK_INFLIGHT=1` / `=2`.
    - Each arm has its own worktree with binaries built on rocky (`scripts/build-bench-on-rocky.sh vulkan`). Each run
      is a full `00_run_fleet.py --only minipc --fresh --phases profile,schedule,run,summary`, then `fit_overhead.py
      --dry-run`.
    - Four rounds, interleaved before → depth 1 → depth 2: 12 runs, all exit 0. Rounds 3–4 were added because after
      two the all-apps GPU ranges did not overlap.
    - Before every run: 5 s settle, `gpu_busy_percent` 0, load 0.46–0.49, no other `/dev/dri` holder, no CI job;
      `powersave`, DPM `auto`, clocks unlocked.
  - Tables in design.md D4 "Results (tasks 4.5)": fitted constants, throughput, tree GPU chunk shapes, and the
    teardown experiment. Means before → depth 1 (before spread):
    - `gpu_vulkan` per_chunk, all apps: 1.4645 → **1.6644** (0.3583). This is ≤ before mean + spread = **1.8228**, so
      **acceptance met**;
    - `gpu_vulkan` per_chunk, tree only: 0.8711 → 0.8664 (0.0131);
    - tree best measured makespan (max chunk): 1.7223 → 1.7324 (0.0364);
    - tree GPU chunk durations for stages 1–2 / 1–3 / 1–4 / 1–6: within their before spreads (≤ +0.0382); stages 1–5
      +0.0778 (spread 0.0281).
  - *Found:* an end-of-run stall in the CPU chunk with the group 4 builds.
    - Of 32 tree schedule runs per arm, 21 at depth 1 and 20 at depth 2 have one of the last six items above 5 ms (up
      to 24 ms); before, 0 (max 4.7 ms).
    - It makes tree's per-task makespan +0.17 ms and its tree-only `cpu` constant +0.077 ms (spread 0.018). The
      steady period is unchanged: 1.5278 → 1.5335.
    - Experiment (not committed): keeping each GPU chunk's `Sequence` alive past its thread's exit cuts the stalls on
      fixed schedules from 10/15 to 3/15 (before 0/15). So the Sequence teardown, with its up to 32 recorded command
      buffers, explains most of it.
    - Follow-up recorded in D4; production steady state is unaffected.
  - *Depth 2:* at depth > 1 the chunk ticks are submit → complete latency, so the fitted constants (8.48 ms all-apps)
    and the max-chunk makespan (2.96 ms) do not measure cost; depth 2 was compared by throughput.
    - It beats depth 1 beyond noise on 5 of 6 metrics, but against before it is level. Its steady-state gain is
      < 1 %: cifar-dense −0.7 %, cifar-sparse −0.5 %, tree inside noise.
    - Enabling it would need constants fitted at depth 2, which today's tooling cannot produce.
    - **The default stays `BT_VK_INFLIGHT=1`.**

## 5. Docs and final gate

- [x] 5.1 Document every new knob (`BT_VK_BARRIER_ALL`, `BT_VK_WEIGHTS_PLACEMENT` if kept, `BT_VK_QUEUE`, `BT_VK_QUEUE_PRIORITY`, `BT_VK_INFLIGHT`, and the solver's `allow_gpu_multi_chunk`) in `docs/instruction-for-ai/03-unit-testing.md` and `06-end-to-end-scheduling.md`. Update `bugs-found.md` §10 to say the single-GPU-chunk guard is lifted, and add a decision record with the measurement tables under `docs/reports-for-human/`. Verify: each knob is documented with its default.
  - *Evidence:* `BT_VK_WEIGHTS_PLACEMENT` is kept, as the override for re-measuring placement.
    - `03-unit-testing.md` "Engine knobs": each of the five `BT_VK_*` knobs, plus `BT_VK_PHASES` and the solver's
      `allow_gpu_multi_chunk`, with its values and its default in brackets.
    - `06-end-to-end-scheduling.md`: step 2 describes `--allow-gpu-multi-chunk` (default off). Step 3 has a table of
      all six with their defaults, the forwarding of `BT_VK_*` to device runs, and the warning that chunk ticks are
      latency at `BT_VK_INFLIGHT` > 1.
  - `bugs-found.md` §10: the guard is lifted for Vulkan (group 3, 4321e25), still rejects repeated CUDA chunks, and
    the once-crashing schedule now runs as `PipelineE2EVk.ThreeVulkanChunksAlternating`.
  - Decision record: `docs/reports-for-human/2026-10-15-vk-engine-runtime-decisions.md`, indexed in that directory's
    README. It has the knob/default table, the measurement tables per topic (barrier counts, fc1/fc2 placement, queue
    A/B, phases, idle gap, 4.5 refit) and the open follow-ups.
- [ ] 5.2 Final gate at the last commit: local, rocky gate, phone gate and sync-val. Verify: all green, with results listed in the PR.
  - *Results at f51f815 (the code is 1ffb88f's; 4aaddd8 and f51f815 change only docs):*
    - local: `ctest --test-dir build/pc -L omp` 9/9; `uv run pytest -q optimizer` 21 passed.
    - rocky gate (`--device minipc`): `ctest -L vulkan` 7/7; `test-kiss-vk` / tree / cifar-dense / cifar-sparse
      29 / 17 / 13 / 13.
    - sync-val (rocky, with the heuristic and `VK_LOADER_DEBUG=layer`): those four plus `test-pipeline-e2e-vk` 5/5 and
      `test-pipeline-e2e-cifar-dense-vk` 3/3. 0 `Validation Error`, 0 `SYNC-HAZARD`; the validation layer was inserted
      40 / 18 / 13 / 13 / 7 / 3 times.
  - **Phone gate: not run, so 5.2 stays open.** Both phones are attached to the build box (`lsusb` lists them), but
    this session's process could not open `/dev/bus/usb` (seccomp, NoNewPrivs; `adb devices` reports both as `no
    permissions`). Running `scripts/run-on-android.sh 3A021JEHN02756 …` and `… R5CY21Y3VEV …` with the four VK test
    targets from a normal shell closes it. The last phone results are 4.4's (all green at depths 1 and 2).
  - No PR was opened. The results are listed here.
