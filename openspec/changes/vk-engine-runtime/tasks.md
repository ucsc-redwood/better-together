# Tasks

Gates used below:
- **"rocky gate":**
  - rsync the tree: `git ls-files -co --exclude-standard -z | rsync -a --from0 --files-from=- ./ doremy@100.97.77.80:bt-vk-campaign/`;
  - then `ssh doremy@100.97.77.80 bash -s` (fish login shell);
  - there: `cmake --preset vulkan && cmake --build --preset vulkan -j8`, then `cd build/vulkan && ctest -L vulkan` plus `test-{kiss,tree,cifar-dense,cifar-sparse}-vk --device minipc`.
- **"phone gate":** `cmake --preset android`, build the four VK test targets, then `scripts/run-on-android.sh 3A021JEHN02756 <targets>` and `scripts/run-on-android.sh R5CY21Y3VEV <targets>`.
- **"local":** `ctest --test-dir build/pc -L omp` and `uv run pytest -q optimizer`.
- **"sync-val":** the VK suites on rocky with `BT_VK_VALIDATION=1` and `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`; layer activity is confirmed with `VK_LOADER_DEBUG=layer`.
- **"refit":** before/after `optimizer/orchestrate/00_run_fleet.py --only minipc --fresh --phases profile,schedule,run,summary` on rocky, at least 2 interleaved runs per arm, then `optimizer/analysis/fit_overhead.py --dry-run`, as in the archived campaign's tasks 4.3/7.7.

## 1. Sync hygiene (topic A)

- [ ] 1.1 Enable `synchronization2` in the 1.3 feature table (A1) and log its support bit per device. Verify: debug logs on rocky, the Pixel 7a and the Samsung show it supported and enabled, and the rocky gate is green.
- [ ] 1.2 Replace every legacy barrier (`sequence.cpp` `cmd_end` and `cmd_memory_barrier`, and the tree dispatcher's inline barriers at `:300/:368/:419`) with `vkCmdPipelineBarrier2`. The host barrier is COMPUTE_SHADER/SHADER_STORAGE_WRITE → HOST/HOST_READ; the inter-stage one is SHADER_STORAGE_WRITE → SHADER_STORAGE_READ|WRITE (A2). Verify: `git grep -n "pipelineBarrier(" platform apps` finds no legacy calls, and the rocky gate, phone gate and sync-val (0 hazards) all pass.
- [ ] 1.3 Move both timestamps to `vkCmdWriteTimestamp2` at COMPUTE_SHADER (A4). Record per-stage GPU times for tree, cifar-dense and cifar-sparse on rocky, before and after. Verify: the before/after table is recorded, and `test-kiss-vk`'s timestamp checks pass.
- [ ] 1.4 Have stages declare their read/write buffers, and emit an inter-stage dependency only when the next stage touches a buffer written since the last barrier (A3). Add the `BT_VK_BARRIER_ALL=1` override. Verify: sync-val reports 0 hazards with and without the override; the rocky gate and phone gate are green; the number of barriers recorded per chunk is logged at debug level for each app, before vs after.
- [ ] 1.5 Enable `computeFullSubgroups` as optional, and set REQUIRE_FULL_SUBGROUPS on subgroup-dependent pipelines. Without the feature, check that the X workgroup size is a multiple of the subgroup size (A5). Verify: tree suites pass exactly at 16 (Pixel), 32 (`BT_VK_SUBGROUP_SIZE=32` on rocky and the Samsung) and 64; a unit test with a non-multiple workgroup size and the feature forced off fails with both sizes named.

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
