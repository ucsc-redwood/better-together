# Tasks

Gates used below:
- **OMP gate:** `ctest --test-dir build/pc -L omp` passes locally.
- **VK gate:**
  - **rocky-ryzen:** rsync tracked files to `doremy@100.97.77.80:bt-vk-campaign/`, then on rocky `bash -s` → `cmake --preset vulkan && cmake --build --preset vulkan`. There `ctest -L vulkan` (NOT `-L vk`) and `test-tree-vk --device minipc` pass, also with `BT_VK_SUBGROUP_SIZE=32`.
  - **Phones:** `scripts/run-on-android.sh <serial> test-tree-vk test-kiss-vk` passes on the Pixel 7a (`3A021JEHN02756`) and the Samsung (`R5CY21Y3VEV`), and on the Samsung also with `BT_VK_SUBGROUP_SIZE=32`.
- **CUDA gate:** `just build-jetson` (`bt-cross:7.2`) then `scripts/run-on-jetson.sh` pass on duck-stable (all `test-*-cu`).

Bake shaders locally and commit the `*_spv.h` files before any rocky or phone run.

## 1. Correctness reproducers (topic A, design A1)

- [ ] 1.1 Add a Morton boundary test: points at `kMinCoord + kRange` and at a value whose scaled product rounds to 1024.0. Assert each axis quantizes to 1023 on OMP, and compare the CUDA/VK codes with OMP. Verify: it FAILS on the current code (record the failing values), and lives in the tree test files.
- [ ] 1.2 Add a leaf-race test: a constructed input where many BRT nodes link leaves into the same octree node. Run OMP and CUDA stage 7 repeatedly (≥ 100×) and compare the leaf masks with a serial reference. Verify: a lost bit is observed at least once on the current code, or record that the race couldn't be triggered, with the iteration count.
- [ ] 1.3 Add an oracle self-test: give `CheckStage7Topology` an output whose leaf masks are reset to "no leaves". Verify: on the current code it either fails (good, the check works) or passes (the check is vacuous). Record which.
- [ ] 1.4 Explain why `TreeDiffVulkan.Stage7_Octree` passes though Vulkan never runs `k_LinkLeafNodes`: inspect `u_oct_child_leaf_mask_s7_out`'s initial value and what Vulkan writes to it. Verify: the explanation is written under this task and in design A3, and the test from 1.3 or 3.2 now fails for Vulkan on the current code.

## 2. Morton clamp (topic A, design A2)

- [ ] 2.1 Clamp the quantized coordinate to `[0, 1023]` in OMP `func_morton.hpp`, CUDA `func_morton.cuh` and `tree_morton.comp`, identically. Verify: 1.1 passes on all three backends, and stage-1 output on the default input is byte-identical to before (diff recorded).
- [ ] 2.2 Run the OMP gate, the VK gate and the CUDA gate. Verify: all green.

## 3. Atomic leaf masks and Vulkan leaf linking (topic A, design A3)

- [ ] 3.1 Make `set_leaf`'s clear atomic: `#pragma omp atomic` in `func_octree.hpp`, `atomicAnd` in `07_octree.cu`. Verify: 1.2 shows no lost bits over ≥ 100 repetitions on OMP and CUDA; the OMP gate and CUDA gate are green.
- [ ] 3.2 If 1.3/1.4 showed the leaf-slot check is vacuous, fix `CheckStage7Topology` so missing leaf links fail. Verify: the oracle self-test fails on a no-leaves output, and OMP and CUDA still pass.
- [ ] 3.3 Wire leaf linking into Vulkan stage 7: a second dispatch after a compute→compute barrier, with `SetLeaf` using `atomicAnd` (design A3). Verify: `TreeDiffVulkan.Stage7_Octree` and `TreeDiffVulkanChained.Stage7_Octree` pass with the fixed oracle on rocky and both phones.
- [ ] 3.4 Measure stage-7 time before and after on rocky (`bm-tree-vk --benchmark_filter='Tree/Stage7/'`) and on duck-stable CUDA. Verify: the numbers are recorded; any increase comes from the added leaf work.
- [ ] 3.5 Add entries to `docs/reports-for-human/bugs-found.md` for the Morton wrap, the leaf race and the missing Vulkan leaf links. Verify: each entry links its reproducer and fix commit.

## 4. Sort baseline and workgroup-count knob (topic B, design B1)

- [ ] 4.1 Record the baseline on rocky: `bm-tree-vk --device minipc --benchmark_filter='Tree/Stage[23]/' --benchmark_repetitions=10`, 3× interleaved, recording the governor, `gpu_busy_percent` and load. Verify: the stage-2/3 medians and spread are written here (expected ≈0.785 / ≈0.179 ms).
- [ ] 4.2 Sweep `kRadixNumWorkgroups` ∈ {32, 64, 128, 256} in both `VkAppData_Safe` and `VkAppData` (`apps/tree/vulkan/vk_appdata.hpp`), measuring as in 4.1. Verify: there is a table of stage-2 medians per value, and `test-tree-vk` passes at each.
- [ ] 4.3 Keep the best value if it beats 256 beyond the spread. Verify: the VK gate is green and the decision recorded. If stage 2 is already ≤ 0.40 ms, record it and confirm with the owner whether groups 5–6 are still wanted.

## 5. Global prefix once per pass (topic B, design B2/B3)

- [ ] 5.1 Add the per-pass histogram-table scan, or the upfront all-digit histogram (whichever 4.x favours; record the choice in design B2), and make `multi_radixsort.comp` read only its own offsets. Verify: `test-tree-vk` is exact on rocky at 64 and forced 32.
- [ ] 5.2 Run the full VK gate. Verify: green, including the Pixel 7a at subgroup 16.
- [ ] 5.3 Measure stage 2 as in 4.1. Verify: before/after recorded, with no regression beyond the spread.

## 6. Scatter and histogram efficiency (topic B, design B4)

- [ ] 6.1 Add a local LDS sort by digit before the scatter, sized for the smallest `maxComputeSharedMemorySize` across the three GPUs (log each value). Verify: the VK gate is green, and the sizes are recorded.
- [ ] 6.2 Add subgroup-aggregated histogram atomics in `multi_radixsort_histograms.comp`. Verify: the VK gate is green.
- [ ] 6.3 Measure stage 2 as in 4.1. Verify: the median is ≤ 0.40 ms (spec "Sort time is bounded"), or the remaining gap is explained and recorded.

## 7. Full subgroups (topic B, design B5)

- [ ] 7.1 Confirm that the sort pipelines (`multi_radixsort`, plus any new subgroup-dependent sort/scan kernel from groups 4–8) are created as subgroup-size-dependent, so they get `REQUIRE_FULL_SUBGROUPS` from the engine-wide mechanism that `vk-engine-runtime` lands first (that change owns the feature enablement and the `Algorithm` flag; do not re-implement it here). Verify: the debug log on each VK target shows the flag on the sort pipelines, and `BT_VK_VALIDATION=1 test-tree-vk` on rocky reports 0 validation errors.
- [ ] 7.2 Check that the tree suites pass exactly on a device path without `computeFullSubgroups`, using the engine change's fallback. Verify: on rocky, force the fallback with the engine's test hook (or note the device coverage if no such hook exists), then run `test-tree-vk` 15/15.

## 8. Stage-3 compaction (topic B, design B6)

- [ ] 8.1 Implement compaction in fewer than 5 dispatches. Keep `record_device_scan` unchanged for stage 6 unless both stages re-verify. Verify: `test-tree-vk` stage 3 and stage 6 are exact on all VK targets.
- [ ] 8.2 Measure stage 3 as in 4.1. Verify: the after median is ≤ the before median, and the dispatch count is recorded (spec "Compaction is not dispatch-bound").

## 9. Final gate and profile refresh

- [ ] 9.1 Run the OMP, VK and CUDA gates at the final commit, plus `BT_VK_VALIDATION=1` runs of `test-tree-vk` on rocky. Verify: all green, with 0 validation errors.
- [ ] 9.2 Refresh tree × VK profiling on rocky (`optimizer/orchestrate/00_run_fleet.py --only minipc --fresh --phases profile`). Verify: the new per-stage times are recorded, with the run noted here.
