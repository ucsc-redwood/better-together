# Quickstart: Validating the Vulkan Chained AppData Migration

Prerequisites: `just setup-hooks` once per clone; an integrated-GPU Vulkan target
(Jetson devkit, Pixel 7a, or Samsung Galaxy — see
`docs/instruction-for-ai/01-hardware.md` for ssh/adb access).

## 1. Standalone correctness proof (User Story 1)

Build the new `TreeDiffVulkanChained` suite (see
[contracts/vulkan-dispatch-chained.md](./contracts/vulkan-dispatch-chained.md)) and the
hybrid CPU+GPU concurrency test (mirrors `test-pipeline-chained-cu`), then run on each
required target:

```bash
# Dev iteration (x86 iGPU, fastest loop, not a required verification target):
cmake --preset vulkan && cmake --build --preset vulkan --target test-tree-vk
./build/vulkan/test-tree-vk --gtest_filter="TreeDiffVulkanChained.*" --device rocky-ryzen

# Required: Jetson (cross-build via bt-cross:7.2, deploy, run — see 02-building.md)
cmake --preset jetson && cmake --build --preset jetson --target test-tree-vk
scripts/run-on-jetson.sh test-tree-vk

# Required: both Android phones (build via android preset, deploy via rocky's adb)
cmake --preset android && cmake --build --preset android --target test-tree-vk
scripts/run-on-android.sh 3A021JEHN02756   # Pixel 7a
scripts/run-on-android.sh R5CY21Y3VEV      # Samsung Galaxy
```

Expected: every stage's output matches the OMP reference on all three targets
independently (SC-001); the hybrid CPU+GPU schedule test shows genuine concurrent
CPU/GPU processing of different items with a correct final result (SC-002).

## 2. Production Vulkan profilers on the chained path (User Story 2)

After `profiler/tree-vk/const.hpp`'s `AppDataT` switch (mirrors
`profiler/tree-cu/const.hpp`'s Phase 2 change):

```bash
scripts/run-on-jetson.sh bm-baseline-tree-vk bm-fully-tree-vk bm-gen-logs-tree-vk bm-prof-tree-vk run-pipe-tree-vk
scripts/run-on-android.sh 3A021JEHN02756 bm-baseline-tree-vk   # repeat per tool, per phone
```

Before trusting any timing output, confirm no competing process is running on the
target (Constitution Principle VI — check for stray servers/builds the way the
`llama-server`-on-Jetson incident this session required).

Expected: every tool runs to completion without error on Jetson and both phones
(SC-003).

## 3. Routine correctness gate migration (User Story 3)

After `test-tree-vk`/`test_main.cpp` and `test-pipeline-e2e-vk`/
`test_pipeline_main_vk.cpp` are switched to `AppTraits<...VkAppData>` + the ref/out
diff pattern (mirrors `apps/tree/cuda/test_pipeline_main_cu.cu`):

```bash
scripts/run-on-jetson.sh test-tree-vk test-pipeline-e2e-vk
scripts/run-on-android.sh 3A021JEHN02756 test-tree-vk test-pipeline-e2e-vk
scripts/run-on-android.sh R5CY21Y3VEV test-tree-vk test-pipeline-e2e-vk
```

Expected: 100% pass on all three targets (SC-004). Also re-run the existing
`VkAppData_Safe`-based suites on the same targets as a zero-regression check (they
should be unaffected — additive change only, per Constitution Principle II).

## Full local regression check (before calling this feature done)

```bash
cmake --build --preset pc && ctest --test-dir build/pc -L omp --output-on-failure
just fmt   # then revert any .specify/ collateral: git checkout -- .specify/
```
