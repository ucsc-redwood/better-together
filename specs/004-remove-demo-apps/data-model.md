# Data Model: Remove Demo Runner Apps

No runtime data entities are involved — this is a build-system/file-removal feature. The
one conceptual entity from the spec, made concrete:

## Demo/Runner Target (the removal manifest)

| Target | Source file | CMakeLists.txt | Helper macro used |
|---|---|---|---|
| `run-tree-cu` | `apps/tree/cuda/main.cu` | `apps/tree/CMakeLists.txt:27-33` | `bt_add_cuda_app` (stays defined — used elsewhere) |
| `run-tree-vk` | `apps/tree/vulkan/run_main.cpp` | same block | `bt_add_vk_run` (removed — orphaned after this) |
| `run-cifar-dense-omp` | `apps/cifar-dense/omp/main.cpp` | `apps/cifar-dense/CMakeLists.txt:26-30` | `bt_add_omp_run` (removed — orphaned after this) |
| `run-cifar-dense-cu` | `apps/cifar-dense/cuda/main.cu` | same block | `bt_add_cuda_app` (stays defined) |
| `run-cifar-sparse-cu` | `apps/cifar-sparse/cuda/main.cu` | `apps/cifar-sparse/CMakeLists.txt:23-26` | `bt_add_cuda_app` (stays defined) |

Each row's target, source file, and CMakeLists.txt block are removed together. The
"Helper macro" column records which of `cmake/bt_targets.cmake`'s helpers each target
used — `bt_add_omp_run` and `bt_add_vk_run` are removed too (Finding 2 in research.md:
each has exactly one caller, one of the targets above); `bt_add_cuda_app` stays because
it's also used by the `bm-*` benchmarks and `profiler/`'s `run-pipe-*-cu` targets.

## Documentation touch-point

| File | Lines | Action |
|---|---|---|
| `specs/001-octomap-real-workload/quickstart.md` | 89-91 | Edit: remove the "or `run-tree-cu` for a smoke run" suggestion and the "smoke runner" description |
| `specs/001-octomap-real-workload/tasks.md` | 210 | No change (historical record) |
| `docs/reports-for-human/cmake-migration-rfc.md` | 201 | No change (dated report) |
| `docs/reports-for-human/project-evaluation-2026-06-19.md` | 99 | No change (dated report) |

## Verification (folds in what would otherwise be quickstart.md)

```bash
# 1. Confirm the targets are gone and nothing else broke, per preset:
cmake --preset pc && cmake --build --preset pc 2>&1 | tee /tmp/build-pc.log
cmake --build --preset pc --target help | grep -E "^run-(tree|cifar)-" \
  && echo "FAIL: a removed target still exists" || echo "OK: no removed targets remain"

# 2. Same check under the CUDA/Vulkan-enabled presets (backend-guarded deletions):
cmake --preset jetson && cmake --build --preset jetson --target help \
  | grep -E "^run-(tree|cifar)-" && echo "FAIL" || echo "OK"
cmake --preset vulkan && cmake --build --preset vulkan --target help \
  | grep -E "^run-(tree|cifar)-" && echo "FAIL" || echo "OK"

# 3. Regression: the routine gate must stay exactly as green as before.
ctest --test-dir build/pc -L omp --output-on-failure

# 4. Confirm the two orphaned CMake helpers are gone and nothing else references them:
grep -rn "bt_add_omp_run\|bt_add_vk_run" cmake/ apps/ profiler/ runtime/ platform/ tools/ \
  && echo "FAIL: a reference remains" || echo "OK: fully removed"
```

Expected: every "OK" branch above; `ctest` 100% pass, identical result to the pre-removal
baseline.
