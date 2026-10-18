# Tasks

Gates used below:
- **"local":** `ctest --test-dir build/pc -L omp`, plus `test-tree-vk` and `test-pipeline-e2e-vk` on the
  build box's integrated GPU.
- **"rocky correctness":** a native build on rocky-ryzen (`ssh doremy@100.97.77.80 bash -s`, glslc from
  `~/android-ndk-r29/shader-tools/linux-x86_64/glslc`), then `test-{kiss,tree}-vk` and
  `test-pipeline-{e2e,chained}-vk --device minipc`. **Correctness only; nothing is profiled on rocky.**
- **"phone gate":** the VK suites plus `test-pipeline-e2e-vk` and `test-pipeline-chained-vk` through
  `scripts/run-on-android.sh` on 3A021JEHN02756 and R5CY21Y3VEV.

## 1. The marker and the rule

- [x] 1.1 Add `bt::kHostSparse` to `platform/util/buffer_placement.hpp`, documenting what qualifies (host reads outside the stages that are rare or a few elements per item), and make `schedule_placement` ignore it. Verify: `ExpectCpuBuffersStayBoundary`, extended with the marker (D3), passes for tree and both cifar apps under local.
  - Evidence (2026-10-18, build box): `kHostSparse = -1`; `schedule_placement` ignores it, except that a buffer
    only the marker touches (no stage) stays boundary (spec: "A buffer that no stage touches is not affected").
    The helper now treats the marker the same way. `Placement.CpuChunkBuffersStayBoundary` passes in
    `test-tree-omp`, `test-cifar-dense-omp`, `test-cifar-sparse-omp`; `ctest -L omp` 11/11; `test-tree-vk` 20/20
    and `test-pipeline-e2e-vk` 5/5 on the build box's RADV iGPU, no skips.
- [x] 1.2 Add a hardware-free test that the cifar apps get the same class for every buffer under every one-GPU-range schedule as before the change. Verify: it passes under local, and it fails when the marker is wrongly treated as a GPU stage for an undeclared accessor (a mutation).
  - Evidence (2026-10-18, build box): `bt_placement_test::ExpectPlacementUnchanged` (asserts no declaration uses
    the marker, then compares every buffer with the pre-change rule) as `Placement.UndeclaredHostAccessKeepsPlacement`
    in both cifar OMP suites: passes; same local gate as 1.1. Mutation: `schedule_placement` treating an undeclared
    host accessor (0) like the marker (`s == bt::kHostSparse || s == 0`) makes it fail in both suites (and
    `CpuChunkBuffersStayBoundary` with it); reverted.

## 2. Tree declarations and the gate

- [x] 2.1 Declare the five octree outputs `{kHostSparse, 7}`, `u_out_idx` `{kHostSparse, 3}` and `u_edge_offset_s6` `{kHostSparse, 6, 7}` (D2). Verify: `Placement.DeclaredStagesMatchBindings` and the pipeline tests pass under local, rocky correctness and the phone gate.
  - D2 re-checked against the current code first (design D2): unchanged.
  - New hardware-free tests in `test-tree-vk`: `Placement.SparseHostReadsDoNotForceBoundary` (spec "Tree on the GPU
    alone" and "A CPU chunk still reads it": all seven buffers GPU-local under GPU 1-7, the octree boundary under
    GPU 1-6) and `Placement.CpuChunkBuffersStayBoundary` on `VkAppData`. Mutation: the declarations back at `0` make
    the first fail; restored.
  - Local (2026-10-18): `ctest -L omp` 11/11; on the RADV iGPU `test-tree-vk` 22/22, `test-pipeline-e2e-vk` 5/5,
    `test-pipeline-chained-vk` 1/1, no skips.
  - Rocky correctness (native build in `~/bt-ohr-gate`; rocky's HTTPS to GitHub returned a self-signed certificate,
    so CPM got the build box's `build/.cpm-cache` via rsync): `test-kiss-vk` 30/30, `test-tree-vk` 22/22,
    `test-pipeline-e2e-vk` 5/5, `test-pipeline-chained-vk` 1/1 with `--device minipc`. Nothing was timed there.
  - Phone gate (`scripts/run-on-android.sh`, dest `/data/local/tmp/bt-ohr`), both phones exit 0: `test-kiss-vk` 29
    passed + 1 skipped (`PipelineCacheTest.FifoAtCachePathIsIgnoredWithoutBlocking`, the OS forbids creating a FIFO in that dir; unrelated to this change),
    `test-tree-vk` 22, `test-cifar-dense-vk` 16, `test-cifar-sparse-vk` 17, `test-pipeline-e2e-vk` 5,
    `test-pipeline-chained-vk` 1. `PipelineE2EVk.AllVulkan` runs stages 1-7 on the GPU, so on the Pixel the counts
    and the octree are in the uncached GPU-local class and each item's `n_octree_nodes` and octree match the oracle;
    `PipelineChainedVk` asserts `n_unique`, `n_brt_nodes` and `n_octree_nodes` against the oracle.
- [x] 2.2 Run the whole-chunk A/B (D4) on both phones and apply the spec's gate. Verify: the medians and spreads are in design.md, and the declarations stay only if the gate passed (else 2.1 is reverted and the numbers are recorded).
  - 2026-10-18, design D4 "Measured": Pixel 7a `dispatch_wall` 29.54 ms (21.30..29.58) -> 17.79 ms (17.71..19.01),
    spreads disjoint; Samsung 9.98 ms (7.22..10.37) -> 9.91 ms (8.35..10.50). Gate passed; the declarations stay.
- [x] 2.3 Document the marker in `docs/instruction-for-ai/03-unit-testing.md` next to the placement knobs, and add the result to `docs/reports-for-human/2026-10-16-vk-data-flow-fixes.md` (follow-up 1). Verify: both docs name the marker and the measured effect.
  - `03-unit-testing.md`: a `bt::kHostSparse` bullet after `BT_VK_BOUNDARY_PLACEMENT` (what qualifies, the rule, the
    tree declarations, Pixel 29.54 -> 17.79 ms, Samsung unchanged, the tests). The data-flow report's follow-up 1
    is marked done with the A/B medians and spreads of both phones.
