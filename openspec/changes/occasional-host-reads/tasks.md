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

- [ ] 1.1 Add `bt::kHostSparse` to `platform/util/buffer_placement.hpp`, documenting what qualifies (host reads outside the stages that are rare or a few elements per item), and make `schedule_placement` ignore it. Verify: `ExpectCpuBuffersStayBoundary`, extended with the marker (D3), passes for tree and both cifar apps under local.
- [ ] 1.2 Add a hardware-free test that the cifar apps get the same class for every buffer under every one-GPU-range schedule as before the change. Verify: it passes under local, and it fails when the marker is wrongly treated as a GPU stage for an undeclared accessor (a mutation).

## 2. Tree declarations and the gate

- [ ] 2.1 Declare the five octree outputs `{kHostSparse, 7}`, `u_out_idx` `{kHostSparse, 3}` and `u_edge_offset_s6` `{kHostSparse, 6, 7}` (D2). Verify: `Placement.DeclaredStagesMatchBindings` and the pipeline tests pass under local, rocky correctness and the phone gate.
- [ ] 2.2 Run the whole-chunk A/B (D4) on both phones and apply the spec's gate. Verify: the medians and spreads are in design.md, and the declarations stay only if the gate passed (else 2.1 is reverted and the numbers are recorded).
- [ ] 2.3 Document the marker in `docs/instruction-for-ai/03-unit-testing.md` next to the placement knobs, and add the result to `docs/reports-for-human/2026-10-16-vk-data-flow-fixes.md` (follow-up 1). Verify: both docs name the marker and the measured effect.
