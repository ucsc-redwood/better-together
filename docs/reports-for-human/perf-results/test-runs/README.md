# Full-matrix test run logs — 2026-06-17

Real-HW runs of the per-stage differential suite **and** the runtime hetero-pipeline
suite (`test-pipeline-e2e-*`) across the whole fleet, captured while completing the
test matrix (apps × backends × HW). Every file is the verbatim stdout of the
`scripts/run-on-*.sh` / `run-mali-oracle.sh` helper; each exits non-zero on any
non-skip failure. All green.

| log | host | backend | what ran |
|---|---|---|---|
| `jetson-cuda.log` | Jetson Orin | CUDA | per-stage (tree/dense/sparse) + pipeline-e2e ×3 |
| `jetson-vulkan-perstage.log` | Jetson Orin | Vulkan | per-stage ×3 (newly run) |
| `jetson-vulkan-pipeline.log` | Jetson Orin | Vulkan | pipeline-e2e ×3 (newly cross-built) |
| `jetson-omp.log` | Jetson Orin | OMP | per-stage ×3 |
| `rocky-vulkan.log` | rocky-ryzen (RADV) | Vulkan | per-stage ×3 + pipeline-e2e ×3 |
| `mali-vulkan.log` | Pixel 7a + Samsung | Vulkan | per-stage ×3 + pipeline-e2e ×3 (incl. cifar-sparse on Mali) |
| `mali-omp.log` | Pixel 7a + Samsung | OMP | per-stage ×3 |
| `mali-omp-pipeline.log` | Pixel 7a + Samsung | OMP | pipeline-e2e incl. big\|medium\|little medium-tier case |

pc OMP (the everyday `ctest -L omp` gate, 8/8) is verified locally and not captured here.

Notes:
- The ex-`DISABLED_AlternatingBoundary` crash was diagnosed here (GPU-assisted
  validation on rocky) as a concurrent shared-command-buffer race, not octree
  re-entry — see [`../../bugs-found.md`](../../bugs-found.md) §10.
- `DISABLED` lines in the older logs are the pre-fix state; the test is now
  `PipelineE2EVk.RejectsMultiGpuChunkSchedule`.
