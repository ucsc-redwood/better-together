# Archive — all test-run results predating the 2026-07-01 fresh start

**Fresh-start policy (2026-07-01):** all previously collected results across the whole
fleet are archived here; every cell gets re-collected from scratch on the current
fleet. Two reasons: (a) both Jetsons were reflashed and are effectively new hardware
targets; (b) a single clean baseline date beats mixing eras.

## The Jetson reflash (why the jetson logs are doubly stale)

- **Device id then:** `jetson` — a single Jetson Orin Nano devkit on JetPack 6.x
  (L4T R36, Ubuntu 22.04, CUDA 12.6, user `yanwen`).
- **What happened:** on **2026-07-01** both Orin Nano Super devkits were reflashed to
  **JetPack 7.2** (L4T R39.2.0, kernel 6.8-tegra, Ubuntu 24.04, CUDA 13.2, MAXN_SUPER,
  user `doremy`) and re-registered as new devices **`duck-stable`** and
  **`duck-naughty`**. Performance on the new stack is expected to differ (new power
  mode, new driver, new CUDA), so the old numbers are **not comparable**.

## What's in this folder — the 2026-06-17 full-matrix sweep

Real-HW runs of the per-stage differential suite and the runtime hetero-pipeline
suite (`test-pipeline-e2e-*`); each file is the verbatim stdout of a
`scripts/run-on-*.sh` / `run-mali-oracle.sh` helper. All green at the time.

| log | host | backend | what ran |
|---|---|---|---|
| `jetson-cuda.log` | Jetson Orin (JetPack 6) | CUDA | per-stage (tree/dense/sparse) + pipeline-e2e ×3 |
| `jetson-vulkan-perstage.log` | Jetson Orin (JetPack 6) | Vulkan | per-stage ×3 |
| `jetson-vulkan-pipeline.log` | Jetson Orin (JetPack 6) | Vulkan | pipeline-e2e ×3 |
| `jetson-omp.log` | Jetson Orin (JetPack 6) | OMP | per-stage ×3 |
| `rocky-vulkan.log` | rocky-ryzen (RADV) | Vulkan | per-stage ×3 + pipeline-e2e ×3 |
| `mali-vulkan.log` | Pixel 7a + Samsung | Vulkan | per-stage ×3 + pipeline-e2e ×3 (incl. cifar-sparse on Mali) |
| `mali-omp.log` | Pixel 7a + Samsung | OMP | per-stage ×3 |
| `mali-omp-pipeline.log` | Pixel 7a + Samsung | OMP | pipeline-e2e incl. big\|medium\|little medium-tier case |

Historical notes attached to these logs:

- The ex-`DISABLED_AlternatingBoundary` crash was diagnosed here (GPU-assisted
  validation on rocky) as a concurrent shared-command-buffer race, not octree
  re-entry — see [`../../../bugs-found.md`](../../../bugs-found.md) §10.
- `DISABLED` lines in the older logs are the pre-fix state; the test is now
  `PipelineE2EVk.RejectsMultiGpuChunkSchedule`.

## Where the rest of the pre-fresh-start data lives

- **Profiling-store snapshot (JSONL):** `data/` was tracked until 2026-06-17; the last
  committed snapshot (incl. `data/profiling/jetson/**`) is recoverable at commit
  **`8d45084^`** (`git show 8d45084^ -- data/profiling` or check out that tree).
- **Post-2026-06-17 fleet-run JSONL** (untracked `data/profiling/`) lived only on the
  machine that ran `00_run_fleet.py`; archive it there if it still exists.
- **Device spec:** `devices/jetson.json` was removed with the retirement; restore it
  from git history (any pre-2026-07 `main`) if you need to render archived stores in
  the dashboard.
- **Paper-era artifacts:** the IISWC 2025 figures/scripts are archived in the
  `iiswc2025-submission` tag (not on this clone's remote list — see the repo of
  record).
