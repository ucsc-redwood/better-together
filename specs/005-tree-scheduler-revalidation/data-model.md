# Data Model: Tree Scheduler Re-Validation Post-AppData Migration

No new data structures — this feature re-populates existing, already-schema'd stores.
The three conceptual entities from the spec, made concrete against real files:

## Device/Backend Combination

One of the four in-scope pairings. Concretely: a `(device, "tree", backend)` triple that
`speedup_summary.py`'s `discover_cells()` finds by scanning
`data/sched_logs/<device>_tree_<be>_<table>/` directory names.

| Combination | `fleet.json` device key | `--device` id passed to binaries | Backend token |
|---|---|---|---|
| Jetson × CUDA | `duck-stable` | `duck-stable` | `cu` |
| Jetson × Vulkan | `duck-stable` | `duck-stable` | `vk` |
| Pixel × Vulkan | `pixel` | `3A021JEHN02756` | `vk` |
| Samsung × Vulkan | `samsung` | `R5CY21Y3VEV` | `vk` |

## Best-Single-Processor Baseline

Not stored separately — computed on demand by `optimizer/smt/baselines.py:get_baseline_for_config(device, "tree", backend)`
from the freshly-collected isolated JSONL store (`data/profiling/<device>/tree/<backend_long>/isolated/*.jsonl`):

- `omp` = sum over stages of the fastest fully-measured CPU tier's per-stage isolated time
- `<backend>` = sum over stages of that backend's column
- `fastest` = `min(omp, <backend>)` — this is "the best-PU baseline" the spec refers to

## Re-Validation Report

`docs/reports-for-human/perf-results/speedup-summary-<DATE>-appdata-migration.md` — a
verbatim copy of the regenerated `data/sched_logs/speedup-summary.md`, whose row shape
(per `speedup_summary.py`'s `compute_rows`/`render_markdown`) is:

| Column | Meaning |
|---|---|
| Device | `DEVICE_LABEL` friendly name (`duck-stable`, `samsung`, `pixel`) |
| App | `tree` (plus incidental `cifar-dense`/`cifar-sparse` rows, out of this feature's scope) |
| Backend | `CUDA` or `VK` |
| Baseline | `<fastest_pu_name> <ms>` (e.g. `OMP 12.34` or `CUDA 8.90`) |
| Best | `<table_type> <ms>` — the best measured makespan and which profiling table (`btpm`/`isolated`) z3 solved on to produce the winning schedule |
| Speedup | `baseline.fastest / best` — this feature's core answer, per combination |

## Verification (folds in what would otherwise be quickstart.md)

```bash
# 0. Constitution Principle VI: confirm no competing load before starting.
ssh doremy@duck-stable 'ps aux --sort=-%cpu | head -10'
ssh rocky-ryzen 'bash -lc "ps aux --sort=-%cpu | head -10"'

# 1. Fresh, full re-collection for the three in-scope devices (tree + incidental
#    cifar-dense/cifar-sparse). Builds, profiles, schedules, runs, and summarizes.
uv run optimizer/orchestrate/00_run_fleet.py --only duck-stable,samsung,pixel --fresh

# 2. Confirm the regenerated summary has all four tree rows.
grep '| tree |' data/sched_logs/speedup-summary.md

# 3. Archive as a dated report (this feature's User Story 3 / FR-005).
cp data/sched_logs/speedup-summary.md \
   docs/reports-for-human/perf-results/speedup-summary-$(date +%Y-%m-%d)-appdata-migration.md
```

Expected: step 2 returns exactly 4 lines (duck-stable×CUDA, duck-stable×VK, pixel×VK,
samsung×VK); each states a real `Speedup` number (which may be below `1.00x` for tree,
per the report's own "Tree losses" section — not a failure condition per SC-003).
