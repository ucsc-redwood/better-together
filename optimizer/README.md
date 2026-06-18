# Collection Scripts

The **canonical JSONL store** (`bm-prof-*` → `profiling_loader.py` /
`render_isolated_table.py` / `coverage.py`) is the source of truth: schema-validated,
keeps the full timing distribution, self-describing `pu`. The `02_*`/`04_*` schedule
scripts feed the paper's z3 pipeline and now read that store directly (the legacy
wide-CSV export shim was retired — Seam 1).

## Canonical JSONL profiling store

The `bm-prof-<app>-<vk|cu>` binaries (one per app×backend cell, sources in
`pipe/<app>-<vk|cu>/bm_prof.{cpp,cu}`, shared `pipe/bm_prof_common.hpp`) each
measure the GPU PU **and** every present CPU tier in isolation, emitting one
self-describing JSONL record per `(stage, pu)` with the full timing distribution
(p50/p95/p99/cv/…) + measured provenance. Sampling is a **calibrated time budget**
(`measure_calibrated`): cheap stages get many samples, slow ones few, and a cell
whose probe already exceeds `BT_PROF_ABANDON_S` is recorded once and flagged
`provenance.abandoned` — so e.g. little-core-on-dense never blows up the run.

Build (per preset), deploy + run capturing stdout to the store layout
`data/profiling/<device>/<app>/<backend>/isolated/run-NNN.jsonl`:

```bash
# device-side (handles fish login shell / adb stdin per 01-hardware.md)
LD_LIBRARY_PATH=. BT_PROF_RUN=1 ./bm-prof-cifar-dense-vk --device minipc > run-001.jsonl
# knobs: BT_PROF_BUDGET_S(0.3) _ABANDON_S(0.25) _MAX_CELL_S(2.0) _MIN_ITERS(5) _MAX_ITERS(2000)
```

### `profiling_loader.py`
Schema-validate (`schemas/profiling-table.schema.json`) + count-weighted aggregate
of one `(device, app, backend)` cell into a stage×pu table; drops throttled /
`cv > --max-cv` runs, fails loud on a cell with `< --min-runs` survivors.

```bash
uv run optimizer/smt/profiling_loader.py --device minipc --app cifar-dense --backend vulkan
```

### `render_isolated_table.py`
Pivots every collected cell into one table **per app** (rows = stage, columns =
`<device>/<pu>`), the isolated-time deliverable.

```bash
uv run optimizer/analysis/render_isolated_table.py --metric p50 --max-cv 1.0 > data/profiling/isolated-tables.txt
```

### `coverage.py`
What of the permutation table (app × device × supported-backend) is collected vs
missing. `--` = hardware lacks that backend (not a gap); `MISSING` = a real gap.

```bash
uv run optimizer/analysis/coverage.py            # 12/12 supported cells when complete
```

## Schedule pipeline (`02` → `03` → `04`)

`02_gen_schedule_merged.py` reads the canonical JSONL store directly (the paper's
_profiling table_) and runs the z3 solver to emit schedule JSONs; `03_run_schedule.py`
runs them on a device; `04_parse_schedules.py` parses the per-task logs. Full
step-by-step:
[`docs/instruction-for-ai/06-end-to-end-scheduling.md`](../../docs/instruction-for-ai/06-end-to-end-scheduling.md).

```bash
uv run optimizer/orchestrate/02_gen_schedule_merged.py --profiling_root data/profiling \
  --device 3A021JEHN02756 --app cifar-sparse --backend vk --table_type btpm \
  --minimize_mode tmax --num_solutions 30 --output_folder data/schedules_btpm
```

