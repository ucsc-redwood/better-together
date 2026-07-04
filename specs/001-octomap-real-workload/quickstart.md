# Quickstart: Validate the Real Octomap Workload

Proves the feature end-to-end: real data loads, scales up per-stage work, stays
deterministic, and leaves today's fast synthetic tests untouched. See
[`contracts/tree-real-data-contract.md`](contracts/tree-real-data-contract.md) for the
exact env var / file / CLI contracts referenced below, and
[`data-model.md`](data-model.md) for entity details.

## Prerequisites

- `cmake --preset pc` already configured/built (`docs/instruction-for-ai/02-building.md`)
- The Freiburg Campus 360 3D scan files available under
  `resources/octomap/freiburgCampus360_3D/` (fetched separately — out of scope for this
  spec's functional requirements per its Assumptions)
- Python env with `numpy` (already a project dependency, `uv run` picks it up)

## 1. Build the real corpus

```bash
uv run scripts/data_prep/oct.py \
  --base_dir resources/octomap/freiburgCampus360_3D \
  --output_dir resources/octomap/data \
  --scan_range 1-77 \
  --recenter --domain_min 0.0 --domain_range 1024.0 \
  --save
```

Expected: `resources/octomap/data/points.npy` exists, shape `(12154589, 3)` — the full
77-scan set, no truncation (default `--concat_target` is now "use everything"; the
on-disk size doesn't drive run-time memory use, `BT_TREE_INPUT_SIZE` does).

## 2. Confirm real-data mode loads, and scales up on capable hardware (User Story 1 + 2 / SC-001, SC-002)

```bash
export BT_TREE_DATA_DIR="$(pwd)/resources/octomap/data"
./build/pc/bm-tree-omp --device pc     # default n_input=500,000 (kRealDataDefaultInputSize)
```

Expected: run completes without error. Per-stage timings at the default (500,000, a
memory-safety floor — see `docs/instruction-for-ai/05-profiling.md`) may not exceed the
synthetic default, because real data has far fewer unique/BRT nodes at equal point
count. To actually see materially higher per-stage times (SC-002), raise
`BT_TREE_INPUT_SIZE` on capable hardware:

```bash
BT_TREE_DATA_DIR="$(pwd)/resources/octomap/data" BT_TREE_INPUT_SIZE=4000000 \
  ./build/pc/bm-tree-omp --device pc
```

## 3. Confirm determinism (SC-003)

```bash
BT_TREE_DATA_DIR="$(pwd)/resources/octomap/data" ./build/pc/bm-tree-omp --device pc > run1.log
BT_TREE_DATA_DIR="$(pwd)/resources/octomap/data" ./build/pc/bm-tree-omp --device pc > run2.log
```

Expected: the loaded point set (and therefore every downstream stage's structural
output — `n_unique`, `n_brt_nodes`, `n_octree_nodes`) is identical between `run1` and
`run2`.

## 4. Confirm today's tests are unaffected (User Story 3 / SC-004)

```bash
unset BT_TREE_DATA_DIR
ctest --test-dir build/pc -L omp --output-on-failure
```

Expected: green, exactly as before this feature — no test requires
`resources/octomap/data/points.npy` to exist.

## 5. Confirm the fixed-size-prefix rule (edge case)

```bash
BT_TREE_DATA_DIR="$(pwd)/resources/octomap/data" BT_TREE_INPUT_SIZE=100000 \
  ./build/pc/bm-tree-omp --device pc
```

Expected: loads successfully with 100,000 points, and those points are an exact prefix
of any larger `BT_TREE_INPUT_SIZE` run against the same corpus (e.g. the first 100,000
rows of the 4,000,000-point run from step 2).

## 6. Deploy to a fleet target (FR-010)

```bash
scripts/deploy-tree-data.sh jetson
scripts/run-on-jetson.sh test-tree-cu   # BT_TREE_DATA_DIR auto-exported, uses the 500k default
```

There is no CUDA benchmark for tree (`apps/tree/CMakeLists.txt` defines no `bm-tree-cu`);
use `test-tree-cu` for the differential correctness check. See
`docs/instruction-for-ai/01-hardware.md` for exact device names/serials and
`docs/instruction-for-ai/05-profiling.md` for the full per-backend profiler command
reference and per-device `BT_TREE_INPUT_SIZE` table. **Verified 2026-07-04** on a real
Jetson (`duck-stable`): `test-tree-cu` passed 7/7 with the real corpus deployed and
auto-exported.
