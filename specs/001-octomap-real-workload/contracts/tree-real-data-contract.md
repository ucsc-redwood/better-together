# Contract: Tree App Real-Data Input

This isn't a network API — the "interface" this feature exposes is (a) two environment
variables the tree app's binaries read at start-up, (b) an on-disk file format contract
those variables point to, and (c) two CLI scripts that produce/deploy that file. All
three are documented here as the stable surface other tooling (fleet scripts, the
profiler, future tasks) should rely on.

## 1. Environment variable contract

| Variable | Required | Default | Effect |
|---|---|---|---|
| `BT_TREE_DATA_DIR` | No | unset (synthetic mode) | If set, `tree::AppData`'s constructor loads `$BT_TREE_DATA_DIR/points.npy` instead of generating synthetic uniform-random points. |
| `BT_TREE_INPUT_SIZE` | No | `500,000` (`kRealDataDefaultInputSize`) when `BT_TREE_DATA_DIR` is set; ignored otherwise | Number of points (`n_input`) to load as a prefix of `points.npy`. Must be `<=` the file's row count or construction throws. |

Consistent with the existing `BT_WEIGHTS_DIR` contract used by `apps/cifar-dense` /
`apps/cifar-sparse` — same naming shape, same fail-loud-on-problem behavior, same
"unset means today's synthetic behavior, byte-for-byte" guarantee.

**Why 500,000, not bigger.** `profiler/tree-{cu,vk}`'s pooled harness (`bm_prof`,
`kPoolSize=32`) allocates 32 `SafeAppData` instances simultaneously — at a measured
~132 bytes/point/instance, that's ~4.2KB/point pooled. 500,000 keeps that under ~2.1GB,
safe on the most constrained fleet target (Jetson, ~7.4GB total RAM) with zero
per-device configuration. This is a memory-safety floor, not a workload target — operators
on capable hosts should raise `BT_TREE_INPUT_SIZE` explicitly. See the per-device table
in `docs/instruction-for-ai/05-profiling.md`. Single-instance tools (`bm-tree-omp`,
`test-tree-cu`/`-vk`) aren't subject to the ×32 multiplier and can safely use much larger
values (even the full corpus) on any target.

## 2. File format contract — `points.npy`

| Property | Required value |
|---|---|
| Format | NumPy `.npy`, version 1 or 2 |
| dtype | `<f4` (little-endian float32) |
| Shape | `(N, 3)`, C-order (`fortran_order: False`) |
| `N` | `12,154,589` for the canonical corpus (the full Freiburg Campus 360 3D set, no truncation — smaller device-specific corpora are valid too, as long as `N >= BT_TREE_INPUT_SIZE` for that target) |
| Value range | Each coordinate already recentered/scaled into `[kMinCoord, kMinCoord + kRange)` = `[0.0, 1024.0)` (see `apps/tree/tree_appdata.hpp`) — the loader performs no transform |
| Row order | Fixed: ascending `scan_NNN_points.dat` numeric order, concatenated. A read of the first `n_input` rows MUST be identical every time, and MUST be a strict prefix of any larger `n_input` read from the same file |

Loaded via `bt::npy::load_prefix(path, "<f4", {n_input, 3}, dst)`
(`platform/util/npy_loader.hpp`). Failure modes (all throw `std::runtime_error` naming
the path and reason — no silent fallback):

- File missing / not a valid `.npy` file
- Wrong dtype (anything other than `<f4`)
- `fortran_order: True` (rejected — C-order only)
- Fewer than `n_input` rows, or trailing shape isn't `(3,)`

## 3. CLI contract

### `scripts/data_prep/oct.py` (extended)

Builds `points.npy` from the Freiburg Campus 360 3D scan files. New behavior on top of
today's per-scan `.npy` export:

```
uv run scripts/data_prep/oct.py \
  --base_dir resources/octomap/freiburgCampus360_3D \
  --output_dir resources/octomap/data \
  --scan_range 1-77 \
  --recenter --domain_min 0.0 --domain_range 1024.0 \
  --save
```

Writes `resources/octomap/data/points.npy` (`<f4`, `(N, 3)`). Default `--concat_target`
is `None` — **no truncation**, every point in `--scan_range` is kept (12,154,589 for the
full 77-scan set). Pass `--concat_target N` to truncate to a smaller on-disk file if
desired; the on-disk size doesn't drive run-time memory use either way — that's what
`BT_TREE_INPUT_SIZE` is for. Scans are consumed in ascending numeric order regardless of
`--scan_range` ordering syntax.

### `scripts/deploy-tree-data.sh` (new, mirrors `scripts/deploy-weights.sh`)

```
scripts/deploy-tree-data.sh jetson
scripts/deploy-tree-data.sh rocky
scripts/deploy-tree-data.sh android <serial>
```

Pushes `$BT_TREE_DATA_SRC/points.npy` (default `BT_TREE_DATA_SRC=resources/octomap/data`)
to:
- `jetson`/`rocky`: `$HOST:/tmp/bt/tree-data/` via `ssh $HOST bash -s` + `scp`
- `android`: `/data/local/tmp/bt/tree-data/` via `adb push` (stdin redirected from
  `/dev/null` on every `adb` call, per this repo's fish/adb gotchas)

### `scripts/run-on-{jetson,rocky,android}.sh` (one-line addition each)

```bash
[ -d /tmp/bt/tree-data ] && export BT_TREE_DATA_DIR=/tmp/bt/tree-data
```

placed alongside the existing `BT_WEIGHTS_DIR` auto-export line — deployed data is
picked up automatically; without a deploy, the tree app keeps its synthetic seeded input.
