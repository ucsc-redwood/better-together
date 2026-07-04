# Data Model: Real Octomap Workload for Tree App

This feature adds one on-disk artifact and one runtime mode selector; it does not
change any in-memory `AppData` field, type, or octree algorithm data structure (FR-006).

## Entities

### RealPointCloudCorpus (on-disk artifact)

The prepared, deployable real-world input. Produced once by the (extended)
`scripts/data_prep/oct.py`, consumed at process start by the tree app's `AppData`
constructor via `bt::npy::load_prefix`.

| Field | Type | Notes |
|---|---|---|
| path | file path | `$BT_TREE_DATA_DIR/points.npy` |
| dtype | `<f4` (little-endian float32) | required by `bt::npy::load_prefix` |
| shape | `(N, 3)`, C-order | `N = 12,154,589` by default (full dataset, no truncation) |
| values | xyz coordinates | pre-recentered/scaled into `[kMinCoord, kMinCoord + kRange)` (`apps/tree/tree_appdata.hpp`) at prep time — no runtime transform |
| ordering | fixed | ascending `scan_NNN_points.dat` numeric order, concatenated; a fixed-size prefix of this file is always a prefix of every larger corpus built the same way |
| provenance | Freiburg Campus 360 3D | source scan set consumed by `scripts/data_prep/oct.py`, columns `[3:6]` (point xyz) of each `scan_NNN_points.dat`; sensor-position columns `[0:3]` are not used |

### InputMode (derived, not stored)

Not a persisted entity — derived at `AppData` construction time from whether
`BT_TREE_DATA_DIR` is set in the process environment.

| Value | Trigger | Behavior |
|---|---|---|
| `synthetic` (default) | `BT_TREE_DATA_DIR` unset | Existing `mt19937(114514)` uniform-random generator, unchanged |
| `real` | `BT_TREE_DATA_DIR` set | Load `RealPointCloudCorpus` via `bt::npy::load_prefix`; missing/short/malformed file throws (fail loud, FR-010) |

### SizeSelector (derived, not stored)

Governs how many points `AppData` allocates/loads (`n_input`), reusing the constructor
parameter that already exists on `tree::AppData`.

| Mode | Source of `n_input` |
|---|---|
| `synthetic` | `kDefaultInputSize` (640×480 ≈ 300k), unless a call site already overrides it (unchanged) |
| `real` | `BT_TREE_INPUT_SIZE` env var if set, else `kRealDataDefaultInputSize` (500,000) |

Constraint: in `real` mode, `n_input` MUST be `<= RealPointCloudCorpus`'s `N`; enforced by
`bt::npy::load_prefix` (throws otherwise) — this is the "fixed-size prefix, deterministic,
strict-subset-of-larger-runs" rule from the clarification session.

`kRealDataDefaultInputSize` (500,000) is a **memory-safety floor**, not a performance
target: `profiler/tree-{cu,vk}`'s pooled harness (`kPoolSize=32`) allocates 32
`SafeAppData` instances simultaneously, so per-instance memory (~132 bytes/point) becomes
~4.2KB/point pooled — 500,000 keeps that under ~2.1GB, safe on the most constrained fleet
target (Jetson, ~7.4GB total RAM) with no configuration. Real data's Morton-key
duplication rate (~31% unique at 4M, vs. ~99.8% for synthetic at the same `n_input`)
means this default alone may not exceed the synthetic baseline's *structural* work;
operators on capable hardware (`rocky-ryzen`, PC) should raise `BT_TREE_INPUT_SIZE`
explicitly — see the per-device table in `docs/instruction-for-ai/05-profiling.md`.
Single-instance tools (`bm-tree-omp`, `test-tree-cu`/`-vk`) aren't subject to the ×32
multiplier.

### Point (unchanged)

`glm::vec4` (x, y, z, w=1.0f) — identical representation and meaning in both input
modes; consumed unmodified by every existing pipeline stage (Morton encode → sort →
unique → BRT build → edge count/offset → octree build).

### ScanFile (prep-time only, never reaches C++ runtime)

`scan_NNN_points.dat` — six whitespace-separated columns per line: sensor xyz, point
xyz. Only exists as an input to `scripts/data_prep/oct.py`; the tree app never parses
this format directly.

## Relationships

```
scan_001_points.dat, scan_002_points.dat, ... (Freiburg Campus 360 3D)
        │  scripts/data_prep/oct.py: concat (ascending order) + recenter
        ▼
   points.npy  (RealPointCloudCorpus, <f4, (N,3), N = 12,154,589)
        │  bt::npy::load_prefix(path, "<f4", {n_input, 3}, ...)
        │  (n_input from BT_TREE_INPUT_SIZE or real-data default)
        ▼
   AppData::u_input_points_s0  (glm::vec4[n_input], w=1.0f)
        │  (unchanged: Morton encode → sort → unique → BRT → edge count/offset → octree)
        ▼
   existing tree pipeline stages, unmodified
```

## State transitions

None — input selection and size are resolved once, at `AppData` construction
(process start); no runtime transitions between modes within a single run.

## Validation rules (from Functional Requirements)

- FR-003 / normalization: enforced entirely in the Python prep step; the C++ loader
  performs no coordinate transform.
- FR-005 / determinism: `points.npy`'s row order is fixed at prep time; `load_prefix`
  reading the same file with the same `n_input` always yields the same points.
- FR-007 / test isolation: no `ctest`-registered test constructs `AppData` with
  `BT_TREE_DATA_DIR` set; the synthetic path remains the only path exercised by
  `ctest -L omp`/`<backend>`.
- FR-010 / fail loud: any missing file, wrong dtype/shape, or `n_input` exceeding the
  file's row count throws `std::runtime_error` naming the file and reason (via
  `bt::npy::detail::fail`) — never a silent fallback to synthetic data.
