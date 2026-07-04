# Feature Specification: Real Octomap Workload for Tree App

**Feature Branch**: `001-octomap-real-workload`

**Created**: 2026-07-03

**Status**: Draft

**Input**: User description: "use real Octomap data sets, real workload to do the tree application. right now the input is kinda uniformly random, we need sparse input from real world datasets, also make real world data set large enough to increase more work to PUs"

## Clarifications

### Session 2026-07-03

- Q: The Freiburg Campus 360 3D dataset ships as many individual scan files (each far
  smaller than the 3-5M point target). How should the target-scale dataset be assembled?
  → A: Deterministically concatenate a fixed, ordered subset of the dataset's scan files
  until the target point count is reached (no replication/tiling; if the full dataset is
  short of target, that is the achieved scale).
- Q: How should a smaller, device-appropriate slice of the real dataset be selected for
  memory-constrained targets, so it stays deterministic and comparable across devices?
  → A: Take a fixed-size prefix (first N points, N configurable per device) of the same
  ordered corpus used for the full dataset — every smaller run is a strict prefix of
  every larger run.
- Q: Should the real-data input mode ever run inside the routine `ctest -L omp` /
  `ctest -L <backend>` correctness gates, or is it strictly a profiling/benchmarking-only
  path? → A: Profiler-only. The everyday `ctest` correctness gates keep using the
  synthetic generator exclusively; real-data mode is exercised only through BT-Profiler
  runs, kept out of scope for CI's differential test gates.

### Session 2026-07-04 (post-implementation correction)

Real-hardware validation surfaced two facts that weren't known during the original
clarification session: (1) real-world data has heavy Morton-key duplication — at equal
point count it produces ~3x *fewer* unique/BRT nodes than synthetic uniform-random data,
so raw point count alone doesn't guarantee more structural work; (2) the pooled profiler
(`profiler/tree-{cu,vk}`, `kPoolSize=32`) allocates 32 `SafeAppData` instances
simultaneously, multiplying per-instance memory by 32x — a flat large default would OOM
on memory-constrained fleet targets (Jetson, ~7.4GB RAM).

- Q: Should the on-disk corpus be truncated to a fixed target (e.g. 4M), or built from
  the full scan set? → A: Full dataset, no truncation (12,154,589 points from all 77
  scans) — the on-disk file size doesn't drive run-time memory use, `BT_TREE_INPUT_SIZE`
  does, so keeping the whole corpus maximizes flexibility at zero cost.
- Q: Given the pooled profiler's ×32 memory multiplier, how should the default vs.
  per-device `BT_TREE_INPUT_SIZE` be chosen? → A: Conservative universal default
  (500,000 — safe on the most constrained fleet target under the pooled profiler with
  zero configuration), with per-device overrides documented for capable hosts
  (`rocky-ryzen` ~2M, PC build box ~4M) to actually get more structural work than the
  synthetic baseline. See `docs/instruction-for-ai/05-profiling.md`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Profile the tree app against a realistic point cloud (Priority: P1)

A benchmark operator profiling the tree (octree-build) app wants the per-stage timings
they collect to reflect a real, spatially-sparse point cloud instead of uniformly random
synthetic points, so that scheduling decisions made from those timings generalize to
real workloads rather than an artificial best/worst case.

**Why this priority**: this is the core ask — without a real-data input source, nothing
else in this feature has value. It's the smallest slice that changes what gets measured.

**Independent Test**: run the tree app's profiler with the new real-data input mode
selected and confirm the resulting profiling records are produced the same way as today
(same schema, same per-stage breakdown), sourced from real Octomap scan points instead
of the random generator.

**Acceptance Scenarios**:

1. **Given** the tree app is configured to use the real-data input mode, **When** a run
   is profiled, **Then** every existing pipeline stage (Morton encode, sort, unique, BRT
   build, edge count/offset, octree build) executes over the real point cloud and
   produces profiling records in the existing schema, with no stage code changed.
2. **Given** the real-data input mode is selected, **When** the same configuration is run
   twice, **Then** both runs consume an identical set of points (byte-for-byte), so
   profiling results are reproducible run-to-run and across backends.

---

### User Story 2 - Scale the real workload to give PUs meaningfully more work (Priority: P2)

The same operator wants the real dataset large enough that each stage's measured
execution time increases noticeably compared to today's ~300k-point synthetic default,
so the z3 optimizer has more meaningful per-stage cost signal when assigning stages to
processing units (PUs).

**Why this priority**: realism alone (P1) doesn't address the "not enough work" problem
called out in the request; the dataset also needs to be big enough to matter.

**Independent Test**: configure the real-data input at its target scale (~10x today's
default, i.e. on the order of 3-5 million points) and confirm every pipeline stage's
measured wall-clock time increases versus a run at the current default size, with no
stage running out of memory on any fleet target it's expected to run on.

**Acceptance Scenarios**:

1. **Given** the real dataset is configured at its target scale, **When** the tree app
   runs each pipeline stage, **Then** the measured time for every stage is materially
   higher than the same stage's time at today's ~300k-point synthetic default.
2. **Given** an operator wants a different workload size, **When** they change the size
   configuration, **Then** the tree app ingests a correspondingly larger or smaller
   real-data point set without any stage code or kernel changes.

---

### User Story 3 - Keep today's fast synthetic tests working (Priority: P3)

A developer running the everyday `ctest -L omp` gate wants the existing fast,
uniformly-random synthetic input to keep working exactly as it does today, so adding a
real-data mode doesn't slow down or destabilize routine correctness testing.

**Why this priority**: protects existing test infrastructure; lower priority than P1/P2
because it's a non-regression guarantee rather than new capability, but a broken gate
blocks all other work on the repo.

**Independent Test**: run the existing tree unit/differential test suite unchanged after
the real-data mode is added, and confirm it still passes using the synthetic generator
as before, without needing the real dataset to be present.

**Acceptance Scenarios**:

1. **Given** the real-data input mode now exists, **When** the existing tree test suite
   runs with no explicit mode selection, **Then** it defaults to the current synthetic
   uniformly-random generator and passes exactly as before.
2. **Given** the real dataset is not present on a given machine, **When** the default
   (synthetic) test suite runs, **Then** it is unaffected and does not require the real
   dataset to be fetched or installed.

---

### Edge Cases

- What happens when the real dataset's coordinate range falls outside the octree
  pipeline's current coordinate domain? Points MUST be normalized/recentered into that
  domain before reaching the Morton-encode stage.
- What happens when a real scan contains duplicate or near-duplicate points (common at
  sensor origins in real scans)? The existing unique/dedup stage MUST continue to handle
  this without special-casing real data.
- What happens when the real dataset (at its configured scale) doesn't fit in memory on
  a smaller fleet target (e.g. a phone)? The chosen size MUST be selectable per target;
  a smaller target's slice MUST be a fixed-size prefix (first N points, N configurable)
  of the same ordered corpus used for the full dataset, so it stays deterministic and is
  a strict subset of every larger run rather than a differently-sampled dataset.
- What happens when the real dataset files are missing on a given machine (not yet
  fetched/provisioned)? The system MUST fail with a clear, actionable error rather than
  silently falling back to synthetic data.
- What happens when an operator selects the real-data mode but doesn't specify a size?
  The system MUST fall back to a documented default scale (see SC-002).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The tree app MUST support selecting between the existing synthetic
  uniformly-random point generator and a new real-world Octomap-derived point-cloud
  input source at run/profiling configuration time, without requiring a rebuild for each
  choice.
- **FR-002**: When real-data mode is selected, the system MUST source points from the
  Freiburg Campus 360 3D Octomap scan as the canonical real-world dataset.
- **FR-003**: The real-data input pipeline MUST normalize/recenter ingested points into
  the same coordinate domain the octree build stage currently assumes, so downstream
  Morton-encode/sort/build stages work unmodified.
- **FR-004**: The system MUST support a real-data corpus assembled by deterministically
  concatenating the full source dataset (all scan files, no truncation by default —
  12,154,589 points for the Freiburg Campus 360 3D set), so that operators can select
  however many points (via `BT_TREE_INPUT_SIZE`) they need to meaningfully exceed the
  current default synthetic size's per-stage measured workload on capable hardware.
  Raw point count alone does not guarantee this: real data's Morton-key duplication rate
  means matching or exceeding synthetic-mode's structural (BRT/octree) work requires a
  real-data point count several times larger than the synthetic default, not merely
  ~10x (see FR-011 and Assumptions).
- **FR-005**: Real-data ingestion MUST be deterministic — the same selected
  dataset/subset/size MUST produce the same point set, in the same concatenation order,
  on every run, so profiling results and any ad hoc cross-backend comparison of
  real-data runs are reproducible.
- **FR-006**: Existing tree pipeline stages (Morton encode, sort, unique, BRT build, edge
  count/offset, octree build) MUST operate on real-data input without modification to
  per-stage logic — only the point source changes.
- **FR-007**: The system MUST preserve the existing synthetic uniformly-random input
  mode as the default, so today's fast unit/differential tests are unaffected by the
  addition of real-data mode and don't require the real dataset to be present. Real-data
  mode is a profiling/benchmarking-only path: it MUST NOT be added to the routine
  `ctest -L omp`/`ctest -L <backend>` correctness gates, which continue to use the
  synthetic generator exclusively.
- **FR-008**: The real-data input's point count MUST be configurable rather than a
  single hardcoded value, so operators can scale the workload per experiment or per
  target device's memory budget.
- **FR-009**: The profiling pipeline MUST be able to profile a real-data-mode run the
  same way it profiles a synthetic-mode run today, producing profiling records
  compatible with the existing profiling schema, so the z3 optimizer can consume
  real-data-derived timings without format changes.
- **FR-010**: The system MUST make clear how the real dataset is provisioned (fetched or
  bundled) so it can be made available consistently on every fleet target expected to run
  real-data-mode profiling, and MUST fail with an actionable error if it is missing at
  run time rather than silently substituting synthetic data.
- **FR-011**: The real-data input's default point count (used when no explicit size is
  configured via FR-008) MUST be chosen conservatively enough to avoid exhausting memory
  on the most memory-constrained fleet target under the heaviest real profiling workload
  the system supports (the concurrently-pooled profiling harness), without requiring
  per-device configuration. Larger, per-device-tuned sizes remain available via explicit
  configuration for operators on more capable hardware who want more workload.

### Key Entities

- **Real-World Point Cloud Dataset**: An Octomap-derived scan corpus (Freiburg Campus
  360 3D), assembled by deterministically concatenating all of its individual scan files
  (12,154,589 points, no truncation by default), providing sparse, spatially-structured
  3D points as raw input to the tree/octree pipeline — distinguished from today's
  synthetic dataset by realistic spatial clustering/sparsity (and, consequently, a much
  higher Morton-key duplication rate) rather than uniform randomness.
- **Input Mode**: The selectable choice between "synthetic uniform-random" and "real
  Octomap dataset" that determines which point source feeds a given tree app run.
- **Point**: A single 3D coordinate (plus homogeneous w) consumed by the octree build
  pipeline; identical representation regardless of input mode.
- **Profiling Record**: The existing per-stage timing entry (schema-defined) produced
  when a tree app run, in either input mode, is profiled.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Tree app profiling runs can be executed end-to-end against a real-world
  Octomap-derived point cloud with zero changes to octree construction stage logic.
- **SC-002**: The real-world dataset file contains at least 10 million points
  (12,154,589 at full scale), letting an operator configure `BT_TREE_INPUT_SIZE` on
  capable hardware to measurably increase the wall-clock time of every existing tree
  pipeline stage relative to a run at today's ~300k synthetic default. (The safe
  zero-configuration default, chosen for memory safety on constrained fleet targets, does
  not by itself guarantee this for every stage — see FR-011 and Assumptions.)
- **SC-003**: Running the tree app twice against the same selected real-data
  configuration produces an identical point input both times (full determinism
  preserved), so profiling results and any ad hoc cross-backend comparison of real-data
  runs are reproducible.
- **SC-004**: 100% of existing synthetic-input tree unit/differential tests continue to
  pass unchanged after the real-data mode is added, and none of them are converted to
  use the real dataset.
- **SC-005**: An operator can switch the tree app between synthetic and real-data input
  modes, and between different real-data sizes, without modifying stage code or
  rebuilding backend kernels.

## Assumptions

- "PUs" (processing units) are the CPU/GPU targets BT-Optimizer's z3 solver assigns
  pipeline stages to across OMP/CUDA/Vulkan; "increase work to PUs" means increasing each
  stage's measured execution cost — but for real data this does NOT follow directly from
  raw point count alone (see next bullet), only from the *unique*/structural point count.
- Real-world scan data has heavy Morton-key duplication: measured at 4,000,000 input
  points, synthetic uniform-random data yields 3,992,616 unique keys (99.8%) vs. only
  1,244,715 (31.1%) for real data — real data at a given `n_input` does roughly 3x *less*
  structural (BRT/octree) work than synthetic at the same `n_input`. Getting meaningfully
  *more* structural work than the synthetic baseline therefore requires a real-data point
  count several times larger than a naive "~10x" would suggest, not just more scans.
- The profiling harness that actually feeds BT-Optimizer (`profiler/tree-{cu,vk}`'s
  `bm_prof`) pools 32 `SafeAppData` instances simultaneously (`kPoolSize=32`), multiplying
  per-instance memory ~32x. A flat large default point count would exhaust memory on
  constrained fleet targets (Jetson, ~7.4GB total RAM) under that harness specifically —
  this is why FR-011 requires a conservative, safe-everywhere default (500,000) separate
  from the larger, per-device-configured sizes needed to actually exceed the synthetic
  baseline's structural work (see `docs/instruction-for-ai/05-profiling.md`'s per-device
  table). Single-instance tools (`bm-tree-omp`, `test-tree-cu`/`-vk`) aren't subject to
  this multiplier.
- The canonical real dataset is the Freiburg Campus 360 3D Octomap scan, fetched from its
  public source (`ais.informatik.uni-freiburg.de`, CC-BY 3.0) and stored, untruncated
  (12,154,589 points), under `resources/octomap/` (gitignored, not committed — regenerable
  via `scripts/data_prep/oct.py`, same convention as `saved_params/export/`).
- The real dataset is added as a selectable alternative input source, not a wholesale
  replacement — the existing synthetic uniform-random generator stays available and
  remains the default for today's fast correctness tests.
- Normalizing/recentering real scan coordinates into the existing octree coordinate
  domain is a data-preparation concern, not a change to octree algorithm stages.
- Provisioning the real dataset consistently across the fleet (PC, Jetsons, rocky-ryzen,
  phones) is handled by `scripts/deploy-tree-data.sh` and the `run-on-*.sh` auto-export
  wiring, verified end-to-end on a real Jetson (`duck-stable`).
