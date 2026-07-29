# Phase 0 Research: Real Octomap Workload for Tree App

All Technical Context items were resolved directly from existing repo precedent — no
open NEEDS CLARIFICATION remain. Each decision below reuses a pattern already proven
elsewhere in BetterTogether (Principle I, Simplicity First: no new abstraction where an
existing one already fits).

## 1. How real vs. synthetic input is selected

**Decision**: An environment variable, `BT_TREE_DATA_DIR`, toggles real-data mode —
identical mechanism to the existing `BT_WEIGHTS_DIR` used by `apps/cifar-dense/appdata.hpp`
and `apps/cifar-sparse/appdata.hpp`. Set → load the real corpus (fail loud on any
problem). Unset → today's synthetic `mt19937(114514)` uniform-random generator, byte-for-byte
unchanged.

**Rationale**: this exact env-var-toggle-with-fail-loud-fallback shape is already
reviewed, tested, and deployed in this codebase for the same class of problem (a large
real dataset that shouldn't block hermetic tests when absent).

**Alternatives considered**:
- CLI flag — rejected: the profiler/bench/test mains across three backends don't
  currently parse app-specific flags; would require new arg-parsing plumbing in six+
  entry points for no benefit over an env var.
- New JSON config file — rejected: no existing convention for a runtime toggle like
  this; would be a second config mechanism alongside `devices/*.json`.

## 2. How real data is loaded into `AppData`

**Decision**: Reuse `bt::npy::load_prefix` (`platform/util/npy_loader.hpp`), the same
loader `cifar-dense`'s `appdata.hpp` uses for its real test batch. It already implements
exactly the semantics the spec's clarification settled on: load the first N rows of a
larger on-disk array, throwing `std::runtime_error` if the file has fewer than N rows.

**Rationale**: this *is* the "fixed-size prefix of the same ordered corpus" rule from
the clarification session — no new loading code needed, just a new call site.

**Alternatives considered**:
- Parse `.dat` scan files directly in C++ at run time — rejected: reinvents a
  general point-cloud loader in C++ when the project already has a slim, dependency-free
  `.npy` reader; also couples runtime performance to a plain-text parse of a
  potentially multi-hundred-MB file on every process start.

## 3. Where scan concatenation & coordinate recentering happen

**Decision**: Entirely in Python, extending the existing (currently unwired)
`scripts/data_prep/oct.py`. The script concatenates a fixed, ascending-scan-number
subset of `scan_NNN_points.dat` files' point columns, recenters/scales them into
`[kMinCoord, kMinCoord + kRange)` (matching `apps/tree/tree_appdata.hpp`'s existing
domain constants), and writes one `points.npy` (`<f4`, shape `(N, 3)`, C-order).

**Rationale**: keeps octree stage kernels and `AppData`'s constructor untouched beyond
the load branch (FR-006); mirrors how real CIFAR weights are pre-exported once
(`saved_params/export/`, see `docs/instruction-for-ai/04-alexnet-cifar-spec.md`) rather
than computed at run time.

**Alternatives considered**:
- Concatenate/recenter in C++ at first run — rejected: duplicates numpy stitching logic,
  and re-running it per-process/per-device risks non-reproducible ordering; adds cost to
  every profiling run for something that only needs to happen once.

## 4. Default corpus scale & ordering

**Decision**: Default target scale is exactly **4,000,000 points** (inside the agreed
3-5M range), assembled by concatenating `scan_NNN_points.dat` files in ascending numeric
order until the running point count reaches or exceeds 4,000,000, then truncating to
exactly that count.

**Rationale**: a single concrete number is needed for SC-002/SC-003 to be testable;
ascending scan-number order is the simplest fixed, arbitrary-but-deterministic order and
needs no extra metadata beyond the filenames the dataset already ships with.

**Alternatives considered**:
- Order by file size or spatial locality — rejected: no benefit over plain numeric
  order, since the ordering only needs to be *fixed*, not *meaningful*.

## 5. Per-device / configurable size

**Decision**: Reuse `AppData`'s existing `n_input` constructor parameter — it already
does exactly this job. `HostTreeManager::initialize()` (and any other call site building
the singleton for profiling) reads an optional `BT_TREE_INPUT_SIZE` env var to choose
`n_input` when `BT_TREE_DATA_DIR` is set (falling back to the 4,000,000 real-data
default if unset); synthetic mode is entirely unaffected and keeps using
`kDefaultInputSize`.

**Rationale**: satisfies FR-008 (configurable point count) with zero new
constructor/API surface. `bt::npy::load_prefix` already enforces "file must have >= N
rows, else throw" — exactly the per-device-slice contract from the clarification session
(fixed-size prefix of the same ordered corpus).

**Alternatives considered**:
- One hardcoded real-data size, no override — rejected: violates FR-008 and can't give a
  memory-constrained phone a smaller prefix per the clarified edge case.

## 6. Provisioning across the fleet

**Decision**: Mirror `scripts/deploy-weights.sh` exactly: a new
`scripts/deploy-tree-data.sh` pushes `points.npy` to `/tmp/bt/tree-data/` on
jetson/rocky/android targets (same ssh/adb branches, same fish-proof `bash -s`
pattern). `scripts/run-on-{jetson,rocky,android}.sh` each gain one line auto-exporting
`BT_TREE_DATA_DIR` when that directory exists — identical in shape to their existing
`BT_WEIGHTS_DIR` line.

**Rationale**: this is a proven, already-reviewed pattern in this exact repo for the
same class of problem (a large real dataset, toggled by env var, deployed on demand).
Reusing it is both Simplicity First and a Surgical Change (one line per script).

**Alternatives considered**:
- Git-LFS or a submodule bundling the dataset in-repo — rejected: goes against the
  existing convention of keeping large derived/trained data out-of-repo
  (`saved_params/export/` is `.gitignore`'d and deployed on demand, not committed).

## 7. Profiling schema / optimizer impact

**Decision**: No changes to `schemas/profiling-table.schema.json` or the z3
optimizer/loader. The schema records per-stage timings, not input-data provenance, and
(per the clarification session) real-data mode never runs inside `ctest` gates, so
nothing about the schema's consumers needs to change.

**Rationale**: keeps blast radius at zero for `optimizer/` and `schemas/`.

**Alternatives considered**:
- Add a `dataset` field to the profiling record — rejected for this feature: no current
  consumer needs it; can be revisited later if the optimizer ever needs to distinguish
  profiling runs by input dataset.

## 8. Post-implementation correction (2026-07-04): default sizing

Real-hardware validation (deploying to and profiling on an actual Jetson, `duck-stable`)
surfaced two facts not known when decisions #4 and #5 above were made:

1. **Real data's Morton-key duplication rate.** Measured at equal `n_input=4,000,000`:
   synthetic uniform-random data yields 3,992,616 unique keys (99.8%); real data yields
   only 1,244,715 (31.1%). Since stages 4-7 (BRT build, edge count/offset, octree build)
   scale with the *unique* count, real data at a given `n_input` does ~3x *less*
   structural work than synthetic at the same `n_input` — the original "~10x raw points"
   framing (decision #4) doesn't translate into "~10x more work" the way it was assumed
   to.
2. **The pooled profiler's memory multiplier.** `profiler/tree-{cu,vk}`'s `bm_prof`
   (the harness that actually feeds BT-Optimizer, distinct from the single-instance
   `bm-tree-omp`/`test-tree-cu` tools used for local dev/differential testing) allocates
   `kPoolSize=32` `SafeAppData` instances **simultaneously** via `runtime/pipeline.hpp`'s
   `make_dataset`. At a measured ~132 bytes/point/instance, that's ~4.2KB/point pooled —
   the original 4,000,000-point default would need **~15.7GB** under this harness, which
   would OOM on the Jetson (7.4GB total RAM). This wasn't caught earlier because neither
   `bm-tree-omp` nor `test-tree-cu` (both single-instance) exercise the pooled path.

**Revised decision**: keep the on-disk corpus **untruncated** (all 77 scans,
12,154,589 points — decision #3's script now defaults `--concat_target` to "no limit"
rather than a fixed 4,000,000, since the file's size doesn't drive run-time memory use).
Change `kRealDataDefaultInputSize` (decision #5's fallback) from 4,000,000 down to
**500,000** — a memory-safety floor sized to fit the pooled harness on the most
constrained fleet target (Jetson) with zero configuration — and document per-device
`BT_TREE_INPUT_SIZE` overrides (`rocky-ryzen` ~2M, PC build box ~4M) for operators who
want to actually exceed the synthetic baseline's structural work on capable hardware.
Verified: rebuilt and re-ran `test-tree-cu` on `duck-stable` with the new default and
full corpus deployed — 7/7 `TreeDiffCuda` passed.

**Rationale**: a single flat default can't be simultaneously "big enough to matter" and
"safe on every fleet target under the heaviest real harness" — those are genuinely
different numbers (500k vs. millions). Splitting them into a safe default + documented
overrides (reusing the already-existing `BT_TREE_INPUT_SIZE` mechanism from decision #5)
resolves this without new API surface.

**Alternatives considered**:
- Keep 4,000,000 as the default, accept the pooled-profiler OOM risk on Jetson/phones —
  rejected: silently unsafe by default is worse than a conservative default with
  documented, explicit opt-in for more workload.
- Make the default device-aware (read `devices/*.json` RAM at runtime) — rejected as
  over-engineering for this feature: `devices/*.json` doesn't currently record RAM, and
  `BT_TREE_INPUT_SIZE` already gives operators a one-line manual override per target,
  consistent with how device-specific tuning already works elsewhere in this repo
  (e.g. `BT_CUDA_ARCH`, `BT_TEST_DEVICE`).

## Technical Context resolution summary

| Item | Resolution |
|---|---|
| Language/Version | C++20 (existing tree app/backends), Python 3.13 (data-prep script, matches `pyproject.toml`) |
| Primary Dependencies | `bt::npy::load_prefix` (`platform/util/npy_loader.hpp`), `numpy` (already a project dependency, used by `scripts/data_prep/`) |
| Storage | One `.npy` file on disk per deployed target (`$BT_TREE_DATA_DIR/points.npy`), not a database |
| Testing | BT-Profiler runs only (manual/quickstart validation); explicitly excluded from `ctest -L omp`/`<backend>` gates per clarification |
| Target Platform | Existing fleet: PC (OMP), Jetson `duck-stable`/`duck-naughty` (CUDA), `rocky-ryzen`/phones (Vulkan) |
| Project Type | Extension of an existing monorepo app (`apps/tree`) + its profiler/data-prep tooling — no new project |
| Performance Goals | N/A directly — the feature's purpose is to inflate the per-stage cost signal the z3 optimizer consumes, not to hit a latency target itself |
| Constraints | Must not modify octree stage kernel logic (FR-006); must stay deterministic (FR-005); must not enter `ctest` correctness gates (FR-007); pooled profiler (kPoolSize=32) memory must stay safe on the most constrained fleet target (FR-011) |
| Scale/Scope | On-disk corpus: 12,154,589 points (~146MB as `<f4` (N,3)), untruncated. Default `n_input`: 500,000 (memory-safety floor for the pooled profiler); per-device overrides up to a few million on capable hardware — see `docs/instruction-for-ai/05-profiling.md` |
