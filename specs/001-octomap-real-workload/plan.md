# Implementation Plan: Real Octomap Workload for Tree App

**Branch**: `001-octomap-real-workload` | **Date**: 2026-07-03 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-octomap-real-workload/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Give the tree (octree-build) app a real, sparse, large-scale point-cloud input —
sourced from the Freiburg Campus 360 3D Octomap scan set — as a selectable alternative
to today's uniformly-random synthetic generator, so BT-Profiler's per-stage timings (and
therefore the z3 optimizer's PU assignment signal) reflect a realistic, meaningfully
larger workload (~4M points vs. today's ~300k).

Technical approach: reuse the exact pattern already proven by `apps/cifar-dense` /
`apps/cifar-sparse` for the same class of problem — an environment variable
(`BT_TREE_DATA_DIR`) toggles between the existing synthetic generator and loading a
pre-built `.npy` corpus via the existing `bt::npy::load_prefix` loader, deployed to the
fleet with a new script mirroring `scripts/deploy-weights.sh`. No octree stage kernel is
touched; the change is confined to `AppData`'s construction path, a Python data-prep
script, and fleet deployment scripts. Real-data mode is profiling-only and is
deliberately excluded from the `ctest -L omp`/`<backend>` correctness gates (per
clarification), so it carries zero CI risk.

## Technical Context

**Language/Version**: C++20 (existing tree app + OMP/CUDA/Vulkan backends), Python 3.13
(data-prep tooling, matches `pyproject.toml`)

**Primary Dependencies**: `bt::npy::load_prefix` (`platform/util/npy_loader.hpp`,
already used by `apps/cifar-dense`); `numpy` (already a project dependency used by
`scripts/data_prep/`)

**Storage**: One `.npy` file per deployed target (`$BT_TREE_DATA_DIR/points.npy`) — not
a database; mirrors `saved_params/export/` + `BT_WEIGHTS_DIR` convention

**Testing**: Manual/quickstart validation via BT-Profiler runs (see `quickstart.md`);
explicitly NOT added to `ctest -L omp`/`<backend>` gates — those continue to exercise
only the synthetic generator, unchanged

**Target Platform**: Existing fleet — PC (OMP, local), Jetson `duck-stable`/`duck-naughty`
(CUDA cross-compiled), `rocky-ryzen`/phones (Vulkan)

**Project Type**: Extension of an existing monorepo app (`apps/tree`) plus its
data-prep/deployment tooling — no new project or service

**Performance Goals**: N/A as a target in itself — the feature's purpose is to increase
the *measured* per-stage cost so the z3 optimizer has more signal (SC-002), not to hit a
latency budget

**Constraints**: Must not modify octree stage kernel logic (FR-006); must remain
deterministic (FR-005); must not enter `ctest` correctness gates (FR-007); per-device
loads must be a strict, fixed-size prefix of the full corpus (clarification)

**Scale/Scope**: On-disk corpus = 12,154,589 points, full Freiburg Campus 360 3D set,
untruncated (~146MB as `<f4` `(N,3)`), assembled by deterministically concatenating
`scan_NNN_points.dat` files in ascending order. Default loaded `n_input` = 500,000 (a
memory-safety floor for the pooled profiler, `kPoolSize=32`); `BT_TREE_INPUT_SIZE`
selects a larger prefix on capable hardware (see `docs/instruction-for-ai/05-profiling.md`'s
per-device table) — amended 2026-07-04 after real-hardware validation, see
`research.md` §8

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Assessment |
|---|---|
| I. Simplicity First | **Pass.** No new abstraction: reuses the existing `BT_WEIGHTS_DIR`-style env var pattern and the existing `bt::npy::load_prefix` loader verbatim. |
| II. Surgical, Traceable Changes | **Pass.** Touches only `AppData`'s construction path, `scripts/data_prep/oct.py`, one new deploy script, and one line each in three `run-on-*.sh` scripts. No stage kernel or existing test is modified. |
| III. OMP-as-Oracle Differential Testing (NON-NEGOTIABLE) | **Pass / not triggered.** No kernel changes are made (FR-006), so there is nothing new to differentially verify. Real-data mode is explicitly excluded from `ctest` gates per the clarification session, and existing `ctest -L omp`/`<backend>` gates are unaffected (SC-004). |
| IV. Goal-Driven Verification | **Pass.** `quickstart.md` defines a runnable end-to-end validation (load, scale-up, determinism, non-regression, fleet deploy) rather than relying on "it builds." |
| V. Data & Docs as Source of Truth | **Pass.** The new env vars, file contract, and CLI scripts are documented in `contracts/tree-real-data-contract.md`; `docs/instruction-for-ai/05-profiling.md` will be updated during implementation to reference the new real-data mode (task, not this plan). |

No violations. **Complexity Tracking is empty — no entries required.**

*Post-design re-check (after Phase 1): unchanged — the data model and contracts above
don't introduce anything beyond what was assessed here; all five principles still pass.*

## Project Structure

### Documentation (this feature)

```text
specs/001-octomap-real-workload/
├── plan.md                          # This file
├── research.md                      # Phase 0 output
├── data-model.md                    # Phase 1 output
├── quickstart.md                    # Phase 1 output
├── contracts/
│   └── tree-real-data-contract.md   # Phase 1 output
└── tasks.md                         # Phase 2 output (/speckit-tasks — not created here)
```

### Source code (repository root)

This is an extension of an existing monorepo app, not a new project — real paths this
feature touches (or adds), with no unused structural options to strip:

```text
apps/tree/
├── tree_appdata.hpp          # doc comment only: document BT_TREE_DATA_DIR / real-data default size
├── tree_appdata.cpp          # AppData ctor: branch on BT_TREE_DATA_DIR — bt::npy::load_prefix(...) vs existing mt19937 synthetic generator
└── safe_tree_appdata.cpp     # HostTreeManager::initialize(): read BT_TREE_INPUT_SIZE (real-data mode only) to choose n_input; unchanged otherwise

scripts/data_prep/
└── oct.py                    # extend: --concat_target / --recenter / --domain_* flags; write points.npy

scripts/
├── deploy-tree-data.sh       # new — mirrors deploy-weights.sh (jetson/rocky ssh+scp, android adb push)
├── run-on-jetson.sh          # +1 line: auto-export BT_TREE_DATA_DIR if /tmp/bt/tree-data exists
├── run-on-rocky.sh           # +1 line, same pattern
└── run-on-android.sh         # +1 line, same pattern

docs/instruction-for-ai/
└── 05-profiling.md           # document the real-data mode alongside existing profiling how-to
```

No changes to: octree stage kernels (`apps/tree/{omp,cuda,vulkan}/*`), `SafeAppData`'s
struct layout, `schemas/profiling-table.schema.json`, `optimizer/` (per research.md #7).

**Structure Decision**: extend the existing `apps/tree` + `scripts/data_prep` +
fleet-deployment layout in place; no new top-level directory, package, or service is
introduced. This matches the feature's actual shape — a new input-data path for an
existing app, not new architecture.

## Complexity Tracking

*No entries — Constitution Check reported no violations.*
