# Implementation Plan: Remove Demo Runner Apps

**Branch**: `004-remove-demo-apps` | **Date**: 2026-07-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/004-remove-demo-apps/spec.md`

## Summary

Delete the five ad-hoc `run-*` demo/smoke binaries (`run-tree-cu`, `run-tree-vk`,
`run-cifar-dense-omp`, `run-cifar-dense-cu`, `run-cifar-sparse-cu`) and their source
files, deregister them from each app's `CMakeLists.txt`, and remove the two CMake
build-helper macros (`bt_add_omp_run`, `bt_add_vk_run`) that become fully orphaned once
their only two callers are gone. Update the one piece of current how-to documentation
that instructs a reader to build/run a removed target
(`specs/001-octomap-real-workload/quickstart.md`); leave dated historical reports and
that same feature's own past-tense task record untouched.

Direct investigation (grep across `.github/`, `scripts/`, `docs/`, `specs/`,
`optimizer/`) confirms zero live tooling depends on any of the five targets — this is a
pure deletion with no logic to port, no behavior to preserve.

## Technical Context

**Language/Version**: C++20 / CMake (repo-wide, unchanged by this feature)

**Primary Dependencies**: none added or removed — this feature only deletes files and
CMake registrations

**Storage**: N/A

**Testing**: no new tests; the existing `ctest -L omp` gate (and backend-specific gates)
serve as the regression check that removal didn't break anything else, per FR-005/SC-002

**Target Platform**: build-system change, applies identically to every preset
(`pc`/`jetson`/`vulkan`/`android`) since each demo target was guarded by
`BT_ENABLE_CUDA`/`BT_ENABLE_VULKAN` for the backends that don't apply

**Project Type**: single existing C++/CMake project — pure removal, no new structure

**Performance Goals**: N/A (removal reduces build surface/time marginally; not a
performance feature)

**Constraints**: MUST NOT touch any `test-*` target, any `bm-*` benchmark (app-level or
`profiler/`-tier), or anything under `profiler/` (FR-003) — confirmed via direct
`bt_add_omp_run`/`bt_add_vk_run` caller search that the two helper macros being removed
have no other callers besides the five targets in scope

**Scale/Scope**: 5 CMake target registrations across 3 files, 5 source files, 2 CMake
helper-macro definitions, 1 doc file with a forward-looking mention needing an edit

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Simplicity First)**: this feature *is* a simplicity/dead-code-removal
  change — no new abstraction introduced, strictly deletion.
- **Principle II (Surgical, Traceable Changes)**: every file touched is touched because
  it registers, defines, or documents one of the five named targets — confirmed by
  direct grep before writing this plan, not assumed. Historical docs (dated reports, the
  same feature's own past-tense task record) are explicitly left alone per the spec's
  User Story 3 — this is the "leave pre-existing content alone unless it's what the
  request is about" instinct applied precisely.
- **Principle III (OMP-as-Oracle Differential Testing, NON-NEGOTIABLE)**: not implicated
  — no `test-*` target, kernel, or dispatch path is touched.
- **Principle IV (Goal-Driven Verification)**: the goal is verifiable and binary — after
  removal, `cmake --build` succeeds with none of the five targets in the target list, and
  `ctest -L omp` (plus the CUDA/Vulkan gates, since backend-guarded source files are also
  being deleted) stays exactly as green as before.
- No violations — Complexity Tracking intentionally omitted.

## Project Structure

### Documentation (this feature)

```text
specs/004-remove-demo-apps/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks, not this command)
```

No `contracts/` — this feature has no external interface (build-system/file deletion
only), matching the plan template's explicit skip condition for purely internal changes.
No `quickstart.md` beyond what's folded into `data-model.md`'s verification section,
since the "how to validate" content is a single short command sequence, not a scenario
guide.

### Source Code (repository root)

Existing single-project C++/CMake layout; no new directories. Files this feature
deletes or edits:

```text
apps/tree/cuda/main.cu                        # DELETE (run-tree-cu source)
apps/tree/vulkan/run_main.cpp                 # DELETE (run-tree-vk source)
apps/tree/CMakeLists.txt                      # EDIT: remove the "---- runners ----" block (lines 27-33)

apps/cifar-dense/omp/main.cpp                 # DELETE (run-cifar-dense-omp source)
apps/cifar-dense/cuda/main.cu                 # DELETE (run-cifar-dense-cu source)
apps/cifar-dense/CMakeLists.txt               # EDIT: remove the "---- runners ----" block (lines 26-30)

apps/cifar-sparse/cuda/main.cu                # DELETE (run-cifar-sparse-cu source)
apps/cifar-sparse/CMakeLists.txt              # EDIT: remove the "---- runners ----" block (lines 23-26)

cmake/bt_targets.cmake                        # EDIT: remove bt_add_omp_run (lines 9-12) and
                                               #       bt_add_vk_run (lines 34-37) -- confirmed
                                               #       zero other callers repo-wide

specs/001-octomap-real-workload/quickstart.md # EDIT: lines 89-91 currently suggest
                                               #       "run-tree-cu for a smoke run" and
                                               #       describe it as "the smoke runner" --
                                               #       update to stop referencing a target
                                               #       that will no longer exist
```

**Structure Decision**: pure deletion + two small doc/CMake edits in the existing
structure — no new files, no new directories, no behavior added anywhere.

## Complexity Tracking

*No violations — table intentionally omitted (see Constitution Check above).*
