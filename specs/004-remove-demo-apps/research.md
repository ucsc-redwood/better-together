# Research: Remove Demo Runner Apps

All findings below came from directly reading `apps/{tree,cifar-dense,cifar-sparse}/CMakeLists.txt`,
`cmake/bt_targets.cmake`, and grepping `.github/`, `scripts/`, `docs/`, `specs/`, and
`optimizer/` for each target name — not from assumption.

## Finding 1: exact CMake registrations to remove

| Target | Source file | CMakeLists.txt block |
|---|---|---|
| `run-tree-cu` | `apps/tree/cuda/main.cu` | `apps/tree/CMakeLists.txt:27-33` ("---- runners ----" block, shared with `run-tree-vk`) |
| `run-tree-vk` | `apps/tree/vulkan/run_main.cpp` | same block as above |
| `run-cifar-dense-omp` | `apps/cifar-dense/omp/main.cpp` | `apps/cifar-dense/CMakeLists.txt:26-30` ("---- runners ----" block, shared with `run-cifar-dense-cu`) |
| `run-cifar-dense-cu` | `apps/cifar-dense/cuda/main.cu` | same block as above |
| `run-cifar-sparse-cu` | `apps/cifar-sparse/cuda/main.cu` | `apps/cifar-sparse/CMakeLists.txt:23-26` (its own "---- runners ----" block; cifar-sparse has no OMP runner) |

Each app's `CMakeLists.txt` has its runners in one contiguous, clearly-commented block —
removing the whole block (not individual lines) is clean in all three files.

## Finding 2: two CMake helper macros become fully orphaned

**Decision**: remove `bt_add_omp_run` and `bt_add_vk_run` from `cmake/bt_targets.cmake`
(lines 9-12 and 34-37 respectively) as part of this change.

**Rationale**: grepped every caller of both macros across `apps/`, `profiler/`,
`runtime/`, `platform/`, `tools/`. Results:
- `bt_add_omp_run`: exactly one caller — `apps/cifar-dense/CMakeLists.txt:27`
  (`run-cifar-dense-omp`). No other app uses it (tree has no OMP runner; cifar-sparse has
  no OMP runner).
- `bt_add_vk_run`: exactly one caller — `apps/tree/CMakeLists.txt:32` (`run-tree-vk`). No
  other app uses it.

Once these two targets are removed, both macros have zero remaining callers. Per FR-002,
leaving them defined-but-unused would be exactly the kind of orphaned dead code this
feature exists to clean up — removing them is in scope, not a drive-by.

**Contrast**: `bt_add_cuda_app`, `bt_add_omp_app`, and `bt_add_vk_app` (the sibling
"_app" helpers) are NOT touched — they remain heavily used by the `bm-*` benchmarks
(app-level and `profiler/`-tier) and by `profiler/`'s own `run-pipe-*-cu` targets, none
of which are in scope.

## Finding 3: zero live tooling references any of the five targets

**Decision**: all five targets can be removed with no coordination needed with CI,
scripts, or the optimizer.

**Rationale**: direct grep of `.github/*.yml`, `scripts/*.sh`, `optimizer/orchestrate/`,
`optimizer/smt/`, `optimizer/analysis/` for all five target names returned zero matches.
`scripts/run-on-{jetson,rocky,android}.sh` operate on `test-*` binaries exclusively (per
their default target lists) and are unaffected. `optimizer/orchestrate/*.py` only
references `profiler/`-tier targets (`bm-prof-*`, `bm-gen-logs-*`), never the app-level
`run-*`/`bm-*` targets in scope or out of scope here.

## Finding 4: documentation references split into "must update" vs "leave alone"

**Decision**: exactly one file needs an edit — `specs/001-octomap-real-workload/quickstart.md`
lines 89-91. Three other references are historical record and must NOT be touched.

**Rationale** (each reference read in context):
- **`specs/001-octomap-real-workload/quickstart.md:89-91`** — current, forward-looking
  instructional text: *"There is no CUDA benchmark for tree (`apps/tree/CMakeLists.txt`
  only defines the smoke runner `run-tree-cu`, no `bm-tree-cu`); use `test-tree-cu` for
  the differential correctness check, or `run-tree-cu` for a smoke run."* This actively
  tells a reader they can run `run-tree-cu` — it will be false after removal. **MUST
  update**: drop the "or `run-tree-cu` for a smoke run" clause and the "smoke runner"
  description, keeping the accurate parts (no CUDA benchmark for tree; use `test-tree-cu`
  for the differential check).
- **`specs/001-octomap-real-workload/tasks.md:210`** — past-tense record: *"...only
  defines a CUDA smoke-runner `run-tree-cu`, no CUDA benchmark — built and ran that..."*
  This documents what was actually done during that feature's own verification. Per the
  spec's User Story 3 / Assumptions, past feature task records stay as accurate history.
  **Leave unchanged.**
- **`docs/reports-for-human/cmake-migration-rfc.md:201`** — a dated RFC recommending
  `run-tree-cu` as a starting point for a *hypothetical future* migration exercise. It's
  a proposal/report, not current build instructions. **Leave unchanged** (dated report).
- **`docs/reports-for-human/project-evaluation-2026-06-19.md:99`** — a dated audit
  mentioning `run-tree-cu` as an example while describing `bt_cuda`'s shared-static-lib
  structure. Historical analysis, not an instruction. **Leave unchanged** (dated report).

`docs/instruction-for-ai/02-building.md` and `03-unit-testing.md` were also checked and
contain no target-name-specific mentions of any of the five (only generic phrases like
"the runners, benchmarks, and tests") — no edit needed there.

## Finding 5: no test/CI regression risk

**Decision**: the standard `ctest -L omp` local run (plus a CUDA/Vulkan preset build
smoke to confirm the backend-guarded deletions don't break anything under
`BT_ENABLE_CUDA`/`BT_ENABLE_VULKAN`) is sufficient verification — no new test is needed
for a pure-deletion change.

**Rationale**: none of the five removed targets are `add_test`-registered (they used the
`_run`/`_app` helpers, which never call `add_test`), so `ctest` was never exercising them
directly. The verification concern is purely "does the build still succeed and do the
existing tests still pass" (FR-005/SC-002), which the existing gate already covers.
