# Feature Specification: Remove Demo Runner Apps

**Feature Branch**: `004-remove-demo-apps`

**Created**: 2026-07-04

**Status**: Draft

**Input**: User description: "Refactor targets, we should remove demo apps."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The build no longer ships ad-hoc, assertion-free demo binaries (Priority: P1)

A developer building the project sees a target list where every binary has a clear
purpose — a correctness test, a genuine timing benchmark, or a production tool — instead
of also containing several `run-*` smoke/dump programs that construct a pipeline, run
some or all of its stages, print or hex-dump output, and exit without checking whether
the result is correct or how fast it was.

**Why this priority**: this is the entire point of the request — the demo binaries add
build time and target-list clutter without providing any of the value a test (pass/fail)
or a benchmark (timing data feeding a decision) provides. Removing them is the whole
scope; everything else here is about doing so safely.

**Independent Test**: after removal, list all build targets and confirm none of the five
identified demo binaries (see Key Entities) remain, and confirm the build still succeeds
end to end.

**Acceptance Scenarios**:

1. **Given** a fresh build of the project, **When** the target list is inspected, **Then**
   none of `run-tree-cu`, `run-tree-vk`, `run-cifar-dense-omp`, `run-cifar-dense-cu`, or
   `run-cifar-sparse-cu` appear.
2. **Given** the same build, **When** it completes, **Then** it succeeds with no errors
   attributable to the removal (no dangling reference to a deleted source file, no
   orphaned build-helper macro left invoked with nothing to build).

---

### User Story 2 - Removing the demo apps doesn't silently break anything that depends on them (Priority: P1)

A developer relying on the project's test gates, CI, documentation, or the optimizer's
orchestration tooling wants confidence that none of them were secretly depending on a
demo binary that's about to disappear.

**Why this priority**: equal priority to User Story 1 — deleting a target is only safe
once it's confirmed nothing else in the project actually needs it. Skipping this
verification is exactly how a "harmless cleanup" turns into a broken CI run or a
silently-stale doc discovered weeks later.

**Independent Test**: after removal, run the project's routine correctness gates and
confirm they remain 100% green, and separately confirm no CI workflow, script, or
optimizer orchestration file references any of the removed target names.

**Acceptance Scenarios**:

1. **Given** the demo binaries removed, **When** the routine local correctness gate runs,
   **Then** it passes exactly as it did before the removal (no regression).
2. **Given** the same change, **When** CI workflow files, deployment/run scripts, and the
   optimizer's orchestration tooling are checked, **Then** none of them reference a
   removed target name.

---

### User Story 3 - Existing documentation doesn't mislead readers about a deleted target (Priority: P2)

A developer reading the project's docs or a past feature's spec/task list, who sees a
mention of one of the removed demo binaries (e.g. instructions that reference building or
running it), isn't left trying to build something that no longer exists without any
indication of why.

**Why this priority**: lower priority than the removal and its safety verification
(User Stories 1-2) — this is about not leaving a confusing trail behind, not about
correctness or safety of the removal itself.

**Independent Test**: search the project's current how-to documentation for mentions of
the removed targets and confirm each mention is either updated to reflect the removal or
clearly framed as historical record (e.g. inside a dated report describing past work),
not as a current instruction.

**Acceptance Scenarios**:

1. **Given** the project's current build/testing how-to docs (as opposed to dated
   historical reports), **When** they are checked after removal, **Then** none instructs
   a reader to build or run a removed target as if it still exists.
2. **Given** a past feature's spec or task list that documents having built/run a removed
   target during its own verification, **When** it is reviewed, **Then** it is left as an
   accurate historical record of what was done at the time, not silently rewritten.

---

### Edge Cases

- What happens to `run-tree-cu`, which (unlike the other four) has a couple of incidental
  mentions in dated historical report docs and in a past feature's spec/task
  list/quickstart guide (as a smoke check that was actually built and run during that
  feature's verification)? Those mentions MUST be left as accurate historical record (User
  Story 3); only current, forward-looking how-to documentation needs updating.
- What happens to the CMake build-helper macros that exist only to register these demo
  binaries, if removing the binaries leaves a helper macro with no remaining caller? This
  MUST be identified and resolved (either removed alongside its last caller or confirmed
  still used elsewhere) rather than left silently orphaned.
- What happens to the per-app, per-stage timing benchmarks (`bm-tree-omp`, `bm-tree-vk`,
  `bm-cifar-dense-omp/-cu/-vk`, `bm-cifar-sparse-omp/-cu/-vk`)? These are explicitly OUT of
  scope — see Assumptions for why they are not "demo apps."
- What happens if removing a demo binary's source file leaves its containing directory
  otherwise empty or its app's `CMakeLists.txt` referencing a now-nonexistent guard/helper
  combination? The build file MUST be cleaned up completely, not left with a dangling
  reference to a deleted file.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST remove the following five demo/runner targets and their
  source files: `run-tree-cu` (`apps/tree/cuda/main.cu`), `run-tree-vk`
  (`apps/tree/vulkan/run_main.cpp`), `run-cifar-dense-omp`
  (`apps/cifar-dense/omp/main.cpp`), `run-cifar-dense-cu` (`apps/cifar-dense/cuda/main.cu`),
  and `run-cifar-sparse-cu` (`apps/cifar-sparse/cuda/main.cu`).
- **FR-002**: The system MUST remove each target's registration from its app's
  `CMakeLists.txt`, and MUST remove any CMake build-helper macro that has no remaining
  caller after this removal.
- **FR-003**: The system MUST NOT remove or alter any `test-*` target, any `bm-*`
  per-stage timing benchmark target (app-level or `profiler/`-tier), or any file under
  `profiler/` — these are outside this feature's scope.
- **FR-004**: The system MUST verify, before removal, that none of the five targets are
  referenced by any CI workflow file, deployment/run script, or optimizer orchestration
  file — and MUST NOT remove a target found to have such a live reference without first
  resolving that reference.
- **FR-005**: The system MUST confirm the routine local correctness gate remains 100%
  green after the removal, as proof the change is behavior-preserving for everything it
  doesn't touch.
- **FR-006**: The system MUST update current, forward-looking how-to documentation that
  instructs building or running a removed target, so it no longer describes a target that
  no longer exists. Dated historical reports and past features' own spec/task
  documentation of work already done MUST be left unchanged.

### Key Entities

- **Demo/Runner Target**: one of the five build targets in scope for removal, each an
  ad-hoc program that constructs a pipeline, dispatches some or all of its stages, and
  prints or hex-dumps output, with no pass/fail assertion, no correctness check against a
  reference, and no timing measurement — distinct from a test (which asserts correctness)
  and a benchmark (which measures and reports timing).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A full build's target list contains zero of the five identified demo/runner
  targets, and the build otherwise succeeds without error.
- **SC-002**: The routine local correctness gate passes at 100%, identical to its
  pre-removal result, confirming no hidden dependency was broken.
- **SC-003**: Zero CI workflow files, scripts, or optimizer orchestration files reference
  any removed target name.
- **SC-004**: Every current how-to doc that previously instructed building or running a
  removed target has been updated; every dated historical report or past feature's own
  documentation of work already done remains unchanged.

## Assumptions

- **"Demo apps" means the five `run-*` targets specifically** (`run-tree-cu`,
  `run-tree-vk`, `run-cifar-dense-omp`, `run-cifar-dense-cu`, `run-cifar-sparse-cu`) — ad
  hoc smoke/dump programs with no assertions, no baseline comparison, and no timing
  measurement (one of them doesn't even dispatch a pipeline stage at all). This
  explicitly does **not** include the per-app `bm-*` Google-Benchmark targets
  (`bm-tree-omp`, `bm-tree-vk`, `bm-cifar-dense-omp/-cu/-vk`,
  `bm-cifar-sparse-omp/-cu/-vk`), which are genuine per-stage timing benchmarks, a
  different category from a "demo." In particular, `bm-tree-omp` is actively referenced
  as a documented validation step by `docs/instruction-for-ai/05-profiling.md` and by the
  `001-octomap-real-workload` feature's spec/quickstart — removing it would break a live,
  current workflow, unlike the five targets in scope. If this reading of "demo apps" is
  wrong, that's the one thing to correct before planning proceeds.
- Investigation (direct repo search across CI workflows, `scripts/`, `docs/`, and
  `optimizer/orchestrate`) found zero live references to four of the five targets
  (`run-tree-vk`, `run-cifar-dense-omp`, `run-cifar-dense-cu`, `run-cifar-sparse-cu`) —
  they appear only in their own `CMakeLists.txt` registration line. `run-tree-cu` has a
  few incidental mentions in two dated historical report docs and in the
  `001-octomap-real-workload` feature's own spec/tasks/quickstart (documenting that it
  was built and run during that feature's verification) — none of these are live
  tooling dependencies, but User Story 3 / FR-006 exist specifically to handle them
  correctly (update forward-looking docs, leave historical records alone).
- `profiler/`'s own tools (`bm-baseline-*`, `bm-fully-*`, `bm-gen-logs-*`, `bm-prof-*`,
  `run-pipe-*-cu`) are a separate, actively-used tier (they feed the optimizer's
  orchestration scripts) and are explicitly out of scope, per FR-003.
- A prior structural refactor (`docs/reports-for-human/target-structure.md`, phases
  P0–P6, all complete) explicitly did not propose removing these targets — its
  acceptance gate was "no target lost." This feature is a new, distinct decision, not a
  continuation of that plan.
- Scope is limited to the `run-*` targets across all three apps (tree, cifar-dense,
  cifar-sparse) since the user said "demo apps" generally, not naming a specific app.
