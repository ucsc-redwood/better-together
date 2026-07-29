# Implementation Plan: Tree Scheduler Re-Validation Post-AppData Migration

**Branch**: `005-tree-scheduler-revalidation` | **Date**: 2026-07-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/005-tree-scheduler-revalidation/spec.md`

## Summary

Re-run this project's already-built profile → schedule (z3) → run → compare cycle for
the tree app on `duck-stable` (both its CUDA and Vulkan backends), `samsung`, and
`pixel`, using `optimizer/orchestrate/00_run_fleet.py --only duck-stable,samsung,pixel
--fresh`, then read tree's four rows out of the regenerated
`data/sched_logs/speedup-summary.md` and archive them as a new dated report under
`docs/reports-for-human/perf-results/`. No new tooling is built — every step (profiling
collection, baseline derivation, z3 schedule generation, on-device schedule execution,
speedup computation, report rendering) already exists and is exercised as-is.

## Technical Context

**Language/Version**: Python 3.13 (the existing `uv` workspace — `optimizer/orchestrate/`,
`optimizer/smt/`, `optimizer/analysis/`); no new code language introduced

**Primary Dependencies**: the existing orchestration stack — `00_run_fleet.py` (fleet
driver), `01_collect_profiling.py`/`02_gen_schedule_merged.py`/`03_run_schedule.py`
(per-cell steps it shells out to), `optimizer/smt/baselines.py` (best-PU baseline
derivation), `optimizer/analysis/speedup_summary.py` (the comparison + Markdown report).
z3 (via `optimizer/smt/solver.py`) generates the candidate schedules; nothing here is
new.

**Storage**: the existing gitignored `data/` tree (`data/profiling/`,
`data/schedules_{btpm,isolated}/`, `data/sched_logs/`) — regenerated, not versioned;
`fleet.json`/`vocab.json` are the existing config this feature reads, not writes.

**Testing**: no new automated tests. Per the spec's Clarification, IF a code defect
blocks data collection, its minimal fix MUST pass the existing OMP-as-oracle
differential gate for the affected backend (`ctest -L <cu|vk>` on the affected target)
before being trusted — this is Principle III, unconditionally, not something this
feature relaxes.

**Target Platform**: `duck-stable` (Jetson Orin, CUDA+Vulkan, `ssh doremy@duck-stable`),
`samsung` (`R5CY21Y3VEV`, Vulkan only, adb via `rocky-ryzen`), `pixel`
(`3A021JEHN02756`, Vulkan only, adb via `rocky-ryzen`) — the exact three `fleet.json`
device keys the spec's Assumptions map to. `duck-naughty`, `minipc` (rocky-ryzen as a
device under test), and any other fleet member are excluded from `--only`.

**Project Type**: existing Python orchestration/analysis tooling + existing C++
production binaries it deploys and runs (`bm-prof-tree-{cu,vk}`, `bm-gen-logs-tree-{cu,vk}`)
— no new project structure.

**Performance Goals**: N/A as a target to hit — the numbers are the *output* of this
feature (an honest current measurement), not a bar it must clear. Per the spec, tree
legitimately losing to its baseline on some combination is an acceptable outcome, not a
failure.

**Constraints**: `00_run_fleet.py` has no per-app filter (it always runs every app in
`vocab.json["app_stages"]` = tree, cifar-dense, cifar-sparse against the selected
devices) — per the spec's Assumptions/Edge Cases, this incidental extra coverage is
accepted, not worked around. Constitution Principle VI (Isolated Measurement
Environment) applies directly: `duck-stable` and the `rocky-ryzen` adb host must be
confirmed free of competing load immediately before running, given `rocky-ryzen` was
mid-job with unrelated work earlier this same session.

**Scale/Scope**: 3 devices, 4 tree device/backend combinations (duck-stable×cu,
duck-stable×vk, pixel×vk, samsung×vk) as the feature's own success criteria — plus 8
incidental cifar-dense/cifar-sparse cells the same fleet run also produces (2 apps × the
same 4 combinations), which are not this feature's concern.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Simplicity First)**: this feature adds no new code path by default —
  it re-runs existing, already-built tooling. The only code change admitted at all is a
  minimally-scoped defect fix (FR-007), and only if collection is actually blocked.
- **Principle II (Surgical, Traceable Changes)**: if FR-007's contingency fires, the fix
  must be scoped to the blocking defect only — this plan does not pre-authorize any
  broader refactor "while we're in there."
- **Principle III (OMP-as-Oracle Differential Testing, NON-NEGOTIABLE)**: any FR-007 fix
  MUST pass the relevant `ctest -L <backend>` gate on the affected target before its
  output is trusted for re-collection — the differential suites are the arbiter of
  "is the fix correct," not the profiling numbers themselves.
- **Principle IV (Goal-Driven Verification)**: directly the basis for FR-007 — "loop
  until the goal (fresh, complete data) is met," not stop at the first crash.
- **Principle VI (Isolated Measurement Environment)**: explicit precondition before any
  profiling/schedule-run step — confirm `duck-stable` and `rocky-ryzen` (the adb host
  for both phones) have no competing process, exactly as this session's constitution
  amendment and its `llama-server` precedent require.

No violations — Complexity Tracking intentionally omitted.

## Project Structure

### Documentation (this feature)

```text
specs/005-tree-scheduler-revalidation/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks, not this command)
```

No `contracts/` (this feature calls existing CLI tools, it doesn't define a new
interface) and no separate `quickstart.md` (the "how to validate" content is the same
short command sequence that *is* the feature, folded into `data-model.md`).

### Source Code (repository root)

No new source files. Files this feature *may* touch, depending on what FR-007's
contingency finds:

```text
data/                                          # REGENERATE (gitignored): profiling,
                                                #   schedules_{btpm,isolated}, sched_logs
                                                #   for duck-stable/samsung/pixel only
docs/reports-for-human/perf-results/
    speedup-summary-<DATE>-appdata-migration.md  # NEW: dated snapshot of the
                                                  #   regenerated data/sched_logs/
                                                  #   speedup-summary.md
apps/tree/{omp,cuda,vulkan}/...                # CONDITIONAL: only touched if FR-007's
                                                #   contingency fires (a real collection-
                                                #   blocking defect is found) -- scope
                                                #   unknowable in advance by definition
```

**Structure Decision**: no new project structure — this feature operates entirely
through the existing `optimizer/orchestrate/00_run_fleet.py` entry point and existing
report-archival convention; the only unplanned file changes would be a narrowly-scoped
bugfix under FR-007, whose exact location can't be known until (if) it's needed.

## Complexity Tracking

*No violations — table intentionally omitted (see Constitution Check above).*
