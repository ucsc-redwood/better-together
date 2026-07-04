# Feature Specification: Tree Scheduler Re-Validation Post-AppData Migration

**Feature Branch**: `005-tree-scheduler-revalidation`

**Created**: 2026-07-04

**Status**: Draft

**Input**: User description: "with the newer change to Appdata Tree. Recollect the
benchmark table and e2e results on the devices (limit it to Jetson and the two android
phones for now). and see if the scheduler is still able to produce schedules that out
perform best-PU baseline"

## Clarifications

### Session 2026-07-04

- Q: If recollecting profiling data on a device hits a code defect that blocks
  collection (e.g., a crash in the migrated dispatch path), is fixing it in scope for
  this feature? → A: Yes — fix minimal blocking defects as part of this feature (scoped
  strictly to what's needed to unblock collection, not a broader refactor), then
  continue.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Fresh, current profiling data replaces stale pre-migration data (Priority: P1)

A developer who just changed how the tree app's CUDA and Vulkan backends dispatch work
(the genuinely-chained `AppData` migration) wants the profiling numbers that describe
"how fast each stage runs on each processor" to reflect that change, not the
golden-decoupled behavior measured before it — because every downstream decision (the
best-single-processor baseline, the scheduler's choices) is only as good as this data.

**Why this priority**: this is the foundation everything else depends on. Scheduling
decisions made from stale data would look plausible but describe a version of the
software that no longer exists.

**Independent Test**: recollect the tree app's profiling data on each target device and
backend, and confirm the resulting data is dated after the migration and free of any
carried-over data from before it.

**Acceptance Scenarios**:

1. **Given** the tree app's CUDA and Vulkan backends have changed how they dispatch
   work, **When** profiling data is recollected on a Jetson devkit (both its CUDA and
   Vulkan backends) and on both Android phones (Vulkan only — phones have no CUDA),
   **Then** the resulting data is fresh (post-migration) and does not mix in any
   leftover data collected before the migration or before other recent hardware changes
   (the JetPack reflash, the prior kernel-optimization pass).
2. **Given** the recollected data, **When** the best-single-processor baseline for each
   device/backend is derived from it, **Then** that baseline reflects current, not
   historical, per-processor performance.

---

### User Story 2 - An honest answer to "does pipelining still pay off for tree" (Priority: P1)

The same developer wants a real, measured answer — not an assumption — to whether the
scheduler can still find a way to split tree's work across a device's processors that
finishes faster than running the whole thing on the single fastest processor, now that
the underlying dispatch behavior has changed.

**Why this priority**: equal priority to User Story 1 — fresh data alone doesn't answer
the question; it has to actually be fed through the scheduler and the result actually run
on real hardware to know.

**Independent Test**: generate schedule candidates from the fresh data, actually run the
top candidates on each target device, and compare the best measured result against that
device's best-single-processor baseline.

**Acceptance Scenarios**:

1. **Given** fresh profiling data for tree on a given device/backend, **When** the
   scheduler generates candidate schedules and the top candidates are run on that real
   device, **Then** the best measured result is compared against the best-single-
   processor baseline for that same device/backend and the outcome (faster, slower, or
   about the same) is reported as a real number, not inferred.
2. **Given** this comparison across all in-scope device/backend combinations (Jetson
   CUDA, Jetson Vulkan, Pixel Vulkan, Samsung Vulkan), **When** the results are
   assembled, **Then** each combination has its own reported outcome — a win on one
   device/backend is never used to imply a win on another.

---

### User Story 3 - The result is a discoverable, dated record (Priority: P2)

A developer or reviewer who wasn't in the room when this re-validation happened wants to
find the answer later without having to re-run anything, in the same report format this
project already uses for this exact kind of result.

**Why this priority**: lower priority than actually producing a correct, current answer
(User Stories 1-2) — this is about not losing that answer once it exists.

**Independent Test**: locate the dated report and confirm it states, per device/backend,
the baseline, the best measured schedule result, and the resulting speedup (or lack of
one), in the project's existing report format for this kind of result.

**Acceptance Scenarios**:

1. **Given** the measured comparison from User Story 2, **When** it is recorded, **Then**
   it appears as a new dated report following this project's existing format for this
   result (the same one prior "speedup summary" reports use), not an ad hoc one-off.
2. **Given** tree does not beat the baseline on some device/backend, **When** the report
   is written, **Then** it says so plainly rather than omitting or obscuring that result.

---

### Edge Cases

- What happens if a device/backend combination shows the scheduler's best schedule
  running *slower* than the single-processor baseline? This MUST be reported honestly,
  not hidden — this project's own prior reports already document that tree specifically
  can legitimately lose to its baseline on some configurations (its per-stage work is
  small enough that per-chunk overhead the cost model doesn't fully capture can matter),
  so a loss here is a valid, reportable outcome, not a failure of this feature.
- What happens to the tree profiling/schedule/log data currently sitting in the local
  working copy? It is dated well before the JetPack reflash, the device renaming, and the
  AppData migration, and uses device names that no longer match the current fleet
  (`jetson` vs. today's `duck-stable`/`duck-naughty`) — it MUST be treated as unusable and
  replaced, not merged with or appended to.
- What happens if the same recollection run also touches the other two apps
  (cifar-dense, cifar-sparse), since this project's existing whole-fleet tooling doesn't
  currently let a run be scoped to one app? That's acceptable incidental overhead, not a
  scope violation — this feature's own success criteria only concern the tree app's
  results.
- What happens for the Jetson's Vulkan backend versus its CUDA backend — are they treated
  as one combined result or two independent ones? Independent: the Jetson runs both
  backends, and each is its own device/backend combination with its own baseline and its
  own scheduler outcome, per User Story 2's second acceptance scenario.
- What happens if a code defect in the migrated dispatch path blocks data collection
  outright (e.g., a crash), rather than just producing a legitimate loss-vs-baseline
  result? Per the Clarifications above, fixing the minimum necessary to unblock
  collection is in scope; the fix itself MUST stay scoped to what's needed to get
  collection working again, not expand into unrelated refactoring.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST recollect the tree app's profiling data from scratch (not
  reuse or merge with any existing local data) on: a Jetson devkit's CUDA backend, that
  same Jetson devkit's Vulkan backend, the Pixel phone's Vulkan backend, and the Samsung
  phone's Vulkan backend.
- **FR-002**: The system MUST derive each of those four device/backend combinations'
  best-single-processor baseline from the freshly recollected data, not from any
  previously-computed baseline.
- **FR-003**: The system MUST generate candidate schedules from the fresh data for each
  of the four combinations and actually run a representative set of the top candidates
  on the corresponding real device.
- **FR-004**: The system MUST compare the best measured schedule result for each
  combination against that combination's own best-single-processor baseline and record
  whether it is faster, slower, or approximately equal.
- **FR-005**: The system MUST produce a dated report, in this project's existing report
  format for this kind of result, stating the baseline, best measured result, and
  resulting outcome for all four combinations — including any combination where tree does
  not beat its baseline.
- **FR-006**: The system MUST NOT alter or delete the existing dated historical reports
  from before this migration (they remain an accurate record of what was true at the
  time).
- **FR-007**: If a code defect blocks data collection for a device/backend combination,
  the system MUST fix the minimum necessary to unblock collection (scoped strictly to
  the blocking defect, not a broader refactor) and continue, rather than reporting that
  combination as permanently blocked.

### Key Entities

- **Device/Backend Combination**: one of the four in-scope pairings (Jetson+CUDA,
  Jetson+Vulkan, Pixel+Vulkan, Samsung+Vulkan), each carrying its own independent
  baseline, scheduler result, and pass/fail-style outcome.
- **Best-Single-Processor Baseline**: the fastest measured time for running tree's
  entire pipeline on one single processor of a given device (already an established,
  precisely-defined concept in this project — not something this feature redefines).
- **Re-Validation Report**: the dated record of this feature's outcome, in the same
  format this project already uses for reporting measured scheduler speedups.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Fresh, current profiling data exists for tree on all four in-scope
  device/backend combinations, with no pre-migration or pre-reflash data mixed in.
- **SC-002**: Each of the four combinations has an independently measured "scheduler
  best vs. single-processor baseline" comparison, backed by results actually run on real
  hardware (not estimated).
- **SC-003**: A dated report exists, in the project's established format, that states
  each combination's outcome plainly — including honestly reporting any combination
  where tree does not beat its baseline.

## Assumptions

- Scope is the tree app only, on exactly four device/backend combinations: a Jetson
  devkit's CUDA backend, that same Jetson's Vulkan backend, the Pixel 7a's Vulkan
  backend, and the Samsung Galaxy's Vulkan backend — matching the user's explicit "limit
  it to Jetson and the two android phones for now." Of the fleet's two physical Jetson
  units, "Jetson" is read as the primary, coverage-gated one; the benchmark-only twin
  devkit is not required by this feature (it may optionally also be refreshed as a
  low-cost bonus, consistent with how it's already grouped alongside the primary unit in
  this project's existing fleet-wide tooling and reports).
- cifar-dense and cifar-sparse are out of scope for this feature's own success criteria.
  If this project's existing whole-fleet recollection tooling incidentally also
  refreshes their data in the same run (since it doesn't currently support running one
  app in isolation), that's acceptable overhead, not scope creep — per the Edge Cases.
- "Best-PU baseline" means this project's already-established "best-single-processor
  whole-pipeline" comparison point, derived from the isolated (non-interference)
  profiling data — this feature does not redefine what a baseline is, only recomputes it
  from current data.
- The rocky-ryzen host is out of scope as a *device under test* here (per the user's
  explicit device list), even though it's used as the network path to reach both Android
  phones' adb — consistent with the established practice from the recent Vulkan
  migration of not running measurement workloads on it while it's busy with other work.
- The result is expected to land as a new dated file following the existing
  `speedup-summary-YYYY-MM-DD-<label>.md` report convention this project already uses,
  not a newly-invented report shape.
- Whether tree's scheduled result beats its baseline on every combination is not assumed
  in advance — this project's own prior reports already document that tree can
  legitimately show a loss on some configurations. This feature's job is to get an
  honest, current answer, not to guarantee a particular outcome.
