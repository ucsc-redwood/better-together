# Feature Specification: CPU/GPU Schedule Permutation & Overlap Coverage for Tree

**Feature Branch**: `002-cpu-gpu-schedule-coverage`

**Created**: 2026-07-04

**Status**: Draft

**Input**: User description: "Test all permutation of CPU -> GPU mixed stages for Tree, to make sure interchanging CPU compute / GPU compute from different stage on same memory (because many stage write data directly to same memory). Make sure the pipeline is still behaviour correctly. i.e., CPU/GPU computation still overlapping . i.e., test different possible mixture schedules and make sure CPU / CPU execute are overlapping, rather than waiting on each other"

## Clarifications

### Session 2026-07-04

- Q: Which GPU backend(s) should this permutation/overlap testing cover, given the
  genuinely-chained AppData design (the "same memory" path this feature targets)
  currently only exists for CUDA — Vulkan's equivalent was explicitly deferred as a
  follow-up in the prior session? → A: CUDA only, on real Jetson hardware. Vulkan
  coverage would first require its own chaining redesign (already tracked separately)
  before this kind of permutation testing is meaningful there.
- Q: How much evidence should be required before a schedule's "CPU/GPU genuinely
  overlap" verdict is trusted, given real hardware timing is noisy? → A: Require the
  overlap signal to hold across the majority of multiple repeated runs of the same
  schedule, not a single observation — consistent with this project's existing
  profiling-hygiene convention of not trusting a single timing sample on real hardware.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Every CPU/GPU stage split still computes the correct result (Priority: P1)

A developer who just changed where the CPU/GPU boundary falls in the tree pipeline
(e.g., moved the GPU chunk from stages 4-7 to stages 3-6) wants confidence that the
pipeline still produces the correct octree — not just for the one or two schedules that
happen to be hand-tested today, but for every valid way of splitting the 7 stages
between CPU and GPU, since stages now read and write shared buffers directly rather than
isolated copies.

**Why this priority**: this is the core risk the request calls out — stages write
directly into shared memory now, so an untested schedule boundary is exactly where a
visibility/ordering bug would hide. Without this, "the pipeline works" is only proven
for the couple of schedules anyone thought to write a test for.

**Independent Test**: generate every valid contiguous CPU/GPU stage-split schedule for
the tree pipeline, run each one through the real concurrent runtime, and confirm each
produces output matching a straightforward sequential reference computation for the same
input.

**Acceptance Scenarios**:

1. **Given** a schedule that assigns a contiguous range of stages to the GPU and the
   remaining stages to the CPU (for every valid choice of that range, including the
   boundary cases of "no GPU stages" and "all stages on GPU"), **When** the pipeline runs
   that schedule on real Jetson hardware, **Then** the final octree output for every
   processed item matches a sequential single-pass reference computation exactly.
2. **Given** a schedule where the GPU range sits in the middle of the stage sequence
   (CPU stages both before and after it), **When** the pipeline runs it, **Then** the
   two CPU→GPU and GPU→CPU handoffs both preserve correct data, not just one direction.

---

### User Story 2 - CPU and GPU genuinely overlap instead of taking turns (Priority: P2)

The same developer wants proof that, for schedules with both a CPU chunk and a GPU
chunk, the pipeline is actually doing the thing it was built for — the CPU processing
one item while the GPU processes a different item at the same time — rather than
silently degrading into "CPU waits for GPU to finish, then GPU waits for CPU," which
would defeat the entire point of pipelining.

**Why this priority**: correctness (User Story 1) proves the *answer* is right; this
proves the *pipeline* is doing its job. A schedule that's correct but fully serialized
is a silent performance regression that no correctness check would ever catch.

**Independent Test**: for each schedule containing both a CPU and a GPU chunk, measure
the wall-clock work windows of the CPU chunk and the GPU chunk during steady-state
processing and confirm they overlap in time for different in-flight items, rather than
running strictly one-after-another.

**Acceptance Scenarios**:

1. **Given** a schedule with both a CPU chunk and a GPU chunk, **When** the pipeline
   processes a steady stream of items, **Then** measured evidence shows the CPU chunk
   working on one item while the GPU chunk is concurrently working on another.
2. **Given** the same schedule, **When** overlap is measured, **Then** the result
   distinguishes "genuinely overlapping" from "fully serialized" rather than only
   reporting total throughput (a throughput number alone can't tell the two apart).

---

### User Story 3 - A failing permutation is actionable, not just a red mark (Priority: P3)

When one specific CPU/GPU schedule fails — either the answer is wrong or the CPU and GPU
never actually overlap — a developer wants to know exactly which schedule failed and
what specifically went wrong, without having to re-run anything with extra
instrumentation bolted on by hand.

**Why this priority**: lower priority than proving the properties (User Stories 1-2)
exist to check, but a failure that just says "some permutation failed" with no further
detail turns every regression into its own investigation from scratch.

**Independent Test**: intentionally review a failing run's output and confirm it names
the exact stage-boundary schedule involved and states whether the failure was a data
mismatch (and where) or a lack of measured overlap.

**Acceptance Scenarios**:

1. **Given** a schedule whose output doesn't match the sequential reference, **When**
   the sweep reports it, **Then** the report names the exact schedule (stage boundaries
   and CPU/GPU assignment) and which buffer/stage produced the mismatch.
2. **Given** a schedule that runs correctly but shows no measured CPU/GPU overlap,
   **When** the sweep reports it, **Then** the report distinguishes this from a
   correctness failure and names the same schedule detail.

---

### Edge Cases

- What happens for the "all stages on CPU" and "all stages on GPU" boundary schedules,
  which have no CPU/GPU handoff at all? They MUST still be checked for correctness (User
  Story 1); overlap verification (User Story 2) does not apply to them since there's
  only one PU in play.
- What happens when the GPU range is a single stage sandwiched between two CPU ranges
  (the most handoff-heavy case: CPU→GPU then GPU→CPU within one schedule)? Both
  handoffs must be independently verified, not just the first one.
- What happens if a permutation genuinely reveals a real memory-visibility defect (the
  scenario this feature exists to catch)? The sweep MUST fail loudly and specifically
  for that schedule (per User Story 3) rather than silently passing or crashing the
  whole sweep.
- What happens if two different items' work windows overlap only very briefly (e.g.,
  right at a handoff boundary) versus substantially through steady-state processing?
  The overlap check MUST be based on steady-state behavior, not a single incidental
  moment, so a fluke isn't mistaken for real pipelining.
- What happens if a schedule shows overlap on some repeated runs but not others (noisy
  hardware, thermal throttling, scheduling jitter)? The verdict MUST be based on the
  outcome holding across the majority of multiple repeated runs, not any single run,
  so hardware noise isn't mistaken for a genuine serialization regression (or vice
  versa).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST be able to generate every valid schedule that assigns the
  tree pipeline's 7 stages to a single contiguous GPU stage-range with the remaining
  stages on CPU, covering every possible placement of that range (including the
  boundary cases where the range is empty — all stages on CPU — or spans all 7 stages —
  all stages on GPU).
- **FR-002**: For each generated schedule, the system MUST run the tree pipeline
  through the real concurrent runtime (the same pool/queue/worker execution path
  production schedules use) rather than a simplified or mocked execution path.
- **FR-003**: For each generated schedule, the system MUST verify the final output for
  every processed item matches a sequential single-pass reference computation over the
  same input, exactly (structural/integer comparison, consistent with this project's
  existing correctness convention).
- **FR-004**: For each generated schedule that contains both a CPU chunk and a GPU
  chunk, the system MUST measure whether the CPU chunk's and GPU chunk's work windows
  overlap in wall-clock time for different in-flight items during steady-state
  processing, and MUST distinguish "overlapping" from "fully serialized" outcomes.
  Because real-hardware timing is noisy, a schedule's overlap verdict MUST be based on
  the outcome holding across the majority of multiple repeated runs of that schedule,
  not a single observation.
- **FR-005**: When a schedule's output fails the correctness check, the system MUST
  report the exact schedule (stage boundaries and CPU/GPU assignment) and identify which
  stage or buffer produced the mismatch.
- **FR-006**: When a schedule fails to show measured CPU/GPU overlap, the system MUST
  report the exact schedule and clearly distinguish this from a correctness failure.
- **FR-007**: The system MUST run this permutation sweep on real CUDA-capable hardware
  (not a simulated or mocked timing model), since genuine concurrency can only be
  demonstrated on real hardware.
- **FR-008**: This permutation/overlap sweep MUST NOT be added to the project's routine
  correctness gates (the everyday CPU and backend-specific test gates) — it is a
  deliberate, on-demand verification suite, consistent with how the underlying
  genuinely-chained execution path is already treated.

### Key Entities

- **Schedule Permutation**: One specific way of splitting the tree pipeline's 7 stages
  into a contiguous CPU range and a contiguous GPU range (either of which may be empty),
  defined by where the GPU range starts and ends.
- **Overlap Window**: A measured time interval showing a CPU chunk actively working on
  one item while a GPU chunk was concurrently and actively working on a different item,
  used to distinguish genuine pipelining from serialized hand-offs.
- **Permutation Result**: The pass/fail outcome for one schedule permutation, covering
  both its correctness verdict and its overlap verdict, plus enough detail to diagnose
  a failure without re-instrumenting.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every valid contiguous CPU/GPU stage-split schedule for the tree pipeline
  (every possible GPU-range placement across the 7 stages, including the all-CPU and
  all-GPU boundary cases) is exercised, and every one produces output matching the
  sequential reference computation exactly.
- **SC-002**: For every schedule containing both a CPU chunk and a GPU chunk, measured
  evidence across the majority of multiple repeated runs confirms the CPU and GPU
  chunks' work windows genuinely overlap in wall-clock time during steady-state
  processing, rather than running serialized or only appearing to overlap by fluke.
- **SC-003**: Any permutation that fails (on correctness or on overlap) is reported with
  enough specific detail — which schedule, which stage/buffer or which measurement — to
  diagnose the failure without adding new instrumentation after the fact.

## Assumptions

- Scope is the tree app only, as stated in the request; other apps (cifar-dense,
  cifar-sparse) are out of scope for this feature.
- Per the clarification above, this feature targets CUDA only, run on real Jetson
  hardware (`duck-stable`/`duck-naughty`) — genuine wall-clock overlap can only be
  demonstrated on real hardware, not simulated. Vulkan coverage is out of scope until its
  own genuinely-chained dispatch path exists (tracked separately).
- "All permutations of CPU→GPU mixed stages" means every valid contiguous single-GPU
  stage-range placement across the 7 stages (the runtime allows at most one active GPU
  chunk per schedule, since the GPU dispatcher's command buffer/fence is shared across
  chunks) — not stage-by-stage alternation, which the current runtime architecture
  doesn't support within a single schedule.
- CPU-side core-tier sub-splitting (big/little/medium) is out of scope here — that
  dimension is already covered by existing tests; this feature isolates the CPU/GPU
  stage-boundary dimension specifically.
- This targets the genuinely-chained `tree::AppData` dispatch path introduced in the
  prior session (not `SafeAppData`'s golden-decoupled path), since that's the design
  where adjacent stages actually share and overwrite the same buffers across a CPU/GPU
  boundary — the exact property this feature is stress-testing.
- Consistent with the prior session's precedent, this permutation/overlap suite is kept
  out of the routine `ctest -L omp`/`ctest -L cuda` correctness gates by default (an
  on-demand, deliberately-excluded verification suite) unless a later decision promotes
  it into a maintained gate.
- "Overlap" is a qualitative/binary determination (do the CPU and GPU chunks'
  measured work windows for different items intersect in wall-clock time during
  steady-state processing) rather than a specific numeric overlap-percentage target —
  but that binary verdict must itself be corroborated across the majority of multiple
  repeated runs per schedule (per the clarification above), not decided from one run.
