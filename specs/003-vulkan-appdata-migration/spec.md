# Feature Specification: Vulkan Genuinely-Chained AppData Migration

**Feature Branch**: `003-vulkan-appdata-migration`

**Created**: 2026-07-04

**Status**: Draft

**Input**: User description: "With the success of CUDA no more SafeApp data, we need to
extend this to Vulkan backend. Do the same on Vulkan, test it on Jetsons, and Androids."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The Vulkan tree pipeline computes correct results without the golden-decoupled scaffold (Priority: P1)

A developer who just built a genuinely-chained (single-buffer-per-stage, no
golden/`_out` duplication) execution path for Vulkan wants proof that dispatching the
tree pipeline's stages through it — where each stage reads what the previous stage
actually wrote, rather than a fixed reference snapshot built once at construction —
still produces the correct octree, on the real GPUs this backend targets.

**Why this priority**: this is the foundational proof everything else depends on. If the
chained path doesn't compute correct results on real Vulkan hardware, wiring it into
production tools or the routine test gates (User Stories 2-3) would just spread a latent
bug further, exactly as would have happened had this step been skipped for CUDA.

**Independent Test**: dispatch the tree pipeline's 7 stages through the new
genuinely-chained Vulkan path on real hardware and confirm the final output for every
processed item matches the OMP reference computation for the same input, run
independently on both a Jetson devkit and an Android phone.

**Acceptance Scenarios**:

1. **Given** the genuinely-chained Vulkan dispatch path and a fixed-seed input, **When**
   each of the 7 stages is dispatched through it on a Jetson's integrated GPU, **Then**
   every stage's output matches the OMP reference computation for the same input.
2. **Given** the same dispatch path, **When** it runs on an Android phone's GPU (both the
   Pixel's and the Samsung's, since they run different GPU shader-subgroup variants),
   **Then** the same per-stage correctness holds on each phone independently.
3. **Given** a schedule that hands work between a CPU chunk and a Vulkan GPU chunk
   concurrently over the same pooled buffers, **When** the pipeline runs it end-to-end,
   **Then** the final result is still correct and the CPU and GPU chunks are genuinely
   processing different items concurrently rather than serialized (mirroring what the
   CUDA migration already proved for its own chained path).

---

### User Story 2 - Production Vulkan profiling tools measure the real production path (Priority: P2)

An engineer using the Vulkan profiling tools to feed real per-stage timing data to the
scheduler wants those tools to measure the same genuinely-chained execution path
production schedules actually run, instead of a golden-decoupled stand-in whose memory
layout and per-stage cost don't match production.

**Why this priority**: depends on User Story 1 being proven correct first. This is where
the correctness fix actually starts paying off in the numbers the scheduler consumes —
without it, the CUDA-side fix's benefit (matching profiled data to production behavior)
doesn't extend to Vulkan targets.

**Independent Test**: run each production Vulkan profiling tool on a Jetson and on an
Android phone and confirm it completes without error and produces plausible per-stage
output using the new chained path.

**Acceptance Scenarios**:

1. **Given** the production Vulkan profiling tools, **When** each is run on a Jetson
   devkit, **Then** it runs to completion without error and its output reflects the
   genuinely-chained path rather than the golden-decoupled one.
2. **Given** the same tools, **When** each is run on an Android phone, **Then** the same
   holds.

---

### User Story 3 - The project's routine Vulkan correctness checks exercise the same path production code uses (Priority: P3)

A developer relying on the project's everyday Vulkan correctness checks wants those
checks to actually exercise the genuinely-chained path production code now uses, not a
golden-decoupled scaffold that only proves an unused code path still works.

**Why this priority**: lowest priority because it's the last mile — the correctness
proof (User Story 1) and the production wiring (User Story 2) both have to land first.
Skipping this would leave the project's ongoing regression gate silently checking the
wrong thing, the same gap the CUDA-side work found and closed.

**Independent Test**: run the project's existing Vulkan correctness test suites after
they've been switched to the chained path, on a Jetson and on an Android phone, and
confirm they pass.

**Acceptance Scenarios**:

1. **Given** the project's Vulkan differential and pipeline-mechanics test suites now
   targeting the chained path, **When** they run on a Jetson devkit, **Then** they pass.
2. **Given** the same suites, **When** they run on an Android phone, **Then** they pass.

---

### Edge Cases

- What happens on the two different Android GPU shader-subgroup variants (Pixel's
  subgroup-16 vs. Samsung's subgroup-32)? Both MUST be verified independently — a defect
  specific to one subgroup width would otherwise hide behind the other phone's pass.
- What happens for the concurrent CPU+GPU hybrid schedule case, where the CPU writes one
  pooled item's buffers while the GPU reads/writes a different item's at the same time?
  This is the unified-memory-visibility scenario that previously surfaced a real defect
  on this backend (see the Mali coherency history) and MUST be re-verified under the
  chained path, not assumed safe by analogy to the golden-decoupled path's prior fix.
- What happens if the chained path is correct on Jetson but not on a phone, or vice
  versa? Each device MUST be treated as an independent pass/fail — passing on one is not
  evidence for the other, consistent with why this backend has historically had per-GPU
  visibility bugs that only reproduced on specific hardware.
- What happens to the existing golden-decoupled Vulkan path once the chained path is
  proven and wired in? Out of scope here — see Assumptions.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a genuinely-chained (single buffer per stage, no
  golden/`_out` duplication) dispatch path for the tree pipeline's Vulkan backend,
  mirroring the design already proven for CUDA.
- **FR-002**: The system MUST verify this chained Vulkan path's per-stage output against
  the OMP reference computation for the same input, independently on a Jetson devkit and
  on both Android phones (Pixel and Samsung).
- **FR-003**: The system MUST verify that a schedule mixing a CPU chunk and a Vulkan GPU
  chunk over shared pooled buffers still produces correct results and that the CPU and
  GPU chunks genuinely process different items concurrently, independently on a Jetson
  devkit and on both Android phones.
- **FR-004**: The system MUST switch the production Vulkan profiling tools from the
  golden-decoupled scaffold to the chained path, and confirm each runs correctly on a
  Jetson devkit and on both Android phones.
- **FR-005**: The system MUST switch the project's routine Vulkan differential and
  pipeline-mechanics test suites from the golden-decoupled scaffold to the chained path,
  and confirm they pass on a Jetson devkit and on both Android phones.
- **FR-006**: When any correctness or overlap check fails, the system MUST report which
  device, which stage, and what specifically diverged, so a failure on one device/phone
  doesn't get mistaken for or masked by a pass on another.

### Key Entities

- **Chained Vulkan AppData**: The genuinely-chained, single-buffer-per-stage data
  structure for the Vulkan backend, mirroring `tree::AppData`'s design (as opposed to the
  existing golden-decoupled `VkAppData_Safe`).
- **Device Target**: One of the three hardware targets this feature must be verified on
  — the Jetson devkit, the Pixel phone, and the Samsung phone — each with its own
  pass/fail verdict.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The chained Vulkan dispatch path produces per-stage output matching the
  OMP reference computation, verified independently on a Jetson devkit and on both
  Android phones.
- **SC-002**: A hybrid CPU+Vulkan-GPU schedule over shared pooled buffers produces
  correct results with genuine CPU/GPU concurrency, verified independently on a Jetson
  devkit and on both Android phones.
- **SC-003**: Every production Vulkan profiling tool runs to completion without error
  using the chained path, on a Jetson devkit and on both Android phones.
- **SC-004**: The project's routine Vulkan correctness test suites pass using the
  chained path, on a Jetson devkit and on both Android phones.

## Assumptions

- Scope is the tree app only, matching the CUDA-side work this extends; other apps
  (cifar-dense, cifar-sparse) are out of scope.
- "Do the same on Vulkan" means mirroring the three stages already completed for CUDA in
  this project: (1) build and prove a genuinely-chained dispatch path standalone, (2)
  wire production profiling tools to it, (3) migrate the routine correctness-gate test
  suites to it — for the Vulkan backend specifically.
- The existing golden-decoupled `VkAppData_Safe` struct is not deleted as part of this
  feature — consistent with how `SafeAppData` itself was left in place (not removed) once
  CUDA's production and test paths moved off it. Removal is a separate, later decision.
- "Test it on Jetsons, and Androids" is read as the required hardware verification
  targets for this feature: the Jetson devkit(s) and both Android phones (Pixel 7a,
  Samsung Galaxy) — the two GPU shader-subgroup variants this project's hardware fleet
  covers. Development iteration may also use the x86 Vulkan host (the easiest Vulkan
  path, no cross-compile), but it is not a required verification target here since the
  user named Jetson and Android specifically.
- Rocky-ryzen (the x86 integrated-GPU Vulkan host) is not named as a required test target
  by the user; it remains available for development iteration but SC-001 through SC-004
  are scoped to Jetson + both Android phones as stated.
- Per this project's existing convention (kept for the CUDA equivalent), this feature's
  new standalone correctness/overlap proof suites are not required to join the routine
  per-commit gate; only the migrated production and routine-gate suites (User Stories 2-3)
  are expected to run as part of normal verification going forward.
