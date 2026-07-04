<!--
Sync Impact Report
- Version change: [TEMPLATE] → 1.0.0 (initial ratification)
- Modified principles: n/a (first fill of template placeholders)
- Added sections:
  - I. Simplicity First (NON-NEGOTIABLE)
  - II. Surgical, Traceable Changes
  - III. OMP-as-Oracle Differential Testing (NON-NEGOTIABLE)
  - IV. Goal-Driven Verification
  - V. Data & Docs as Source of Truth
  - Backend & Build Constraints (Section 2)
  - Development Workflow (Section 3)
  - Governance
- Removed sections: none (placeholders only)
- Templates requiring updates:
  - ✅ .specify/templates/plan-template.md (generic "[Gates determined based on constitution file]" placeholder is filled per-feature at plan time; no edit needed now)
  - ✅ .specify/templates/spec-template.md (no constitution-specific placeholders found)
  - ✅ .specify/templates/tasks-template.md (no constitution-specific placeholders found)
  - ✅ .specify/templates/checklist-template.md (no constitution-specific placeholders found)
  - No command files under .specify/templates/commands/ exist in this repo
- Follow-up TODOs: none
-->

# BetterTogether Constitution

## Core Principles

### I. Simplicity First (NON-NEGOTIABLE)
Write the minimum code that solves the problem — nothing speculative. No abstractions
for single-use code, no unrequested "flexibility", no error handling for impossible
cases or inputs that cannot occur given internal/framework guarantees. If 200 lines
could be 50, rewrite it. Validate only at real system boundaries (user input, external
APIs) — trust internal code and framework guarantees everywhere else.
**Rationale**: this is a research/systems framework spanning three languages and three
backends; speculative abstraction compounds across that surface faster than it would in
a single-stack app, and has repeatedly been the source of dead code and drift (see
`docs/reports-for-human/` cleanup history).

### II. Surgical, Traceable Changes
Touch only what the task requires; match existing style; do not refactor or reformat
adjacent code that isn't broken. Every changed line MUST trace to the request. Remove
orphans your own change created; leave pre-existing dead code in place but call it out —
do not silently delete it as a drive-by. State assumptions and surface competing
interpretations instead of silently picking one; stop and ask when something is
genuinely unclear before writing code.
**Rationale**: the codebase is actively used for hardware experiments across a live
device fleet; unrelated churn makes diffs unreviewable and risks masking the actual
behavioral change under test.

### III. OMP-as-Oracle Differential Testing (NON-NEGOTIABLE)
OMP (CPU) is the reference oracle for every stage and every backend. New or changed
CUDA/Vulkan kernels MUST be verified against the OMP output at a fixed seed (`114514`):
exact comparison for integer/structural stages, `NearEqual` for floating-point stages.
`ctest -L omp` is the everyday gate and MUST be green before a change is considered
mergeable; a `GTEST_SKIP` counts as a pass. Backend-specific gates (`ctest -L cuda`,
`ctest -L vulkan`, etc.) apply additionally whenever that backend is touched or
reachable from the change, run on the correct target hardware (see
`docs/instruction-for-ai/01-hardware.md`).
**Rationale**: with three backends implementing the same kernels, differential testing
against a trusted CPU reference is the only tractable way to catch backend-specific
numerical or structural bugs; this method has already found real defects (Mali
coherency, CUDA managed-memory visibility) that unit tests alone missed.

### IV. Goal-Driven Verification
Turn every task into a verifiable goal and loop until it is met, not until code merely
compiles. "Fix the bug" means: reproduce it first, then make the reproduction pass.
"Refactor X" means: capture the tests green before the change and keep them green after.
A change is not done until the relevant `ctest -L <backend>` is green on its target — at
minimum `ctest -L omp` locally — and, for UI/runnable-behavior changes, the feature has
actually been exercised (started, screenshotted, or driven end-to-end), not just
type-checked.
**Rationale**: "looks right" and "builds" are not the bar on a profiling/scheduling
system where correctness is numerical and cross-backend; only running the gate proves
anything.

### V. Data & Docs as Source of Truth
Device topology, kernel/backend availability, and hardware access facts live in
`devices/*.json` and `docs/instruction-for-ai/` — not in ad hoc knowledge, memory, or
prose that has drifted. Where `devices/*.json` or `conf.cpp` disagrees with the README or
any narrative doc, the data/code wins and the doc MUST be corrected. Status, audits, and
decision history belong in `docs/reports-for-human/`; how-to and load-bearing operational
knowledge belongs in `docs/instruction-for-ai/`. Do not duplicate one into the other.
**Rationale**: the fleet (Jetsons, rocky-ryzen, phones) has been reflashed and
reconfigured multiple times; stale narrative docs have previously caused wrong deploy
targets and non-comparable benchmark results (e.g. pre/post JetPack-7.2 reflash data).

## Backend & Build Constraints

**Build system**: CMake (`cmake/` + presets `pc`/`jetson`/`vulkan`/`android`) is the only
supported build system. No new build tooling is introduced without a constitution
amendment.

**Backend placement rules** (do not violate without an explicit, documented exception):
- OMP runs anywhere and is the reference oracle (Principle III).
- CUDA is cross-compiled to the Jetson fleet only (`duck-stable`/`duck-naughty`); use the
  cross image matching the target's JetPack/CUDA version (`bt-cross:7.2` for JetPack 7.2 /
  CUDA 13.2, `bt-cross:6.1` for legacy JetPack-6 targets). Pre-reflash Jetson results are
  archived and MUST NOT be compared against post-reflash results.
- Vulkan requires an integrated GPU (`kiss-vk` hard-selects `eIntegratedGpu`); run it on
  `rocky-ryzen` or the phones, never on a discrete-only host.

**Formatting gate**: run `just fmt` before committing. CI's `fmt-check` job rejects
unformatted code; `just setup-hooks` installs the local pre-commit hook that catches this
before it reaches CI. Generated code is excluded from formatting review.

**Canonical model shapes**: kernels implementing the AlexNet/CIFAR pipeline MUST match
`docs/instruction-for-ai/04-alexnet-cifar-spec.md` exactly across all three backends.

## Development Workflow

**Branch model**: `dev` is the active development branch; `main` is verified-stable and
public-facing (default branch). Both are protected with required Tier-0 checks. Promote
`dev` → `main` only by PR, only after Tier-0 CI and the relevant fleet gate are green.

**CI gates**: the hosted Tier-0 gate (`fmt-check` + `omp-subset` + `pytest`) is required
on every PR. Self-hosted fleet Tier-1 checks (CUDA/Vulkan/Android) apply when the change
touches that backend or is being promoted to `main`.

**Review discipline**: reviews check for compliance with Principles I–V above in
addition to functional correctness — unrequested abstraction, unrelated diffs, or
untested backend-specific code are grounds for requesting changes, not just style nits.

## Governance

This constitution is the ratified rule set for how work on BetterTogether is done; the
repo-root `CLAUDE.md` is the day-to-day operational playbook derived from it and MUST
stay consistent with it. Where the two conflict, this constitution governs and
`CLAUDE.md` MUST be updated to match.

**Amendments**: proposed via the `/speckit-constitution` workflow (or an equivalent
manual edit followed by the same Sync Impact Report + propagation checklist). Every
amendment MUST update the version per semantic versioning:
- **MAJOR** — backward-incompatible principle removal or redefinition.
- **MINOR** — a new principle or materially expanded section.
- **PATCH** — clarification, wording, or non-semantic fix.

Every amendment MUST re-run the consistency propagation checklist against
`.specify/templates/plan-template.md`, `spec-template.md`, `tasks-template.md`,
`checklist-template.md`, and any command files, updating stale references before the
amendment is considered complete.

**Compliance review**: every plan produced by `/speckit-plan` MUST pass the
"Constitution Check" gate against the current version of this document before Phase 0
research begins, and again after Phase 1 design. Every PR is expected to satisfy
Principles I–V; deviations require explicit justification recorded in the plan's
Complexity Tracking section, not silent omission.

**Version**: 1.0.0 | **Ratified**: 2026-07-03 | **Last Amended**: 2026-07-03
