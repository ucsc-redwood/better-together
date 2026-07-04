---
description: "Task list for: Real Octomap Workload for Tree App"
---

# Tasks: Real Octomap Workload for Tree App

**Input**: Design documents from `/specs/001-octomap-real-workload/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/tree-real-data-contract.md](./contracts/tree-real-data-contract.md), [quickstart.md](./quickstart.md)

**Tests**: Not requested in the spec — this feature is explicitly kept OUT of the
`ctest -L omp`/`<backend>` correctness gates (FR-007, clarification session), so no new
automated test files are added. Verification is via the runnable `quickstart.md`
scenarios instead (Constitution Principle IV, Goal-Driven Verification).

**Execution notes (2026-07-03)**: this environment initially had neither the raw
Freiburg Campus 360 3D scan files nor reachable CUDA/Vulkan fleet hardware. The scan
dataset was fetched from its public source
(`http://ais.informatik.uni-freiburg.de/projects/datasets/fr360/freiburgCampus360_3D.zip`,
CC-BY 3.0) and used to complete T006-T010 and T015 with real evidence (hashes, byte
diffs, timing tables) rather than assumptions. SSH access to the Jetson
(`doremy@duck-stable`) was then made available, which unblocked T016 and T019 too: the
real corpus was cross-compiled, deployed, and run on actual Jetson Orin hardware — 7/7
CUDA differential tests green in both real-data and synthetic mode. **21 of 22 tasks are
now complete and verified with real evidence.** Only Vulkan-on-rocky-ryzen (the second
half of T019) remains outstanding — that hardware was not made reachable this session.

**Post-completion amendment (2026-07-04)**: real-hardware validation surfaced two facts
that changed the shipped defaults after the tasks above were verified: (1) real data's
Morton-key duplication rate means it does ~3x *less* structural work than synthetic data
at equal `n_input`; (2) the pooled profiler (`profiler/tree-{cu,vk}`, `kPoolSize=32`)
multiplies per-instance memory ~32x, which would OOM the Jetson (7.4GB RAM) even at the
originally-shipped 4M default. Corrected: the on-disk corpus is now the full untruncated
dataset (12,154,589 points), and `kRealDataDefaultInputSize` dropped from 4,000,000 to
500,000 (a memory-safety floor, with per-device `BT_TREE_INPUT_SIZE` overrides documented
for capable hardware). Re-verified after the change: local build/tests green
(`ctest -L omp` unaffected), and `test-tree-cu` re-run on `duck-stable` with the new
default and full corpus — 7/7 `TreeDiffCuda` passed. See `research.md` §8, `spec.md`'s
2026-07-04 clarification session, and `docs/instruction-for-ai/05-profiling.md`'s
per-device table for full details. The specific measured numbers quoted in T006-T010 and
T015 below reflect the *original* 4,000,000-point default at the time they were run —
still accurate as measurements, just no longer the shipped default value.

**Organization**: Tasks are grouped by user story (from spec.md) to enable independent
implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on an incomplete task)
- **[Story]**: US1 / US2 / US3, per spec.md's prioritized user stories
- File paths are exact, repo-relative

## Path Conventions

This is an extension of an existing monorepo C++/Python app (`apps/tree`), not a new
project — all paths below are real, existing or planned repo paths (see plan.md's
Project Structure section). There is no `src/`/`tests/` split to choose between.

---

## Phase 1: Setup

**Purpose**: Fix the one pre-existing gap that would otherwise block this feature's own
data-prep step.

- [X] T001 [P] Add `tabulate` as a Python dependency (it's imported by
  `scripts/data_prep/oct.py` but isn't declared anywhere in `pyproject.toml` /
  `optimizer/pyproject.toml` / `uv.lock` today, so the script currently fails at import
  time) — add it to `pyproject.toml`'s dependencies (or `optimizer/pyproject.toml`,
  matching wherever the rest of `scripts/data_prep/`'s deps live) and update `uv.lock`
  via `uv lock`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The shared real-data loading mechanism every user story depends on — the
env-var toggle, the size-selection knob, and the data-prep script that produces the file
they all load. Mirrors the existing `BT_WEIGHTS_DIR` pattern in `apps/cifar-dense`.

**⚠️ CRITICAL**: No user story task can be validated until this phase is complete.

- [X] T002 [P] In `apps/tree/tree_appdata.hpp`: add a `kRealDataDefaultInputSize =
  4'000'000` constant next to `kDefaultInputSize`, and a doc comment describing the
  `BT_TREE_DATA_DIR` / `BT_TREE_INPUT_SIZE` contract (point to
  `specs/001-octomap-real-workload/contracts/tree-real-data-contract.md`).
- [X] T003 [P] Extend `scripts/data_prep/oct.py` with `--concat_target` (int, default
  4000000), `--recenter` (flag), `--domain_min` (float, default 0.0), and
  `--domain_range` (float, default 1024.0) options: consume `scan_NNN_points.dat` files
  in ascending numeric order (independent of `--scan_range`'s listed order), concatenate
  their point columns (`[3:6]`), recenter/scale into `[domain_min, domain_min +
  domain_range)` when `--recenter` is set, truncate to exactly `--concat_target` rows
  once reached, and write the result as `points.npy` (`<f4`, shape `(N, 3)`, C-order) in
  `--output_dir`.
- [X] T004 [P] In `apps/tree/tree_appdata.cpp`: in `AppData`'s constructor, branch on
  `std::getenv("BT_TREE_DATA_DIR")` — if set, load `$BT_TREE_DATA_DIR/points.npy` into
  `u_input_points_s0` via `bt::npy::load_prefix(path, "<f4", {n_input, 3}, ...)`
  (`platform/util/npy_loader.hpp`), filling each point's `w` with `1.0f`; if unset, keep
  today's `mt19937(114514)` synthetic generator exactly as-is. Depends on T002 for the
  new constant's name.
- [X] T005 [P] In `apps/tree/safe_tree_appdata.cpp`'s `HostTreeManager::initialize()`:
  when `std::getenv("BT_TREE_DATA_DIR")` is set, read `BT_TREE_INPUT_SIZE` (falling back
  to `kRealDataDefaultInputSize` if unset) and pass it as `AppData`'s `n_input`; when
  unset, keep passing `kDefaultInputSize` exactly as today. Depends on T002 for the new
  constant's name.

**Checkpoint**: real-data mode compiles; once T003 has produced a `points.npy`, T004/T005
can be exercised end-to-end by any user story below.

---

## Phase 3: User Story 1 - Profile the tree app against a realistic point cloud (Priority: P1) 🎯 MVP

**Goal**: Real-data mode loads correctly, produces profiling records in the existing
schema, is fully deterministic, and fails loud (never silently substitutes synthetic
data) when misconfigured.

**Independent Test**: with `BT_TREE_DATA_DIR` set to a generated `points.npy`, run the
tree app's OMP profiler/benchmark twice and confirm both runs load an identical point
set and complete without error, producing the same profiling-record shape as today's
synthetic runs.

### Implementation for User Story 1

- [X] T006 [US1] Generate a local `points.npy` via the extended
  `scripts/data_prep/oct.py` (quickstart.md step 1): fetched the real dataset from
  `http://ais.informatik.uni-freiburg.de/projects/datasets/fr360/freiburgCampus360_3D.zip`
  (145MB, CC-BY 3.0, 77 scans / 12,154,589 total points), extracted to
  `resources/octomap/freiburgCampus360_3D/` (gitignored, not committed), then ran `uv run
  scripts/data_prep/oct.py --base_dir resources/octomap/freiburgCampus360_3D
  --output_dir resources/octomap/data --scan_range 1-77 --concat_target 4000000
  --recenter --domain_min 0.0 --domain_range 1024.0 --save`. Verified:
  `resources/octomap/data/points.npy` is `(4000000, 3)`, dtype `<f4`, C-contiguous,
  recentered into `[6.05, 1024.0) x [0, 1019.3) x [493.5, 756.7)` (within the
  `[0, 1024)` domain as required).
- [X] T007 [US1] Build (`cmake --preset pc && cmake --build --preset pc`) and run
  `bm-tree-omp` (or the tree profiler binary per
  `docs/instruction-for-ai/05-profiling.md`) with `BT_TREE_DATA_DIR` exported to
  `resources/octomap/data`; confirm it completes without error and its profiling output
  matches the existing per-stage schema/breakdown used by synthetic-mode runs
  (quickstart.md step 2, correctness half). Verified: `BT_TREE_DATA_DIR=.../data
  ./bm-tree-omp --device pc` runs all 7 stages to completion, producing standard
  google-benchmark output identical in shape to synthetic-mode runs.
- [X] T008 [US1] Run the same binary twice with `BT_TREE_DATA_DIR` set (quickstart.md
  step 3) and confirm `n_unique`, `n_brt_nodes`, and `n_octree_nodes` — and the loaded
  `u_input_points_s0` array itself — are identical between the two runs. Verified via a
  scratch harness (built against the compiled `libbt_core.a`, not committed) that
  constructs `tree::AppData` with `BT_TREE_DATA_DIR` set and hashes every loaded point:
  two independent process runs produced identical hash `640d1bd88ec49369` and identical
  first/last points.
- [X] T009 [US1] Point `BT_TREE_DATA_DIR` at a directory with no `points.npy` (or a
  corrupted/wrong-shape one) and confirm `AppData` construction throws
  `std::runtime_error` naming the path and reason, rather than silently falling back to
  the synthetic generator (FR-010 edge case). Verified: `BT_TREE_DATA_DIR=/tmp/empty_dir
  ./test-tree-omp` throws `npy: /tmp/empty_dir/points.npy: cannot open file` for every
  stage test — no silent fallback, no crash.

**Checkpoint**: User Story 1 is fully functional and independently demonstrable — real
data loads, is deterministic, and fails loud on misconfiguration.

---

## Phase 4: User Story 2 - Scale the real workload to give PUs meaningfully more work (Priority: P2)

**Goal**: The real corpus (~4M points, ~10x today's synthetic default) measurably
increases every stage's wall-clock time, its size is configurable per device, and it can
be deployed to and run on fleet targets without exceeding memory.

**Independent Test**: profile the tree app with the T006 corpus and confirm every
stage's measured time is materially higher than a synthetic-mode run at today's default;
confirm a smaller `BT_TREE_INPUT_SIZE` yields an exact-prefix subset; confirm the
dataset deploys to and runs on a real fleet target.

### Implementation for User Story 2

- [X] T010 [US2] Using the T006 corpus, profile the tree app with `BT_TREE_DATA_DIR` set
  vs. unset (quickstart.md step 2, scale-effect half) and confirm every existing
  pipeline stage's measured wall-clock time is materially higher than at today's
  ~300k-point synthetic default (SC-002). Verified (`bm-tree-omp --device pc`, ms,
  real-data-4M vs synthetic-300k): Stage1 5.90 vs 0.427 (13.8x), Stage2 21.5 vs 1.26
  (17x), Stage3 3.46 vs 0.067 (51x), Stage4 23.2 vs 5.76 (4.0x), Stage5 0.790 vs 0.194
  (4.1x), Stage6 0.220 vs 0.055 (4.0x), Stage7 11.9 vs 3.10 (3.8x) — every stage
  materially higher.
- [X] T011 [P] [US2] Add `scripts/deploy-tree-data.sh`, mirroring
  `scripts/deploy-weights.sh`: pushes `$BT_TREE_DATA_SRC/points.npy` (default
  `BT_TREE_DATA_SRC=resources/octomap/data`) to `/tmp/bt/tree-data/` on `jetson`/`rocky`
  (via `ssh $HOST bash -s` + `scp`) and to `/data/local/tmp/bt/tree-data/` on `android
  <serial>` (via `adb push`, stdin redirected from `/dev/null` on every `adb` call per
  this repo's fish/adb gotchas).
- [X] T012 [P] [US2] Add `[ -d /tmp/bt/tree-data ] && export
  BT_TREE_DATA_DIR=/tmp/bt/tree-data` to `scripts/run-on-jetson.sh`, alongside its
  existing `BT_WEIGHTS_DIR` auto-export line.
- [X] T013 [P] [US2] Add the same `BT_TREE_DATA_DIR` auto-export line to
  `scripts/run-on-rocky.sh`.
- [X] T014 [P] [US2] Add the same `BT_TREE_DATA_DIR` auto-export line to
  `scripts/run-on-android.sh` (using `/data/local/tmp/bt/tree-data`).
- [X] T015 [US2] Run locally with `BT_TREE_INPUT_SIZE=500000` (quickstart.md step 5) and
  confirm the loaded 500,000 points are an exact prefix of the full T006 corpus's first
  500,000 points. Verified two ways: (1) a scratch harness loading 500,000 points
  directly byte-diffed against the first 500,000 lines of a 4,000,000-point load —
  identical; (2) `BT_TREE_DATA_DIR=... BT_TREE_INPUT_SIZE=500000 ./bm-tree-omp --device
  pc` completes in 0.703ms for Stage1 vs 5.90ms at 4M points — proportionally consistent
  with an ~8x-smaller prefix.
- [X] T016 [US2] Deploy to one real fleet target and confirm end-to-end (quickstart.md
  step 6): `scripts/deploy-tree-data.sh jetson` then `scripts/run-on-jetson.sh
  test-tree-cu` on `doremy@duck-stable` (an actual Jetson Orin). Verified: the corpus
  staged to `/tmp/bt/tree-data/points.npy` (48MB) on the device; `BT_TREE_DATA_DIR`
  auto-exported correctly (confirmed by checking the deployed dir); the CUDA binary ran
  to completion with no OOM (there is no `bm-tree-cu` target — `apps/tree/CMakeLists.txt`
  only defines a CUDA smoke-runner `run-tree-cu`, no CUDA benchmark — built and ran that
  too). Note: since the corpus was already deployed, this run exercised real-data mode
  directly on Jetson hardware — see T019 for the result.

**Checkpoint**: User Stories 1 AND 2 both work independently — real data loads
deterministically (US1) and demonstrably increases per-stage cost while remaining
configurable and deployable (US2).

---

## Phase 5: User Story 3 - Keep today's fast synthetic tests working (Priority: P3)

**Goal**: Adding real-data mode causes zero regression to today's synthetic-mode
correctness tests, on every backend.

**Independent Test**: with `BT_TREE_DATA_DIR` unset, the existing tree test suite passes
exactly as before, on every backend it runs on today.

### Implementation for User Story 3

- [X] T017 [US3] With `BT_TREE_DATA_DIR` unset, run `ctest --test-dir build/pc -L omp
  --output-on-failure` (quickstart.md step 4) and confirm every existing tree test
  (`test-tree-omp`, `test-pipeline-e2e-omp`) passes unchanged (SC-004). Verified: 9/9
  `ctest -L omp` tests passed, including `test-tree-omp` (0.52s) and
  `test-pipeline-e2e-omp` (20.29s).
- [X] T018 [P] [US3] Grep-audit `apps/tree/{omp,cuda,vulkan}/test_main*.cpp` and
  `test_pipeline_main*.cpp` to confirm none of them set or depend on
  `BT_TREE_DATA_DIR`/`BT_TREE_INPUT_SIZE` — structurally guaranteeing FR-007 (real-data
  mode never enters a `ctest`-registered test) rather than relying on it being true by
  accident. Verified: `grep -rn "BT_TREE_DATA_DIR\|BT_TREE_INPUT_SIZE" apps/tree/`
  matches only the three intentional call sites (`tree_appdata.hpp/.cpp`,
  `safe_tree_appdata.cpp`) — no test file references either var.
- [X] T019 [US3] Where CUDA/Vulkan hardware is available (Jetson / rocky-ryzen per
  `docs/instruction-for-ai/01-hardware.md`), also run `ctest -L cuda` and `ctest -L
  vulkan` for the tree app and confirm they remain green and unaffected by this change.
  **CUDA — verified on `doremy@duck-stable` (real Jetson Orin)**: cross-built
  `test-tree-cu` (bypassing an unrelated, pre-existing `tools/` probe-target /
  stale-CMakeCache libgomp-path issue in the cross container, neither caused by nor part
  of this feature) and ran it two ways: (1) with the real corpus present/auto-exported —
  `TreeDiffCuda` 7/7 PASSED (5962ms total, correctly slower — real 4M-point data); (2)
  with the corpus temporarily moved aside — `TreeDiffCuda` 7/7 PASSED (1015ms total,
  clean synthetic-mode baseline). Both scenarios green, zero regression either way; the
  deployed corpus was restored afterward. **Vulkan (rocky-ryzen)**: still not reachable
  from this session — not run.

**Checkpoint**: All three user stories are independently functional — non-regression is
confirmed across every backend's correctness gate.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and formatting cleanup that spans all three stories.

- [X] T020 [P] Document `BT_TREE_DATA_DIR`/`BT_TREE_INPUT_SIZE`, the `points.npy`
  contract, and `scripts/deploy-tree-data.sh` in
  `docs/instruction-for-ai/05-profiling.md`, alongside the existing profiling how-to
  (Constitution Principle V, Data & Docs as Source of Truth).
- [X] T021 Run `just fmt` across every file touched by T002-T020, then confirm `just
  fmt-check` is clean (CLAUDE.md's quickstart requirement). `clang-format` (C++) and
  `ruff` (Python) both report clean for this feature's files; `shfmt -d` run directly
  against the four touched shell scripts reports no diff. Repo-wide `just fmt-check`
  still exits 1, but solely due to pre-existing formatting debt in vendored
  `.specify/scripts/bash/*.sh` and `.specify/*.json` — `just fmt` touched those too, but
  that reformatting was reverted (`git restore`) as out-of-scope collateral, per this
  repo's Surgical Changes principle; no CMake or JSON file was touched by this feature.
- [X] T022 Run the full `quickstart.md` validation end-to-end, in order (steps 1-6), as
  a final sanity pass before considering the feature done. All 6 steps ran clean: (1)
  real corpus built from the actual Freiburg dataset, 4,000,000 points; (2) load +
  scale-effect confirmed (every stage materially slower at 4M vs 300k); (3) determinism
  confirmed (identical hash across independent runs); (4) `ctest -L omp` 9/9 green with
  the env var unset; (5) `BT_TREE_INPUT_SIZE` exact-prefix rule confirmed; (6) deployed
  to and ran on a real Jetson Orin (`duck-stable`) via `scripts/deploy-tree-data.sh` +
  `scripts/run-on-jetson.sh`, 7/7 CUDA differential tests green in both real-data and
  synthetic mode. Vulkan (rocky-ryzen) was not reached in this session.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup (T001, so `oct.py` in T003 can actually
  run) — BLOCKS all user stories.
- **User Stories (Phase 3-5)**: All depend on Foundational (Phase 2) completion.
  - US1 (T006-T009) has no dependency on US2 or US3.
  - US2 (T010-T016) reuses the T006 corpus generated in US1 but adds no new dependency
    on US1's *code* — only on the artifact. If US2 is worked before US1, T010/T015/T016
    would first need to (re)run T006's generation step themselves.
  - US3 (T017-T019) has no dependency on US1 or US2's artifacts — it only needs
    Foundational's code changes to exist, since it's verifying *those* didn't break
    anything.
- **Polish (Phase 6)**: Depends on all three user stories being complete.

### Within Each Phase

- Foundational: T002 (constants/doc) before T004/T005 (both read the new constant);
  T003 (data-prep script) is independent of T002/T004/T005.
- US1: T006 (generate corpus) before T007/T008/T009 (all need `points.npy` to exist).
- US2: T010/T015 need T006's corpus (already produced in US1); T011-T014 (deploy
  plumbing) are independent of T010/T015 and of each other; T016 depends on T011-T014.
- US3: T017-T019 are independent of each other and of every other phase's artifacts.

### Parallel Opportunities

- T001 (Setup) has no co-tasks to parallelize with.
- T002 and T003 (Foundational) can run in parallel — different files, no shared
  dependency.
- T004 and T005 (Foundational) can run in parallel once T002 is done — different files.
- T011, T012, T013, T014 (US2 deploy plumbing) can all run in parallel — four different
  files.
- T018 (US3 grep-audit) can run in parallel with T017/T019 (different activity, no
  shared file).
- T020 (Polish, docs) can run in parallel with T021 (formatting).

---

## Parallel Example: Foundational Phase

```bash
# Launch once T001 (tabulate dependency) is done:
Task: "In apps/tree/tree_appdata.hpp: add kRealDataDefaultInputSize constant + doc comment"
Task: "Extend scripts/data_prep/oct.py with --concat_target/--recenter/--domain_* flags"

# Then, once the above constant exists:
Task: "In apps/tree/tree_appdata.cpp: AppData ctor branches on BT_TREE_DATA_DIR"
Task: "In apps/tree/safe_tree_appdata.cpp: HostTreeManager::initialize() reads BT_TREE_INPUT_SIZE"
```

## Parallel Example: User Story 2 deploy plumbing

```bash
Task: "Add scripts/deploy-tree-data.sh mirroring scripts/deploy-weights.sh"
Task: "Add BT_TREE_DATA_DIR auto-export line to scripts/run-on-jetson.sh"
Task: "Add BT_TREE_DATA_DIR auto-export line to scripts/run-on-rocky.sh"
Task: "Add BT_TREE_DATA_DIR auto-export line to scripts/run-on-android.sh"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001).
2. Complete Phase 2: Foundational (T002-T005) — the shared loading mechanism.
3. Complete Phase 3: User Story 1 (T006-T009) — real data loads, deterministically,
   failing loud on misconfiguration.
4. **STOP and VALIDATE**: run quickstart.md steps 1-3 independently.
5. This is a demonstrable MVP: the tree app can be profiled against real Octomap data.

### Incremental Delivery

1. Setup + Foundational → shared mechanism ready.
2. Add User Story 1 → validate independently → MVP demo-able.
3. Add User Story 2 → validate independently (scale effect confirmed, deployable to
   fleet).
4. Add User Story 3 → validate independently (non-regression confirmed on every
   backend).
5. Polish → docs + formatting + full quickstart re-run.

### Parallel Team Strategy

With multiple contributors, once Foundational (T002-T005) is done:

- Contributor A: User Story 1 (T006-T009).
- Contributor B: User Story 2's deploy plumbing (T011-T014) — doesn't need US1's
  corpus to exist yet, since it's just adding scripts.
- Contributor C: User Story 3 (T017-T019) — can start as soon as Foundational compiles,
  doesn't need any corpus at all.

---

## Notes

- [P] tasks touch different files and have no dependency on an incomplete task.
- [Story] labels (US1/US2/US3) map every task to its spec.md user story for
  traceability.
- No new automated tests are added — per FR-007 and the clarification session,
  real-data mode is deliberately kept out of `ctest`; verification is the runnable
  `quickstart.md` scenarios (T006-T009, T010, T015-T017, T019, T022).
- Commit after each task or logical group, per this repo's Surgical, Traceable Changes
  principle — each task above already traces to a specific FR/SC/clarification.
- Stop at any checkpoint (end of Phase 2, 3, 4, or 5) to validate that increment
  independently before continuing.
