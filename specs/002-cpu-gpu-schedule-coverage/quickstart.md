# Quickstart: Validate CPU/GPU Schedule Permutation & Overlap Coverage

Proves the feature end-to-end: every valid CPU/GPU schedule for the tree pipeline is
generated, run through the real concurrent runtime on real Jetson hardware, checked for
correctness, and checked for genuine CPU/GPU overlap across repeated runs. See
[`data-model.md`](data-model.md) for entity details and [`research.md`](research.md) for
why each piece is implemented the way it is.

## Prerequisites

- The prior session's genuinely-chained `tree::AppData` dispatch overloads already land
  on this branch (`apps/tree/omp/dispatchers.{hpp,cpp}`, `apps/tree/cuda/dispatchers.{cuh,cu}`)
  — this feature is a new test file, not a redo of that work.
- `bt-cross:7.2` Docker image available (or buildable) for cross-compiling to the Jetson.
- SSH access to `duck-stable` (or `duck-naughty`), per
  `docs/instruction-for-ai/01-hardware.md`.

## 1. Cross-build the new test

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/workspace/build \
  -v "$PWD:/workspace" -w /workspace bt-cross:7.2 bash -lc \
  'cmake --preset jetson && cmake --build --preset jetson --target test-schedule-permutation-cu -j"$(nproc)"'
```

Expected: builds cleanly, producing `build/jetson/test-schedule-permutation-cu`.

## 2. Deploy and run on real Jetson hardware

```bash
scripts/run-on-jetson.sh test-schedule-permutation-cu
```

Expected: the binary generates and prints all 29 schedule permutations, runs each one
5 times through the real concurrent runtime, and reports a correctness + overlap
verdict per schedule. Total run time is bounded (29 schedules × 5 runs) but not
SLA'd — this is an on-demand tool, not part of `ctest -L cuda`.

## 3. Confirm the report is actionable (User Story 3)

Expected in the output for every schedule:
- Its exact GPU stage range (or "all-CPU"/"all-GPU" for the boundary cases).
- A correctness verdict (pass, or which stage/buffer mismatched).
- An overlap verdict (pass / fail / not-applicable for the two boundary schedules),
  based on >= 3 of 5 repeated runs showing measured concurrent execution time.

## 4. Confirm this doesn't affect the routine gates

```bash
ctest --test-dir build/jetson -L cuda --output-on-failure
```

Expected: `test-schedule-permutation-cu` does **not** appear in this run (its `LABELS`
is `experimental`, not `cuda`) — confirming FR-008: this sweep stays out of the routine
correctness gate.
