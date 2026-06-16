# CLAUDE.md — BetterTogether

Profile-guided software-pipelining framework for heterogeneous edge SoCs. An app is a
sequence of **stages**; each stage has a kernel in up to three backends (**OMP** CPU,
**CUDA**, **Vulkan**). Stages are profiled under interference, assigned to processing
units by a z3 SMT solver, and pipelined. It's really three tools talking through files:
**BT-Profiler → CSV → BT-Optimizer (Python/z3) → schedule JSON → BT-Implementer (C++)**.

## Start here

> **Before doing anything on this repo, read
> [`docs/instruction-for-ai/README.md`](docs/instruction-for-ai/README.md).**

The canonical, load-bearing how-to lives in [`docs/instruction-for-ai/`](docs/instruction-for-ai/):

- [`00-project-goal.md`](docs/instruction-for-ai/00-project-goal.md) — what it is, the apps, what "done" means
- [`01-hardware.md`](docs/instruction-for-ai/01-hardware.md) — the test fleet: specs, roles, **and how to ssh/adb in & deploy**
- [`02-building.md`](docs/instruction-for-ai/02-building.md) — CMake presets, cross-compile recipes
- [`03-unit-testing.md`](docs/instruction-for-ai/03-unit-testing.md) — how to run & write tests (OMP-as-oracle)
- [`04-alexnet-cifar-spec.md`](docs/instruction-for-ai/04-alexnet-cifar-spec.md) — canonical model shapes
- [`05-profiling.md`](docs/instruction-for-ai/05-profiling.md) — CLI/agent-driven runtime-overhead profiling: tools per backend/target

Status, audits, decision logs, and roadmaps (the *why* and *where we are*) are in
[`docs/reports-for-human/`](docs/reports-for-human/). Newcomer overview: root
[`README.md`](README.md).

## Essential facts (don't trip on these)

- **Build system:** `xmake` is the **source of truth** (ships the paper results); the
  CMake build (`cmake/` + presets) is a **work-in-progress migration**. Don't touch
  `xmake.lua` unless asked.
- **Where each backend runs:** OMP runs **anywhere** and is the **reference oracle**.
  CUDA is **cross-compiled to the Jetson** (the PC is build-only — CUDA 13 breaks the
  build via CUB removal). Vulkan needs an **integrated GPU** — `kiss-vk` hard-selects
  `eIntegratedGpu`, so the discrete RTX throws "No integrated GPU found"; run Vulkan on
  `rocky-ryzen` or the phones.
- **Remote shell gotcha:** `rocky-ryzen`'s login shell is **fish** → wrap remote commands
  in `bash -lc '…'`. Hosts/serials/access: [`01-hardware.md`](docs/instruction-for-ai/01-hardware.md).
- **Testing = OMP-as-oracle differential**, fixed seed `114514`: exact compare for
  integer/structural stages (tree/octree), `NearEqual` for float (cifar). `ctest -L omp`
  is the everyday gate; a `GTEST_SKIP` counts as a pass.
- **Canonical model not yet migrated:** [`04-alexnet-cifar-spec.md`](docs/instruction-for-ai/04-alexnet-cifar-spec.md)
  makes `AlexNetCIFAR` (11 stages) canonical, but the C++ kernels still implement the old
  `SmallAlexNet` (9 stages) — they are **not** shape/weight-compatible.
- **Device topology source of truth** is `devices/*.json` (schema-validated), **not** the
  README — where they disagreed, `conf.cpp`/the JSON win.

## Build & test (quickstart)

```bash
cmake --preset pc          # CPU/OpenMP only; deps auto-fetched via CPM
cmake --build --preset pc
ctest --test-dir build/pc -L omp --output-on-failure   # the everyday gate
```

Jetson (CUDA), rocky-ryzen (Vulkan), and Android recipes:
[`02-building.md`](docs/instruction-for-ai/02-building.md) ·
[`03-unit-testing.md`](docs/instruction-for-ai/03-unit-testing.md).

## How to work here

**Think before coding.** State assumptions; if multiple interpretations exist, surface
them instead of silently picking; if a simpler approach exists, say so. When something is
unclear, stop and ask.

**Simplicity first.** Minimum code that solves the problem — nothing speculative. No
abstractions for single-use code, no unrequested "flexibility", no error handling for
impossible cases. If 200 lines could be 50, rewrite it.

**Surgical changes.** Touch only what the task requires; match existing style. Don't
refactor or reformat adjacent code that isn't broken. Remove orphans *your* change
created; leave pre-existing dead code (mention it, don't delete it). Every changed line
should trace to the request.

**Goal-driven execution.** Turn the task into a verifiable goal and loop until it's met —
"fix the bug" → write a test that reproduces it, then make it pass; "refactor X" → tests
green before and after. On this repo the loop is concrete: a change isn't done until the
relevant `ctest -L <backend>` is green on its target (at minimum `ctest -L omp` locally).
