# CLAUDE.md — BetterTogether

Profile-guided software-pipelining framework for heterogeneous edge SoCs. An app is a
sequence of **stages**; each stage has a kernel in up to three backends (**OMP** CPU,
**CUDA**, **Vulkan**). Stages are profiled under interference, assigned to processing
units by a z3 SMT solver, and pipelined. It's really three tools talking through files:
**BT-Profiler → JSONL profiling store → BT-Optimizer (Python/z3) → schedule JSON → BT-Implementer (C++)**.

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

- **Build system:** **CMake** (`cmake/` + presets `pc`/`jetson`/`vulkan`/`android`) is
  the build system. xmake was retired 2026-06-16; no `xmake.lua` remains. (The unported
  volk diagnostics + NNAPI targets were removed as dead code 2026-06-18 — recover from
  git history if ever needed.)
- **Where each backend runs:** OMP runs **anywhere** and is the **reference oracle**.
  CUDA is **cross-compiled to the Jetsons** — two Orin Nano Super devkits,
  `duck-stable`/`duck-naughty` (`ssh doremy@duck-{stable,naughty}`), **reflashed to
  JetPack 7.2 / CUDA 13.2 on 2026-07-01**; pre-reflash results are archived and not
  comparable. Default cross image: `bt-cross:7.2` (CUDA 13.2, official SBSA cross
  toolchain + arm64 Vulkan — matches the fleet; validated 2026-07-02).
  `bt-cross:6.1` (CUDA 12.6) is the legacy image for JetPack-6 targets.
  Vulkan needs an **integrated GPU** — `kiss-vk` hard-selects
  `eIntegratedGpu`, so the discrete RTX throws "No integrated GPU found"; run Vulkan on
  `rocky-ryzen` or the phones.
- **Remote shell gotcha:** `rocky-ryzen`'s login shell is **fish** → wrap remote commands
  in `bash -lc '…'`. Hosts/serials/access: [`01-hardware.md`](docs/instruction-for-ai/01-hardware.md).
- **Testing = OMP-as-oracle differential**, fixed seed `114514`: exact compare for
  integer/structural stages (tree), `NearEqual` for float (cifar). `ctest -L omp`
  is the everyday gate; a `GTEST_SKIP` counts as a pass.
- **Canonical model MIGRATED (2026-07-02):** both cifar apps implement the canonical
  11-stage `AlexNetCIFAR` of [`04-alexnet-cifar-spec.md`](docs/instruction-for-ai/04-alexnet-cifar-spec.md)
  (verified on all three backends on real HW). Real trained weights live in
  `saved_params/export/` (dense 90.48% / sparse@25%-density 90.58% test acc, regenerate
  via `scripts/data_prep/{alexnet_cifar10,prune_alexnet_cifar10}.py`); shipped AppData
  still seeds synthetic weights — .npy loading is the remaining wiring.
- **Device topology source of truth** is `devices/*.json` (schema-validated), **not** the
  README — where they disagreed, `conf.cpp`/the JSON win.

## Build & test (quickstart)

```bash
just setup-hooks           # once per clone: pre-commit then runs `just fmt-check`
cmake --preset pc          # CPU/OpenMP only; deps auto-fetched via CPM
cmake --build --preset pc
ctest --test-dir build/pc -L omp --output-on-failure   # the everyday gate
```

Run `just fmt` before committing — CI's fmt-check job rejects unformatted code
(the pre-commit hook catches it locally once installed).

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
