# Reports for humans

Status snapshots, audits, decision logs, and forward-looking design docs — the
"why" and "where we are", written for people. These are **dated narratives**, not
instructions; for how to actually build/test/deploy, see
[`../instruction-for-ai/`](../instruction-for-ai/README.md).

| Doc | What it is |
|---|---|
| [`testing-status.md`](testing-status.md) | Audit of the current test suite + the (app × stage × backend × target) coverage matrix and the T0–T4 roadmap |
| [`bugs-found.md`](bugs-found.md) | Real defects surfaced by the differential-oracle test suite, with root cause + fix status |
| [`planning-notes-2026-06-15.md`](planning-notes-2026-06-15.md) | Decision log from the re-architecture working session (reproducibility, hardware, CI, build-system) |
| [`rearchitecture.md`](rearchitecture.md) | The framework vision — 5-layer target architecture, "add a device = drop a data file" |
| [`cmake-migration-rfc.md`](cmake-migration-rfc.md) | RFC: migrate the build from xmake → CMake (build matrix, toolchains, phased plan) |

The project overview / quick-start for newcomers is the repo-root
[`README.md`](../../README.md).
