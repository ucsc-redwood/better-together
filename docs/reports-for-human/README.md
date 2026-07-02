# Reports for humans

Status snapshots, audits, decision logs, and forward-looking design docs — the
"why" and "where we are", written for people. These are **dated narratives**, not
instructions; for how to actually build/test/deploy, see
[`../instruction-for-ai/`](../instruction-for-ai/README.md).

| Doc | What it is |
|---|---|
| [`session-2026-07-02-jetpack72-fresh-start-and-overhead-model.md`](session-2026-07-02-jetpack72-fresh-start-and-overhead-model.md) | **Latest:** JetPack 7.2 fleet refresh (duck-stable/naughty), fresh-start archive + 12/12 matrix re-verified, CUDA-13 CUB port + `bt-cross:7.2` default, fresh baseline, and the fitted per-chunk overhead cost model (P2) |
| [`testing-status.md`](testing-status.md) | Audit of the current test suite + the (app × stage × backend × target) coverage matrix and the T0–T4 roadmap |
| [`bugs-found.md`](bugs-found.md) | Real defects surfaced by the differential-oracle test suite, with root cause + fix status |
| [`planning-notes-2026-06-15.md`](planning-notes-2026-06-15.md) | Decision log from the re-architecture working session (reproducibility, hardware, CI, build-system) |
| [`rearchitecture.md`](rearchitecture.md) | The framework vision — 5-layer target architecture, "add a device = drop a data file" |
| [`target-structure.md`](target-structure.md) | **Next TODO:** concrete directory-level target structure + phased migration (P0–P6, gated) realizing the rearchitecture vision |
| [`cmake-migration-rfc.md`](cmake-migration-rfc.md) | RFC: migrate the build from xmake → CMake (build matrix, toolchains, phased plan) |
| [`perf-results/`](perf-results/README.md) | Dated performance-measurement campaigns on the fleet (method + results + insights), e.g. RGA static shader analysis on the Radeon 780M |

The project overview / quick-start for newcomers is the repo-root
[`README.md`](../../README.md).
| [`code-improvement-plan.md`](code-improvement-plan.md) | **Resume here:** phased execution plan for the review findings (gates per change, risk order) |
