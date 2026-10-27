# Reports for humans

Status snapshots, audits, decision logs, and forward-looking design docs — the
"why" and "where we are", written for people. These are **dated narratives**, not
instructions; for how to actually build/test/deploy, see
[`../instruction-for-ai/`](../instruction-for-ai/README.md).

| Doc | What it is |
|---|---|
| [`2026-10-23-coordinated-corun.md`](2026-10-23-coordinated-corun.md) | **Latest:** the coordinated co-run study's go/no-go (Samsung llama.cpp 1B/3B, Pixel ExecuTorch 1B): GPU pacing, prefill ubatch and CPU decode width swept for the specialized pipelines and the data-parallel baseline alike, trade-off curves, and the prediction grid with the best coordinated settings on both sides. No-go: at most 0.99 against the 1.2 gate (1B, P=512, D=128, CPU prefill and GPU decode at duty 1.0); coordination raises data-parallel enough that the best gain falls (1.09 with uncoordinated settings), and the largest effect is the CPU decode's core set (keeping off the prime core), not pacing |
| [`2026-10-21-llm-phase-affinity.md`](2026-10-21-llm-phase-affinity.md) | the LLM phase-affinity study's go/no-go on both phones (llama.cpp and ExecuTorch, Llama 3.2 1B/3B): per-PU prefill and decode rates, co-run interference at full overlap, read bandwidth, and the predicted specialized-pipeline vs data-parallel grid. Go, narrowly: 1.23 against the 1.2 gate for the Samsung, ExecuTorch, 3B, P=2048, D=512, CPU prefill and GPU decode (1.18 against the GPU alone); every other cell is at most 1.03 |
| [`2026-10-20-end-to-end-refresh.md`](2026-10-20-end-to-end-refresh.md) | the October campaign end to end on both phones (re-profile, solve, execute, baselines): before (2026-10-17) vs now for every (phone, app, table) cell, which changes moved which cells, what got worse, and the largest remaining error (the GPU-chunk price) |
| [`2026-10-19-tree-cpu-pinning-gate.md`](2026-10-19-tree-cpu-pinning-gate.md) | the tree-cpu-pinning acceptance gate on both phones: with tree's CPU chunks pinned to their tier and tree re-profiled, its CPU-chunk error falls in all four (phone, table) cells and no best period is worse; the remaining tree error is the btpm GPU column |
| [`2026-10-18-cpu-chunk-costs-gate.md`](2026-10-18-cpu-chunk-costs-gate.md) | the cpu-chunk-costs acceptance gate on both phones: tree stage 2 profiled on intact input lowers tree's CPU-chunk error in every cell; the fitted CPU overhead fails the held-out check on both phones and is not shipped; the cifar bound fails as written in 5 cells |
| [`2026-10-17-gpu-chunk-cost-model-gate.md`](2026-10-17-gpu-chunk-cost-model-gate.md) | the gpu-chunk-cost-model acceptance gate on both phones: range-priced GPU chunks vs the per-stage sum, every (phone, app, table) cell; the per-stage model stays the default |
| [`2026-10-16-mali-conv-kernels.md`](2026-10-16-mali-conv-kernels.md) | the Mali (Pixel 7a) paths of the two cifar 3x3 convs: output blocking shipped (dense -47 %, sparse -51 % of the stage sum), three shared-memory variants measured and rejected, with every table |
| [`2026-10-16-vk-data-flow-profile.md`](2026-10-16-vk-data-flow-profile.md) | where the Vulkan data-flow time goes: host cache maintenance, GPU-thread placement, a coherent-memory experiment, per device (two phones, not rocky) |
| [`2026-10-15-vk-shader-profile.md`](2026-10-15-vk-shader-profile.md) | Vulkan shader profile and the ranked optimization plan, with the per-device results of the three that were applied |
| [`2026-10-15-vk-engine-runtime-decisions.md`](2026-10-15-vk-engine-runtime-decisions.md) | the vk-engine-runtime change's decisions with their measurements: conditional barriers, Pixel weights placement, queue defaults, several Vulkan GPU chunks, phase timing, in-flight depth, the overhead refit |
| [`session-2026-07-02-jetpack72-fresh-start-and-overhead-model.md`](session-2026-07-02-jetpack72-fresh-start-and-overhead-model.md) | JetPack 7.2 fleet refresh (duck-stable/naughty), fresh-start archive + 12/12 matrix re-verified, CUDA-13 CUB port + `bt-cross:7.2` default, fresh baseline, and the fitted per-chunk overhead cost model (P2) |
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
