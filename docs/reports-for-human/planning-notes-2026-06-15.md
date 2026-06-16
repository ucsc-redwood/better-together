# BetterTogether — Planning & Design Notes (2026-06-15)

A decision log from a working session on turning BetterTogether from a published
research artifact into a reproducible, extensible framework. This captures the
findings, decisions, and roadmap so future work (human or agentic) has context.

Related: [`rearchitecture.md`](rearchitecture.md) (the framework plan).

---

## 0. Decisions made this session

| Question | Decision |
|---|---|
| Refactor ambition | **Re-architect into a reusable framework** (not just cleanup). |
| Timeline driver | **Already published; improving for the future** — no deadline. |
| Primary extension axis | **New devices / SoCs** — "add a device = drop in a data file." |
| Migration style | **Strangler-fig**, never a big-bang rewrite; every step test-guarded. |
| Build system | **Leaning toward migrating xmake → canonical CMake** — but only *after* a characterization test net exists. |
| Work location | Branch `refactor/framework-device-axis`. |

---

## 1. Project recap

BetterTogether (IISWC 2025, Xu et al., UCSC / Microsoft Research) is an
interference-aware framework for fine-grained software pipelining on
heterogeneous edge SoCs. It maps the stages of streaming vision workloads across
CPU (big/medium/little via OpenMP), GPU (Vulkan / CUDA), and accelerators (TPU
via NNAPI). Core contribution: **profiling each stage under realistic background
load** (interference-aware), which predicts real pipeline latency far better
than isolated profiling, then solving stage→PU assignment with z3 (SMT).
Reported: **2.14× geomean, up to 7.59×** over homogeneous GPU baselines.

It is really **three tools talking through files**: BT-Profiler → (profiling
table CSV) → BT-Optimizer (Python/z3) → (schedule JSON) → BT-Implementer
(C++ runtime with SPSC-queue dispatchers + UMA buffers).

## 2. Reproducibility assessment

**Working:** deterministic seeded inputs (tree/octree use seed `114514`; CIFAR
weights + input batches committed); 245 committed data files (the paper's
measurements); pre-compiled Vulkan `.spv`/`.h` shaders; `uv.lock` pins the
Python side; xmake auto-resolves C++ deps; figure scripts + outputs ship.

**Missing / fixed this session:**
- No `LICENSE` despite MIT claims everywhere → **added** (MIT).
- README Quick Start commands (`just collect`, `just gen-schedule`, …) **do not
  exist** in the justfile — doc/justfile contract is broken.
- `conf.cpp` and `README.md` **disagree** on several device topologies — code is
  authoritative.
- Hardcoded host/paths: `yanwen@android-dev.ucsc`, `http://192.168.1.64:8080`,
  NDK path, device IDs baked into recipes.
- No CI, no tests-as-gate, no env/version capture, no `CITATION.cff`.
- Stray `CLAUDE.md` claims the repo is Rust (it is not).

**Reproduction tiers** (the framing to adopt):
- **Tier 1 — analysis only, any laptop:** run z3 on shipped tables → schedules →
  regenerate figures. Validates the core scientific claim with no edge hardware.
  *Make this one command.*
- **Tier 2 — pipeline on *a* supported device.**
- **Tier 3 — reproduce the paper numbers** (needs the exact phones + Jetson).

## 3. Hardware inventory & data coverage

| Device ID | Model / SoC | Backend | Data ships? |
|---|---|---|---|
| `3A021JEHN02756` | Google Pixel 7a (Tensor G2; X1/A78/A55; Mali-G710) | Vulkan (+ Edge TPU expt) | yes |
| `9b034f1b` | OnePlus 11 (SD 8 Gen 2; X3/A715/A710/A510; Adreno 740) | Vulkan | yes |
| `jetson` | NVIDIA Jetson Orin Nano (6× A78AE; Ampere) | CUDA | yes |
| `jetsonlowpower` | Jetson Orin Nano, 7W (4× A78AE) | CUDA | **NO data committed** ⚠ |
| `R5CY21Y3VEV` | Samsung Galaxy (reported SM-S926B) | Vulkan | yes (**not in paper figures** ⚠) |

Two gaps worth knowing: the **Jetson low-power** results are in the paper but the
data is **not in the repo**; the **Samsung Galaxy** has full data but is **not
reported** in the paper.

Host/orchestration: a Linux x86 box driving phones over `adb` (the `justfile`
`connect` SSHes to `yanwen@android-dev.ucsc`); Android NDK r29.

## 4. Re-architecture plan (summary)

See [`rearchitecture.md`](rearchitecture.md). The spine is **Layer 0: explicit,
versioned, validated data contracts** (device spec, profiling table, schedule,
app manifest) shared by C++ and Python — because the framework is born when the
implicit file contracts between the three tools become explicit and swappable.

Phases: (0) lock behavior with a characterization net → (1) device-spec contract
[**done this session**] → (2) C++ device loader reads `devices/*.json`, delete
the hardcoded table → (3) remaining contracts → (4) Stage/Kernel/Backend
registry + delete the `pipe/` duplication farm → (5) data-driven CLI, optimizer
behind an interface, docs/CI/AGENTS.md.

**Honest constraint:** the framework can abstract the *harness* (registration,
dispatch, buffer lifecycle, timing) but **not the kernel math** — a Vulkan
shader, a CUDA `__global__`, and an OpenMP loop are irreducibly different code.

### Phase 1 delivered (this branch)
- `schemas/device-spec.schema.json` — the device contract.
- `devices/*.json` (×9) — extracted verbatim from `conf.cpp@109bcf1`.
- `scripts/validate_devices.py` — structural + golden characterization check (runs with no extra deps; currently green).
- `devices/README.md`, `LICENSE`, `REARCHITECTURE.md`.

## 5. Reproduction & containerization (Docker / dev container)

Containers solve **Tier 1 + builds**, never the mobile-GPU measurement.
- **Win:** Python analysis (z3/pandas/matplotlib, already pinned) and the C++
  build + CPU/OpenMP path containerize cleanly. A devcontainer doubles as a
  consistent, "agent-ready" toolchain.
- **Wall:** Mali/Adreno are inside the phones (unreachable); CUDA-in-Docker is
  Linux-x86-NVIDIA only; Vulkan needs ICD passthrough.
- **Recommendation:** ship a `Dockerfile` (substrate) + `.devcontainer/`
  (wrapper) scoped to Tier-1 + build; keep Android/Jetson as documented
  bring-your-own-device flows.

## 6. Testing strategy

**Key idea — the CPU/OpenMP path is the oracle.** A backend kernel is correct iff
it matches the CPU reference on the same (seeded) input within tolerance. This
**decouples correctness from hardware**: you need the real GPU only for *speed*,
not for *correctness*. Integer/structural stages (Morton, radix tree, octree,
sort) → exact equality; float stages (conv/linear) → `EXPECT_NEAR(tol)`.

Trust chain: ground truth (CIFAR labels / octree invariants) → validates CPU
oracle → differential check validates CUDA / Vulkan / NNAPI.

The Stage/Kernel registry (Phase 4) makes the differential harness **generic** —
it enumerates `(stage, backend)` and a new backend gets tested for free.

## 7. CI/CD runner topology

CI machine assignment = GitHub **hosted** vs **self-hosted** runners + **labels**.
It is **not** "N backends = N servers": all *builds* and the software-testable
backends fit on one free hosted runner.

| Tier | Runner | Triggers | Covers |
|---|---|---|---|
| 0 | hosted `ubuntu-latest` | every PR | Python/data/optimizer; CPU build+run; **all backends build-only**; Vulkan via **lavapipe**; shader/`spirv-val` checks; sanitizers (CPU) |
| 1 | self-hosted **NVIDIA dGPU** box | nightly / manual | **All three backends at once**: CPU + CUDA (CUDA driver) + Vulkan (NVIDIA ICD) |
| 2 | Jetson + adb-host (phones) | manual / release | Real-device profile→schedule→execute, perf numbers |

**A single NVIDIA dGPU Linux server runs/correctness-tests all three backends**
(it provides both a CUDA driver and a Vulkan ICD). Caveats: it only exercises the
**warp32** subgroup variant (the `radixsort_warp16/64` shaders need their target
mobile GPUs), and it cannot catch Mali/Adreno-driver-specific bugs — but those
are Layer-2 concerns.

**Two layers / public-private split:**
- **Layer 1 — correctness:** generic, can be public (with gating); the NVIDIA box.
- **Layer 2 — actual data / perf:** **private**; Jetson + phones on own servers.
  **Android must not be public.** Cleanest: a private repo (or private,
  `workflow_dispatch`-only workflows) drives the device farm; the public repo
  holds code + Layer-1 correctness.
- **Security:** a self-hosted runner on a public repo must **never** be triggered
  by fork `pull_request`; gate behind manual/scheduled triggers + environment
  protection.

## 8. Build system: xmake → CMake?

**Leaning yes** for a public, long-lived, reproducible framework — but for the
*right* reasons:
- The strong reasons are **external**: CMake ubiquity (reviewer familiarity =
  lower reproduction barrier), CTest/CI maturity, first-class CUDA
  (`CMAKE_CUDA_ARCHITECTURES`), `find_package(Vulkan)`, the NDK's native
  `android.toolchain.cmake`, and `compile_commands.json` (better clangd/agent
  introspection).
- "AI makes CMake easy" is *not* the reason (AI could maintain xmake too) — AI
  merely **removes the historical cost** that justified xmake.
- **What you lose:** xmake's superb package manager. Closest CMake ergonomics =
  **CPM.cmake** (or vcpkg manifest).
- **Non-negotiable sequencing:** do **not** migrate the build before the
  characterization net exists — a build migration perturbs flags / arch codes /
  optimization / link order, which can silently change numerical results. Net
  first → migrate one target (CPU) → tests prove parity → expand.
- If the project were private/personal, staying on xmake is fine; the public
  framework goal is what justifies the switch.

## 9. Existing tests (current state)

There **is** a real gtest suite — 13 files (~5,500 lines, ~722 assertions), one
`test_main.{cpp,cu}` per (app × backend), registered as xmake `test-<app>-<backend>`
targets (`set_group("test")`). **But it is almost entirely smoke + shape:**
- ~320 `EXPECT_NO_THROW` ("it ran, didn't crash").
- `EXPECT_EQ` mostly on tensor dimensions and integer/structural results.
- `EXPECT_TRUE(is_different)` ("the buffer was written").
- **Zero** float-tolerance comparisons (`EXPECT_NEAR`/`FLOAT_EQ` count = 0).
- **No oracle / cross-backend comparison** — each backend self-tests in isolation;
  nothing checks that CUDA/Vulkan match CPU or any golden.
- No aggregate runner / CTest / CI; run ad hoc via `xmake run test-...`.

So the skeleton (per-stage, per-backend) is already there; the upgrade is to turn
`EXPECT_NO_THROW` into differential-vs-oracle checks with tolerance.

## 10. Roadmap / recommended next steps

1. **Phase 0 net + Tier-0 CI (verifiable on a laptop):** Python/schema/optimizer
   tests + figure smoke + a **pure-CPU build preset** (missing today). Stand up
   `.github/workflows/` Tier-0.
2. **Differential oracle harness:** convert one app's OMP test (e.g. cifar-dense)
   to golden + tolerance as the template; generalize across backends.
3. **Phase 2:** C++ device loader consumes `devices/*.json`; delete the hardcoded
   `DeviceRegistry`.
4. **Add the NVIDIA box** as the Layer-1 self-hosted runner (all 3 backends).
5. **Private Layer-2** for Jetson + phones (`workflow_dispatch` only).
6. **Then** the CMake migration (CPM.cmake), CPU target first, tests prove parity.
7. Fix doc drift (README↔justfile↔conf.cpp), add `CITATION.cff`, replace `CLAUDE.md`.

## 11. Open questions

- Where will the Jetson low-power data come from (re-collect, or is it archived)?
- Is the Samsung Galaxy intentionally unreported, or should it be added?
- Private Layer-2: separate private repo vs. gated single-repo workflows?
- CMake dependency strategy: CPM.cmake vs vcpkg manifest?
