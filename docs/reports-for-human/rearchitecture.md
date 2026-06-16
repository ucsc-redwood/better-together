# BetterTogether — Re-architecture Plan

Status: **in progress** · Branch: `refactor/framework-device-axis` · Primary axis: **new devices / SoCs**

This document is the north star for turning BetterTogether from a published
research artifact into an extensible framework that other researchers can build
on. The published code (commit `109bcf1`) remains the artifact of record; this
work happens alongside it and is migrated in small, test-guarded steps.

---

## 1. The core insight

BetterTogether is not one program. It is **three tools that communicate through
artifacts on disk**:

```
   BT-Profiler  ──profiling table (CSV)──▶  BT-Optimizer  ──schedule (JSON)──▶  BT-Implementer
   (C++ runtime)                            (Python/z3)                          (C++ runtime)
```

Today those contracts are *implicit and unvalidated*: the CSV column layout, the
schedule JSON shape, and the device/affinity format are conventions duplicated
across C++ and Python with no schema and no validation. **A framework is born
the moment those contracts become explicit, versioned, and validated.** Once the
schemas are real, each tool becomes independently swappable — which is exactly
what lets a follow-up paper reuse the profiler but swap the scheduler, target a
new device, or replace the runtime.

So the spine of the framework is **Layer 0: the contracts** — not a language.

## 2. Target architecture

```
Layer 0  CONTRACTS  (language-neutral JSON Schema, shared by C++ and Python)
         • Device + affinity spec     ← this branch
         • Application / Stage manifest
         • Profiling table
         • Schedule
              validated on both sides; versioned

Layer 1  CORE RUNTIME  (libbettertogether, C++)
         • Stage / Kernel interface + registry  (register a kernel for (stage, backend))
         • Backend interface: OpenMP | Vulkan | CUDA | NNAPI  (uniform dispatch + lifecycle)
         • UsmBuffer memory abstraction  (formalize the existing pmr + allocators)
         • Pipeline executor — the SPSC-queue dispatcher, generic over a Schedule
         • Device registry — loaded from Layer-0 device specs, not hardcoded
         • Profiler driver — runs the registry under isolated / interference load → table

Layer 2  APPLICATIONS AS PLUGINS   tree | cifar-dense | cifar-sparse
         each = its stages + per-backend kernels + appdata, *registered*.
         New app = implement interface + register. No benchmark-harness copy-paste.

Layer 3  OPTIMIZER  (Python package)
         Optimizer interface; the current z3 gapness/tmax solver is ONE strategy.
         ILP / heuristic / dynamic / learned cost models plug in behind the contract.

Layer 4  CLI / ORCHESTRATION
         one `bt profile|optimize|run` CLI, fully parameterized; the justfile
         becomes a thin wrapper. Global state and hardcoded device IDs are removed.
```

**Honest constraint.** The framework can abstract the *harness* (registration,
dispatch, buffer lifecycle, timing, the `pipe/` benchmark scaffolding — the
biggest dedup win). It **cannot** abstract the *kernel math*: a Vulkan compute
shader, a CUDA `__global__`, and an OpenMP loop are irreducibly different code.
The value is a clean, uniform boundary *around* each kernel, not a single
implementation. Dedup has a floor at roughly "everything except the kernel
bodies."

## 3. Why "device" is the first axis

The chosen extension axis is **adding new SoCs**. Today that requires:

- editing and recompiling `builtin-apps/conf.cpp` (the hardcoded `DeviceRegistry`),
- editing hardcoded device IDs in the `justfile` recipes, and
- knowing undocumented conventions (Android paths, the serve→fetch topology).

Goal: **adding a device = dropping a data file**, no recompile. The device spec
(Layer 0) is therefore the first contract to formalize.

While extracting it we found that the top-level `README.md` and `conf.cpp`
**disagree** about several devices (e.g. the OnePlus `9b034f1b` has no super-core
in code; `jetson` is 6 little cores in code vs "4× big" in the README). `conf.cpp`
— the code that produced the published results — is authoritative, and the
extracted specs reflect it. A single data source eliminates this class of drift.

## 4. Migration phases (each shippable, test-guarded)

| Phase | Work | Verifiable how |
|------|------|----------------|
| **0** | Tag `109bcf1` as the reference; build the characterization net (golden device topology, golden schedule JSON for shipped tables, figure regeneration). | tests green |
| **1 (this branch)** | Extract the **device spec** contract: `schemas/device-spec.schema.json`, `devices/*.json`, and `scripts/validate_devices.py` locking the published `conf.cpp` values. | `validate_devices.py` exits 0 |
| **2** | C++ device loader: `DeviceRegistry` reads `devices/*.json` at runtime; delete the hardcoded table in `conf.cpp`. Prove identical core lists vs the golden. | parity test on device + build |
| **3** | Formalize the remaining Layer-0 contracts (profiling table, schedule, app manifest) + Python/C++ validators. | round-trip shipped `data/` |
| **4** | Stage/Kernel/Backend registry in the C++ core; migrate one app (tree); delete the `pipe/` duplication farm. | golden correctness parity |
| **5** | Data-drive the CLI/justfile (no hardcoded IDs/hosts); optimizer behind an interface; docs, CI, `AGENTS.md`, "add your own device/app/backend" tutorials. | CI green |

## 5. Status of this branch

Done (Phase 1, device axis):

- `schemas/device-spec.schema.json` — the formal device contract.
- `devices/*.json` — all 9 devices extracted verbatim from `conf.cpp`.
- `scripts/validate_devices.py` — structural + golden characterization check (runs with no extra deps).
- `devices/README.md` — how to add a device.
- `LICENSE` — MIT (was referenced everywhere but missing).

Next (Phase 2): add the C++ loader so `conf.cpp` consumes `devices/*.json`, then
delete the hardcoded `DeviceRegistry` body. This is the step that makes "add a
device = drop a file" real at runtime, and needs the C++ toolchain to verify.

## 6. Planned schema extensions

- **GPU / accelerator descriptors** in the device spec (backend availability,
  warp/subgroup size, memory). Today the GPU is implicit per backend (phones=vk,
  jetson=cu); the device axis will eventually make it explicit and data-driven.
