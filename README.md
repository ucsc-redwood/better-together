# BetterTogether

<div align="center">

![BetterTogether](better-together.png)

**Profile-Guided Software Pipelining for Heterogeneous Edge SoCs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake ≥ 3.25](https://img.shields.io/badge/CMake-%E2%89%A53.25-blue.svg)](https://cmake.org/)

### Supported Backends

<p align="center">
  <img src="https://img.shields.io/badge/Vulkan-AC162C?style=for-the-badge&logo=vulkan&logoColor=white" alt="Vulkan"/>
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/OpenMP-0071C5?style=for-the-badge&logo=openmp&logoColor=white" alt="OpenMP"/>
</p>

**Keywords**: Edge Computing • Heterogeneous Computing • GPU Computing • Pipeline Parallelism • Performance Modeling • Scheduling Algorithms

</div>

---

## Table of Contents

- [Overview](#overview)
- [Evaluated Applications](#evaluated-applications)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration Files](#configuration-files)
- [Advanced Usage](#advanced-usage)
- [Publications & Citation](#publications--citation)

---

## Overview

**BetterTogether** is a flexible scheduling framework that enables fine-grained software pipelining on heterogeneous edge SoCs. While large-scale data processing typically occurs in the cloud, modern mobile and edge devices are increasingly capable—offering benefits like lower latency, reduced energy consumption, and offline availability. However, efficiently utilizing edge SoCs is challenging: they integrate diverse processing units (PUs) such as big.LITTLE CPU architectures, GPUs, and AI accelerators, each with distinct performance characteristics. Furthermore, execution on one PU can interfere with others, complicating performance modeling.

BetterTogether addresses these challenges through **profile-guided performance modeling** that captures intra-application interference, enabling accurate schedule prediction across diverse edge devices. Applications are provided as a sequence of stages (each with CPU and GPU implementations), which are then pipelined across the SoC's processing units.

### Key Highlights

- 🎯 **Accurate Performance Modeling**: Novel profiling technique that accounts for intra-application interference on integrated SoCs
- ⚡ **Pipeline Parallelism**: Maps application stages to the most efficient processing units (CPU big/medium/little cores, GPU, TPU)
- 🧠 **SMT-Based Optimization**: Uses constraint solving (Z3) to generate optimal static pipeline schedules
- 🚀 **Proven Performance**: Achieves **2.14× geomean speedup** (up to **7.59×**) over homogeneous GPU baselines
- 📱 **Cross-Vendor Portability**: Evaluated on Google Pixel 7a (Arm Mali), OnePlus 11 (Qualcomm Adreno), and NVIDIA Jetson Nano
- 🔧 **Multi-Backend Support**: OpenMP (CPU), CUDA (NVIDIA), and Vulkan (cross-platform GPU)
- 🤖 **Extensible**: Demonstrated integration with Google EdgeTPU for AI acceleration

### Motivation

**Why Edge Computing?**
- ⚡ **Lower Latency**: Process data locally without cloud round-trip
- 🔋 **Energy Efficiency**: Reduced data transmission and optimized on-device compute
- 🔒 **Privacy**: Sensitive data stays on device
- 📶 **Offline Operation**: Works without internet connectivity

**The Challenge**:
Modern edge SoCs integrate diverse processing units (big.LITTLE CPUs, GPUs, AI accelerators), but efficiently utilizing them is difficult:
1. **Performance heterogeneity**: Different stages run best on different PUs (see example below)
2. **Interference effects**: Execution on one PU affects others on integrated SoCs
3. **Device diversity**: Optimal schedules differ across devices (ARM, Qualcomm, NVIDIA)

**Example - 3D Octree Construction on Google Pixel 7a**:
- **Sorting** runs fastest on big/medium CPU cores (~3ms)
- **Radix tree building** runs fastest on GPU (~2ms)
- **Octree construction** performs similarly on big cores and GPU (~4ms)

Simply running everything on GPU or CPU leaves significant performance on the table. BetterTogether's pipeline approach achieves **3.5× speedup** over GPU-only for this workload by intelligently mapping stages to optimal PUs.

---

### Project Statistics

Source only (build artifacts, virtualenvs, and the regenerable `data/` store excluded;
`tokei`):

```
===============================================================================
 Language              Files        Lines         Code     Comments       Blanks
===============================================================================
 C++                      56         8649         5749         1365         1535
 C++ Header               70         8192         5375         1663         1154
 C Header                 29         9368         9368            0            0
 CUDA                     40         3585         2196          800          589
 GLSL (compute)           29         2418         1650          366          402
 Python                   47         7026         5174          818         1034
 CMake                    18          754          488          197           69
 JSON / schemas           21          943          940            0            3
 Shell / Just              9          613          360          190           63
 Web (dashboard)           7         1872         1771           64           37
 Markdown                 30         5196            0         4174         1022
===============================================================================
```

---

## Evaluated Applications

BetterTogether has been evaluated on three computer vision workloads with varying computational characteristics:

| Application | Description | Stages | Backends | Compare mode |
|------------|-------------|--------|----------|--------------|
| **cifar-dense** | Dense AlexNet CNN inference on CIFAR-10 | 9 today → 11 canonical | OMP · CUDA · Vulkan | float (`NearEqual`) |
| **cifar-sparse** | Pruned/sparse AlexNet (irregular memory access) | 9 today → 11 canonical | OMP · CUDA · Vulkan | float (`NearEqual`) |
| **tree** | 3D octree construction (morton → sort → unique → radix-tree → edge-count → prefix-sum → octree-build) | 7 | OMP · CUDA · Vulkan | exact (integer/structural) |

> Stage counts reflect the **currently implemented** kernels (`vocab.json` is the single
> source of truth). The canonical `AlexNetCIFAR` spec
> ([`docs/instruction-for-ai/04-alexnet-cifar-spec.md`](docs/instruction-for-ai/04-alexnet-cifar-spec.md))
> is 11 stages; the C++ kernels still implement the 9-stage `SmallAlexNet` and are not yet
> migrated to it.

### Workload Properties

These applications are well-suited for pipeline parallelism because they:

1. **Decompose into stages**: Each application consists of distinct computational phases
2. **Process streaming input**: Operate on independent inputs (e.g., image frames) that arrive continuously
3. **Exhibit heterogeneity**: Different stages have different optimal PUs (as shown in the paper's Figure 1)

### Backends

Each application stage has multiple implementations:
- **OpenMP (CPU)**: Executes on big/medium/little CPU cores with thread affinity
- **CUDA**: For NVIDIA GPUs (Jetson Nano)
- **Vulkan Compute**: Cross-platform GPU compute (Arm Mali, Qualcomm Adreno)

Each application is divided into stages that can be independently scheduled across different processors, enabling BetterTogether to find the optimal mapping for each workload-platform combination.

---

## Requirements

### Core Dependencies

- **[CMake](https://cmake.org/) ≥ 3.25** - Build system (presets-based; deps auto-fetched via CPM)
- **[uv](https://astral.sh/uv)** - Fast Python package manager
- **[just](https://github.com/casey/just)** - Command runner (Rust-based)
- **Python 3.13+** - For scheduling and analysis scripts

### Optional Dependencies

- **CUDA Toolkit** - For NVIDIA GPU support
- **Vulkan SDK** - For cross-platform GPU compute
- **Android NDK** - For Android device support
- **ADB** - For Android device deployment

### Installation

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install just (command runner)
cargo install just

# CMake ≥ 3.25 (build system) — via your package manager, e.g. apt/brew/pip
```

---

## Quick Start

### 1. Build the Project

For x86_64 PC (CPU/OpenMP — the everyday build & test):
```bash
cmake --preset pc && cmake --build --preset pc
ctest --test-dir build/pc -L omp
```

For Vulkan (x86 build, runs on an integrated-GPU box), NVIDIA Jetson (CUDA, cross-compiled
in the container), and Android (ARM64):
```bash
cmake --preset vulkan  && cmake --build --preset vulkan    # iGPU box (e.g. AMD Radeon, Intel)
cmake --preset jetson  && cmake --build --preset jetson    # via bt-cross:6.1 container
cmake --preset android && cmake --build --preset android   # needs ANDROID_NDK_HOME
```

Convenience wrappers (`just build-x86`, `just build-jetson`, `just build-android`, then
`just test`) build and run the unit-test matrix across the fleet. See
[`docs/instruction-for-ai/02-building.md`](docs/instruction-for-ai/02-building.md) for the
full preset/cross-build/deploy details.

### 2. Profile → Schedule → Run → Compare

The paper's "three tools talking through files" — BT-Profiler (C++) → JSONL store →
BT-Optimizer (Python/z3) → schedule JSON → BT-Implementer (C++ runtime). The end-to-end
procedure with every gotcha is in
[`docs/instruction-for-ai/06-end-to-end-scheduling.md`](docs/instruction-for-ai/06-end-to-end-scheduling.md);
the short version:

```bash
# 1. Profile: run the per-(app×backend) bm-prof binary on the device, capture stdout
#    (pure JSONL) to the canonical store: data/profiling/<device>/<app>/<backend>/<scenario>/run-NNN.jsonl
ssh <host> 'cd /tmp/bt && LD_LIBRARY_PATH=. BT_PROF_SCENARIO=interference \
  ./bm-prof-cifar-sparse-vk --device 3A021JEHN02756 2>/dev/null' \
  > data/profiling/3A021JEHN02756/cifar-sparse/vulkan/interference/run-001.jsonl

# 2. Generate schedules: z3 reads the JSONL store directly (no CSV step) and emits candidates
uv run optimizer/orchestrate/02_gen_schedule_merged.py --profiling_root data/profiling \
  --device 3A021JEHN02756 --app cifar-sparse --backend vk --table_type btpm \
  --minimize_mode tmax --num_solutions 10 --output_folder data/schedules_btpm

# 3. Run the schedule(s) on the device and capture per-task timing logs
uv run optimizer/orchestrate/03_run_schedule.py --device 3A021JEHN02756 --app cifar-sparse \
  --backend vk --table-type btpm --minimize-mode tmax --ssh-host <host> --build-dir build/vulkan \
  --log-folder data/sched_logs/3A021JEHN02756_cifar-sparse_vk

# 4. Parse / measure makespan vs the single-PU baseline → speedup
uv run optimizer/orchestrate/04_parse_schedules.py data/sched_logs/3A021JEHN02756_cifar-sparse_vk
```

Everything under `data/` (the JSONL profiling store, generated schedules, run logs, CIFAR
dataset, trained weights) is **regenerable and git-ignored** — it is the output of this
pipeline, not source.

---

## Configuration Files

### Device Configuration

Devices are the framework's primary extension axis: **adding a device = dropping in a data
file.** Each target is a schema-validated JSON in `devices/<id>.json`
([`schemas/device-spec.schema.json`](schemas/device-spec.schema.json)) — the **source of
truth**. At build time `scripts/embed_device_specs.py` codegens these into the C++ device
registry (`platform/registry/generated/device_specs_embedded.hpp`); do **not** hand-edit
`platform/registry/conf.cpp`.

```json
{
  "id": "3A021JEHN02756",
  "description": "Google Pixel 7a (Tensor G2): 4 little + 2 medium + 2 big.",
  "cores": [
    { "id": 0, "type": "little", "pinnable": true },
    { "id": 4, "type": "medium", "pinnable": true },
    { "id": 6, "type": "big",    "pinnable": true }
  ],
  "gpu": { "backend": "vulkan", "name": "Mali-G710", "subgroup_size": 16 }
}
```

**Note**: Find a phone's device ID with `adb devices`. Core types are
`little`/`medium`/`big` (the solver's CPU tiers); the GPU `backend` is `vulkan` or `cuda`.

### Schedule Format

Schedules are the cross-tool contract between BT-Optimizer (z3) and BT-Implementer (C++),
validated against [`schemas/schedule.schema.json`](schemas/schedule.schema.json). A file is
an **array of candidate schedules**; each partitions the application's stages into
contiguous chunks across PUs. Stage numbering is **1-based and inclusive**, and the chunks
must contiguously cover `[1, n_stages]`:

```json
[
  {
    "uid": "SCH-0001",
    "solution_id": 1,
    "chunks": [
      { "core_type": "GPU", "start_stage": 1, "end_stage": 3, "hardware": "gpu_vulkan" },
      { "core_type": "Big", "start_stage": 4, "end_stage": 7 }
    ],
    "metrics": { "max_time": 3.21 }
  }
]
```

`core_type` is one of `Little`/`Medium`/`Big`/`GPU`; a `GPU` chunk additionally requires
`hardware` (`gpu_cuda` or `gpu_vulkan`). `metrics` is advisory — the runtime ignores it.

---

## Project Structure

The tree is **component-first** (the old `builtin-apps/`, `pipe/`, `utility/` were dissolved
into these components by the 2026-06 refactor):

```
better-together/
├── apps/                  # Per-application kernels: omp/ cuda/ vulkan/ + differential oracle
│   ├── cifar-dense/       #   each app provides the same stages in up to 3 backends,
│   ├── cifar-sparse/      #   plus appdata + a *_diff_oracle.hpp for OMP-as-oracle tests
│   └── tree/
├── platform/              # Backend-agnostic substrate
│   ├── engine/            #   cuda/ + vulkan/ compute engines (kiss-vk, UMA buffers)
│   ├── mem/               #   memory-resource helpers
│   ├── registry/          #   device registry (generated/ from devices/*.json)
│   ├── vocab/             #   generated/ vocabulary header (from vocab.json)
│   └── util/              #   ndarray, npy loader, resource paths, debug logging
├── runtime/               # BT-Implementer: app-agnostic pipeline engine
│   │                      #   (SPSC queues, schedule/config readers, task dispatch)
│   └── tests/
├── profiler/              # BT-Profiler: bm-prof / bm-baseline / bm-gen-logs drivers
│                          #   + per-(app×backend) sources (cifar-dense-vk/, tree-cu/, …)
├── optimizer/             # BT-Optimizer (Python package): z3 SMT scheduler
│   ├── smt/               #   solver, constraints, profiling loader, vocab
│   ├── orchestrate/       #   02_gen_schedule / 03_run_schedule / 04_parse / 05_timeline
│   └── analysis/          #   coverage + isolated-table rendering
├── tools/                 # Standalone probes (affinity, cpuinfo, Vulkan version, stress)
├── devices/               # Per-device topology specs (*.json, schema-validated, the SoT)
├── schemas/               # JSON Schemas: device-spec, profiling-table, schedule (contracts)
├── vocab.json             # Single source of truth for PU tiers / backends / stage counts
├── cmake/                 # Toolchains, CPM, per-backend target helpers
├── scripts/               # Build/deploy wrappers, codegen, device-spec validation, figures
├── resources/             # Model weights / CIFAR params (regenerable; see scripts/data_prep)
├── dashboard/             # Static offline analysis site (devices / apps / profiling / schedules)
├── docs/                  # instruction-for-ai/ (how-to) + reports-for-human/ (status)
└── data/                  # Profiling store + schedules + run logs (git-ignored, regenerable)
```

> **Working on the code (human or AI)?** Start with
> [`docs/instruction-for-ai/README.md`](docs/instruction-for-ai/README.md) —
> project goal, hardware & access, build, test, and the canonical model spec.
> Status, audits, and roadmaps are in [`docs/reports-for-human/`](docs/reports-for-human/).

---

## Advanced Usage

### Custom Devices

To add a new device, drop a schema-validated JSON into `devices/<id>.json` (no C++ edits) —
the registry is codegenned from it at build time:

```bash
# 1. write devices/my_device_id.json (cores[] + gpu{}, per schemas/device-spec.schema.json)
# 2. validate it
uv run scripts/validate_devices.py
# 3. rebuild — scripts/embed_device_specs.py regenerates the embedded registry
cmake --build --preset pc
```

### Optimization Modes

BetterTogether supports multiple optimization strategies:

| Mode | Description |
|------|-------------|
| `btpm + gapness` | Better-Together Profiling Model, minimize scheduling gaps |
| `btpm + tmax` | Better-Together Profiling Model, minimize maximum chunk time |
| `isolated + gapness` | Isolated execution model, minimize gaps |
| `isolated + tmax` | Isolated execution model, minimize max time |

### Visualization

Visualize a generated schedule:

```bash
uv run scripts/view/view_schedule.py data/schedules_btpm/<device>/<app>/<backend>/schedules_btpm_tmax.json
```

Render an execution timeline from the run logs:

```bash
uv run optimizer/orchestrate/05_timeline.py data/sched_logs/<device>_<app>_<backend>
```

### Analysis Dashboard

The repo ships a self-contained, **offline static dashboard** (`dashboard/`) that
cross-references the whole project: the device fleet, the per-app stage breakdowns, the
collected profiling tables, and a schedule explorer (z3 chunk assignments with measured
speedup-over-baseline). It builds to a single bundle and can be served locally (e.g. over
Tailscale) with no backend. See [`docs/reports-for-human/`](docs/reports-for-human/) for how
to generate and serve it.

---

## Publications & Citation

Presented at **IISWC 2025** (Xu et al., UCSC / Microsoft Research). The paper is included in
this repo: [`IISWC_2025_BetterTogether_Yanwen.pdf`](IISWC_2025_BetterTogether_Yanwen.pdf).

```bibtex
@inproceedings{xu2025bettertogether,
  title     = {BetterTogether: Profile-Guided Software Pipelining for Heterogeneous Edge SoCs},
  author    = {Xu, Yanwen and others},
  booktitle = {IEEE International Symposium on Workload Characterization (IISWC)},
  year      = {2025}
}
```

---

## License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

<div align="center">

**[Documentation](docs/instruction-for-ai/README.md) • [Issues](https://github.com/ucsc-redwood/better-together/issues) • [Paper](IISWC_2025_BetterTogether_Yanwen.pdf)**

</div>
