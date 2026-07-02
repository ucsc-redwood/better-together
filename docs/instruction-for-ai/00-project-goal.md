# Project goal

**BetterTogether** is a profile-guided software-pipelining framework for
heterogeneous edge SoCs (phones, Jetson). Edge SoCs pack diverse processing units
(big/medium/little CPU cores via OpenMP, GPU via Vulkan/CUDA, AI accelerators), and
running a whole workload on one unit leaves performance on the table — but the units
**interfere** with each other, so naive offloading mispredicts latency.

The framework's contribution: **profile each application stage under realistic
background load** (interference-aware), which predicts real pipeline latency far
better than isolated profiling, then solve stage→processing-unit assignment with an
SMT solver. Reported result: **2.14× geomean, up to 7.59×** speedup over homogeneous
GPU-only baselines (IISWC 2025, Xu et al., UCSC / Microsoft Research).

## The shape: three tools talking through files

An application is a sequence of **stages**; each stage has a kernel implemented in up
to three backends — **OMP** (CPU), **CUDA** (NVIDIA), **Vulkan** (cross-platform GPU,
GLSL compute). The pipeline is really three tools passing files:

```
BT-Profiler  ──(JSONL profiling store)──▶  BT-Optimizer (Python / z3 SMT)
                                                │
                                          (schedule JSON)
                                                ▼
                                   BT-Implementer (C++ runtime:
                                   SPSC-queue dispatchers + UMA buffers)
```

The primary extension axis is **devices**: "add a device = drop in a data file"
(`devices/<id>.json`). See [`01-hardware.md`](01-hardware.md) for the target fleet.

## Evaluated applications

| App | What it is | Stages | Compare mode |
|---|---|---|---|
| **cifar-dense** | AlexNet inference on CIFAR-10 | 11 | float (`NearEqual`) |
| **cifar-sparse** | Pruned/sparse AlexNet (irregular memory) | 9 → 11 | float (`NearEqual`) |
| **tree** | 3D octree construction (morton → sort → unique → radix-tree → edge-count → prefix-sum → octree-build) | 7 | exact (integer/structural) |

The canonical AlexNet shapes are load-bearing for kernel work — see
[`04-alexnet-cifar-spec.md`](04-alexnet-cifar-spec.md).

## What "done" looks like

Every stage of every application is **numerically correct on every backend, on every
target hardware**, enforced as a CI gate (OMP is the in-process reference oracle).
How to build: [`02-building.md`](02-building.md). How to test:
[`03-unit-testing.md`](03-unit-testing.md). Current status & roadmap:
[`../reports-for-human/testing-status.md`](../reports-for-human/testing-status.md).
