# Spec Delta

## Purpose

Defines what cifar-sparse's GPU convolution kernels must produce, and how fast they must run
relative to the measured baseline, so kernel rewrites stay correct and measurably useful.

## ADDED Requirements

### Requirement: Sparse convolutions match the OMP oracle
Every pruned convolution stage computed on the GPU SHALL match the OMP reference within the existing
cifar-sparse Vulkan tolerance (rtol 1e-3, atol 1e-4), for both synthetic and real pruned weights.

#### Scenario: Differential suite on every Vulkan target
- **WHEN** `test-cifar-sparse-vk` runs on rocky-ryzen, the Pixel 7a and the Samsung
- **THEN** every stage passes against the OMP oracle

#### Scenario: Real pruned weights
- **WHEN** the suite runs with `BT_WEIGHTS_DIR` set to the real magnitude-pruned export
- **THEN** every convolution stage still matches the oracle within the same tolerance

### Requirement: CSR decode is independent of the kernel shape
The kernel SHALL produce correct results for any CSR row length, including empty rows and rows whose
nonzeros all fall in the padded border for a given output pixel.

#### Scenario: Empty and border-only rows
- **WHEN** a weight matrix contains an output channel with zero nonzeros, or with nonzeros that all
  read the padded border at the image edge
- **THEN** that channel's output is bias plus ReLU only, matching the oracle

### Requirement: Sparse convolutions are faster than the CSR baseline
On rocky-ryzen, the GPU time of each of the four large pruned convolutions (conv2 to conv5) SHALL be
at most half its 2026-10-15 baseline (17.54 / 27.43 / 38.20 / 23.72 ms at N=128), measured under the
same governor and clock conditions.

#### Scenario: Per-stage re-measurement
- **WHEN** `bm-cifar-sparse-vk --benchmark_format=csv` runs on rocky-ryzen before and after the change
- **THEN** each conv stage's GPU time (the `real_time` / manual-time column) is at most 50% of its
  baseline

#### Scenario: Phones do not regress
- **WHEN** the same benchmark runs on the Pixel 7a and the Samsung
- **THEN** no conv stage is slower than its pre-change time beyond run-to-run noise
