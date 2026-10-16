# Spec Delta

## Purpose

Defines what cifar-dense's GPU 3×3 convolution kernels must compute in each precision mode, how
their variants are selected, and how fast they must run, so kernel optimizations stay correct and
measurable.

## ADDED Requirements

### Requirement: FP32 paths match the OMP oracle
Every FP32 GPU convolution variant SHALL match the OMP reference within the existing cifar-dense
Vulkan tolerance (rtol 1e-3, atol 1e-4) on synthetic and real weights.

#### Scenario: Differential suite per variant
- **WHEN** `test-cifar-dense-vk` runs on rocky-ryzen, the Pixel 7a and the Samsung with each FP32 variant selected
- **THEN** every conv stage passes against the OMP oracle

### Requirement: Reduced-precision paths are bounded and opt-in
An FP16 or cooperative-matrix convolution path SHALL be used only when explicitly enabled. It SHALL
pass a documented per-stage tolerance against the OMP oracle, and SHALL keep end-task accuracy on the
real trained weights within 0.5 percentage points of the FP32 path.

#### Scenario: Default stays FP32
- **WHEN** no precision mode is requested
- **THEN** the FP32 path runs, and results match the FP32 oracle tolerance

#### Scenario: FP16 enabled
- **WHEN** the FP16 path is enabled on a device that supports `shaderFloat16` and 16-bit storage
- **THEN** every conv stage passes the FP16 tolerance, and real-weight end-task accuracy is within
  0.5 points of FP32

### Requirement: Variants are selected from device capabilities
A convolution variant that needs an optional device capability (`shaderFloat16`, 16-bit storage,
`cooperativeMatrix`) SHALL only be selected when the device reports it. Otherwise the engine SHALL
fall back to a variant the device supports, without failing.

#### Scenario: No cooperative matrix
- **WHEN** the device does not support `VK_KHR_cooperative_matrix` (e.g. Mali-G710)
- **THEN** a non-cooperative-matrix variant runs and the suite passes

### Requirement: Convolutions are faster than the 2026-10-15 baseline
On rocky-ryzen, the combined FP32 GPU time of conv2–conv5 SHALL be at most half of the baseline
(0.883 + 1.989 + 2.426 + 1.576 = 6.874 ms at N=16), under the same governor and clock conditions.
No conv stage on either phone SHALL be slower than its pre-change time beyond run-to-run noise.

#### Scenario: Per-stage re-measurement
- **WHEN** `bm-cifar-dense-vk --benchmark_format=csv` runs before and after on rocky-ryzen
- **THEN** the sum of conv2–conv5 `real_time` is ≤ 3.437 ms

#### Scenario: Phones do not regress
- **WHEN** the benchmark runs on the Pixel 7a and the Samsung before and after
- **THEN** no conv stage is slower beyond run-to-run noise
