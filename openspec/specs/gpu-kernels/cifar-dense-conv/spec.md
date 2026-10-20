# gpu-kernels/cifar-dense-conv Specification

## Purpose
Defines what cifar-dense's GPU 3×3 convolution kernels must compute in each precision mode, how
their variants are selected, and how fast they must run, so kernel optimizations stay correct and
measurable.

## Requirements

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

### Requirement: Convolutions are faster than the recorded baseline
On the Samsung, where the subgroup-32-and-wider path runs, the combined FP32 GPU time of conv2–conv5 SHALL be
lower than its baseline beyond run-to-run noise. The baseline is recorded on the phones before any change of this
work. No conv stage on either phone SHALL be slower than its baseline beyond noise. Performance is measured on the
phones only; rocky-ryzen runs correctness tests.

#### Scenario: Samsung A/B
- **WHEN** `bm-prof-cifar-dense-vk` (isolated scenario) runs alternately with the baseline and the changed kernel, 3 rounds, 20 s apart on the Samsung
- **THEN** the changed conv2–conv5 sum has a lower median, with non-overlapping run spreads

#### Scenario: Phones do not regress
- **WHEN** the same A/B runs on the Pixel 7a (Mali path, unchanged by this work) and the Samsung
- **THEN** no conv stage is slower beyond run-to-run noise
