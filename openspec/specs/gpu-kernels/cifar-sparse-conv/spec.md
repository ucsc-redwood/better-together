# gpu-kernels/cifar-sparse-conv Specification

## Purpose
Defines what cifar-sparse's GPU convolution kernels must produce, and how fast they must run
relative to the measured baseline, so kernel rewrites stay correct and measurably useful.

## Requirements

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
On the Samsung, where the subgroup-32-and-wider path runs, the GPU time of each of the four large pruned
convolutions (conv2 to conv5) SHALL be lower than its baseline beyond run-to-run noise. The baseline is recorded on
the phones before any change of this work. Performance is measured on the phones only; rocky-ryzen runs
correctness tests.

#### Scenario: Samsung A/B
- **WHEN** `bm-prof-cifar-sparse-vk` (isolated scenario) runs alternately with the baseline and the changed kernel, 3 rounds, 20 s apart on the Samsung
- **THEN** each of conv2–conv5 has a lower median, with non-overlapping run spreads

#### Scenario: Phones do not regress
- **WHEN** the same A/B runs on the Pixel 7a (Mali path, unchanged by this work) and the Samsung
- **THEN** no conv stage is slower than its baseline beyond run-to-run noise
