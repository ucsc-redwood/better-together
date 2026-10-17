# Spec Delta

## Purpose

Defines how the cifar 3x3 convolution kernels behave on GPUs whose subgroup size is below 32 and that
have no scalar unit (Mali): which code path such a GPU runs, what it must produce, and the measured
condition under which a Mali path replaces the previous kernel.

## ADDED Requirements

### Requirement: The conv path follows the running GPU's subgroup size
The cifar-dense and cifar-sparse 3x3/stride-1/pad-1 convolution kernels SHALL run their Mali path
when the subgroup size of the GPU actually selected is below 32, and their other path otherwise. The
choice SHALL come from the running device, not from `--device` or `devices/*.json`.

#### Scenario: Pixel 7a takes the Mali path
- **WHEN** the conv kernels run on the Pixel 7a (Mali-G710, subgroup 16)
- **THEN** both kernels execute their Mali path

#### Scenario: Wider subgroups keep the other path
- **WHEN** the conv kernels run on the Samsung (Xclipse 940) or the Radeon 780M (subgroup 64)
- **THEN** both kernels execute the path they executed before this change

### Requirement: The subgroup-32-and-wider path does not change
Adding the Mali path SHALL NOT change the output values or the speed of the path taken by GPUs with a
subgroup size of 32 or more.

#### Scenario: Same outputs on the Samsung
- **WHEN** the cifar-dense and cifar-sparse differential suites run on the Samsung before and after the
  change with the same inputs
- **THEN** every conv stage produces the same output values

#### Scenario: No slowdown on the Samsung
- **WHEN** per-stage GPU time of the five convs of each app is measured on the Samsung before and after
  the change (3 interleaved rounds)
- **THEN** no conv stage is slower beyond run-to-run noise

### Requirement: The Mali path matches the reference at any shape
The Mali path SHALL match a double-precision reference within the cifar Vulkan tolerance (rtol 1e-3,
atol 1e-4) for any batch size, channel count and map size it accepts, including maps whose size is not
a multiple of the subgroup size or of any tile the path uses.

#### Scenario: Ragged map sizes
- **WHEN** the Mali path runs a 3x3 conv on 6x6, 7x5 and 8x8 maps with ReLU and bias, on the Pixel
- **THEN** every output matches the double-precision reference within tolerance

#### Scenario: CSR rows that contribute nothing
- **WHEN** the sparse Mali path runs a weight matrix with an output channel that has no nonzeros, and
  one whose nonzeros all read the padded border at the image edge
- **THEN** those channels' outputs are bias plus ReLU only, matching the reference

#### Scenario: Whole-model differential suites
- **WHEN** `test-cifar-dense-vk` and `test-cifar-sparse-vk` run on the Pixel 7a
- **THEN** every stage passes against the OMP oracle, with synthetic and with real weights

### Requirement: A Mali path ships only if it is measured faster
A Mali path SHALL replace the previous kernel on subgroup-below-32 GPUs only when, on the Pixel 7a,
each of the four largest convs of its app (stages 3, 5, 6, 7) is faster beyond run-to-run noise and no
conv stage is slower. Otherwise the previous kernel SHALL stay in use.

#### Scenario: Per-stage A/B on the Pixel
- **WHEN** `bm-prof-cifar-{dense,sparse}-vk` (isolated scenario) runs on the Pixel 7a with the previous
  and the new kernel, 3 rounds interleaved with a cooldown between runs
- **THEN** the change is enabled only if stages 3, 5, 6 and 7 each have a lower median than before,
  with non-overlapping run spreads, and no conv stage has a higher median beyond its spread

#### Scenario: Measurement stays off rocky
- **WHEN** a Mali path is measured for this gate
- **THEN** the measurement runs on the phones, and rocky-ryzen is used only for correctness tests
