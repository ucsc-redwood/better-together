# Spec Delta

## Purpose

Defines what the cifar apps' GPU fully-connected kernels must compute, which weight precisions are
allowed and with what accuracy bound, and how much weight traffic they may generate, since these
layers are DRAM-bandwidth-bound.

## ADDED Requirements

### Requirement: FP32 fully-connected layers match the OMP oracle
Every FP32 GPU fully-connected stage of cifar-dense and cifar-sparse SHALL match the OMP reference
within each app's existing Vulkan tolerance (rtol 1e-3, atol 1e-4) on synthetic and real weights.

#### Scenario: Differential suites
- **WHEN** `test-cifar-dense-vk` and `test-cifar-sparse-vk` run on rocky-ryzen, the Pixel 7a and the Samsung
- **THEN** the fc1, fc2 and fc3 stages pass against the OMP oracle

### Requirement: Weights are streamed once per batch
A fully-connected stage SHALL read each weight element from DRAM at most once per batch, whatever the
batch size, so weight traffic doesn't grow with the number of images.

#### Scenario: cifar-sparse at N=128
- **WHEN** `bm-prof-cifar-sparse-vk` (isolated scenario) runs alternately with the baseline and the changed kernel, 3 rounds, 20 s apart on the Pixel 7a and on the Samsung
- **THEN** fc1 and fc2 have a lower median on both phones beyond run-to-run noise, and each phone's ratio of
  cifar-sparse fc1 (N=128) to cifar-dense fc1 (N=16) is recorded (about 7.4× on the Radeon 780M before this
  change: 6.41 vs 0.87 ms)

### Requirement: Reduced-precision weights are bounded and opt-in
FP16 or INT8 weight storage for fully-connected layers SHALL be used only when explicitly enabled. It
SHALL pass a documented per-stage tolerance against the OMP oracle, and SHALL keep end-task accuracy
on the real trained weights within 0.5 percentage points of FP32 for each app (dense 90.48%, sparse
90.58% in FP32).

#### Scenario: Default stays FP32
- **WHEN** no weight-precision mode is requested
- **THEN** FP32 weights are used, and results match the FP32 oracle tolerance

#### Scenario: INT8 weights enabled
- **WHEN** INT8 weight storage is enabled (8-bit storage access is already a required engine feature)
- **THEN** fc stages pass the INT8 tolerance, and real-weight end-task accuracy is within 0.5 points
  of FP32 for both apps

#### Scenario: Unsupported device
- **WHEN** a reduced-precision mode is requested on a device without the needed capability
- **THEN** the FP32 path runs, and a warning names the missing capability

### Requirement: Reduced-precision weights cut weight time
On each phone that supports a reduced weight precision, fc1 and fc2 with that precision SHALL be faster than with
FP32 weights beyond run-to-run noise. The ratio to FP32 is recorded per phone. Nothing is timed on rocky-ryzen.

#### Scenario: Re-measured on the phones
- **WHEN** `bm-prof-cifar-dense-vk` (isolated scenario) runs alternately with FP32 and each reduced weight
  precision on the Pixel 7a and the Samsung, 3 rounds, 20 s apart
- **THEN** fc1 and fc2 have a lower median with the reduced precision on every phone that supports it, and the
  FP16 / FP32 and INT8 / FP32 ratios are recorded
