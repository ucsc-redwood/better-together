# Spec Delta

## Purpose

Defines how compute pipelines get their subgroup size, workgroup size and compiled-pipeline cache.
The aim is that kernels adapt to the GPU they run on, rather than to per-device data files or
duplicated shader sources.

## ADDED Requirements

### Requirement: Subgroup size comes from the running device
Shaders whose correctness depends on the subgroup size SHALL be specialized at pipeline creation
with the subgroup size of the GPU actually selected. It SHALL NOT come from the `--device` argument or `devices/*.json`.

#### Scenario: Board without a registry subgroup entry
- **WHEN** the tree Vulkan pipeline runs on a GPU whose `devices/*.json` entry has no `gpu.subgroup_size`
- **THEN** the radix sort runs with the device-reported subgroup size and matches the OMP oracle exactly

#### Scenario: Every fleet subgroup width
- **WHEN** `ctest -L vk` runs on the Pixel 7a (16), the Samsung (32) and rocky-ryzen (64)
- **THEN** the tree suites pass exactly on all three, using one radix-sort shader source

### Requirement: Subgroup size is pinned where the device allows it
When the device supports subgroup size control for compute, the engine SHALL require the chosen
subgroup size on each subgroup-dependent pipeline, so the size cannot vary per dispatch. Otherwise
it SHALL use the device's reported default subgroup size.

#### Scenario: Device with subgroup size control
- **WHEN** the selected GPU supports subgroup size control for the compute stage
- **THEN** subgroup-dependent pipelines are created with the chosen size as a required subgroup size

#### Scenario: Device without subgroup size control
- **WHEN** the selected GPU does not support subgroup size control
- **THEN** pipelines are specialized with the reported default subgroup size and still pass the oracle tests

### Requirement: Registry subgroup value is a cross-check only
If a device's registry entry declares `gpu.subgroup_size` and it differs from the size the engine
uses, the engine SHALL log a warning naming both values and continue with the device-derived size.

#### Scenario: Registry disagrees with hardware
- **WHEN** `devices/<id>.json` declares a subgroup size different from the device-derived one
- **THEN** a warning naming both values is logged, and the run proceeds with the device-derived size

### Requirement: Declared workgroup size is applied to the shader
A workgroup size the host declares for a kernel SHALL be the workgroup size the shader executes
with, for every shader that exposes it as a specialization constant. A mismatch SHALL NOT pass silently.

#### Scenario: Host-declared size reaches the shader
- **WHEN** a dispatcher declares a workgroup size for a shader that exposes it as a specialization constant
- **THEN** the pipeline executes with that workgroup size

#### Scenario: Declared size exceeds device limits
- **WHEN** the declared workgroup size exceeds the device's compute workgroup limits
- **THEN** pipeline creation fails with an error naming the size and the limit

### Requirement: Compiled pipelines persist across runs
The engine SHALL reuse compiled-pipeline data from previous runs on the same device and driver when
it is available. It SHALL fall back to a cold compile when cached data is missing, unwritable or
incompatible, without failing.

#### Scenario: Warm start
- **WHEN** a Vulkan binary runs a second time on the same device and driver
- **THEN** pipeline creation reuses the cached data written by the first run

#### Scenario: Stale or foreign cache
- **WHEN** the cache file was written by a different device or driver version, or is corrupt
- **THEN** it is ignored, the pipelines compile cold, and results are unchanged

#### Scenario: Read-only location
- **WHEN** the cache location is not writable (e.g. a restricted Android directory)
- **THEN** the run completes normally without persisting the cache
