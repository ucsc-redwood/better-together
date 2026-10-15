# Spec Delta

## Purpose

Defines how the Vulkan engine creates its logical device: which device features it requires and
enables, how it fails when they are missing, and when validation layers are active.

## ADDED Requirements

### Requirement: Enable exactly the required device features
The engine SHALL enable on the logical device exactly the set of device features its shaders and
allocator require, and no other optional features, regardless of what else the device supports.

#### Scenario: Device supports more features than required
- **WHEN** the selected GPU supports Vulkan 1.2 features beyond the required set
- **THEN** only the required features are enabled on the created logical device

#### Scenario: Device supports all required features
- **WHEN** the selected GPU supports every required feature
- **THEN** device creation succeeds and every Vulkan test suite (`ctest -L vk`) passes with unchanged results

### Requirement: Fail loudly on a missing required feature
If the selected GPU lacks any required feature, the engine SHALL fail device creation with an error
that names each missing feature. It SHALL NOT continue with that feature disabled.

#### Scenario: Required feature unsupported
- **WHEN** the selected GPU does not support a required feature (e.g. `shaderInt8`)
- **THEN** engine construction throws, and the error message names that feature

#### Scenario: Integrated-GPU probe on an incapable device
- **WHEN** the integrated-GPU availability probe runs on a GPU missing a required feature
- **THEN** the probe reports the engine as unavailable, so differential tests skip rather than crash

### Requirement: Validation layers are opt-in
The engine SHALL request the Khronos validation layer only when explicitly enabled by the
`BT_VK_VALIDATION=1` environment variable. By default no validation layer is requested.

#### Scenario: Default run
- **WHEN** a Vulkan binary runs without `BT_VK_VALIDATION` set, on a host with validation layers installed
- **THEN** no validation layer is enabled on the instance

#### Scenario: Validation explicitly requested
- **WHEN** a Vulkan binary runs with `BT_VK_VALIDATION=1` and the layer is installed
- **THEN** the validation layer is enabled on the instance

#### Scenario: Validation requested but not installed
- **WHEN** `BT_VK_VALIDATION=1` is set but the layer is not installed
- **THEN** the engine warns and continues without validation
