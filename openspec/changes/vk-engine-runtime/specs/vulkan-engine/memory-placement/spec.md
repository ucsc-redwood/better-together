# Spec Delta

## Purpose

Defines how the Vulkan engine chooses memory properties for each buffer from its usage class, and
what each class guarantees to host code, so GPU-heavy data can live in the fastest host-mappable
memory without breaking zero-copy hand-off between CPU and GPU chunks.

## ADDED Requirements

### Requirement: Placement follows the buffer's usage class
The engine SHALL let each buffer be allocated with a usage class: weights, GPU-local, or boundary.
A buffer with no class SHALL get today's placement: host-visible, preferring host-cached.

#### Scenario: Unclassified buffer
- **WHEN** an AppData buffer is allocated without a class
- **THEN** it is host-visible, host-cached where available, and mapped, exactly as before this change

#### Scenario: Weight buffer
- **WHEN** a weight buffer is allocated
- **THEN** it is placed in the memory type selected for the weights class on that device, and the first dispatch reads exactly the host-provided weights

### Requirement: Every class stays host-mapped
Every buffer SHALL remain host-visible and persistently mapped whatever its class, because AppData
buffers are `std::pmr::vector`s whose allocator must return a host pointer. Classes SHALL differ only
in which host-visible memory type they use, e.g. device-local or write-combined instead of cached.

#### Scenario: Weights written through a write-combined mapping
- **WHEN** weights are loaded into a weights-class buffer placed in uncached (write-combined) memory
- **THEN** the load completes, and the GPU reads the loaded values

### Requirement: Boundary buffers keep host-cached placement
Any buffer that a CPU chunk reads under the configured schedule SHALL use a host-cached memory type
where the device offers one. The host-visibility guarantee of `vulkan-engine/execution-sync` SHALL
hold for it.

#### Scenario: Hybrid schedule hand-off
- **WHEN** a schedule hands an item from a GPU chunk to a CPU chunk
- **THEN** every buffer the CPU chunk reads is host-cached, and its values match the OMP oracle

### Requirement: Placement is decided by measurement
A class SHALL be moved off the default placement on a device only if a before/after measurement on
that device shows a gain beyond run-to-run noise. The measurement SHALL be recorded.

#### Scenario: No measurable gain
- **WHEN** the fc1 weight-placement experiment shows no gain beyond noise on a device
- **THEN** that device keeps the default placement for the weights class, and the result is recorded
