# vulkan-engine/execution-sync Specification

## Purpose
Guarantees that results a submitted Vulkan command sequence writes to host-visible buffers are
safely readable by host code once the host has waited for that sequence to complete.

## Requirements

### Requirement: GPU writes are host-visible after completion wait
Every command sequence the engine submits SHALL include a memory dependency that makes all of its
compute-shader writes available to the host domain. Once the host's completion wait returns, the
host SHALL observe every value the sequence wrote, on coherent and non-coherent memory alike.

#### Scenario: Host reads results after a chunk completes
- **WHEN** a dispatcher submits one or more stages as a single sequence and the host waits for it to complete
- **THEN** the host reads the values the shaders wrote, not stale contents

#### Scenario: Validation reports no synchronization hazard
- **WHEN** a Vulkan test suite runs with `BT_VK_VALIDATION=1` and synchronization validation enabled
- **THEN** no write-after-read or read-after-write hazard is reported between shader writes and host reads

#### Scenario: Non-coherent mobile GPU
- **WHEN** the differential suites run on the Pixel 7a (Mali, host-cached non-coherent memory)
- **THEN** every stage matches the OMP oracle within the existing tolerances
