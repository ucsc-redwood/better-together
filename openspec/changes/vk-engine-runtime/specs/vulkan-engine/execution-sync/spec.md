# Spec Delta

## MODIFIED Requirements

### Requirement: GPU writes are host-visible after completion wait
Every command sequence the engine submits SHALL include a memory dependency that makes all of its
compute-shader writes available to the host domain. Once the host's completion wait returns, the
host SHALL observe every value the sequence wrote, on coherent and non-coherent memory alike. The
dependency SHALL be expressed with synchronization2: source COMPUTE_SHADER / SHADER_STORAGE_WRITE,
destination HOST / HOST_READ.

#### Scenario: Host reads results after a chunk completes
- **WHEN** a dispatcher submits one or more stages as a single sequence and the host waits for it to complete
- **THEN** the host reads the values the shaders wrote, not stale contents

#### Scenario: Validation reports no synchronization hazard
- **WHEN** a Vulkan test suite runs with `BT_VK_VALIDATION=1` and synchronization validation enabled
- **THEN** no write-after-read or read-after-write hazard is reported between shader writes and host reads

#### Scenario: Non-coherent mobile GPU
- **WHEN** the differential suites run on the Pixel 7a (Mali, host-cached non-coherent memory)
- **THEN** every stage matches the OMP oracle within the existing tolerances

## ADDED Requirements

### Requirement: Inter-stage dependencies cover RAW and WAW
When stages recorded in one command buffer share a buffer, the engine SHALL insert a memory
dependency between them. Its source SHALL include SHADER_STORAGE_WRITE, and its destination SHALL
include SHADER_STORAGE_READ and SHADER_STORAGE_WRITE, so read-after-write and write-after-write
hazards are both covered.

#### Scenario: Ping-pong radix sort passes
- **WHEN** the tree radix sort records its passes, each writing a buffer an earlier pass wrote, into one command buffer
- **THEN** synchronization validation reports no read-after-write or write-after-write hazard, and the sort output matches the oracle exactly

### Requirement: Dependencies only where stages depend
The engine SHALL NOT insert an inter-stage dependency between consecutive recorded stages that share
no buffer written by the earlier one and where the later one writes no buffer the earlier one reads,
unless an earlier access still has to be ordered.

(Amended during implementation: the first draft named only buffers the earlier stage writes. A later
stage that writes a buffer the earlier stage reads is a write-after-read hazard, which needs an
execution dependency (design A2); without it the earlier draft would have forbidden a required
barrier. With synchronization validation's shader-access analysis on, dropping that barrier is
reported as `SYNC-HAZARD-WRITE-AFTER-READ`; see tasks 1.4.)

#### Scenario: Independent consecutive stages
- **WHEN** two consecutive recorded stages share no buffer that the first writes, and the second writes no buffer the first reads
- **THEN** no barrier is recorded between them, and synchronization validation still reports no hazard
