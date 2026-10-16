# Spec Delta

## Purpose

Defines how a schedule's GPU chunks execute alongside CPU chunks: how many chunks one GPU may serve,
how their submissions are synchronized, and how the optimizer may model them.

## ADDED Requirements

### Requirement: One GPU may serve several chunks of a schedule
The runtime SHALL execute a schedule in which the same GPU backend appears in more than one chunk.
Each such chunk SHALL record into its own command buffer and fence. Submissions to the shared queue
SHALL be externally synchronized.

#### Scenario: Two Vulkan chunks around a CPU chunk
- **WHEN** a schedule assigns stages 1–2 to Vulkan, 3–5 to CPU and 6–N to Vulkan, and the pipeline runs
- **THEN** every item's outputs match the OMP oracle, and the run completes without device loss

#### Scenario: Validation under concurrent GPU chunks
- **WHEN** that schedule runs with `BT_VK_VALIDATION=1`
- **THEN** no validation error is reported, including no queue external-synchronization error

### Requirement: Cache maintenance is per chunk
The set of buffers flushed before a submit and invalidated after its fence SHALL be tracked per
command sequence. It SHALL NOT be shared across concurrently recording chunks.

#### Scenario: Non-coherent device with two GPU chunks
- **WHEN** the two-GPU-chunk schedule runs on the Pixel 7a (non-coherent cached memory)
- **THEN** every item matches the oracle

### Requirement: Optimizer models GPU time-sharing
When the optimizer is allowed to place one GPU in more than one chunk, it SHALL model those chunks
as sharing the device: their combined time bounds throughput together, not as independent parallel
chunks. The mode SHALL be opt-in; by default the solver keeps one contiguous block per PU.

#### Scenario: Default solver mode
- **WHEN** the optimizer runs without the multi-chunk option
- **THEN** every emitted schedule has at most one chunk per PU, as today

#### Scenario: Multi-chunk mode cost
- **WHEN** the optimizer emits a schedule with two GPU chunks of predicted times a and b
- **THEN** the predicted pipeline period is at least a + b
