# Spec Delta

## Purpose

Defines how GPU work for a chunk is recorded, submitted and waited on: when recorded work is reused,
how many items may be in flight, and how the per-chunk overhead is measured. The fixed cost of
running a chunk on the GPU must be visible and as small as the hardware allows.

## ADDED Requirements

### Requirement: Per-chunk overhead is measured by phase
The engine SHALL be able to report, per GPU chunk execution, the host time spent recording,
submitting and waiting, the GPU execution interval, and the host-to-GPU start latency. Host and GPU
timestamps SHALL be correlated with calibrated timestamps where the device supports them.

#### Scenario: Overhead breakdown on rocky-ryzen
- **WHEN** a measurement run executes tree × Vulkan chunks on rocky-ryzen
- **THEN** each chunk's record, submit, wait, GPU-execution and start-latency times are reported, and they sum to the measured wall time within the timer resolution

#### Scenario: Device without calibrated timestamps
- **WHEN** the device does not support calibrated timestamps
- **THEN** the host-side phases are still reported, and the GPU interval comes from the existing timestamp queries, with a note that the domains are uncorrelated

### Requirement: Recorded work is reused per pooled item
For a given chunk and pooled item, the engine SHALL record the chunk's command buffer once and
resubmit it on later executions, unless the item's buffer bindings or the push-constant values change.

#### Scenario: Steady-state execution
- **WHEN** the same pooled item passes through the same GPU chunk a second time with unchanged bindings
- **THEN** no command-buffer recording happens on that execution, and the outputs still match the OMP oracle

#### Scenario: Bindings change
- **WHEN** an item's buffers or push constants for the chunk differ from the recorded ones
- **THEN** the command buffer is re-recorded before submission

### Requirement: Several items may be in flight per GPU chunk
A GPU chunk SHALL be able to have more than one item submitted before waiting on the oldest one,
while handing items to the next chunk in submission order. The depth SHALL be configurable,
including depth 1, which is today's behaviour.

#### Scenario: Depth 2 preserves order and results
- **WHEN** a schedule runs with in-flight depth 2 on its GPU chunk
- **THEN** items leave the chunk in the order they entered, and every item matches the OMP oracle

#### Scenario: Depth 1 is the old behaviour
- **WHEN** the depth is 1
- **THEN** each item is submitted and waited on before the next is dequeued, as today

### Requirement: Overhead changes are judged by refit
Any change to submission SHALL be accepted only if a `fit_overhead` refit from before/after fleet
runs on rocky-ryzen shows the fitted per-chunk GPU overhead not increasing beyond the before-runs'
spread. The result SHALL be recorded.

#### Scenario: Refit after the change
- **WHEN** the before/after fleet runs are refit with `fit_overhead`
- **THEN** the after per-chunk `gpu_vulkan` constant is at most the before mean plus the before spread, and both are recorded
