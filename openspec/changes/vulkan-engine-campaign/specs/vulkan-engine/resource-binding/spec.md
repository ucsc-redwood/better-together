# Spec Delta

## Purpose

Defines how a stage's buffers are bound to its compute dispatch. Several stages, possibly sharing
one kernel, must be able to be recorded into one command buffer with no per-stage descriptor-set
bookkeeping in the app dispatchers.

## ADDED Requirements

### Requirement: Bindings are recorded per dispatch
A stage's buffer bindings SHALL be captured in the command buffer at record time, so each recorded
dispatch uses exactly the buffers bound for it. This holds even when several dispatches of the
same kernel share one command buffer.

#### Scenario: Same kernel used twice in one chunk
- **WHEN** a multi-stage chunk records two stages that use the same kernel with different buffers into one command buffer
- **THEN** each dispatch reads and writes its own buffers, and results match the OMP oracle

#### Scenario: Multi-pass radix sort in one submit
- **WHEN** the tree radix sort records all of its passes into one command buffer
- **THEN** each pass uses that pass's source and destination buffers, and the sort output matches the oracle exactly

### Requirement: Dispatchers need no descriptor-set capacity planning
App dispatchers SHALL NOT need to declare in advance how many binding sets a kernel needs, or
assign set indices per stage.

#### Scenario: Adding a stage that reuses a kernel
- **WHEN** a developer adds a new stage that reuses an existing kernel
- **THEN** no per-kernel set count or per-stage set index has to be chosen or updated

### Requirement: Portable binding fallback
On a GPU without the extension used for per-dispatch binding, the engine SHALL fall back to a
binding method with the same observable behaviour. It SHALL NOT fail.

#### Scenario: Extension unavailable
- **WHEN** the selected GPU does not support the per-dispatch binding extension
- **THEN** every Vulkan suite still passes with identical results

### Requirement: Binding overhead does not regress
Per-dispatch host-side binding cost SHALL be no greater than the descriptor-update path it replaces.

#### Scenario: µs-scale app re-measured
- **WHEN** tree × Vulkan is profiled on rocky-ryzen before and after the change, under the same governor and clock conditions
- **THEN** the measured per-chunk makespan does not increase beyond run-to-run noise
