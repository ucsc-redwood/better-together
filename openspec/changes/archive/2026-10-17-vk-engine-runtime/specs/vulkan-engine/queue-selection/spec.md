# Spec Delta

## Purpose

Defines which Vulkan queue family and priority the engine uses for compute, how the choice is made
and reported, and how unavailable choices are handled, so queue placement is a controlled variable
in the interference experiments.

## ADDED Requirements

### Requirement: Queue choice is reported
At device creation, the engine SHALL log at debug level every queue family it saw (flags, queue
count, timestamp valid bits), the family it selected, and the queue priority in effect.

#### Scenario: Debug log on rocky-ryzen
- **WHEN** a Vulkan binary runs with debug logging on rocky-ryzen
- **THEN** the log lists each queue family and names the selected family and priority

### Requirement: Queue family is selectable
The engine SHALL let a run select a universal (graphics+compute) or a compute-only queue family. The
default selection SHALL stay what it is today, unless a recorded measurement justifies changing it.

#### Scenario: Compute-only family requested and available
- **WHEN** a run requests the compute-only family on a GPU that exposes one
- **THEN** compute work is submitted to a queue of that family, and every Vulkan suite still passes

#### Scenario: Compute-only family requested but absent
- **WHEN** a run requests the compute-only family on a GPU without one
- **THEN** the engine warns, names the family it uses instead, and continues

### Requirement: Queue priority is best-effort
When a run requests a global queue priority, the engine SHALL apply it if the device supports it,
and otherwise SHALL warn and continue at the default priority. That includes a
`VK_ERROR_NOT_PERMITTED` rejection.

#### Scenario: Priority not permitted
- **WHEN** a high global priority is requested and the driver refuses it
- **THEN** device creation succeeds at the default priority, with a warning naming the refused priority
