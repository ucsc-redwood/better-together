# Spec Delta

## Purpose

Defines the contract for the device-wide sort (tree stage 2) and stream compaction (tree stage 3)
GPU kernels: they reproduce the OMP oracle exactly at every subgroup width the fleet runs, and they
stay within a stated bound of the memory-bandwidth floor.

## ADDED Requirements

### Requirement: Sort output matches the oracle exactly at every subgroup width
The tree stage-2 Vulkan sort SHALL produce the same key sequence as the OMP oracle, bit for bit,
for every subgroup size the engine can run it at.

#### Scenario: Default subgroup sizes
- **WHEN** the tree VK suites run on the Pixel 7a (subgroup 16), the Samsung and rocky-ryzen (default 64)
- **THEN** the stage-2 output equals the oracle's sorted keys exactly

#### Scenario: Pinned subgroup 32
- **WHEN** the tree VK suites run on rocky-ryzen and the Samsung with `BT_VK_SUBGROUP_SIZE=32`
- **THEN** the stage-2 output equals the oracle's sorted keys exactly

### Requirement: Compaction output matches the oracle exactly
The tree stage-3 Vulkan compaction SHALL produce the same unique keys and the same unique count as
the OMP oracle.

#### Scenario: Unique keys and count
- **WHEN** the tree VK suites run on any fleet VK target
- **THEN** `n_unique` and the first `n_unique` unique keys equal the oracle's exactly

### Requirement: Sort time is bounded relative to the bandwidth floor
On rocky-ryzen, at the default input size (n = 307,200), the stage-2 sort SHALL take no more than
0.40 ms of GPU time (the baseline is 0.785 ms; the 4-pass DRAM floor is ≈0.18 ms).

#### Scenario: Stage-2 GPU time on rocky-ryzen
- **WHEN** `bm-tree-vk --benchmark_filter='Tree/Stage2/'` runs on rocky-ryzen under the same governor and clock conditions as the baseline
- **THEN** the median stage-2 GPU timestamp time is ≤ 0.40 ms

### Requirement: Compaction is not dispatch-bound
The stage-3 compaction SHALL use fewer dispatches than today's five, and it SHALL NOT be slower than
the 0.179 ms baseline on rocky-ryzen.

#### Scenario: Stage-3 GPU time on rocky-ryzen
- **WHEN** `bm-tree-vk --benchmark_filter='Tree/Stage3/'` runs on rocky-ryzen before and after
- **THEN** the after median is ≤ the before median, and the recorded dispatch count is below 5

### Requirement: The sort runs with full subgroups where the device can guarantee it
Subgroup-dependent sort kernels index shared memory by `gl_SubgroupID`, so they SHALL be created with
`REQUIRE_FULL_SUBGROUPS` on devices that support `computeFullSubgroups`. Where that is unsupported, the
engine SHALL log that full subgroups are not guaranteed, and the oracle tests SHALL still pass.

#### Scenario: Device supports full subgroups
- **WHEN** the selected GPU supports `computeFullSubgroups`
- **THEN** the radix-sort pipelines are created with `REQUIRE_FULL_SUBGROUPS`, and `BT_VK_VALIDATION=1` runs of the tree suite report 0 validation errors

#### Scenario: Device without the feature
- **WHEN** the selected GPU does not support `computeFullSubgroups`
- **THEN** a log line says full subgroups are not guaranteed, and the tree suite still matches the oracle exactly
