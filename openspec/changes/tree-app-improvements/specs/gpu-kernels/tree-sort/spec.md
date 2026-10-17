# Spec Delta

## Purpose

Defines the contract for the device-wide sort (tree stage 2) and stream compaction (tree stage 3)
GPU kernels: they reproduce the OMP oracle exactly at every subgroup width the fleet runs, and they
get measurably faster on the phones than the baseline recorded at the start of this work.

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

### Requirement: The sort is faster on the phones
On the Pixel 7a and on the Samsung, at the default input size (n = 307,200), the stage-2 sort's GPU
time SHALL be lower than its baseline beyond run-to-run noise. The baseline is recorded before any
sort change of this work, with the 64 radix workgroups that are already in place.

#### Scenario: Stage-2 A/B on both phones
- **WHEN** `bm-prof-tree-vk` (isolated scenario) runs alternately with the baseline and the changed
  sort on each phone, 3 rounds, 20 s apart
- **THEN** the changed sort's stage-2 median is lower on both phones, with non-overlapping run spreads

### Requirement: Compaction is not dispatch-bound
The stage-3 compaction SHALL use fewer dispatches than today's five, and it SHALL NOT be slower than
its baseline beyond run-to-run noise on the Pixel 7a or on the Samsung.

#### Scenario: Stage-3 A/B on both phones
- **WHEN** `bm-prof-tree-vk` (isolated scenario) runs alternately with the baseline and the changed
  compaction on each phone, 3 rounds, 20 s apart
- **THEN** the after median is not higher beyond the spread on either phone, and the recorded dispatch
  count is below 5

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
