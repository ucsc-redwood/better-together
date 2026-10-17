# Spec Delta

## MODIFIED Requirements

### Requirement: Subgroup size is pinned where the device allows it
When the device supports subgroup size control for compute, the engine SHALL require the chosen
subgroup size on each subgroup-dependent pipeline, so the size cannot vary per dispatch. Otherwise
it SHALL use the device's reported default subgroup size. When the device supports
`computeFullSubgroups`, subgroup-dependent pipelines SHALL also require full subgroups. When it
doesn't, the engine SHALL require that the workgroup's X size be a multiple of the subgroup size in
use, and fail pipeline creation otherwise.

#### Scenario: Device with subgroup size control
- **WHEN** the selected GPU supports subgroup size control for the compute stage
- **THEN** subgroup-dependent pipelines are created with the chosen size as a required subgroup size

#### Scenario: Device without subgroup size control
- **WHEN** the selected GPU does not support subgroup size control
- **THEN** pipelines are specialized with the reported default subgroup size and still pass the oracle tests

#### Scenario: Another supported size is pinned for testing
- **WHEN** `BT_VK_SUBGROUP_SIZE=<n>` is set and the GPU can pin compute pipelines to `n` (a power of
  two within its `minSubgroupSize`..`maxSubgroupSize`)
- **THEN** subgroup-dependent pipelines are specialized with, and pinned to, `n`
- **AND** if the GPU cannot pin to `n`, engine creation fails with an error naming `n` and the supported range

#### Scenario: Full subgroups available
- **WHEN** the selected GPU supports `computeFullSubgroups`
- **THEN** subgroup-dependent pipelines are created requiring full subgroups, and the tree suites match the oracle exactly at every fleet subgroup width

#### Scenario: Full subgroups unavailable and size mismatched
- **WHEN** `computeFullSubgroups` is unsupported and a subgroup-dependent pipeline's workgroup X size is not a multiple of the subgroup size in use
- **THEN** pipeline creation fails with an error naming both sizes
