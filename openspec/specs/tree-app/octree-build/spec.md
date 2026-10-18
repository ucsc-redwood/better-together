# tree-app/octree-build Specification

## Purpose
Defines the stage-7 octree contract: every backend produces the same internal topology, leaf links,
masks and geometry; concurrent updates to a node's masks never lose bits; and the differential oracle
actually detects a missing or wrong leaf link.

## Requirements

### Requirement: Mask updates are atomic
Every backend SHALL update an octree node's child-node mask and child-leaf mask with atomic
read-modify-write operations, so concurrent links into the same node never lose a bit.

#### Scenario: Many leaves linked into one node concurrently
- **WHEN** a test input makes several BRT nodes link leaves into the same octree node in parallel
- **THEN** that node's leaf mask has every linked octant's bit cleared, on OMP, CUDA and Vulkan

### Requirement: Every backend links leaves
The Vulkan stage-7 kernel SHALL produce leaf links (child slots and leaf-mask bits), as the OMP and
CUDA backends do.

#### Scenario: Vulkan leaf slots
- **WHEN** tree stage 7 runs on any fleet Vulkan target
- **THEN** the set of leaf-populated (octree node, octant) slots equals the OMP oracle's

### Requirement: The oracle detects missing leaf links
The stage-7 differential check SHALL fail when a backend's output is missing leaf links that the
oracle has.

#### Scenario: Leaf links removed from an output
- **WHEN** the check is given an output whose leaf masks were reset to "no leaves"
- **THEN** `CheckStage7Topology` reports a failure naming the missing leaf slots
