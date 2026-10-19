# vulkan-engine/memory-placement Specification

## Purpose
Defines how the Vulkan engine chooses memory properties for each buffer from its usage class, and
what each class guarantees to host code, so GPU-heavy data can live in the fastest host-mappable
memory without breaking zero-copy hand-off between CPU and GPU chunks.

## Requirements

### Requirement: Placement follows the buffer's usage class
The engine SHALL let each buffer be allocated with a usage class: weights, GPU-local, or boundary.
A buffer with no class SHALL get today's placement: host-visible, preferring host-cached.

#### Scenario: Unclassified buffer
- **WHEN** an AppData buffer is allocated without a class
- **THEN** it is host-visible, host-cached where available, and mapped, exactly as before this change

#### Scenario: Weight buffer
- **WHEN** a weight buffer that no CPU chunk reads under the configured schedule is allocated
- **THEN** it is placed in the memory type selected for the weights class on that device, and the first dispatch reads exactly the host-provided weights

(Amended during implementation: the first draft said "a weight buffer", which conflicts with "Boundary buffers keep
host-cached placement" whenever a CPU chunk runs a weighted stage. On the Pixel 7a, the one device where the weights
class moved, it is uncached for the CPU, so the boundary requirement wins for such weights; see design B3 and tasks
2.4.)

### Requirement: Every class stays host-mapped
Every buffer SHALL remain host-visible and persistently mapped whatever its class, because AppData
buffers are `std::pmr::vector`s whose allocator must return a host pointer. Classes SHALL differ only
in which host-visible memory type they use, e.g. device-local or write-combined instead of cached.

#### Scenario: Weights written through a write-combined mapping
- **WHEN** weights are loaded into a weights-class buffer placed in uncached (write-combined) memory
- **THEN** the load completes, and the GPU reads the loaded values

### Requirement: Boundary buffers keep host-cached placement
Any buffer that a CPU chunk reads under the configured schedule SHALL use a host-cached memory type
where the device offers one. The host-visibility guarantee of `vulkan-engine/execution-sync` SHALL
hold for it.

#### Scenario: Hybrid schedule hand-off
- **WHEN** a schedule hands an item from a GPU chunk to a CPU chunk
- **THEN** every buffer the CPU chunk reads is host-cached, and its values match the OMP oracle

### Requirement: Placement is decided by measurement
A class SHALL be moved off the default placement on a device only if a before/after measurement on
that device shows a gain beyond run-to-run noise. The measurement SHALL be recorded.

#### Scenario: No measurable gain
- **WHEN** the fc1 weight-placement experiment shows no gain beyond noise on a device
- **THEN** that device keeps the default placement for the weights class, and the result is recorded

### Requirement: Sparse host reads do not force the boundary class
A buffer whose only accessors outside the GPU stages of the schedule are declared sparse host reads
SHALL be placed as if no host code touched it. A buffer that no stage touches is not affected.

#### Scenario: Tree on the GPU alone
- **WHEN** the tree app runs every stage on the GPU, with its final octree and its two read-back
  counts declared as sparse host reads
- **THEN** those seven buffers are allocated from the GPU-local class, not the boundary class

#### Scenario: A CPU chunk still reads it
- **WHEN** a schedule runs stage 7 on a CPU chunk, so a CPU stage touches the final octree
- **THEN** the octree buffers are allocated from the boundary class, as "Boundary buffers keep
  host-cached placement" requires

### Requirement: Undeclared host access keeps today's placement
Host code outside any stage that is not declared a sparse host read SHALL keep a buffer in the
boundary class, exactly as before this change.

#### Scenario: Existing declarations
- **WHEN** the cifar apps, whose declarations do not use the sparse marker, are placed under any
  schedule
- **THEN** every buffer gets the same class as before this change

### Requirement: Sparse host reads stay correct in every class
After the completion wait of the chunk that wrote it, host code SHALL read the values the GPU wrote to
a buffer declared as a sparse host read, whichever class that buffer was placed in.

#### Scenario: Read-back counts on a non-coherent device
- **WHEN** the tree pipeline runs on the Pixel 7a with the counts in the GPU-local class
- **THEN** the unique-key count read after stage 3 and the octree-node count read after stage 6 equal
  the OMP oracle's, and every item's octree matches the oracle

#### Scenario: Results read by a check
- **WHEN** a test reads the final octree of a pipelined item on the Pixel 7a and on the Samsung
- **THEN** it matches the oracle

### Requirement: The tree declarations ship only if measured faster
The tree app SHALL declare its final octree and its read-back counts as sparse host reads only when a
whole tree GPU chunk is measured faster beyond run-to-run noise on the Pixel 7a and not slower beyond
noise on the Samsung.

#### Scenario: Whole-chunk A/B
- **WHEN** a tree chunk of stages 1-7 runs alternately with and without the declarations (host thread
  on the big cores, 3 rounds, 20 s apart) on both phones
- **THEN** the declarations are kept only if the Pixel's median is lower with non-overlapping spreads
  and the Samsung's median is not higher beyond its spread; the numbers are recorded either way
