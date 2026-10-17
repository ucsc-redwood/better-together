# Spec Delta

## ADDED Requirements

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
