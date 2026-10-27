# profiling/cost-table Specification

## Purpose
Defines what the profiling store records for the scheduler, and with what provenance: the per-stage
records it has always held, measurements of whole GPU stage ranges, and the GPU operating frequency
behind each record.

## Requirements

### Requirement: GPU stage ranges are measured as whole chunks
For a Vulkan cell, the profiler SHALL record the time of every contiguous stage range [a, b] of the
app dispatched as one GPU chunk, through the same dispatch path a pipeline's GPU chunk uses, including
its submissions, host cache maintenance and any host read-back between stages.

#### Scenario: Every range of a 7-stage app
- **WHEN** range profiling runs for the tree app on a phone
- **THEN** the store holds one record for each of the 28 ranges [a, b] with 1 <= a <= b <= 7

#### Scenario: A range that needs a host read-back
- **WHEN** the range [1, 7] of the tree app is measured
- **THEN** its recorded time includes both submissions and the read-back of the unique-key count
  between stages 3 and 4

### Requirement: Range records state the idle time before each chunk
Each range record SHALL state the idle gap that preceded every measured submission, and the profiler
SHALL record each range at least back-to-back (no gap) and with a non-zero gap.

#### Scenario: Two cadences per range
- **WHEN** range profiling completes for a cell
- **THEN** every range has a record with a zero gap and a record with a non-zero gap, each naming its
  gap in milliseconds

### Requirement: Range records do not break existing stores
Per-stage records SHALL keep their current form and meaning. Range records SHALL be distinguishable
from per-stage records, and a reader that ignores range records SHALL read a store exactly as before.

#### Scenario: Old store, new reader
- **WHEN** a store written before this change is loaded
- **THEN** it loads without error and yields the same per-stage table as before

#### Scenario: New store, per-stage reader
- **WHEN** a store holding range records is loaded for the per-stage table only
- **THEN** the per-stage table equals the one built from the per-stage records alone

### Requirement: Records carry the GPU operating frequency where the device exposes it
Every GPU record SHALL carry the GPU's frequency at measurement time in MHz when the device exposes it
to an unprivileged process, and SHALL mark it unknown otherwise.

#### Scenario: Pixel 7a and Samsung
- **WHEN** a GPU record is written on the Pixel 7a or on the Samsung
- **THEN** its frequency field holds a positive value read from the device while the measured kernel
  runs, not the unknown marker and not the 0 an idle, power-gated GPU reports

#### Scenario: A device that hides it
- **WHEN** no readable frequency source exists
- **THEN** the record marks the frequency unknown, and the profiling run does not fail

### Requirement: A stage is profiled on unmodified input
A CPU stage kernel SHALL NOT modify the buffers it reads as input, so that the profiler's repeated runs of
one stage on one AppData see the same input a pipeline item does. A stage that needs scratch space SHALL use
its output or a dedicated scratch buffer.

#### Scenario: Running a stage twice
- **WHEN** an OMP test runs stage s of tree, cifar-dense or cifar-sparse twice on the same AppData
- **THEN** the stage's input buffers after each run equal their contents before the first run, and both runs
  produce identical outputs

#### Scenario: Tree stage 2 timing
- **WHEN** `bm-prof-tree-vk` profiles stage 2 on a CPU tier of a phone
- **THEN** its median is within the run-to-run spread of the same stage timed with fresh input (stage 1 re-run
  before every sample, outside the timed region), on the same tier and phone
