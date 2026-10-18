# Spec Delta

## Purpose

Defines where a pipeline's CPU chunk runs: on the cores of the tier the schedule gives it, with that tier's thread
count, so profiled tier costs and executed chunks describe the same hardware.

## ADDED Requirements

### Requirement: A CPU chunk runs only on its tier's cores
When a CPU chunk of any app runs a stage range with a core list and a thread count, every thread that executes
stage work SHALL run on a core from that list, and the number of threads SHALL NOT exceed the given count. Stage
results SHALL be the same as without pinning.

#### Scenario: Tree on one tier
- **WHEN** an OMP test runs tree stages 1 to 7 through `dispatch_multi_stage(cores, n, AppData&, 1, 7)` with a
  one-tier core list
- **THEN** every thread that runs stage work reports, from inside the stage, an affinity mask contained in that core
  list, at most n threads are used, and the outputs equal the unpinned run's

#### Scenario: Every app
- **WHEN** the same test runs for cifar-dense and cifar-sparse
- **THEN** the same holds

### Requirement: Tree's tier records describe their tier
The tree profiler's `big`, `medium` and `little` records SHALL be measured with the stage pinned to that tier.

#### Scenario: Profiling a tree stage on one tier
- **WHEN** `bm-prof-tree-vk` profiles a tree stage on the `little` tier of a phone
- **THEN** its median is within the run-to-run spread of the same stage timed in a process pinned to the little
  cores with `taskset`
