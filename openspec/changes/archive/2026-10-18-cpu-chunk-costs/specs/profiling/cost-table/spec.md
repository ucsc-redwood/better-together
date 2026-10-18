# Spec Delta

## ADDED Requirements

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
