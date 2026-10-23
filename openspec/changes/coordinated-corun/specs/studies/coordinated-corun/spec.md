# Spec Delta

## Purpose

Defines what the coordinated co-run study must measure, compare and decide, so that the go/no-go on coordinated
CPU/GPU phase-specialized LLM pipelining rests on reproducible trade-off curves from the phones.

## ADDED Requirements

### Requirement: The paced harness matches the reference rates
The study SHALL use a paced-execution harness built against the same llama.cpp version as `llm-phase-affinity`. With
no gap, the harness SHALL reproduce the `llm-phase-affinity` alone prefill and decode rates for the same model, PU
configuration and prompt length within 5%, and its achieved duty SHALL be reported for every paced run.

#### Scenario: Calibration
- **WHEN** the harness prefills a 512-token prompt on the Samsung GPU with gap 0, and decodes at depth 0 on mask `3f0`
- **THEN** both rates are within 5% of the `llm-phase-affinity` medians, or the difference is recorded and the
  harness's own alone rates are used as the reference for every ratio

#### Scenario: Pacing takes effect
- **WHEN** a GPU prefill loop runs at target duty 0.5
- **THEN** the report gives the achieved duty (compute time over wall time from the per-ubatch timestamps) and the
  resulting rate

### Requirement: Coordination trade-offs are measured on both sides of the comparison
The study SHALL measure, for each coordination setting, the rates of both co-running phases, for the specialized
pairs (GPU prefill with CPU decode) and for the data-parallel pair (GPU decode with CPU decode), on the Samsung with
Llama 3.2 1B, at prompt lengths 512 and 2048 for the prefill side. Each value SHALL be the median of 3 trials with
its min-max, the masks, the GPU clock and the thermal status, measured as `llm-phase-affinity` measured its co-runs.

#### Scenario: GPU pacing sweep
- **WHEN** the GPU prefill is paced at duty 1.0, 0.8, 0.67, 0.5 and 0.33 next to a CPU decode on mask `3f0`
- **THEN** the results give, per duty, the GPU prefill rate, the CPU decode rate, and each as a ratio to its alone
  rate

#### Scenario: Baseline coordinated the same way
- **WHEN** the GPU decode is paced at the same duties next to a CPU decode
- **THEN** the data-parallel pair's rates are reported per duty, so the gate compares coordinated against coordinated

### Requirement: Throughput is predicted with the best coordinated settings
For each request mix of `llm-phase-affinity` (prompt 128/512/2048 x output 32/128/512), the study SHALL compute the
specialized-pipeline throughput and the data-parallel throughput with the `llm-phase-affinity` D5 model, choosing for
each the measured coordination setting that maximizes its throughput, and SHALL report both next to the uncoordinated
values of `llm-phase-affinity`.

#### Scenario: One mix
- **WHEN** the mix is P = 512, D = 128 on the Samsung with llama.cpp 1B
- **THEN** the grid lists the best coordinated specialized throughput with its setting, the best coordinated
  data-parallel throughput with its setting, the uncoordinated values, and the gain

### Requirement: A go/no-go decision is recorded
The study SHALL end with a decision record. The outcome SHALL be "go" only if a coordinated specialized pipeline is
predicted to reach at least 1.2x the best coordinated data-parallel throughput on at least one request mix; otherwise
"no-go", with the best margin, its setting, and what limits it.

#### Scenario: Coordination helps both sides equally
- **WHEN** pacing raises the data-parallel throughput as much as the specialized one
- **THEN** the record says so and the gain is computed against the coordinated baseline, not the uncoordinated one

### Requirement: Measurements stay on the phones
No profiling, benchmark or timing run of this study SHALL execute on rocky-ryzen or the build box. Existing files on
the phones that the study did not create SHALL NOT be modified or deleted, and no GPU or CPU frequency setting SHALL be
changed.

#### Scenario: Building the harness
- **WHEN** the harness is built
- **THEN** it is built on the build box against the unmodified llama.cpp sources and pushed to a directory of the
  study's own under `/data/local/tmp`, which is removed at the end
