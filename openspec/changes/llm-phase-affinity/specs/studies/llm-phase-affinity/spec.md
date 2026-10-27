# Spec Delta

## Purpose

Defines what the LLM phase-affinity study must measure, predict and decide, so that the go/no-go on phase-specialized
CPU/GPU pipelining of on-device LLM inference rests on reproducible numbers from the phones.

## ADDED Requirements

### Requirement: Prefill and decode rates are measured per processing unit
The study SHALL measure prefill tokens/s and decode tokens/s separately for each model, quantization, runtime and PU
configuration in scope, at prompt lengths 128, 512 and 2048, on the Pixel 7a and the Samsung S24+. Each value SHALL be
the median of 3 runs taken 20 s apart, with its min-max spread, the CPU cores used, and the GPU clock where the device
exposes it.

#### Scenario: One model on one phone
- **WHEN** Llama 3.2 1B is measured on the Samsung with llama.cpp on the big CPU tier and with llama.cpp on Vulkan
- **THEN** the results list prefill and decode tokens/s for each of the three prompt lengths, each with its median,
  spread, cores and clock

#### Scenario: A configuration that does not run
- **WHEN** a configuration fails to load or run (out of memory, unsupported operator)
- **THEN** it is recorded as failed with the error, and no number is reported for it

### Requirement: Co-run interference is measured
The study SHALL measure each phase's rate while the other PU runs concurrently, for three pairings: GPU prefill with
CPU decode, CPU prefill with GPU decode, and both PUs decoding. The concurrent runs SHALL overlap for the whole measured
interval.

#### Scenario: Specialized pairing
- **WHEN** the GPU runs prefill of a 512-token prompt in a loop while the CPU runs decode in a loop
- **THEN** both rates are reported, as is each rate's ratio to the same configuration run alone

### Requirement: Memory bandwidth is characterized
The study SHALL report the read bandwidth each PU reaches alone and while the other PU also streams memory, on each
phone.

#### Scenario: Explaining decode rates
- **WHEN** decode rates are compared across PUs
- **THEN** the report gives each PU's decode rate as a fraction of the bytes-per-token bound implied by its measured
  bandwidth

### Requirement: Throughput is predicted per request mix
For each phone, model and request mix (prompt 128/512/2048 tokens x output 32/128/512 tokens), the study SHALL compute
the request throughput of:
- the best single PU;
- the ideal data-parallel bound, as the sum of the PUs' request rates, both from rates measured alone and from the
  both-decoding co-run rates;
- every specialized pipeline (prefill on one PU, decode on the other), using the specialized co-run rates and including
  the time to hand the KV cache from one PU to the other.

#### Scenario: A specialized pipeline's period
- **WHEN** prefill runs on the GPU and decode on the CPU for prompt length P and output length D
- **THEN** the predicted period is the larger of the GPU prefill time and the CPU decode time (from the co-run rates)
  plus the KV hand-off time, and the throughput is its inverse

### Requirement: A go/no-go decision is recorded
The study SHALL end with a decision record listing every measured and predicted value. The outcome SHALL be "go" only if
some specialized pipeline is predicted to reach at least 1.2x the co-run-corrected data-parallel bound on at least one
phone and one request mix, and SHALL otherwise be "no-go" with the largest margin found.

#### Scenario: No mix qualifies
- **WHEN** no specialized pipeline reaches 1.2x the co-run-corrected data-parallel bound
- **THEN** the record states "no-go", the best margin with its phone, model and mix, and which measured effect limited it

### Requirement: Measurements stay on the phones
No profiling, benchmark or timing run of this study SHALL execute on rocky-ryzen. Existing files on the phones that the
study did not create SHALL NOT be modified or deleted.

#### Scenario: Building binaries
- **WHEN** a binary or model file is built for the study
- **THEN** it is built on the build box (or in a container) and pushed to a directory of the study's own under
  `/data/local/tmp`, which is removed at the end
