# optimizer/cost-model Specification

## Purpose
Defines how the scheduler predicts the time of a schedule's chunks from the profiling store, and how
the accuracy of those predictions is measured against executed runs, so cost-model changes are judged
by whether they predict what the pipeline does.

## Requirements

### Requirement: A GPU chunk is priced by its measured range
When the cost table holds range records for a GPU stage range, the solver SHALL price a GPU chunk
covering exactly that range from those records only, without per-stage times or a fitted GPU
overhead. Without them, it SHALL price the chunk as today.

#### Scenario: Range data present
- **WHEN** a schedule's GPU chunk covers stages [4, 7] and the table has [4, 7] range records
- **THEN** the chunk's predicted time lies between the [4, 7] back-to-back time and the [4, 7]
  idle-gap time, and does not depend on any per-stage record

#### Scenario: Range data absent
- **WHEN** the table has no range records for the cell
- **THEN** every chunk is priced as the sum of its stages plus the fitted overhead, and the solver's
  output equals what it produced before this change

### Requirement: CPU chunks are priced as before
A chunk on a CPU tier SHALL be priced as the sum of its stages' times plus the fitted CPU overhead,
whether or not GPU range records exist.

#### Scenario: Mixed schedule
- **WHEN** a schedule has a GPU chunk with range data and a CPU chunk
- **THEN** the CPU chunk's predicted time equals its value under the per-stage model

### Requirement: The cost model is selectable
The solver SHALL accept a choice between the range model and the per-stage model, and SHALL record in
each schedule which model priced it.

#### Scenario: Forcing the per-stage model
- **WHEN** the solver is run with the per-stage model selected on a store that holds range records
- **THEN** it ignores the range records, and the written schedules name the per-stage model

### Requirement: Several GPU chunks of one schedule are priced by their ranges
When a schedule may give the GPU several chunks, the bound on the pipeline period from the GPU's time
sharing SHALL be the sum of those chunks' range times.

#### Scenario: Two GPU chunks
- **WHEN** a schedule places the GPU on [1, 2] and on [5, 7], both with range records
- **THEN** the period the solver must respect is at least the [1, 2] time plus the [5, 7] time

### Requirement: Prediction accuracy is reported for executed schedules
For every executed schedule, the fleet summary SHALL report the measured period, the predicted
period, and their ratio, and SHALL give per (device, app, backend, cost model, table) the median of
|measured / predicted - 1| over the executed schedules.

#### Scenario: A summary after a run
- **WHEN** schedules solved with the range model and with the per-stage model have been executed on a
  phone
- **THEN** the summary lists each schedule's ratio and one median absolute error per model and table

### Requirement: The range model must predict better than the per-stage model
The range model SHALL become the default only if, on the Pixel 7a and on the Samsung, for each of tree,
cifar-dense and cifar-sparse, its median absolute error over executed schedules is lower than the
per-stage model's on the same table. In addition, the best measured schedule it finds SHALL NOT be
slower beyond run-to-run noise.

#### Scenario: Acceptance run on the phones
- **WHEN** both models' schedules are generated from the same store and executed on the Pixel 7a and
  the Samsung
- **THEN** for each app the range model's median absolute error is lower, and its best measured period
  is not worse beyond noise; otherwise the per-stage model stays the default and the result is recorded

### Requirement: CPU chunk predictions are checked against executed schedules
After this change, on the Pixel 7a and on the Samsung, the median CPU-chunk |measured/predicted - 1| of executed
per-stage schedules, priced as the store ships them, SHALL be lower than the 2026-10-17 gate's value for tree on
both tables, and SHALL NOT be higher by more than 0.1 for cifar-dense or cifar-sparse. The best measured period per (phone, app, table) SHALL NOT be worse beyond
run-to-run noise.

#### Scenario: Acceptance run on the phones
- **WHEN** schedules are solved from the refreshed store and executed on both phones
- **THEN** the summary's per-chunk CPU error meets the bounds above for every (phone, app, table), and the
  decision record lists every cell with its before and after values

#### Scenario: Re-executing unchanged schedules
- **WHEN** the same cifar schedules are executed again at the same prices in a later session
- **THEN** their CPU error may move by up to 0.1 without failing the bound (the Pixel 7a's cifar-sparse btpm cell moved
  from 0.134 to 0.215 this way)

### Requirement: Pinned tree chunks are predicted at least as well
After tree's CPU chunks are pinned and tree is re-profiled, the median CPU-chunk |measured/predicted - 1| of executed
per-stage tree schedules SHALL NOT be higher than the 2026-10-18 cpu-chunk-costs gate's value by more than 0.1, on
either phone and either table. The best measured tree period SHALL NOT be worse beyond run-to-run noise.

#### Scenario: Acceptance run on the phones
- **WHEN** tree schedules are solved from the refreshed store and executed on both phones as in the 2026-10-18 gate
- **THEN** each (phone, table) cell meets both bounds, and the decision record lists the before / after CPU error,
  best period and the best schedule's run-to-run spread
