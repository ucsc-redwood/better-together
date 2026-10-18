# Spec Delta

## ADDED Requirements

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
