# Spec Delta

## ADDED Requirements

### Requirement: Each phone ships a fitted CPU chunk overhead
For every phone in the fleet the profiling store SHALL hold an `overhead.json` fitted by
`optimizer/analysis/fit_overhead.py` from that phone's executed schedules, and the per-stage model SHALL apply it.
The constants SHALL be accepted only if they lower the CPU-chunk error on executions that were not used to fit
them.

#### Scenario: Held-out check
- **WHEN** the overhead is fitted on one set of executed schedules and evaluated on a second, independently solved
  and executed set
- **THEN** the second set's median CPU-chunk |measured/predicted - 1| with the overhead is lower than without it

### Requirement: CPU chunk predictions are checked against executed schedules
After this change, on the Pixel 7a and on the Samsung, the median CPU-chunk |measured/predicted - 1| of executed
per-stage schedules SHALL be lower than the 2026-10-17 gate's value for tree on both tables, and SHALL NOT be higher
for cifar-dense or cifar-sparse. The best measured period per (phone, app, table) SHALL NOT be worse beyond
run-to-run noise.

#### Scenario: Acceptance run on the phones
- **WHEN** schedules are solved from the refreshed store with the fitted overhead and executed on both phones
- **THEN** the summary's per-chunk CPU error meets the bounds above for every (phone, app, table), and the
  decision record lists every cell with its before and after values
