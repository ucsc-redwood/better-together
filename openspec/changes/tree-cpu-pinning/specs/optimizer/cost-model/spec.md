# Spec Delta

## ADDED Requirements

### Requirement: Pinned tree chunks are predicted at least as well
After tree's CPU chunks are pinned and tree is re-profiled, the median CPU-chunk |measured/predicted - 1| of executed
per-stage tree schedules SHALL NOT be higher than the 2026-10-18 cpu-chunk-costs gate's value by more than 0.1, on
either phone and either table. The best measured tree period SHALL NOT be worse beyond run-to-run noise.

#### Scenario: Acceptance run on the phones
- **WHEN** tree schedules are solved from the refreshed store and executed on both phones as in the 2026-10-18 gate
- **THEN** each (phone, table) cell meets both bounds, and the decision record lists the before / after CPU error,
  best period and the best schedule's run-to-run spread
