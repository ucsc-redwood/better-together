# Data Model: CPU/GPU Schedule Permutation & Overlap Coverage for Tree

This feature adds no new persisted data or product-facing types — it's a verification
tool operating entirely on existing runtime types (`Schedule`, `tree::AppData`,
`Logger`). The entities below are conceptual, realized as in-memory C++ values inside
the new test file, not new structs added to the product.

## Entities

### Schedule Permutation

One member of the 29-schedule space this feature exhaustively covers.

| Field | Type | Notes |
|---|---|---|
| gpu_start, gpu_end | `int` (1-7) or absent | The contiguous GPU stage range; absent = all-CPU schedule |
| Realized as | `Schedule` (`runtime/schedule.hpp`) | 1-3 `ChunkConfig`s: optional CPU-before `[1, gpu_start-1]`, the GPU chunk `[gpu_start, gpu_end]` (absent for all-CPU), optional CPU-after `[gpu_end+1, 7]` |
| Validity | Enforced by existing `validate_schedule_coverage`/`first_concurrent_gpu_chunk` (`runtime/schedule.hpp`) — reused unchanged, not reimplemented |

Generation: loop `gpu_start` from 1..7, `gpu_end` from `gpu_start`..7 (28 combinations),
plus one explicit all-CPU schedule = 29 total.

### Pool Item

One `tree::AppData` instance cycling through a schedule's SPSC ring
(`make_dataset<tree::AppData>`, per the prior session). Unchanged from that session —
this feature doesn't add fields, only runs more schedules through the existing type.

### Timing Record (existing, reused verbatim)

`runtime/record.hpp`'s `Logger<kNumToProcess>::records_[task][chunk] ->
{start, end}` (cycle counts) — populated by `worker_with_record`, already public,
read directly by this feature's overlap analysis. No new field added.

### Overlap Verdict

Derived, not stored — computed per (schedule, run) from that run's `Logger::records_`:

| Field | Type | Notes |
|---|---|---|
| schedule | Schedule Permutation | which of the 29 |
| run_index | 0-4 | which of the 5 repeated runs |
| concurrent_time_ms | `double` | wall-time where >= 2 chunks' intervals overlapped (steady-state window only), via the ported `_coverage_time` sweep |
| overlapping | `bool` | `concurrent_time_ms > 0` for schedules with both a CPU and GPU chunk; N/A for the two all-one-PU boundary schedules |

### Permutation Result

The final pass/fail record this feature reports per schedule (FR-005/006):

| Field | Type | Notes |
|---|---|---|
| schedule | Schedule Permutation | |
| correctness | pass / fail(stage, buffer index, mismatch detail) | from the reused `CheckItemChained` diff against the sequential OMP reference |
| overlap | pass / fail / not-applicable | "pass" = `overlapping == true` in >= 3 of 5 runs; not-applicable for all-CPU/all-GPU schedules |

## Relationships

```
29 Schedule Permutations (generated, not read from file)
        │  for each: 5x [ make_dataset<tree::AppData> -> worker_with_record ring -> Logger ]
        ▼
   Logger::records_[task][chunk] = {start, end}  (existing, unchanged type)
        │                                   │
        │ (per item, post-run)              │ (per run, post-run)
        ▼                                   ▼
  CheckItemChained vs. sequential      ported _coverage_time sweep
  OMP reference (reused from prior          → concurrent_time_ms
  session) → correctness verdict            → overlapping (bool)
        │                                   │
        └───────────────┬───────────────────┘
                         ▼
              Permutation Result (reported per schedule)
```

## State transitions

None — each schedule/run is independent and stateless; results accumulate into a final
report, no persisted state across the sweep's lifetime.

## Validation rules (from Functional Requirements)

- FR-001 / exhaustiveness: exactly 29 schedules generated, verified via
  `validate_schedule_coverage` (existing) before running.
- FR-002 / real runtime: every schedule runs through `make_dataset` +
  `worker_with_record` (existing production primitives), never a mock.
- FR-003 / correctness: every pooled item's final octree diffed exactly against the
  sequential OMP reference (reused `CheckItemChained`).
- FR-004 / overlap with repeated-run evidence: `overlapping` requires `concurrent_time_ms
  > 0` in >= 3 of 5 runs, only for schedules with both a CPU and a GPU chunk.
- FR-005/FR-006 / actionable failure: `Permutation Result` always carries the exact
  schedule (gpu_start/gpu_end) plus, on failure, either the mismatching stage/buffer
  index or which runs lacked overlap.
