# Research: Tree Scheduler Re-Validation Post-AppData Migration

All findings below came from directly reading `optimizer/orchestrate/00_run_fleet.py`,
`optimizer/smt/baselines.py`, `optimizer/analysis/speedup_summary.py`, `fleet.json`,
`vocab.json`, and the three prior `speedup-summary-2026-07-02-*.md` reports plus the
newest `docs/reports-for-human/2026-07-03-kernel-wave-and-definitive-baseline.md` — not
from assumption.

## Finding 1: the entire pipeline already exists as one command

**Decision**: use `optimizer/orchestrate/00_run_fleet.py --only
duck-stable,samsung,pixel --fresh --phases build,profile,schedule,run,summary` as the
single driver for this feature; do not hand-roll the per-cell steps.

**Rationale**: `00_run_fleet.py` already implements exactly the sequence the spec
describes — per device, per (app × backend): build → `01_collect_profiling.py` → for
each of two profiling tables (`btpm`, `isolated`): `02_gen_schedule_merged.py` (z3, 10
solutions) → `03_run_schedule.py` (runs the top-N candidates, capped at 4 by default,
`fleet.json` can override per-device/app) — then, once every device finishes,
`optimizer/analysis/speedup_summary.py` renders `data/sched_logs/speedup-summary.md`.
`--only` restricts to the three device keys this feature needs (`duck-stable` already
carries `backends: ["cu", "vk"]` in `fleet.json`, so both Jetson backends run under one
device name — no separate flag needed for that). `--fresh` deletes exactly
`data/profiling/<dev>`, `data/schedules_{btpm,isolated}/<dev>`, and
`data/sched_logs/<dev>_*` for the selected devices before running — its own docstring
calls this "the cure for kernel/runtime changes, where stale cells from the OLD
implementation would otherwise linger" — precisely this feature's situation.

**Alternatives considered**: manually invoking `01`/`02`/`03` per device/backend/table
(8 device/backend/table combinations: duck-stable×{cu,vk}×{btpm,isolated}, plus
samsung/pixel×vk×{btpm,isolated}) — rejected as needless manual repetition of what
`00_run_fleet.py` already automates correctly, with more chances for a missed `--fresh`
wipe on one cell leaving stale data behind.

## Finding 2: no per-app filter exists — accepted as incidental scope, not worked around

**Decision**: accept that the same `--only duck-stable,samsung,pixel` run also profiles
and schedules cifar-dense and cifar-sparse on those three devices (`load_apps()` reads
all of `vocab.json["app_stages"]` unconditionally — there is no `--app` flag on
`00_run_fleet.py`). Do not add one.

**Rationale**: the spec's own Edge Cases/Assumptions already settled this — adding an
app filter would be scope creep relative to "recollect tree's data," and this project's
existing `speedup_summary.py` already discovers cells generically and reports every
app it finds, so tree's four rows can simply be read out of the resulting table without
needing the other two apps' rows suppressed. Per FR-007's fix-scope constraint (nothing
beyond what's needed), inventing a new CLI flag here would itself be out of scope unless
collection is actually blocked without it (it isn't).

## Finding 3: "best-PU baseline" is derived, not looked up, and already precisely defined

**Decision**: no new baseline logic — `optimizer/smt/baselines.py:get_baseline_for_config`
already computes exactly what the spec calls the "best-single-processor baseline,"
reading straight from the freshly-collected isolated JSONL store.

**Rationale**: `omp_time` = sum of the fastest fully-measured CPU tier's per-stage
isolated times; `gpu_time` = sum of the backend's (vk/cu) column; `fastest` =
`min(omp_time, gpu_time)` over whichever are present. This function is called both by
`02_gen_schedule_merged.py` (to give z3 something to beat) and by
`speedup_summary.py` (to compute the reported speedup) — so re-running `00_run_fleet.py`
on fresh data automatically recomputes the baseline from that fresh data; there is
nothing to separately "re-derive."

## Finding 4: the report format is `speedup_summary.py`'s Markdown output, archived by convention

**Decision**: after the fleet run's `summary` phase regenerates
`data/sched_logs/speedup-summary.md`, copy it verbatim to
`docs/reports-for-human/perf-results/speedup-summary-<DATE>-appdata-migration.md` —
matching the exact naming convention of the three prior instances
(`speedup-summary-2026-07-02-{fresh-start,overhead-model,union-sweep}.md`).

**Rationale**: `render_markdown()` in `speedup_summary.py` already produces: a title, a
one-sentence definition of "measured speedup," a `| Device | App | Backend | Baseline |
Best | Speedup |` table (rows discovered from `data/sched_logs/` directory names, so it
naturally includes only what was actually run), a "Reading the table" section, a "Tree
losses" section that already explains *why* tree can show speedup < 1 (small per-stage
work vs. per-chunk GPU submit/fence overhead — a framework-overhead property, not a
kernel bug) and a "Caveats" section. This is the exact document shape User Story 3 wants
— nothing new needs designing.

## Finding 5: local `data/` is unusable as a starting point — confirmed, not assumed

**Decision**: treat local `data/profiling/`, `data/schedules_*/`, `data/sched_logs/` as
fully stale; `--fresh` must be used (not skipped as an optimization).

**Rationale**: every tree-related file under local `data/` is dated 2026-06-20 and keyed
under the device name `jetson` — which predates (a) the 2026-07-01 JetPack 7.2 reflash
that renamed/retired that device id in favor of `duck-stable`/`duck-naughty`, (b) the
2026-07-02 fresh-start re-collection, (c) the 2026-07-03 kernel-optimization wave (whose
own report says its data "stores on the benchmarking host," i.e. not this local
checkout), and (d) today's AppData migration. There is no `duck-stable`- or
`duck-naughty`-keyed tree data locally at all to be tempted to reuse.

## Finding 6: Constitution Principle VI applies concretely to this feature's own execution

**Decision**: before starting the fleet run, confirm `duck-stable` (ssh) and
`rocky-ryzen` (the adb host for both phones — even though it's not a device under test,
profiling/schedule-run traffic for both phones physically transits it) have no
competing process, using the same check this session already used to catch the
`llama-server` incident and the later in-progress `executorch` export job.

**Rationale**: this is not hypothetical — this exact session already found rocky-ryzen
mid-job with unrelated work once already. A repeat would silently distort the phones'
profiling/schedule-run numbers (adb traffic shares rocky's CPU/network stack) without
failing any test.
