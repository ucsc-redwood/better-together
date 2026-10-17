#!/usr/bin/env python3
"""Loader for the canonical JSONL profiling store.

Reads ``data/profiling/<device>/<app>/<backend>/<scenario>/run-*.jsonl`` (produced
by the ``bm-prof-*`` profiler), validates every record against
``schemas/profiling-table.schema.json``, filters out throttled / high-CV samples,
and count-weighted-aggregates a chosen metric (default p50) per (stage, pu).

This replaces the old ``regex-scrape -> groupby.mean`` path: same data, but the
contract is explicit and it FAILS LOUD on schema-invalid data, on a fully-filtered
cell, or when fewer than ``min_runs`` survive -- instead of silently feeding noise
to the solver.

NB: the records store summaries (p50/p95/...), not raw samples, so aggregating a
percentile across runs is a count-weighted point estimate, not a true pooled
percentile. For a faithful pooled percentile, increase per-run iterations rather
than relying on cross-run combination.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

SCHEMA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "schemas", "profiling-table.schema.json")
)
# GPU stage-range records (range-NNN.jsonl, bm-ranges-*): their own files and schema, so a
# reader of run-*.jsonl never sees one (design D1).
RANGE_SCHEMA_PATH = os.path.join(os.path.dirname(SCHEMA_PATH), "profiling-range.schema.json")


def _load_schema(path=SCHEMA_PATH):
    with open(path) as f:
        return json.load(f)


def read_records(paths):
    """Parse JSONL files into a flat list of record dicts (fails loud on bad JSON)."""
    records = []
    for p in paths:
        with open(p) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{p}:{lineno}: invalid JSON: {e}") from e
    return records


def validate(records, schema_path=SCHEMA_PATH):
    """Validate every record against the schema; raise with all errors if any fail."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(_load_schema(schema_path))
    errors = []
    for i, r in enumerate(records):
        where = r.get("stage", f"{r.get('start_stage', '?')}-{r.get('end_stage', '?')}")
        for e in validator.iter_errors(r):
            errors.append(f"  record {i} ({r.get('pu', '?')}/stage{where}): {e.message}")
    if errors:
        raise ValueError("schema validation failed:\n" + "\n".join(errors))


def load_profiling(
    root, device, app, backend, scenario, metric="p50", min_runs=1, max_cv=0.1, cv_exempt_pus=()
):
    """Return (table, report).

    table: {(stage, pu): {"value", "n_runs", "count"}} -- aggregated metric.
    report: provenance about what was read, dropped, or came up short.

    ``cv_exempt_pus``: PUs whose rows pass the CV gate regardless of noise (an
    explicit throttle flag still drops them). Rationale: the GPU target PU must
    exist for EVERY stage or the whole cell is unschedulable, and its p50 is
    robust to the interference spikes that inflate CV -- with sub-ms optimized
    kernels one preempted sample blows past any sane gate, and losing the cell
    is strictly worse than trusting the median. Exempted-but-noisy cells are
    listed in ``report["cv_exempted"]`` so callers can warn loudly.
    """
    pattern = os.path.join(root, device, app, backend, scenario, "run-*.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no run-*.jsonl found under {pattern}")

    records = read_records(paths)
    validate(records)

    cells = defaultdict(list)
    for r in records:
        cells[(r["stage"], r["pu"])].append(r)

    table, dropped, insufficient, cv_exempted = {}, [], [], []
    for (stage, pu), rs in sorted(cells.items()):
        # Drop measured-and-known-bad samples: explicit thermal throttle, or
        # noise above the CV gate. A field absent from provenance is never
        # assumed -- `throttled` missing means "not measured", not "False".
        kept = [
            r
            for r in rs
            if not r["provenance"].get("throttled", False)
            and (r["timing"]["cv"] <= max_cv or pu in cv_exempt_pus)
        ]
        if pu in cv_exempt_pus:
            noisy = [r["timing"]["cv"] for r in kept if r["timing"]["cv"] > max_cv]
            if noisy:
                cv_exempted.append((stage, pu, max(noisy)))
        if len(kept) < len(rs):
            dropped.append((stage, pu, len(rs) - len(kept)))
        if len(kept) < min_runs:
            insufficient.append((stage, pu, len(kept)))
            continue
        weight = sum(r["timing"]["count"] for r in kept)
        value = sum(r["timing"][metric] * r["timing"]["count"] for r in kept) / weight
        table[(stage, pu)] = {"value": value, "n_runs": len(kept), "count": weight}

    report = {
        "paths": paths,
        "n_records": len(records),
        "metric": metric,
        "max_cv": max_cv,
        "min_runs": min_runs,
        "dropped": dropped,
        "insufficient": insufficient,
        "cv_exempted": cv_exempted,
    }
    return table, report


def load_ranges(root, device, app, backend, scenario, metric="p50", min_runs=1):
    """Return (ranges, report) from the cell's range-*.jsonl (schema-validated).

    ranges: {(start_stage, end_stage): {"t0", "t1", "gap_ms", "submissions", "n_runs"}}
    with t0 the back-to-back time (gap_ms == 0) and t1 the time after the idle gap
    (gap_ms > 0), each the count-weighted ``metric`` over the runs; gap_ms is the mean
    gap of the t1 records. A range needs both cadences from >= ``min_runs`` runs
    (explicitly throttled records dropped), else it is listed in report["insufficient"].
    No CV gate: the GPU p50 is what the solver uses (cf. cv_exempt_pus above).

    A cell without range files yields ({}, report) -- an old store is not an error.
    """
    pattern = os.path.join(root, device, app, backend, scenario, "range-*.jsonl")
    paths = sorted(glob.glob(pattern))
    records = read_records(paths)
    validate(records, RANGE_SCHEMA_PATH)

    cells = defaultdict(list)
    for r in records:
        if not r["provenance"].get("throttled", False):
            kind = "t0" if r["gap_ms"] == 0 else "t1"
            cells[(r["start_stage"], r["end_stage"], kind)].append(r)

    def weighted(rs, key):
        weight = sum(r["timing"]["count"] for r in rs)
        return sum(key(r) * r["timing"]["count"] for r in rs) / weight

    ranges, insufficient = {}, []
    for a, b in sorted({(a, b) for a, b, _ in cells}):
        hot, idle = cells.get((a, b, "t0"), []), cells.get((a, b, "t1"), [])
        if len(hot) < min_runs or len(idle) < min_runs:
            insufficient.append((a, b, len(hot), len(idle)))
            continue
        ranges[(a, b)] = {
            "t0": weighted(hot, lambda r: r["timing"][metric]),
            "t1": weighted(idle, lambda r: r["timing"][metric]),
            "gap_ms": sum(r["gap_ms"] for r in idle) / len(idle),
            "submissions": max(r["submissions"] for r in hot + idle),
            "n_runs": min(len(hot), len(idle)),
        }

    report = {
        "paths": paths,
        "n_records": len(records),
        "metric": metric,
        "min_runs": min_runs,
        "insufficient": insufficient,
    }
    return ranges, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/profiling")
    ap.add_argument("--device", required=True)
    ap.add_argument("--app", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--scenario", default="isolated")
    ap.add_argument("--metric", default="p50", choices=["p50", "p95", "p99", "mean"])
    ap.add_argument("--max-cv", type=float, default=0.1)
    ap.add_argument("--min-runs", type=int, default=1)
    args = ap.parse_args()

    table, rep = load_profiling(
        args.root,
        args.device,
        args.app,
        args.backend,
        args.scenario,
        metric=args.metric,
        min_runs=args.min_runs,
        max_cv=args.max_cv,
    )

    print(
        f"loaded {rep['n_records']} records from {len(rep['paths'])} run file(s)  "
        f"(metric={rep['metric']}, max_cv={rep['max_cv']}, min_runs={rep['min_runs']})"
    )

    stages = sorted({s for s, _ in table})
    pus = sorted({p for _, p in table})
    print("stage " + " ".join(f"{p:>9}" for p in pus))
    for s in stages:
        cells = " ".join(
            (f"{table[(s, p)]['value']:>9.4f}" if (s, p) in table else f"{'--':>9}") for p in pus
        )
        print(f"{s:>5} {cells}")

    if rep["dropped"]:
        print(
            "\ndropped (cv>max_cv or throttled): "
            + ", ".join(f"stage{s}/{p}x{n}" for s, p, n in rep["dropped"])
        )
    if rep["insufficient"]:
        print(
            "INSUFFICIENT (< min_runs survived, omitted from table): "
            + ", ".join(f"stage{s}/{p}({n})" for s, p, n in rep["insufficient"])
        )

    ranges, rrep = load_ranges(
        args.root,
        args.device,
        args.app,
        args.backend,
        args.scenario,
        metric=args.metric,
        min_runs=args.min_runs,
    )
    if rrep["paths"]:
        print(f"\nloaded {rrep['n_records']} range records from {len(rrep['paths'])} range file(s)")
        print("range       t0 (gap 0)  t1 (after gap)    gap_ms  submissions  runs")
        for (a, b), r in ranges.items():
            print(
                f"{a:>2}-{b:<2} {r['t0']:>14.4f} {r['t1']:>15.4f} {r['gap_ms']:>9.4f} "
                f"{r['submissions']:>12} {r['n_runs']:>5}"
            )
        if rrep["insufficient"]:
            print(
                "INSUFFICIENT ranges (runs with gap 0 / gap > 0): "
                + ", ".join(f"{a}-{b}({h}/{i})" for a, b, h, i in rrep["insufficient"])
            )


if __name__ == "__main__":
    main()
