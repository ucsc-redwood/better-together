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


def _load_schema():
    with open(SCHEMA_PATH) as f:
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


def validate(records):
    """Validate every record against the schema; raise with all errors if any fail."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(_load_schema())
    errors = []
    for i, r in enumerate(records):
        for e in validator.iter_errors(r):
            errors.append(f"  record {i} ({r.get('pu', '?')}/stage{r.get('stage', '?')}): {e.message}")
    if errors:
        raise ValueError("schema validation failed:\n" + "\n".join(errors))


def load_profiling(root, device, app, backend, scenario,
                   metric="p50", min_runs=1, max_cv=0.1):
    """Return (table, report).

    table: {(stage, pu): {"value", "n_runs", "count"}} -- aggregated metric.
    report: provenance about what was read, dropped, or came up short.
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

    table, dropped, insufficient = {}, [], []
    for (stage, pu), rs in sorted(cells.items()):
        # Drop measured-and-known-bad samples: explicit thermal throttle, or
        # noise above the CV gate. A field absent from provenance is never
        # assumed -- `throttled` missing means "not measured", not "False".
        kept = [r for r in rs
                if not r["provenance"].get("throttled", False)
                and r["timing"]["cv"] <= max_cv]
        if len(kept) < len(rs):
            dropped.append((stage, pu, len(rs) - len(kept)))
        if len(kept) < min_runs:
            insufficient.append((stage, pu, len(kept)))
            continue
        weight = sum(r["timing"]["count"] for r in kept)
        value = sum(r["timing"][metric] * r["timing"]["count"] for r in kept) / weight
        table[(stage, pu)] = {"value": value, "n_runs": len(kept), "count": weight}

    report = {
        "paths": paths, "n_records": len(records),
        "metric": metric, "max_cv": max_cv, "min_runs": min_runs,
        "dropped": dropped, "insufficient": insufficient,
    }
    return table, report


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
        args.root, args.device, args.app, args.backend, args.scenario,
        metric=args.metric, min_runs=args.min_runs, max_cv=args.max_cv,
    )

    print(f"loaded {rep['n_records']} records from {len(rep['paths'])} run file(s)  "
          f"(metric={rep['metric']}, max_cv={rep['max_cv']}, min_runs={rep['min_runs']})")

    stages = sorted({s for s, _ in table})
    pus = sorted({p for _, p in table})
    print("stage " + " ".join(f"{p:>9}" for p in pus))
    for s in stages:
        cells = " ".join(
            (f"{table[(s, p)]['value']:>9.4f}" if (s, p) in table else f"{'--':>9}")
            for p in pus
        )
        print(f"{s:>5} {cells}")

    if rep["dropped"]:
        print("\ndropped (cv>max_cv or throttled): "
              + ", ".join(f"stage{s}/{p}x{n}" for s, p, n in rep["dropped"]))
    if rep["insufficient"]:
        print("INSUFFICIENT (< min_runs survived, omitted from table): "
              + ", ".join(f"stage{s}/{p}({n})" for s, p, n in rep["insufficient"]))


if __name__ == "__main__":
    main()
