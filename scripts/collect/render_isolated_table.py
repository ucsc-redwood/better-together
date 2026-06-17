#!/usr/bin/env python3
"""Render the isolated per-stage profiling table for each application across the
whole (device x backend x pu) matrix, straight from the canonical JSONL store.

Reuses profiling_loader.load_profiling (schema-validates, drops throttled / high-cv
runs, count-weighted aggregate) for every (device, app, backend) cell discovered
under --root, then pivots into one table per app: rows = stage, columns =
"<device>/<pu>" (the GPU backend PU plus every present CPU tier), value = chosen
metric in ms.

    uv run python scripts/collect/render_isolated_table.py --metric p50 --max-cv 0.5
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profiling_loader import load_profiling  # noqa: E402

# Stable column ordering: devices in fleet order, PUs GPU-first then big->little.
DEVICE_ORDER = ["jetson", "minipc", "R5CY21Y3VEV", "3A021JEHN02756"]
PU_ORDER = ["cuda", "vulkan", "big", "medium", "little"]


def discover(root, scenario):
    """{app: [(device, backend)]} for every cell with run-*.jsonl present."""
    cells = {}
    for path in glob.glob(os.path.join(root, "*", "*", "*", scenario, "run-*.jsonl")):
        rel = os.path.relpath(path, root).split(os.sep)
        device, app, backend = rel[0], rel[1], rel[2]
        cells.setdefault(app, set()).add((device, backend))
    return {app: sorted(dc) for app, dc in cells.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/profiling")
    ap.add_argument("--scenario", default="isolated")
    ap.add_argument("--metric", default="p50", choices=["p50", "p95", "p99", "mean"])
    ap.add_argument("--max-cv", type=float, default=0.5)
    ap.add_argument("--min-runs", type=int, default=1)
    args = ap.parse_args()

    apps = discover(args.root, args.scenario)
    if not apps:
        raise SystemExit(f"no run-*.jsonl found under {args.root}/*/*/*/{args.scenario}")

    dev_rank = {d: i for i, d in enumerate(DEVICE_ORDER)}
    pu_rank = {p: i for i, p in enumerate(PU_ORDER)}

    for app in sorted(apps):
        # (stage, (device, pu)) -> value, plus the set of columns and stages seen.
        grid, columns, stages, notes = {}, set(), set(), []
        for device, backend in apps[app]:
            table, rep = load_profiling(
                args.root, device, app, backend, args.scenario,
                metric=args.metric, min_runs=args.min_runs, max_cv=args.max_cv,
            )
            for (stage, pu), cell in table.items():
                grid[(stage, (device, pu))] = cell["value"]
                columns.add((device, pu))
                stages.add(stage)
            if rep["dropped"]:
                n = sum(d[2] for d in rep["dropped"])
                notes.append(f"{device}/{backend}: dropped {n} run-records (cv>{args.max_cv})")

        cols = sorted(columns, key=lambda dp: (dev_rank.get(dp[0], 99), pu_rank.get(dp[1], 99)))
        hdr = [f"{d}/{p}" for d, p in cols]
        w = max(12, *(len(h) for h in hdr))

        print(f"\n=== {app}  ({args.scenario}, {args.metric} ms) ===")
        print(f"{'stage':>5} " + " ".join(f"{h:>{w}}" for h in hdr))
        for s in sorted(stages):
            row = " ".join(
                (f"{grid[(s, c)]:>{w}.4f}" if (s, c) in grid else f"{'--':>{w}}")
                for c in cols
            )
            print(f"{s:>5} {row}")
        for note in notes:
            print(f"  note: {note}")


if __name__ == "__main__":
    main()
