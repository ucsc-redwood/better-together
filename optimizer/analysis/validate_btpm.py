#!/usr/bin/env python3
"""Validate the paper's central claim: does the contended (BTPM/interference) cost table
predict the REAL pipeline makespan better than the isolated table?

For each (device, app, backend, cost model) and each table_type in {btpm, isolated}, join —
per executed z3 candidate schedule — the **predicted** makespan (the schedule JSON's
metrics.max_time, which z3 computed from that table) against the **measured** makespan (the
max-chunk time from data/sched_logs/, the same number speedup_summary uses). The join lives
in analysis/accuracy.py; this script is its CLI.
The accuracy metric is the measured/predicted ratio: ratio≈1 means the table predicts
reality; ratio far from 1 means it doesn't. We then report which table predicts better.

This is the empirical check the interference audit asked for: if BTPM's ratios aren't
closer to 1 than isolated's, the contended table isn't earning its keep (and given the
known DVFS corruption of the BTPM GPU column, expect isolated to often predict better).

Usage: uv run python optimizer/analysis/validate_btpm.py
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.accuracy import accuracy_rows  # noqa: E402

BE_LABEL = {"vk": "VK", "cu": "CUDA"}
DEVICE_LABEL = {
    "duck-stable": "duck-stable",
    "duck-naughty": "duck-naughty",
    "minipc": "minipc",
    "R5CY21Y3VEV": "samsung",
    "3A021JEHN02756": "pixel",
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sched-logs", default="data/sched_logs")
    ap.add_argument("--schedules-root", default="data", help="parent of schedules_<table>/")
    args = ap.parse_args()

    # The measured/predicted join is analysis.accuracy (also used by speedup_summary).
    groups = {}
    for r in accuracy_rows(args.schedules_root, args.sched_logs):
        key = (r["device"], r["app"], r["be"], r["cost_model"])
        groups.setdefault(key, {}).setdefault(r["table"], []).append(r["ratio"])

    print(
        f"{'device':8} {'app':13} {'be':4} {'model':10} {'table':9} {'n':>2}  "
        f"{'measured/predicted ratio':>26}  accuracy"
    )
    print("-" * 89)
    btpm_better = iso_better = tie = 0
    for (device, app, be, model), by_table in sorted(groups.items()):
        medians = {}
        for tt in ("btpm", "isolated"):
            ratios = by_table.get(tt)
            label = (
                f"{DEVICE_LABEL.get(device, device):8} {app:13} {BE_LABEL[be]:4} {model:10} {tt:9}"
            )
            if not ratios:
                print(f"{label} {'0':>2}  (no executed schedules)")
                continue
            med = statistics.median(ratios)
            medians[tt] = med
            print(
                f"{label} {len(ratios):>2}  median {med:6.2f}x [{min(ratios):.2f}–{max(ratios):.2f}]  "
                f"|1-ratio|={abs(med - 1.0):.2f}"
            )
        if len(medians) == 2:
            b, i = abs(medians["btpm"] - 1), abs(medians["isolated"] - 1)
            if abs(b - i) < 0.05:
                tie += 1
            elif b < i:
                btpm_better += 1
            else:
                iso_better += 1

    print("-" * 89)
    print(
        f"Per-(cell, model) winner (predicted closest to measured): "
        f"btpm better on {btpm_better}, isolated better on {iso_better}, tie on {tie}."
    )
    print(
        "Paper's claim (BTPM predicts reality better than isolated) holds only if btpm wins "
        "clearly. Given the DVFS corruption of the BTPM GPU column, expect it NOT to until "
        "the harness controls GPU clocks."
    )


if __name__ == "__main__":
    main()
