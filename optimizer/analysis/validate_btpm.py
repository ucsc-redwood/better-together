#!/usr/bin/env python3
"""Validate the paper's central claim: does the contended (BTPM/interference) cost table
predict the REAL pipeline makespan better than the isolated table?

For each (device, app, backend) and each table_type in {btpm, isolated}, join — per z3
candidate schedule UID — the **predicted** makespan (the schedule JSON's metrics.max_time,
which z3 computed from that table's per-stage costs) against the **measured** makespan (the
steady-state max-chunk time from data/sched_logs/, the same number speedup_summary uses).
The accuracy metric is the measured/predicted ratio: ratio≈1 means the table predicts
reality; ratio far from 1 means it doesn't. We then report which table predicts better.

This is the empirical check the interference audit asked for: if BTPM's ratios aren't
closer to 1 than isolated's, the contended table isn't earning its keep (and given the
known DVFS corruption of the BTPM GPU column, expect isolated to often predict better).

Usage: uv run python optimizer/analysis/validate_btpm.py
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrate.case import Case  # noqa: E402

from analysis.results.log_parser import find_log_files, process_log_file  # noqa: E402
from analysis.results.statistics import calculate_aggregated_statistics  # noqa: E402

TABLES = ["btpm", "isolated"]
BE_LABEL = {"vk": "VK", "cu": "CUDA"}
DEVICE_LABEL = {
    "jetson": "jetson",
    "minipc": "minipc",
    "R5CY21Y3VEV": "samsung",
    "3A021JEHN02756": "pixel",
}
APP_ORDER = ["tree", "cifar-dense", "cifar-sparse"]


def predicted_by_uid(sched_path):
    """{uid: predicted max_time(ms)} from a z3 schedule JSON (list of candidates)."""
    if not os.path.isfile(sched_path):
        return {}
    out = {}
    for s in json.load(open(sched_path)):
        uid = s.get("uid")
        mk = (s.get("metrics") or {}).get("max_time")
        if uid and mk is not None:
            out[uid] = mk
    return out


def measured_by_uid(log_folder):
    """{uid: measured makespan(ms)} = max chunk avg per schedule UID from sched_logs."""
    logs = find_log_files(log_folder) if os.path.isdir(log_folder) else []
    data = []
    with contextlib.redirect_stdout(io.StringIO()):  # log_parser is chatty on stdout
        for lg in logs:
            data.extend(process_log_file(lg))
    if not data:
        return {}
    agg, _ = calculate_aggregated_statistics(data)
    out = {}
    for uid, st in agg.items():
        chunks = st.get("chunks") or {}
        if chunks:
            out[uid] = max(c["avg_duration_ms"] for c in chunks.values())
    return out


def discover_cells(sched_logs_root):
    cells = set()
    for name in sorted(os.listdir(sched_logs_root)) if os.path.isdir(sched_logs_root) else []:
        if not os.path.isdir(os.path.join(sched_logs_root, name)):
            continue
        parts = name.split("_")
        if len(parts) < 4 or parts[-1] not in TABLES or parts[-2] not in BE_LABEL:
            continue
        device, app, be = "_".join(parts[:-3]), parts[-3], parts[-2]
        if app in APP_ORDER:
            cells.add((device, app, be))
    return sorted(cells)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sched-logs", default="data/sched_logs")
    ap.add_argument("--schedules-root", default="data", help="parent of schedules_<table>/")
    args = ap.parse_args()

    print(
        f"{'device':8} {'app':13} {'be':4} {'table':9} {'n':>2}  "
        f"{'measured/predicted ratio':>26}  accuracy"
    )
    print("-" * 78)
    btpm_better = iso_better = tie = 0
    for device, app, be in discover_cells(args.sched_logs):
        ratios_by_table = {}
        for tt in TABLES:
            sched = Case(device, app, be).schedule_path(
                os.path.join(args.schedules_root, f"schedules_{tt}"), tt, "tmax"
            )
            pred = predicted_by_uid(sched)
            meas = measured_by_uid(os.path.join(args.sched_logs, f"{device}_{app}_{be}_{tt}"))
            uids = sorted(set(pred) & set(meas))
            ratios = [meas[u] / pred[u] for u in uids if pred[u] > 0]
            if ratios:
                med = statistics.median(ratios)
                ratios_by_table[tt] = med
                lo, hi = min(ratios), max(ratios)
                acc = abs(med - 1.0)  # 0 = perfect
                print(
                    f"{DEVICE_LABEL.get(device, device):8} {app:13} {BE_LABEL[be]:4} {tt:9} "
                    f"{len(uids):>2}  median {med:6.2f}x [{lo:.2f}–{hi:.2f}]  |1-ratio|={acc:.2f}"
                )
            else:
                print(
                    f"{DEVICE_LABEL.get(device, device):8} {app:13} {BE_LABEL[be]:4} {tt:9} "
                    f"{'0':>2}  (no matched run UIDs)"
                )
        if len(ratios_by_table) == 2:
            b, i = abs(ratios_by_table["btpm"] - 1), abs(ratios_by_table["isolated"] - 1)
            if abs(b - i) < 0.05:
                tie += 1
            elif b < i:
                btpm_better += 1
            else:
                iso_better += 1

    print("-" * 78)
    print(
        f"Per-cell winner (predicted closest to measured): "
        f"btpm better on {btpm_better}, isolated better on {iso_better}, tie on {tie}."
    )
    print(
        "Paper's claim (BTPM predicts reality better than isolated) holds only if btpm wins "
        "clearly. Given the DVFS corruption of the BTPM GPU column, expect it NOT to until "
        "the harness controls GPU clocks."
    )


if __name__ == "__main__":
    main()
