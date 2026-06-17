#!/usr/bin/env python3
"""Bridge the canonical JSONL store to the legacy wide-CSV the z3 consumer expects.

The SMT scheduler (02_gen_schedule_merged.py -> smt/data_loader.py) reads
`<root>/<device>/<app>/<backend>/{isolated,btpm}.csv` with columns
    stage,little,medium,big,vulkan,cuda,device,run
grouped-by-stage and averaged. This script regenerates those CSVs from the
schema-validated JSONL store, so the new profiler feeds z3 unchanged:

    isolated   scenario -> isolated.csv
    interference scenario -> btpm.csv   (the paper's BTPM / interference table)

One CSV row per (run, stage); each PU column is that run's chosen metric (p50 by
default), 0.0 for a PU the device lacks (matching the legacy "absent = 0.0" that
data_loader's use_cuda test relies on). Validates every record before writing.

    uv run scripts/collect/export_btpm_csv.py --root data/profiling-wall
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profiling_loader import read_records, validate  # noqa: E402

PU_COLS = ["little", "medium", "big", "vulkan", "cuda"]
SCEN_OUT = {"isolated": "isolated.csv", "interference": "btpm.csv"}
# Canonical store backend dir -> legacy dir name the z3 consumer's --backend uses.
BACKEND_OUT = {"vulkan": "vk", "cuda": "cu"}


def export_cell(root, out_root, device, app, backend, scenario, metric):
    paths = sorted(glob.glob(f"{root}/{device}/{app}/{backend}/{scenario}/run-*.jsonl"))
    if not paths:
        return None
    records = read_records(paths)
    validate(records)

    # (run, stage) -> {pu: metric}
    rows = {}
    for r in records:
        rows.setdefault((r["run"], r["stage"]), {})[r["pu"]] = r["timing"][metric]

    out_dir = os.path.join(out_root, device, app, BACKEND_OUT.get(backend, backend))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, SCEN_OUT[scenario])
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage"] + PU_COLS + ["device", "run"])
        for run, stage in sorted(rows):
            d = rows[(run, stage)]
            w.writerow([stage] + [f"{d.get(c, 0.0):.6f}" for c in PU_COLS] + [device, run])
    return out, len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/profiling")
    ap.add_argument("--out-root", default="data/btpm_export",
                    help="where the legacy CSVs land (device/app/{vk,cu}/{isolated,btpm}.csv)")
    ap.add_argument("--metric", default="p50", choices=["p50", "p95", "p99", "mean"])
    args = ap.parse_args()

    # discover every (device, app, backend, scenario) cell with run files
    cells = set()
    for p in glob.glob(f"{args.root}/*/*/*/*/run-*.jsonl"):
        rel = os.path.relpath(p, args.root).split(os.sep)
        cells.add((rel[0], rel[1], rel[2], rel[3]))  # device, app, backend, scenario

    n = 0
    for device, app, backend, scenario in sorted(cells):
        if scenario not in SCEN_OUT:
            continue
        res = export_cell(args.root, args.out_root, device, app, backend, scenario, args.metric)
        if res:
            out, nrows = res
            print(f"{out}  ({nrows} rows, metric={args.metric})")
            n += 1
    print(f"\nexported {n} CSV files under {args.out_root}")


if __name__ == "__main__":
    main()
