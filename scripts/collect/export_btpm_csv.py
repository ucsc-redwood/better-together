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
from case import Case, to_short_backend  # noqa: E402
from profiling_loader import read_records, validate  # noqa: E402

PU_COLS = ["little", "medium", "big", "vulkan", "cuda"]
# Scenarios that map to a wide CSV (isolated -> isolated.csv, interference -> btpm.csv,
# via Case.csv_path); other scenarios in the store are skipped.
VALID_SCENARIOS = {"isolated", "interference"}


def export_cell(root, out_root, device, app, backend, scenario, metric):
    # `backend` here is the profiling-store long name (cuda/vulkan); Case owns the layout.
    case = Case(device, app, to_short_backend(backend))
    paths = sorted(glob.glob(case.profiling_glob(root, scenario)))
    if not paths:
        return None
    records = read_records(paths)
    validate(records)

    # (run, stage) -> {pu: metric}
    rows = {}
    for r in records:
        rows.setdefault((r["run"], r["stage"]), {})[r["pu"]] = r["timing"][metric]

    out = case.csv_path(out_root, scenario)
    os.makedirs(os.path.dirname(out), exist_ok=True)
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
        if scenario not in VALID_SCENARIOS:
            continue
        res = export_cell(args.root, args.out_root, device, app, backend, scenario, args.metric)
        if res:
            out, nrows = res
            print(f"{out}  ({nrows} rows, metric={args.metric})")
            n += 1
    print(f"\nexported {n} CSV files under {args.out_root}")


if __name__ == "__main__":
    main()
