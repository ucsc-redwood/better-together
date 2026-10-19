#!/usr/bin/env python3
"""Per-chunk prediction error of executed schedules: the measure of the cpu-chunk-costs and
tree-cpu-pinning gates (docs/reports-for-human/2026-10-18-cpu-chunk-costs-gate.md, section 1).

For every executed schedule of a cell (analysis/accuracy.py: runs matched to schedules by
chunk assignment and pooled over both tables' log folders) and every chunk of it:

  measured   the median over the runs of the chunk's mean duration over its tasks
  predicted  the chunk's `time` in the schedule JSON, its price as solved
  ratio      measured / predicted

Per (device, app, backend, cost model, table):

  CPU / GPU error   the median |ratio - 1| over the CPU / GPU chunks of its executed schedules
  period error      the median |measured period / metrics.max_time - 1| (accuracy.py)
  best              the lowest measured period, [min-max] over its runs, and its uid

--overhead-root prices every chunk also without the overhead its schedules were solved under:
the chunk's time minus per_chunk_ms + n_stages * per_stage_ms of its class in
<overhead-root>/<device>/overhead.json (classes `cpu`, `gpu_vulkan`, `gpu_cuda`).

  uv run optimizer/analysis/chunk_accuracy.py --schedules-root data --sched-logs data/sched_logs \\
    --app tree
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.accuracy import (  # noqa: E402
    APP_ORDER,
    TABLES,
    cell_rows,
    discover_cells,
    schedule_files,
    signature,
)
from analysis.results.log_parser import find_log_files, process_log_file  # noqa: E402

DEVICE_LABEL = {"3A021JEHN02756": "Pixel 7a", "R5CY21Y3VEV": "Samsung"}


def chunk_durations(log_folders, by_uid):
    """{signature: [[mean duration of chunk i, i = 0..] per log]} from every log in log_folders."""
    out = {}
    for folder in log_folders:
        if not os.path.isdir(folder):
            continue
        with contextlib.redirect_stdout(io.StringIO()):  # log_parser is chatty on stdout
            runs = [run for log in sorted(find_log_files(folder)) for run in process_log_file(log)]
        for run in runs:
            sched = by_uid.get(run["schedule_uid"])
            if sched is None or len(run["chunks"]) != len(sched["chunks"]):
                continue
            durations = [run["chunks"][i]["avg_duration"] for i in range(len(sched["chunks"]))]
            out.setdefault(signature(sched), []).append(durations)
    return out


def overhead_ms(overhead, chunk):
    """What the overhead model added to the chunk's price."""
    if not overhead:
        return 0.0
    cls = chunk.get("hardware", "gpu_vulkan") if chunk["core_type"] == "GPU" else "cpu"
    entry = overhead.get(cls) or {}
    n_stages = chunk["end_stage"] - chunk["start_stage"] + 1
    return float(entry.get("per_chunk_ms", 0.0)) + n_stages * float(entry.get("per_stage_ms", 0.0))


def cell_summaries(device, app, be, schedules_root, sched_logs_root, overhead_root, cost_model):
    """One summary per table of the cell, for the schedules priced by `cost_model`."""
    by_uid = {}
    for f in schedule_files(schedules_root, device, app, be):
        with open(f["path"]) as fh:
            by_uid.update({s["uid"]: s for s in json.load(fh)})
    folders = [os.path.join(sched_logs_root, f"{device}_{app}_{be}_{t}") for t in TABLES]
    durations = chunk_durations(folders, by_uid)
    overhead = None
    if overhead_root:
        path = os.path.join(overhead_root, device, "overhead.json")
        if os.path.exists(path):
            with open(path) as fh:
                overhead = json.load(fh)

    with contextlib.redirect_stdout(io.StringIO()):  # find_log_files reports every folder
        rows = cell_rows(device, app, be, schedules_root, sched_logs_root)
    out = []
    for table in TABLES:
        rs = [r for r in rows if r["table"] == table and r["cost_model"] == cost_model]
        if not rs:
            continue
        err = {"cpu": [], "gpu": [], "cpu_plain": [], "gpu_plain": []}
        for r in rs:
            sched = by_uid[r["uid"]]
            per_chunk = list(zip(*durations[r["signature"]]))
            for chunk, measured in zip(sched["chunks"], per_chunk):
                m = statistics.median(measured)
                kind = "gpu" if chunk["core_type"] == "GPU" else "cpu"
                err[kind].append(abs(m / chunk["time"] - 1))
                err[kind + "_plain"].append(
                    abs(m / (chunk["time"] - overhead_ms(overhead, chunk)) - 1)
                )
        best = min(rs, key=lambda r: r["measured"])
        out.append(
            {
                "device": device,
                "app": app,
                "table": table,
                "n": len(rs),
                "cpu_n": len(err["cpu"]),
                **{
                    k: statistics.median(v) if v else None
                    for k, v in err.items()
                    if overhead or not k.endswith("_plain")
                },
                "period_error": statistics.median(abs(r["ratio"] - 1) for r in rs),
                "best": best["measured"],
                "best_min": best["measured_min"],
                "best_max": best["measured_max"],
                "best_runs": best["n_logs"],
                "best_uid": best["uid"],
            }
        )
    return out


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--schedules-root", default="data", help="holds schedules_<table>/")
    ap.add_argument("--sched-logs", default="data/sched_logs")
    ap.add_argument("--overhead-root", help="<root>/<device>/overhead.json the schedules used")
    ap.add_argument("--app", choices=APP_ORDER, help="only this app")
    ap.add_argument("--device", help="only this device")
    ap.add_argument("--cost-model", default="per-stage")
    args = ap.parse_args()

    plain = " | CPU error, no overhead | GPU error, no overhead" if args.overhead_root else ""
    print(
        f"| Phone | App | Table | Executed | CPU chunks | CPU error | GPU error{plain} "
        "| Period error | Best ms [min-max] | Runs | Best schedule |"
    )
    print("|---" * (11 + (2 if args.overhead_root else 0)) + "|")
    for device, app, be in discover_cells(args.schedules_root, args.sched_logs):
        if (args.app and app != args.app) or (args.device and device != args.device):
            continue
        for s in cell_summaries(
            device,
            app,
            be,
            args.schedules_root,
            args.sched_logs,
            args.overhead_root,
            args.cost_model,
        ):
            extra = (
                f" | {fmt(s['cpu_plain'])} | {fmt(s['gpu_plain'])}" if args.overhead_root else ""
            )
            print(
                f"| {DEVICE_LABEL.get(device, device)} | {app} | {s['table']} | {s['n']} | {s['cpu_n']} "
                f"| {fmt(s['cpu'])} | {fmt(s['gpu'])}{extra} | {fmt(s['period_error'])} "
                f"| {s['best']:.2f} [{s['best_min']:.2f}-{s['best_max']:.2f}] | {s['best_runs']} "
                f"| {s['best_uid']} |"
            )


if __name__ == "__main__":
    main()
