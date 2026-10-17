"""Prediction accuracy of executed schedules (optimizer/cost-model spec "Prediction
accuracy is reported for executed schedules"; design D6).

For every schedule a cost model proposed and the device executed:

  predicted  metrics.max_time of the schedule JSON (the period the solver priced)
  measured   the period from data/sched_logs: per log, each chunk's mean duration
             over its tasks, summed per PU (one device serves all chunks of its PU) and
             maxed over PUs -- the max-chunk makespan speedup_summary and validate_btpm
             always used, extended to several GPU chunks; the median over the logs
  ratio      measured / predicted

Measured runs are matched to schedules by their chunk assignment (PU, start, end per
chunk), not by uid: the uid hashes the predicted chunk times, so the same assignment
proposed by two cost models or tables has two uids but is one pipeline. A run measured
once therefore counts for every schedule file of the cell that proposed it, and every log
of the cell that ran the assignment counts for it (the folders of both tables).

Schedule files: schedules_<table>_<mode>[_<model>].json for both minimize modes (tmax,
gapness). A schedule's cost model and table are its recorded metrics.cost_model /
metrics.table (02 records the model that actually priced it: a range model without range
records falls back to per-stage); schedules written before those fields existed take them
from the file name. One row per (cost model, table, assignment): the same assignment priced
the same way in several files (both modes, or a per-stage fallback beside the per-stage
file) is one schedule, kept from the first file in the order tmax before gapness, then
per-stage, range-hot, range-idle by file name.

Per (device, app, backend, cost model, table) the summary gives the number of executed
schedules, the median of |ratio - 1|, and the best (lowest) measured period.
"""

import contextlib
import glob
import io
import json
import os
import re
import statistics

from analysis.results.log_parser import find_log_files, process_log_file

APP_ORDER = ["tree", "cifar-dense", "cifar-sparse"]
TABLES = ["btpm", "isolated"]
MODES = ["tmax", "gapness"]
COST_MODELS = ["per-stage", "range-hot", "range-idle"]  # smt.constraints.COST_MODELS
_SCHEDULE_FILE = re.compile(
    r"schedules_(btpm|isolated)_(tmax|gapness)(?:_(range-hot|range-idle))?\.json$"
)


def signature(schedule):
    """The chunk assignment of a schedule: what the executor actually runs."""
    return tuple((c["core_type"], c["start_stage"], c["end_stage"]) for c in schedule["chunks"])


def schedule_files(schedules_root, device, app, be):
    """A cell's schedule files under <schedules_root>/schedules_<table>/<device>/<app>/<be>/
    (02's --output_folder layout), both minimize modes: [{path, table, mode, cost_model}],
    the table and cost model as named by the file, ordered by table, mode, then cost model."""
    out = []
    pattern = os.path.join(schedules_root, "schedules_*", device, app, be, "schedules_*.json")
    for path in sorted(glob.glob(pattern)):
        m = _SCHEDULE_FILE.search(os.path.basename(path))
        if m:
            out.append(
                {
                    "path": path,
                    "table": m.group(1),
                    "mode": m.group(2),
                    "cost_model": m.group(3) or "per-stage",
                }
            )
    out.sort(
        key=lambda f: (
            TABLES.index(f["table"]),
            MODES.index(f["mode"]),
            COST_MODELS.index(f["cost_model"]),
        )
    )
    return out


def log_periods(log_folders, by_uid):
    """{signature: [period per log]} from every schedule_run_*.log in log_folders.
    by_uid maps each uid to its schedule (any file of the cell); unknown uids are skipped."""
    out = {}
    for folder in log_folders:
        for log in sorted(find_log_files(folder)) if os.path.isdir(folder) else []:
            with contextlib.redirect_stdout(io.StringIO()):  # log_parser is chatty on stdout
                runs = process_log_file(log)
            for run in runs:
                sched = by_uid.get(run["schedule_uid"])
                if sched is None or not run["chunks"]:
                    continue
                per_pu = {}
                for chunk_id, c in run["chunks"].items():
                    pu = sched["chunks"][chunk_id]["core_type"]
                    per_pu[pu] = per_pu.get(pu, 0.0) + c["avg_duration"]
                out.setdefault(signature(sched), []).append(max(per_pu.values()))
    return out


def cell_rows(device, app, be, schedules_root, sched_logs_root):
    """One row per (cost model, table, assignment) of the cell that was executed."""
    loaded = []
    for f in schedule_files(schedules_root, device, app, be):
        with open(f["path"]) as fh:
            loaded.append((f, json.load(fh)))
    by_uid = {s["uid"]: s for _, sols in loaded for s in sols}
    folders = sorted(glob.glob(os.path.join(sched_logs_root, f"{device}_{app}_{be}_*")))
    measured = log_periods(folders, by_uid)

    rows, seen = [], set()
    for f, sols in loaded:
        for s in sols:
            metrics = s.get("metrics") or {}
            model = metrics.get("cost_model", f["cost_model"])  # what priced it (02 fallback)
            table = metrics.get("table", f["table"])
            periods = measured.get(signature(s))
            predicted = metrics.get("max_time")
            if not periods or not predicted or (model, table, signature(s)) in seen:
                continue
            seen.add((model, table, signature(s)))
            med = statistics.median(periods)
            rows.append(
                {
                    "device": device,
                    "app": app,
                    "be": be,
                    "cost_model": model,
                    "table": table,
                    "mode": f["mode"],
                    "uid": s["uid"],
                    "signature": signature(s),
                    "predicted": predicted,
                    "measured": med,
                    "measured_min": min(periods),
                    "measured_max": max(periods),
                    "n_logs": len(periods),
                    "ratio": med / predicted,
                }
            )
    return rows


def discover_cells(schedules_root, sched_logs_root):
    """(device, app, be) cells with both schedule files and a sched_logs folder."""
    cells = set()
    for path in glob.glob(os.path.join(schedules_root, "schedules_*", "*", "*", "*")):
        device, app, be = path.split(os.sep)[-3:]
        if app in APP_ORDER and glob.glob(os.path.join(sched_logs_root, f"{device}_{app}_{be}_*")):
            cells.add((device, app, be))
    return sorted(cells, key=lambda c: (c[0], APP_ORDER.index(c[1]), c[2]))


def accuracy_rows(schedules_root, sched_logs_root):
    rows = []
    for device, app, be in discover_cells(schedules_root, sched_logs_root):
        rows.extend(cell_rows(device, app, be, schedules_root, sched_logs_root))
    return rows


def summarize(rows):
    """Per (device, app, be, cost_model, table): n, median |ratio - 1|, best measured."""
    groups = {}
    for r in rows:
        groups.setdefault((r["device"], r["app"], r["be"], r["cost_model"], r["table"]), []).append(
            r
        )
    out = []
    for (device, app, be, model, table), rs in groups.items():
        best = min(rs, key=lambda r: r["measured"])
        out.append(
            {
                "device": device,
                "app": app,
                "be": be,
                "cost_model": model,
                "table": table,
                "n": len(rs),
                "median_abs_error": statistics.median(abs(r["ratio"] - 1) for r in rs),
                "best_measured": best["measured"],
                "best_min": best["measured_min"],
                "best_max": best["measured_max"],
                "best_uid": best["uid"],
            }
        )
    return out
