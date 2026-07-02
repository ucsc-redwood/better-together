#!/usr/bin/env python3
"""Fit the per-chunk framework-overhead constants from measured schedule runs.

The z3 cost model sums per-stage KERNEL times, so it under-predicts real chunks,
which additionally pay a per-chunk constant (SPSC handoff, thread wake) and a
per-stage dispatch tax (GPU submit + fence per stage). This script fits

    measured_chunk_ms  ~=  table_stage_sum  +  per_chunk_ms  +  n_stages * per_stage_ms

per (device, PU class) by least squares over every (schedule, chunk) the fleet run
measured, and writes  <profiling>/<device>/overhead.json  for the solver
(smt/overhead.py; 02_gen_schedule_merged applies it unless --no-overhead).

Classes: "cpu" (all CPU tiers), "gpu_cuda", "gpu_vulkan". The prediction side is
recomputed from the profiling table the schedule was solved on (btpm/isolated), NOT
from the schedule JSON's "time" field -- so refitting stays correct even after
schedules start carrying overhead-inclusive predictions. Parameters are clamped at
>= 0: a negative fit means the contended table over-predicts (e.g. DVFS boost),
which is a table problem, not negative framework overhead.

Usage:
  uv run optimizer/analysis/fit_overhead.py                # fit every device found
  ... --sched-logs data/sched_logs --profiling data/profiling --dry-run
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrate.case import table_to_scenario  # noqa: E402
from smt.bt_vocab import CORE_TYPES  # noqa: E402
from smt.data_loader import UNAVAILABLE, load_stage_timings  # noqa: E402

from analysis.results.log_parser import find_log_files, process_log_file  # noqa: E402
from analysis.results.statistics import calculate_aggregated_statistics  # noqa: E402

TABLES = ["btpm", "isolated"]
BACKENDS = ["cu", "vk"]
APPS = ["tree", "cifar-dense", "cifar-sparse"]
GPU_CLASS = {"cu": "gpu_cuda", "vk": "gpu_vulkan"}


def discover_cells(sched_logs_root):
    """{(device, app, be, table): log_folder} from <device>_<app>_<be>_<table> dirs."""
    cells = {}
    if not os.path.isdir(sched_logs_root):
        return cells
    for name in sorted(os.listdir(sched_logs_root)):
        path = os.path.join(sched_logs_root, name)
        if not os.path.isdir(path) or name.startswith("_"):
            continue
        parts = name.split("_")
        if len(parts) < 4:
            continue
        table, be, app, device = parts[-1], parts[-2], parts[-3], "_".join(parts[:-3])
        if table in TABLES and be in BACKENDS and app in APPS:
            cells[(device, app, be, table)] = path
    return cells


def measured_chunks(log_folder):
    """{schedule_uid: {chunk_id(int): avg_duration_ms}} for one cell's run logs."""
    logs = find_log_files(log_folder)
    data = []
    for lg in logs:
        data.extend(process_log_file(lg))
    if not data:
        return {}
    agg, _ = calculate_aggregated_statistics(data)
    out = {}
    for uid, stats in agg.items():
        out[uid] = {
            int(cid): c["avg_duration_ms"] for cid, c in (stats.get("chunks") or {}).items()
        }
    return out


def load_schedules(schedules_root, table, device, app, be):
    path = os.path.join(
        schedules_root.format(table=table), device, app, be, f"schedules_{table}_tmax.json"
    )
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        sols = json.load(f)
    return {s["uid"]: s for s in sols if "uid" in s}


def collect_samples(args):
    """[(device, class, n_stages, residual_ms)] across every measured chunk."""
    samples = []
    timings_cache = {}
    for (device, app, be, table), folder in discover_cells(args.sched_logs).items():
        key = (device, app, be, table)
        if key not in timings_cache:
            try:
                timings_cache[key] = load_stage_timings(
                    args.profiling, device, app, be, table_to_scenario(table)
                )[0]
            except Exception as e:  # noqa: BLE001 -- a missing/gated table just skips the cell
                print(f"  skip {device}/{app}/{be}/{table}: {e}")
                timings_cache[key] = None
        timings = timings_cache[key]
        if timings is None:
            continue

        scheds = load_schedules(args.schedules_root, table, device, app, be)
        for uid, chunks in measured_chunks(folder).items():
            sol = scheds.get(uid)
            if sol is None:
                continue
            for chunk in sol.get("chunks", []):
                meas = chunks.get(chunk["id"])
                if meas is None:
                    continue
                col = CORE_TYPES.index(chunk["core_type"])
                stage_sum = sum(
                    timings[k][col] for k in range(chunk["start_stage"] - 1, chunk["end_stage"])
                )
                if stage_sum >= UNAVAILABLE:
                    continue
                n = chunk["end_stage"] - chunk["start_stage"] + 1
                cls = chunk.get("hardware", GPU_CLASS[be]) if chunk["core_type"] == "GPU" else "cpu"
                samples.append((device, cls, n, meas - stage_sum))
    return samples


def fit(samples):
    """{device: {class: {per_chunk_ms, per_stage_ms, n_samples, resid_*}}}.

    Per class, pick the model that minimizes the mean |residual| among three
    candidates -- (0) no overhead, (1) robust intercept (median residual), (2)
    least-squares intercept+slope -- with parameters clamped at >= 0. Model
    selection matters: CPU residuals are dominated by skewed co-execution /
    DVFS effects that a constant does NOT explain (subtracting the mean makes
    most chunks WORSE); such classes must honestly fall back to zero rather
    than distort small-chunk decisions. The consistently-reproducible GPU
    submit/fence tax survives selection.
    """
    grouped = defaultdict(list)
    for device, cls, n, r in samples:
        grouped[(device, cls)].append((n, r))

    fits = defaultdict(dict)
    for (device, cls), pts in sorted(grouped.items()):
        ns = np.array([p[0] for p in pts], dtype=float)
        rs = np.array([p[1] for p in pts], dtype=float)

        candidates = [(0.0, 0.0, "none")]
        candidates.append((max(0.0, float(np.median(rs))), 0.0, "median"))
        if len(pts) >= 4 and np.ptp(ns) >= 1:
            coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(ns), ns]), rs, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            if a >= 0 and b >= 0:
                candidates.append((a, b, "lstsq"))

        def mean_abs(a, b):
            return float(np.abs(rs - (a + b * ns)).mean())

        a, b, model = min(candidates, key=lambda t: mean_abs(t[0], t[1]))
        fits[device][cls] = {
            "per_chunk_ms": round(a, 4),
            "per_stage_ms": round(b, 4),
            "model": model,
            "n_samples": len(pts),
            "mean_abs_resid_before_ms": round(mean_abs(0.0, 0.0), 3),
            "mean_abs_resid_after_ms": round(mean_abs(a, b), 3),
        }
    return fits


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sched-logs", default="data/sched_logs")
    ap.add_argument("--profiling", default="data/profiling")
    ap.add_argument(
        "--schedules-root",
        default="data/schedules_{table}",
        help="per-table schedule store; '{table}' expands to btpm/isolated",
    )
    ap.add_argument("--dry-run", action="store_true", help="print fits, write nothing")
    args = ap.parse_args()

    samples = collect_samples(args)
    if not samples:
        sys.exit("no (schedule, chunk) samples found -- run the fleet benchmark first")
    fits = fit(samples)

    print(f"\n{len(samples)} chunk samples across {len(fits)} device(s)\n")
    print(
        f"{'device':14s} {'class':11s} {'n':>4s} {'chunk ms':>9s} {'stage ms':>9s} {'|r| before':>11s} {'after':>7s}"
    )
    for device, classes in fits.items():
        for cls, e in classes.items():
            print(
                f"{device:14s} {cls:11s} {e['n_samples']:4d} {e['per_chunk_ms']:9.3f} "
                f"{e['per_stage_ms']:9.3f} {e['mean_abs_resid_before_ms']:11.3f} "
                f"{e['mean_abs_resid_after_ms']:7.3f}"
            )

    if args.dry_run:
        return
    for device, classes in fits.items():
        out = dict(classes)
        out["_provenance"] = {
            "fitted_from": args.sched_logs,
            "n_samples": sum(e["n_samples"] for e in classes.values()),
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        path = os.path.join(args.profiling, device, "overhead.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
