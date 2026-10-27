#!/usr/bin/env python3
"""Compute the measured fleet speedups and emit data/sched_logs/speedup-summary.md.

ONE canonical method for every (device, app, backend) cell:

    speedup = baseline.fastest (ms/task)  /  best measured makespan (ms/task)

  - baseline.fastest: fastest single-PU whole-pipeline time, summed from the isolated
    profiling store (smt.baselines.get_baseline_for_config) -- OMP = fastest CPU tier,
    VK/CUDA = the GPU column.
  - measured makespan: min over the z3 candidate schedules that were RUN of the
    steady-state bottleneck (max chunk avg ms), taken over both profiling tables
    (btpm/isolated). Reuses the same log aggregation 04_parse_schedules.py prints
    (analysis.results log_parser + statistics) -- no matplotlib.

Cells are DISCOVERED by scanning data/sched_logs/<device>_<app>_<be>_<table>/ so the
table reflects exactly what ran. Output is the Markdown dashboard/generate.py parses.

The "Prediction accuracy" section (optimizer/cost-model spec) lists, for every executed
schedule of every cost model and table, measured period / predicted period, and per
(device, app, backend, cost model, table) the median |measured / predicted - 1| and the
best measured period (analysis/accuracy.py; schedules from <schedules-root>/schedules_*/).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smt.baselines import get_baseline_for_config  # noqa: E402

from analysis.accuracy import accuracy_rows, summarize  # noqa: E402
from analysis.results.log_parser import find_log_files, process_log_file  # noqa: E402
from analysis.results.statistics import calculate_aggregated_statistics  # noqa: E402

# device id (== the --device token / sched_log dir prefix) -> friendly summary label.
# dashboard/generate.py SUMMARY_DEVICE_ALIAS maps these labels back to ids.
DEVICE_LABEL = {
    "duck-stable": "duck-stable",
    "duck-naughty": "duck-naughty",
    "minipc": "minipc",
    "R5CY21Y3VEV": "samsung",
    "3A021JEHN02756": "pixel",
}
DEVICE_ORDER = ["duck-stable", "duck-naughty", "samsung", "minipc", "pixel"]
APP_ORDER = ["tree", "cifar-dense", "cifar-sparse"]
BE_LABEL = {"vk": "VK", "cu": "CUDA"}
TABLES = ["btpm", "isolated"]


def measured_makespan(log_folder):
    """min over run schedules of (max chunk steady-state avg ms). None if no logs."""
    logs = find_log_files(log_folder)
    if not logs:
        return None
    data = []
    for lg in logs:
        data.extend(process_log_file(lg))
    if not data:
        return None
    agg, _ = calculate_aggregated_statistics(data)
    best = None
    for stats in agg.values():
        chunks = stats.get("chunks") or {}
        if not chunks:
            continue
        mk = max(c["avg_duration_ms"] for c in chunks.values())
        best = mk if best is None or mk < best else best
    return best


def discover_cells(sched_logs_root):
    """Return {(device, app, be): {table: log_folder}} from the sched_logs dir names
    (<device>_<app>_<be>_<table>; device/app tokens never contain '_')."""
    cells = {}
    if not os.path.isdir(sched_logs_root):
        return cells
    for name in sorted(os.listdir(sched_logs_root)):
        path = os.path.join(sched_logs_root, name)
        if not os.path.isdir(path):
            continue
        parts = name.split("_")
        if len(parts) < 4:
            continue
        table, be, app, device = parts[-1], parts[-2], parts[-3], "_".join(parts[:-3])
        if table not in TABLES or be not in BE_LABEL or app not in APP_ORDER:
            continue
        cells.setdefault((device, app, be), {})[table] = path
    return cells


def compute_rows(sched_logs_root, prof_root):
    cells = discover_cells(sched_logs_root)
    rows = []
    for (device, app, be), tbl_paths in cells.items():
        try:
            base = get_baseline_for_config(device, app, be, prof_root)
        except Exception as e:  # noqa: BLE001
            print(f"  baseline FAIL {device}/{app}/{be}: {e}")
            base = None
        if not base or "fastest" not in base:
            print(f"  no baseline {device}/{app}/{be}")
            continue
        fast = base["fastest"]
        fast_pu = "OMP" if abs(base.get("omp", float("inf")) - fast) < 1e-6 else BE_LABEL[be]
        cand = {tt: measured_makespan(p) for tt, p in tbl_paths.items()}
        valid = {k: v for k, v in cand.items() if v}
        if not valid:
            print(f"  no makespan {device}/{app}/{be} {cand}")
            continue
        best_tt = min(valid, key=lambda k: valid[k])
        best = valid[best_tt]
        rows.append(
            {
                "device": device,
                "label": DEVICE_LABEL.get(device, device),
                "app": app,
                "be": be,
                "base_pu": fast_pu,
                "base": fast,
                "best": best,
                "best_tt": best_tt,
                "speedup": fast / best,
            }
        )
    # deterministic order: device, then app, then backend (cu before vk)
    rows.sort(
        key=lambda r: (
            DEVICE_ORDER.index(r["label"]) if r["label"] in DEVICE_ORDER else 99,
            APP_ORDER.index(r["app"]) if r["app"] in APP_ORDER else 99,
            r["be"],
        )
    )
    return rows


def _cell_order(r):
    return (
        DEVICE_ORDER.index(r["label"]) if r["label"] in DEVICE_ORDER else 99,
        APP_ORDER.index(r["app"]) if r["app"] in APP_ORDER else 99,
        r["be"],
        r["cost_model"],
        r["table"],
    )


def render_accuracy(acc_rows):
    """The prediction-accuracy section: per-(cell, cost model, table) medians, then every
    executed schedule's ratio."""
    for r in acc_rows:
        r["label"] = DEVICE_LABEL.get(r["device"], r["device"])
    summary = summarize(acc_rows)
    for r in summary:
        r["label"] = DEVICE_LABEL.get(r["device"], r["device"])
    out = [
        "## Prediction accuracy",
        "",
        "Measured period = per run log, each chunk's mean duration summed per PU, maxed over "
        "PUs; the median over the logs (spread [min-max]). Predicted = the schedule's "
        "metrics.max_time. Error = |measured / predicted - 1|. A run is matched to every "
        "schedule file of its cell that proposed the same chunk assignment, and an assignment's "
        "logs are pooled over the cell's log folders (Logs). Cost model and table are the ones "
        "the schedule records (metrics.cost_model / metrics.table); an assignment priced the "
        "same way in several files (both minimize modes, a per-stage fallback) counts once.",
        "",
        "| Device | App | Backend | Cost model | Table | Executed | Median abs error | "
        "Best measured ms [min-max] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(summary, key=_cell_order):
        out.append(
            f"| {r['label']} | {r['app']} | {BE_LABEL[r['be']]} | {r['cost_model']} | "
            f"{r['table']} | {r['n']} | {r['median_abs_error']:.3f} | {r['best_measured']:.2f} "
            f"[{r['best_min']:.2f}-{r['best_max']:.2f}] |"
        )
    out += [
        "",
        "### Executed schedules",
        "",
        "| Device | App | Backend | Cost model | Table | Mode | Schedule | Predicted ms | "
        "Measured ms [min-max] | Logs | Measured / predicted |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(acc_rows, key=lambda r: (_cell_order(r), r["predicted"])):
        out.append(
            f"| {r['label']} | {r['app']} | {BE_LABEL[r['be']]} | {r['cost_model']} | "
            f"{r['table']} | {r['mode']} | {r['uid']} | {r['predicted']:.2f} | "
            f"{r['measured']:.2f} [{r['measured_min']:.2f}-{r['measured_max']:.2f}] | "
            f"{r['n_logs']} | {r['ratio']:.3f} |"
        )
    return out


def render_markdown(rows, acc_rows=()):
    out = ["# Measured pipeline speedups (BetterTogether)", ""]
    out.append(
        "Measured speedup = fastest single-PU whole-pipeline baseline (ms/task, summed from "
        "isolated profiling) / best measured pipeline makespan (max-chunk steady-state, min over "
        "the z3 btpm/isolated tmax candidates that were run)."
    )
    out.append("")
    out.append("| Device | App | Backend | Baseline | Best | Speedup |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['label']} | {r['app']} | {BE_LABEL[r['be']]} | "
            f"{r['base_pu']} {r['base']:.2f} | {r['best_tt']} {r['best']:.2f} | {r['speedup']:.2f}x |"
        )
    out += [
        "",
        "## Reading the table",
        "- Baseline is the *fastest single processing unit* running the whole pipeline alone "
        "(OMP = the fastest CPU tier; VK/CUDA = the GPU); the cell names that PU and its ms/task.",
        "- Best is the best *measured* pipelined makespan across the z3 candidate schedules that "
        "were run, and which profiling table (btpm/isolated) z3 solved on.",
        "- Speedup > 1 means software-pipelining across CPU+GPU beat the best single PU.",
        "",
        "## Tree losses",
        "tree is a tiny integer pipeline (sub-ms stages); per-task framework overhead (per-stage "
        "GPU submit + fence round-trips) is a large fraction of its kernel work, so on devices "
        "with higher GPU overhead the pipelined makespan can exceed the fastest single PU "
        "(speedup < 1). This is a framework-overhead property of a tiny workload, not a kernel bug.",
        "",
        "## Caveats",
        "- Phone (samsung/pixel) cifar-sparse: only the best-predicted schedule(s) were swept "
        "where CPU-only candidates were too slow to run all ten.",
    ]
    if acc_rows:
        out += [""] + render_accuracy(list(acc_rows))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sched-logs", default="data/sched_logs")
    ap.add_argument("--profiling", default="data/profiling")
    ap.add_argument("--schedules-root", default="data", help="parent of schedules_<table>/")
    ap.add_argument("--out", default=None, help="default <sched-logs>/speedup-summary.md")
    args = ap.parse_args()

    rows = compute_rows(args.sched_logs, args.profiling)
    acc_rows = accuracy_rows(args.schedules_root, args.sched_logs)
    for r in rows:
        print(
            f"{r['label']:8s} {r['app']:13s} {BE_LABEL[r['be']]:5s}: base {r['base_pu']} "
            f"{r['base']:.2f}  best {r['best']:.2f} ({r['best_tt']})  speedup {r['speedup']:.2f}x"
        )
    out_path = args.out or os.path.join(args.sched_logs, "speedup-summary.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(render_markdown(rows, acc_rows))
    print(f"\nwrote {out_path} ({len(rows)} rows, {len(acc_rows)} executed schedules)")


if __name__ == "__main__":
    main()
