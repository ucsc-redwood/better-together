#!/usr/bin/env python3
"""BEFORE vs NOW per (phone, app, table) for the end-to-end refresh
(docs/reports-for-human/2026-10-20-end-to-end-refresh.md).

Each side is a set of executed schedules (schedules root + sched_logs) and the isolated profiling
store its baseline is summed from. Per cell and side, from analysis/chunk_accuracy.py (the measure
of the 2026-10-17 to 2026-10-19 gates, per-stage cost model):

  best      the lowest measured period, [min-max] over its runs
  errors    median |measured / predicted - 1| of the period, the CPU chunks and the GPU chunks
  baseline  the fastest single PU over the whole app, summed from the isolated store
            (smt.baselines.get_baseline_for_config, as speedup_summary.py)
  speedup   baseline / best

`--before-isolated <app>=<subdir>` reads an app's BEFORE baseline from
<before-profiling>/<device>/<app>/vulkan/isolated/<subdir>/ (a superseded store).

  uv run optimizer/analysis/e2e_refresh.py \\
    --before-schedules-root data/archive/2026-10-17-gpu-chunk-cost-model \\
    --before-sched-logs data/archive/2026-10-17-gpu-chunk-cost-model/sched_logs \\
    --before-isolated tree=superseded-2026-10-17 --before-isolated cifar-dense=superseded-2026-10-18 \\
    --before-isolated cifar-sparse=superseded-2026-10-18
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smt.baselines import get_baseline_for_config  # noqa: E402
from smt.profiling_loader import load_profiling  # noqa: E402

from analysis.accuracy import APP_ORDER, discover_cells  # noqa: E402
from analysis.chunk_accuracy import DEVICE_LABEL, cell_summaries, fmt  # noqa: E402


def isolated_store(root, device, app, be, subdir=""):
    """Baseline (fastest ms, PU label) and the whole-app sum per PU {pu: ms} of the isolated
    store, optionally read from a subdirectory of it. The baseline is speedup_summary.py's: it
    drops a cell with cv > 1 on every PU, so a noisy GPU stage removes the GPU column. The sums
    keep the GPU's p50 regardless, as the solver does (smt.data_loader)."""
    with tempfile.TemporaryDirectory() as tmp:
        if subdir:
            link = os.path.join(tmp, device, app, "vulkan")
            os.makedirs(link)
            src = os.path.join(os.path.abspath(root), device, app, "vulkan", "isolated", subdir)
            os.symlink(src, os.path.join(link, "isolated"))
            root = tmp
        with contextlib.redirect_stdout(io.StringIO()):
            base = get_baseline_for_config(device, app, be, root)
            table, _ = load_profiling(
                root, device, app, "vulkan", "isolated", max_cv=1.0, cv_exempt_pus=("vulkan",)
            )
    sums, n = {}, {}
    for (_, pu), cell in table.items():
        sums[pu] = sums.get(pu, 0.0) + cell["value"]
        n[pu] = n.get(pu, 0) + 1
    sums = {pu: v for pu, v in sums.items() if n[pu] == max(n.values())}  # every stage measured
    if not base:
        return None, "-", sums
    pu = "OMP" if abs(base.get("omp", float("inf")) - base["fastest"]) < 1e-6 else "VK"
    return base["fastest"], pu, sums


def side(schedules_root, sched_logs, profiling, subdirs):
    """{(device, app, table): summary + baseline} and {(device, app): sum per PU} for one side."""
    out, stores = {}, {}
    for device, app, be in discover_cells(schedules_root, sched_logs):
        base, pu, stores[(device, app)] = isolated_store(
            profiling, device, app, be, subdirs.get(app, "")
        )
        for s in cell_summaries(device, app, be, schedules_root, sched_logs, None, "per-stage"):
            s.update(base=base, base_pu=pu, speedup=base / s["best"] if base else None)
            out[(device, app, s["table"])] = s
    return out, stores


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--before-schedules-root", required=True)
    ap.add_argument("--before-sched-logs", required=True)
    ap.add_argument("--before-profiling", default="data/profiling")
    ap.add_argument("--before-isolated", action="append", default=[], metavar="APP=SUBDIR")
    ap.add_argument("--schedules-root", default="data")
    ap.add_argument("--sched-logs", default="data/sched_logs")
    ap.add_argument("--profiling", default="data/profiling")
    ap.add_argument("--baselines", help="<dir>/<device>/<app>.json from bm-baseline-<app>-vk")
    args = ap.parse_args()

    before, before_stores = side(
        args.before_schedules_root,
        args.before_sched_logs,
        args.before_profiling,
        dict(a.split("=", 1) for a in args.before_isolated),
    )
    now, now_stores = side(args.schedules_root, args.sched_logs, args.profiling, {})

    def best(s):
        return f"{s['best']:.2f} [{s['best_min']:.2f}-{s['best_max']:.2f}]"

    def base(s):
        return f"{s['base_pu']} {s['base']:.2f}" if s["base"] else "-"

    def speedup(s):
        return f"{s['speedup']:.2f}x" if s["speedup"] else "-"

    print(
        "| Phone | App | Table | Best ms [min-max] before | now | Baseline ms before | now "
        "| Speedup before | now | Period error before | now | CPU error before | now "
        "| GPU error before | now | Best schedule now | Runs |"
    )
    print("|---" * 18 + "|")
    order = {a: i for i, a in enumerate(APP_ORDER)}
    for key in sorted(set(before) | set(now), key=lambda k: (k[0], order[k[1]], k[2])):
        b, n = before.get(key), now.get(key)
        device, app, table = key
        cols = []
        for get in (best, base, speedup):
            cols += [get(b) if b else "-", get(n) if n else "-"]
        for k in ("period_error", "cpu", "gpu"):
            cols += [fmt(b[k]) if b else "-", fmt(n[k]) if n else "-"]
        print(
            f"| {DEVICE_LABEL.get(device, device)} | {app} | {table} | {' | '.join(cols)} "
            f"| {n['best_uid'] if n else '-'} | {n['best_runs'] if n else '-'} |"
        )

    print(
        "\nWhole app in the isolated store as the solver reads it (GPU cells kept past the CV "
        "gate), sum of the per-stage p50 per PU, ms (before / now):\n"
    )
    pus = ["vulkan", "big", "medium", "little"]
    print("| Phone | App | " + " | ".join(pus) + " |")
    print("|---" * (2 + len(pus)) + "|")
    for device, app in sorted(
        set(before_stores) | set(now_stores), key=lambda k: (k[0], order[k[1]])
    ):
        b, n = before_stores.get((device, app), {}), now_stores.get((device, app), {})
        cols = [f"{b[p]:.2f} / {n[p]:.2f}" if p in b and p in n else "-" for p in pus]
        print(f"| {DEVICE_LABEL.get(device, device)} | {app} | " + " | ".join(cols) + " |")

    if args.baselines:
        print_measured_baselines(args.baselines, now, order)


BASELINE_PUS = {"": "VK", "/BigCores": "big", "/MediumCores": "medium", "/LittleCores": "little"}


def measured_baselines(path):
    """{PU: (median, min, max) real_time ms over the repetitions} of one bm-baseline JSON
    (google-benchmark JSON after the log lines bm-baseline prints first)."""
    with open(path) as fh:
        text = fh.read()
    runs = {}
    for b in json.loads(text[text.index("{") :])["benchmarks"]:
        if b["run_type"] != "iteration":
            continue
        name = b["name"]
        if name.startswith("OMP/") and name.endswith("/Baseline"):
            pu = "OMP all cores"
        else:
            pu = BASELINE_PUS.get(name[name.index("/Baseline") + len("/Baseline") :])
        runs.setdefault(pu, []).append(b["real_time"])
    return {pu: (statistics.median(v), min(v), max(v)) for pu, v in runs.items()}


def print_measured_baselines(root, now, order):
    """bm-baseline's whole app per PU (Step 5), and the best measured period of each table
    against the fastest single PU (VK or one CPU tier; `OMP all cores` is not one PU)."""
    pus = ["VK", "big", "medium", "little", "OMP all cores"]
    print("\nMeasured whole-app baseline (bm-baseline), ms, median [min-max] over repetitions:\n")
    print(
        "| Phone | App | "
        + " | ".join(pus)
        + " | Fastest single PU | Speedup btpm | Speedup isolated |"
    )
    print("|---" * (len(pus) + 5) + "|")
    for device, app in sorted({k[:2] for k in now}, key=lambda k: (k[0], order[k[1]])):
        path = os.path.join(root, device, f"{app}.json")
        if not os.path.exists(path):
            continue
        m = measured_baselines(path)
        cols = [f"{m[p][0]:.2f} [{m[p][1]:.2f}-{m[p][2]:.2f}]" if p in m else "-" for p in pus]
        fastest = min((m[p][0], p) for p in pus[:4] if p in m)
        speed = [
            f"{fastest[0] / now[(device, app, t)]['best']:.2f}x" if (device, app, t) in now else "-"
            for t in ("btpm", "isolated")
        ]
        print(
            f"| {DEVICE_LABEL.get(device, device)} | {app} | {' | '.join(cols)} "
            f"| {fastest[1]} {fastest[0]:.2f} | {' | '.join(speed)} |"
        )


if __name__ == "__main__":
    main()
