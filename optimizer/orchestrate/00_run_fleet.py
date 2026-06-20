#!/usr/bin/env python3
"""Fleet benchmark orchestrator -- run the whole BT scheduler benchmark across every
hardware target concurrently, with live progress.

Each target is a physically independent machine (jetson over ssh, minipc=rocky local-ssh,
two phones over adb-on-rocky), so their pipelines run in PARALLEL: one worker thread per
device, each shelling out to the existing per-device steps. The only shared resources are
rocky (hosts minipc + both phones' adb) and this box (z3 + parsing); the heavy work is in
subprocesses/ssh, so the GIL is a non-issue.

Per device, per (app x backend in fleet.json):
  [build] -> 01 collect profiling -> for table in {btpm,isolated}: [02 gen schedule] +
  [03 run schedule]  -> (after all devices) speedup_summary.py.

Config:
  fleet.json           -- transport/build/caps per device (the source of truth for HOW
                          to reach each target). The (app x backend x hw) gate matrix is
                          fleet-coverage.json; what we BENCHMARK is app x device.backends.

Examples:
  uv run --project optimizer optimizer/orchestrate/00_run_fleet.py            # full fleet, all phases
  ... --only jetson,samsung                                                   # subset of devices
  ... --phases summary                                                        # just recompute the table
  ... --only jetson --phases profile --runs 3                                 # re-collect jetson profiling
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORCH = os.path.join(REPO, "optimizer", "orchestrate")
S01 = os.path.join(ORCH, "01_collect_profiling.py")
S02 = os.path.join(ORCH, "02_gen_schedule_merged.py")
S03 = os.path.join(ORCH, "03_run_schedule.py")
SUMMARY = os.path.join(REPO, "optimizer", "analysis", "speedup_summary.py")
FLEET_JSON = os.path.join(REPO, "fleet.json")
COVERAGE = os.path.join(REPO, "fleet-coverage.json")
FLEET_LOG_DIR = os.path.join(REPO, "data", "sched_logs", "_fleet")
PY = sys.executable
TABLES = ["btpm", "isolated"]
N_SOLUTIONS = 10
BUILD_RECIPE = {
    "vulkan": "build-bench-x86",
    "android": "build-bench-android",
    "jetson": "build-bench-jetson",
}

from rich.console import Console  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.table import Table  # noqa: E402

TTY = sys.stdout.isatty()  # non-TTY (CI / piped) -> plain log lines, no Live table
console = Console()

state = {}  # device name -> status dict
bstate = {}  # build artifact -> status dict


# -- helpers ------------------------------------------------------------------
def load_apps():
    cov = json.load(open(COVERAGE))
    return sorted(
        {c["app"] for c in cov["cells"]}, key=["tree", "cifar-dense", "cifar-sparse"].index
    )


def transport_args(dev):
    t = dev["transport"]
    if t["kind"] == "ssh":
        return ["--ssh-host", t["host"]]
    if t["kind"] == "adb":
        a = ["--adb-serial", t["serial"]]
        if t.get("adb_host"):
            a += ["--adb-host", t["adb_host"]]
        return a
    raise ValueError(f"unknown transport kind: {t['kind']}")


def sh(cmd, logf, on_line=None):
    """Run cmd, streaming stdout+stderr line-by-line to logf; return returncode.
    on_line(line) is called per line so the caller can surface live sub-step detail."""
    with open(logf, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert p.stdout is not None  # stdout=PIPE -> always a stream
        for line in p.stdout:
            f.write(line)
            f.flush()
            if on_line:
                on_line(line.rstrip("\n"))
        p.stdout.close()
        return p.wait()


def fmt(sec):
    return f"{int(sec) // 60:d}:{int(sec) % 60:02d}"


def wipe_device_results(dev_id):
    """Delete ALL regenerable result data for a device (profiling + schedules + run logs)
    so a re-run starts from scratch -- the cure for kernel/runtime changes, where stale
    cells from the OLD implementation would otherwise linger in the aggregated table."""
    removed = []
    for p in (
        os.path.join(REPO, "data", "profiling", dev_id),
        os.path.join(REPO, "data", "schedules_btpm", dev_id),
        os.path.join(REPO, "data", "schedules_isolated", dev_id),
        *glob.glob(os.path.join(REPO, "data", "sched_logs", f"{dev_id}_*")),
    ):
        if os.path.isdir(p):
            shutil.rmtree(p)
            removed.append(os.path.relpath(p, REPO))
    return removed


# -- step command builders ----------------------------------------------------
def cmd_01(dev, app, be, runs):
    return [
        PY,
        S01,
        "--device",
        dev["device_id"],
        "--app",
        app,
        "--backend",
        be,
        *transport_args(dev),
        "--build-dir",
        f"build/{dev['build']}",
        "--runs",
        str(runs),
    ]


def cmd_02(dev, app, be, tt):
    return [
        PY,
        S02,
        "--profiling_root",
        "data/profiling",
        "--device",
        dev["device_id"],
        "--app",
        app,
        "--backend",
        be,
        "--table_type",
        tt,
        "--minimize_mode",
        "tmax",
        "-n",
        str(N_SOLUTIONS),
        "-o",
        f"data/schedules_{tt}",
    ]


def cmd_03(dev, app, be, tt, cap, repeat):
    return [
        PY,
        S03,
        "--device",
        dev["device_id"],
        "--app",
        app,
        "--backend",
        be,
        *transport_args(dev),
        "--build-dir",
        f"build/{dev['build']}",
        "--table-type",
        tt,
        "--minimize-mode",
        "tmax",
        "--schedules-root",
        f"data/schedules_{tt}",
        "--log-folder",
        f"data/sched_logs/{dev['device_id']}_{app}_{be}_{tt}",
        "--repeat",
        str(repeat),
        "--n-schedules-to-run",
        str(cap),
    ]


# -- workers ------------------------------------------------------------------
def device_worker(name, dev, apps, phases, runs, repeat):
    s = state[name]
    s.update(status="running", t0=time.time())
    online = device_on_line(name)
    logf = os.path.join(FLEET_LOG_DIR, f"{name}.log")
    open(logf, "w").close()
    runs = dev.get("runs", runs)  # per-device override (e.g. noisy minipc -> more runs)
    cells = [(app, be) for app in apps for be in dev["backends"]]
    s["total"] = len(cells)

    def step(phase, cmd, label):
        s["phase"], s["detail"] = phase, ""
        if sh(cmd, logf, online):
            s["fail"].append(label)

    for app, be in cells:
        s["cell"] = f"{app}/{be}"
        if "profile" in phases:
            step("profile", cmd_01(dev, app, be, runs), f"profile {app}/{be}")
        for tt in TABLES:
            if "schedule" in phases:
                step(f"sched:{tt}", cmd_02(dev, app, be, tt), f"sched {app}/{be}/{tt}")
            if "run" in phases:
                cap = dev.get("caps", {}).get(app, 0)
                step(f"run:{tt}", cmd_03(dev, app, be, tt, cap, repeat), f"run {app}/{be}/{tt}")
        s["done"] += 1
        if not TTY:
            print(f"[{name}] {app}/{be} done ({s['done']}/{s['total']})", flush=True)
    s["status"] = "fail" if s["fail"] else "done"
    s["phase"] = "-"
    return s["fail"]


def build_worker(art):
    b = bstate[art]
    b.update(status="running", t0=time.time())
    logf = os.path.join(FLEET_LOG_DIR, f"build_{art}.log")
    open(logf, "w").close()
    if not TTY:
        print(f"[build] {art} ...", flush=True)
    rc = sh(["just", BUILD_RECIPE[art]], logf, build_on_line(art))
    b["status"] = "done" if rc == 0 else "fail"
    if not TTY:
        print(f"[build] {art} {b['status']}", flush=True)
    return rc


# -- live sub-step detail (parsed from each step's streamed stdout) ------------
def device_on_line(name):
    """Surface fine-grained progress in the device's Detail cell: profiling run
    (01 prints '>> <scenario> run N/M'), z3 candidate (02 prints 'Solution K:')."""
    s = state[name]

    def f(line):
        m = re.search(r">> (isolated|interference) run (\d+)/(\d+)", line)
        if m:
            s["detail"] = f"{m.group(1)[:3]} {m.group(2)}/{m.group(3)}"
        elif m := re.search(r"Solution (\d+):", line):
            s["detail"] = f"z3 {m.group(1)}/{N_SOLUTIONS}"
        elif "schedule record blocks" in line:
            s["detail"] = "ran candidates"

    return f


def build_on_line(art):
    """Surface cmake's [ NN%] progress in the build artifact's Detail cell."""
    b = bstate[art]

    def f(line):
        if m := re.search(r"\[\s*(\d+)%\]", line):
            b["detail"] = f"{m.group(1)}%"

    return f


# -- live rendering -----------------------------------------------------------
_COLOR = {"pending": "dim", "running": "yellow", "done": "green", "fail": "red"}


def render(title, rows, kind):
    t = Table(title=title, expand=False)
    if kind == "device":
        for c in ["Device", "Status", "Phase", "Cell", "Detail", "Progress", "Elapsed"]:
            t.add_column(c)
        for name in rows:
            s = state[name]
            el = fmt(time.time() - s["t0"]) if s["t0"] else ""
            t.add_row(
                name,
                f"[{_COLOR[s['status']]}]{s['status']}[/]",
                s["phase"],
                s["cell"],
                s["detail"],
                f"{s['done']}/{s['total']}",
                el,
            )
    else:
        for c in ["Build artifact", "Status", "Detail", "Elapsed"]:
            t.add_column(c)
        for art in rows:
            b = bstate[art]
            el = fmt(time.time() - b["t0"]) if b["t0"] else ""
            t.add_row(art, f"[{_COLOR[b['status']]}]{b['status']}[/]", b["detail"], el)
    return t


def drive(futures, title, rows, kind):
    if TTY:
        with Live(render(title, rows, kind), console=console, refresh_per_second=4) as live:
            while not all(f.done() for f in futures):
                time.sleep(0.25)
                live.update(render(title, rows, kind))
            live.update(render(title, rows, kind))
    else:
        for _ in as_completed(futures):
            pass


# -- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", help="comma-separated subset of fleet.json device names")
    ap.add_argument(
        "--phases",
        default="build,profile,schedule,run,summary",
        help="comma-separated subset of build,profile,schedule,run,summary",
    )
    ap.add_argument("--runs", type=int, default=3, help="profiling runs per scenario (step 01)")
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run each schedule this many times (step 03); the makespan is averaged across "
        "repeats. Use >1 to sample run-to-run variance of the chaotic environment (NOT to "
        "control it -- repetition characterizes the distribution, it doesn't lock anything).",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="START FROM SCRATCH: delete the selected devices' profiling/schedules/run-logs "
        "before running, so a kernel/runtime change can't leave stale old-implementation results "
        "in the aggregated table. (No --only -> wipes the whole fleet.)",
    )
    args = ap.parse_args()

    os.chdir(REPO)
    os.makedirs(FLEET_LOG_DIR, exist_ok=True)
    fleet = json.load(open(FLEET_JSON))["devices"]
    apps = load_apps()
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    names = [n.strip() for n in args.only.split(",")] if args.only else list(fleet)
    bad = [n for n in names if n not in fleet]
    if bad:
        sys.exit(f"unknown device(s): {bad}; known: {list(fleet)}")
    devices = {n: fleet[n] for n in names}
    print(f"fleet: {names} | apps: {apps} | phases: {phases} | logs: {FLEET_LOG_DIR}\n")

    if args.fresh:
        if not any(p in phases for p in ("profile", "schedule", "run")):
            sys.exit(
                "--fresh wipes results but no regenerating phase (profile/schedule/run) selected"
            )
        print("--fresh: wiping previous results for", names)
        for n in names:
            for p in wipe_device_results(devices[n]["device_id"]):
                print(f"  rm {p}")
        print()

    failures = {}

    # build phase (parallel artifact builds; only those the selected devices need)
    if "build" in phases:
        arts = sorted({d["build"] for d in devices.values()})
        for a in arts:
            bstate[a] = {"status": "pending", "t0": None, "detail": ""}
        with ThreadPoolExecutor(max_workers=len(arts)) as ex:
            futs = {ex.submit(build_worker, a): a for a in arts}
            drive(list(futs), "Build (benchmark binaries)", arts, "build")
        bad_builds = [a for a in arts if bstate[a]["status"] == "fail"]
        if bad_builds:
            print(
                f"\n!! build failed: {bad_builds} (see {FLEET_LOG_DIR}/build_*.log) -- aborting run"
            )
            sys.exit(2)

    # run phase (parallel device workers)
    if any(p in phases for p in ("profile", "schedule", "run")):
        for n in names:
            state[n] = {
                "status": "pending",
                "phase": "",
                "cell": "",
                "detail": "",
                "done": 0,
                "total": 0,
                "t0": None,
                "fail": [],
            }
        with ThreadPoolExecutor(max_workers=len(devices)) as ex:
            futs = {
                ex.submit(device_worker, n, devices[n], apps, phases, args.runs, args.repeat): n
                for n in names
            }
            drive(list(futs), "Fleet benchmark", names, "device")
            for f in futs:
                fl = f.result()
                if fl:
                    failures[futs[f]] = fl

    # summary phase (local, once)
    if "summary" in phases:
        slog = os.path.join(FLEET_LOG_DIR, "summary.log")
        open(slog, "w").close()
        rc = sh([PY, SUMMARY], slog)
        md = os.path.join(REPO, "data", "sched_logs", "speedup-summary.md")
        if os.path.isfile(md):
            print("\n" + open(md).read())
        if rc:
            print(f"!! summary failed (see {slog})")
            failures["summary"] = ["speedup_summary.py"]

    if failures:
        print("\n=== FAILURES ===")
        for k, v in failures.items():
            print(f"  {k}: {v}")
        sys.exit(1)
    print("\nfleet benchmark OK")


if __name__ == "__main__":
    main()
