#!/usr/bin/env python3
"""Collect bm-prof profiling JSONL from one target into the canonical store.

Step 01 of the BT pipeline (the one that used to be done by hand): deploy the
prebuilt `bm-prof-<app>-<be>` to the target, run it for every scenario x run, and
capture its stdout (reserved for JSONL; logs go to stderr) into

  data/profiling/<device>/<app>/<backend_long>/<scenario>/run-00N.jsonl

`--tools ranges` runs `bm-ranges-<app>-vk` instead (every GPU stage range as one chunk,
schemas/profiling-range.schema.json) into range-00N.jsonl beside them; `--tools prof,ranges`
runs both, alternating, per (scenario, run). BT_PROF_* knobs set for this script (e.g.
BT_PROF_BUDGET_S, BT_PROF_MIN_ITERS) reach the device run.

The binary reads its cost model from the embedded device registry via `--device`,
and takes the run id / scenario / warmup from env (BT_PROF_RUN / BT_PROF_SCENARIO /
BT_PROF_WARMUP). Same deploy channels (ssh / adb-on-host) as 03_run_schedule.py.

Examples:
  # Jetson (ssh), CUDA, 3 runs of both scenarios
  uv run optimizer/orchestrate/01_collect_profiling.py --device duck-stable --app tree \
    --backend cu --ssh-host doremy@duck-stable --build-dir build/jetson
  # Samsung (adb on rocky), Vulkan
  ... --device R5CY21Y3VEV --backend vk --adb-serial R5CY21Y3VEV --adb-host rocky-ryzen \
      --build-dir build/android
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrate.case import Case  # noqa: E402
from orchestrate.transport import VK_QUIET_ENV, Target  # noqa: E402

DEFAULT_BUILD = {"vk": "build/vulkan", "cu": "build/jetson"}
DEFAULT_SCENARIOS = ["isolated", "interference"]
# tool -> (binary prefix, store file prefix): per-stage runs and GPU stage-range records.
TOOLS = {"prof": ("bm-prof", "run"), "ranges": ("bm-ranges", "range")}


def first_line_is_json(text):
    for line in text.splitlines():
        if line.strip():
            try:
                json.loads(line)
                return True
            except json.JSONDecodeError:
                return False
    return False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--device", required=True, help="value passed to the binary's --device")
    ap.add_argument("--app", required=True)
    ap.add_argument("--backend", required=True, choices=["vk", "cu"])
    ap.add_argument("--build-dir", help="dir holding bm-prof-<app>-<be> (default by backend)")
    ap.add_argument(
        "--ssh-host",
        help="deploy+run over ssh (jetsons=doremy@duck-{stable,naughty}, minipc=rocky-ryzen)",
    )
    ap.add_argument("--adb-serial", help="deploy+run over adb (phones)")
    ap.add_argument("--adb-host", help="run adb on this ssh host (phones attached to rocky-ryzen)")
    ap.add_argument(
        "--dest",
        help="device directory to deploy to and run in (default /tmp/bt, "
        "/data/local/tmp/bt on adb)",
    )
    ap.add_argument("--root", default="data/profiling", help="profiling-store root")
    ap.add_argument(
        "--tools",
        default="prof",
        help="comma-separated: prof (bm-prof -> run-NNN.jsonl), ranges (bm-ranges, Vulkan only "
        "-> range-NNN.jsonl)",
    )
    ap.add_argument(
        "--cooldown-s", type=float, default=0, help="idle seconds before every measured run"
    )
    ap.add_argument("--runs", type=int, default=3, help="runs per (scenario) cell")
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS), help="comma-separated")
    ap.add_argument("--warmup", type=int, default=20, help="BT_PROF_WARMUP (provenance)")
    ap.add_argument(
        "--keep",
        action="store_true",
        help="keep existing run-*.jsonl (additive); default wipes the scenario dir first so a "
        "re-collection cleanly REPLACES the previous one (the loaders average all run-*.jsonl, "
        "so a stale file from a prior --runs/implementation would otherwise mix in)",
    )
    args = ap.parse_args()

    if not args.ssh_host and not args.adb_serial:
        ap.error("need a transport: --ssh-host (jetson/minipc) or --adb-serial (phones)")

    tools = [t for t in args.tools.split(",") if t]
    if any(t not in TOOLS for t in tools) or ("ranges" in tools and args.backend != "vk"):
        ap.error(f"--tools: choose from {sorted(TOOLS)} (ranges: Vulkan only)")
    build_dir = args.build_dir or DEFAULT_BUILD[args.backend]
    binaries = {
        t: os.path.join(build_dir, f"{TOOLS[t][0]}-{args.app}-{args.backend}") for t in tools
    }
    for binary in binaries.values():
        if not os.path.exists(binary):
            sys.exit(f"missing: {binary}")

    if args.ssh_host:
        target = Target("ssh", host=args.ssh_host, dest=args.dest)
    else:
        target = Target("adb", serial=args.adb_serial, adb_host=args.adb_host, dest=args.dest)

    case = Case(args.device, args.app, args.backend)
    scenarios = [s for s in args.scenarios.split(",") if s]
    knobs = {
        k: v
        for k, v in os.environ.items()
        if k.startswith("BT_PROF_") and k not in ("BT_PROF_RUN", "BT_PROF_SCENARIO")
    }

    print(
        f"binaries {list(binaries.values())}\ntarget   {args.device} ({target.kind}, {target.dest})\n"
        f"scenarios {scenarios} x {args.runs} runs\n"
    )
    target.push(list(binaries.values()))

    fail = 0
    for scenario in scenarios:
        outdir = case.profiling_dir(args.root, scenario)
        os.makedirs(outdir, exist_ok=True)
        if not args.keep:
            # clean replace: drop any stale file of the collected kinds (e.g. from a larger prior
            # --runs or a different implementation) so the loaders don't average old data in.
            for t in tools:
                for stale in glob.glob(os.path.join(outdir, f"{TOOLS[t][1]}-*.jsonl")):
                    os.remove(stale)
        for run in range(1, args.runs + 1):
            for tool in tools:
                if args.cooldown_s > 0:
                    time.sleep(args.cooldown_s)
                # progress marker parsed by 00_run_fleet.py's live Detail column
                print(f">> {scenario} run {run}/{args.runs}", flush=True)
                out_path = os.path.join(outdir, f"{TOOLS[tool][1]}-{run:03d}.jsonl")
                env = {
                    **VK_QUIET_ENV,
                    **knobs,
                    "BT_PROF_RUN": run,
                    "BT_PROF_SCENARIO": scenario,
                    "BT_PROF_WARMUP": args.warmup,
                    # Real weights, fail-loud (deploy-weights.sh first): sparse stage
                    # timings depend on the real CSR pattern.
                    "BT_WEIGHTS_DIR": f"{target.dest}/weights",
                }
                bname = os.path.basename(binaries[tool])
                cmd = f"LD_LIBRARY_PATH=. ./{bname} --device {args.device}"
                t0 = time.monotonic()
                out = target.exec(cmd, env=env)
                n = sum(1 for line in out.splitlines() if line.strip())
                ok = n > 0 and first_line_is_json(out)
                with open(out_path, "w") as f:
                    f.write(out)
                status = "ok" if ok else "EMPTY/INVALID"
                print(f"  -> {out_path} ({n} records, {time.monotonic() - t0:.0f} s) {status}")
                if not ok:
                    fail = 1

    print(f"\ndone (fail={fail}).")
    sys.exit(fail)


if __name__ == "__main__":
    main()
