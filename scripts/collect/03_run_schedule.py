#!/usr/bin/env python3
"""Run z3-generated schedules on a target and capture the per-task logs.

Replaces the retired xmake + HTTP-schedule-server path. The executor
`bm-gen-logs-<app>-<vk|cu>` now reads the schedule from a LOCAL file
(schedule_source.hpp; no libcurl/IP), so this script:

  1. picks the schedule JSON 02 wrote
       <schedules-root>/<device>/<app>/<backend>/schedules_<table>_<mode>.json
  2. deploys the prebuilt binary + that JSON to the target (scp over ssh, or
     adb push -- the same channels run-on-*.sh uses; fish/adb gotchas handled)
  3. runs `./bm-gen-logs-<app>-<be> --device <device> --schedule-file <path>
     --n-schedules-to-run N`, capturing stdout to <log-folder>/schedule_run_<i>.log
     (the name 04_parse_schedules.py expects), repeat times.

Examples:
  # Jetson (ssh), btpm table, tmax mode
  uv run scripts/collect/03_run_schedule.py --device jetson --app cifar-dense \
    --backend vk --ssh-host duck-naughty --build-dir build/jetson \
    --table-type btpm --minimize-mode tmax --log-folder data/sched_logs --repeat 1
  # MiniPC (ssh)
  ... --device minipc --backend vk --ssh-host rocky-ryzen --build-dir build/vulkan ...
  # Samsung (adb running on rocky)
  ... --device R5CY21Y3VEV --backend vk --adb-serial R5CY21Y3VEV --adb-host rocky-ryzen --build-dir build/android ...
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case import Case  # noqa: E402

DEFAULT_BUILD = {"vk": "build/vulkan", "cu": "build/jetson"}


def run(cmd, check=True, **kw):
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, check=check, **kw)


def deploy_and_run_ssh(host, binary, schedule, device, dest, n_sched):
    """scp binary+schedule to host:dest, run the executor, return stdout text."""
    run(["ssh", host, "bash", "-s"], input=f"mkdir -p {dest}\n", text=True)
    run(["scp", binary, schedule, f"{host}:{dest}/"])
    bname, sname = os.path.basename(binary), os.path.basename(schedule)
    # quoted heredoc-equivalent: fish login shell bypassed via `bash -s` over stdin
    script = (
        f"cd {dest}\n"
        f"LD_LIBRARY_PATH=. ./{bname} --device {device} "
        f"--schedule-file {sname} --n-schedules-to-run {n_sched}\n"
    )
    # check=False: VK executors can segfault on TEARDOWN (Tegra, bugs-found §9) after
    # the records are already on stdout -- keep the captured output regardless of exit.
    p = run(["ssh", host, "bash", "-s"], input=script, text=True, stdout=subprocess.PIPE, check=False)
    return p.stdout


def deploy_and_run_adb(serial, adb_host, binary, schedule, device, n_sched):
    """adb push + run on the phone; adb itself may run on a remote host (--adb-host)."""
    dest = "/data/local/tmp/bt"
    bname, sname = os.path.basename(binary), os.path.basename(schedule)

    if adb_host:
        # binary/schedule live on THIS box -> stage to the adb host, then push
        run(["scp", binary, schedule, f"{adb_host}:/tmp/bt-android/"])
        sh = (f"adb -s {serial} shell 'mkdir -p {dest}' </dev/null\n"
              f"adb -s {serial} push /tmp/bt-android/{bname} /tmp/bt-android/{sname} {dest}/ </dev/null\n"
              f"adb -s {serial} shell 'chmod 755 {dest}/{bname}' </dev/null\n"
              f"adb -s {serial} shell \"cd {dest} && LD_LIBRARY_PATH=. ./{bname} "
              f"--device {device} --schedule-file {sname} --n-schedules-to-run {n_sched}\" </dev/null\n")
        p = run(["ssh", adb_host, "bash", "-s"], input=sh, text=True, stdout=subprocess.PIPE, check=False)
        return p.stdout.replace("\r", "")

    adb = ["adb", "-s", serial]
    run(adb + ["shell", f"mkdir -p {dest}"], stdin=subprocess.DEVNULL)
    run(adb + ["push", binary, schedule, dest + "/"], stdin=subprocess.DEVNULL)
    run(adb + ["shell", f"chmod 755 {dest}/{bname}"], stdin=subprocess.DEVNULL)
    p = run(adb + ["shell",
                   f"cd {dest} && LD_LIBRARY_PATH=. ./{bname} --device {device} "
                   f"--schedule-file {sname} --n-schedules-to-run {n_sched}"],
            stdin=subprocess.DEVNULL, text=True, stdout=subprocess.PIPE, check=False)
    return p.stdout.replace("\r", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, help="value passed to the binary's --device")
    ap.add_argument("--app", required=True)
    ap.add_argument("--backend", required=True, choices=["vk", "cu"])
    ap.add_argument("--table-type", default="btpm", choices=["isolated", "btpm"])
    ap.add_argument("--minimize-mode", default="tmax", choices=["gapness", "tmax"])
    ap.add_argument("--schedules-root", default="data/schedules_btpm")
    ap.add_argument("--schedule-file", help="explicit path; overrides schedules-root construction")
    ap.add_argument("--build-dir", help="dir holding bm-gen-logs-<app>-<be> (default by backend)")
    ap.add_argument("--ssh-host", help="deploy+run over ssh/scp (jetson=duck-naughty, minipc=rocky-ryzen)")
    ap.add_argument("--adb-serial", help="deploy+run over adb (phones)")
    ap.add_argument("--adb-host", help="run adb on this ssh host (both phones are attached to rocky-ryzen)")
    ap.add_argument("--log-folder", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--n-schedules-to-run", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    if not args.ssh_host and not args.adb_serial:
        ap.error("need a transport: --ssh-host (jetson/minipc) or --adb-serial (phones)")

    build_dir = args.build_dir or DEFAULT_BUILD[args.backend]
    binary = os.path.join(build_dir, f"bm-gen-logs-{args.app}-{args.backend}")
    schedule = args.schedule_file or Case(
        args.device, args.app, args.backend
    ).schedule_path(args.schedules_root, args.table_type, args.minimize_mode)
    for p in (binary, schedule):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    os.makedirs(args.log_folder, exist_ok=True)
    print(f"binary   {binary}\nschedule {schedule}\ntarget   {args.device}\n")

    for i in range(1, args.repeat + 1):
        print(f"=== run {i}/{args.repeat} ===")
        if args.adb_serial:
            out = deploy_and_run_adb(args.adb_serial, args.adb_host, binary, schedule,
                                     args.device, args.n_schedules_to_run)
        else:
            out = deploy_and_run_ssh(args.ssh_host, binary, schedule, args.device,
                                     "/tmp/bt", args.n_schedules_to_run)
        log = os.path.join(args.log_folder, f"schedule_run_{i}.log")
        with open(log, "w") as f:
            f.write(out)
        print(f"  -> {log}  ({out.count('### Python Begin ###')} schedule record blocks)")

    print(f"\ndone. parse with: uv run scripts/collect/04_parse_schedules.py {args.log_folder}")


if __name__ == "__main__":
    main()
