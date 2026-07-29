#!/usr/bin/env python3
"""Shared deploy+exec channels to the fleet (ssh / adb / local).

One place for the subtle gotchas that 01_collect_profiling.py and 03_run_schedule.py
both depend on:

  - rocky-ryzen's login shell is **fish** (the reflashed Jetsons are bash) -> we pipe a
    bash script over stdin via `ssh HOST bash -s` for every ssh target (never
    `ssh HOST '...'`, which fish re-parses).
  - `adb shell` reads the script's stdin, so every adb call gets `</dev/null`.
  - both phones are attached to rocky-ryzen, so adb runs on a remote `adb_host`.
  - `VK_LOADER_LAYERS_DISABLE='~all~'` keeps the desktop Vulkan validation layers off
    the measured path (no-op where absent, e.g. phones/CUDA).
  - capture survives teardown: VK executors can segfault on teardown (Tegra) AFTER the
    records are already on stdout, so exec never raises on a non-zero exit.
"""

import os
import subprocess

# env that keeps Vulkan validation layers from inflating measured times (desktop loader).
VK_QUIET_ENV = {"VK_LOADER_LAYERS_DISABLE": "~all~"}


def _envprefix(env):
    # quote values so e.g. '~all~' is not tilde-expanded by the remote shell.
    return "".join(f"{k}='{v}' " for k, v in (env or {}).items())


def _run(cmd, **kw):
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, **kw)


class Target:
    """A deploy+exec channel to one fleet device. kind in {ssh, adb, local}."""

    def __init__(self, kind, host=None, serial=None, adb_host=None, dest=None):
        self.kind = kind
        self.host = host
        self.serial = serial
        self.adb_host = adb_host
        self.dest = dest or ("/data/local/tmp/bt" if kind == "adb" else "/tmp/bt")

    @classmethod
    def from_config(cls, t):
        """Build from a fleet.json transport dict: {kind, host|serial, adb_host?}."""
        return cls(
            t["kind"],
            host=t.get("host"),
            serial=t.get("serial"),
            adb_host=t.get("adb_host"),
        )

    # -- deploy ---------------------------------------------------------------
    def push(self, files, dest=None):
        """Copy local files into dest on the target (chmod +x on adb)."""
        dest = dest or self.dest
        files = list(files)
        if self.kind == "ssh":
            _run(
                ["ssh", self.host, "bash", "-s"], input=f"mkdir -p {dest}\n", text=True, check=True
            )
            _run(["scp", *files, f"{self.host}:{dest}/"], check=True)
        elif self.kind == "local":
            os.makedirs(dest, exist_ok=True)
            for f in files:
                if os.path.abspath(os.path.dirname(f) or ".") != os.path.abspath(dest):
                    _run(["cp", f, dest + "/"], check=True)
        elif self.kind == "adb":
            bnames = [os.path.basename(f) for f in files]
            if self.adb_host:
                _run(["scp", *files, f"{self.adb_host}:/tmp/bt-android/"], check=True)
                staged = " ".join(f"/tmp/bt-android/{b}" for b in bnames)
                chmods = "".join(
                    f"adb -s {self.serial} shell 'chmod 755 {dest}/{b}' </dev/null\n"
                    for b in bnames
                )
                sh = (
                    f"adb -s {self.serial} shell 'mkdir -p {dest}' </dev/null\n"
                    f"adb -s {self.serial} push {staged} {dest}/ </dev/null\n" + chmods
                )
                _run(["ssh", self.adb_host, "bash", "-s"], input=sh, text=True, check=True)
            else:
                adb = ["adb", "-s", self.serial]
                _run(adb + ["shell", f"mkdir -p {dest}"], stdin=subprocess.DEVNULL, check=True)
                _run(adb + ["push", *files, dest + "/"], stdin=subprocess.DEVNULL, check=True)
                for b in bnames:
                    _run(
                        adb + ["shell", f"chmod 755 {dest}/{b}"],
                        stdin=subprocess.DEVNULL,
                        check=True,
                    )
        else:
            raise ValueError(f"unknown transport kind: {self.kind}")

    # -- run ------------------------------------------------------------------
    def exec(self, cmd, dest=None, env=None, capture=True):
        """Run `cmd` in dest (with env prefixed); return stdout text. Never raises on
        non-zero exit so a teardown segfault doesn't lose the already-emitted records."""
        dest = dest or self.dest
        full = f"cd {dest} && {_envprefix(env)}{cmd}"
        out = subprocess.PIPE if capture else None
        if self.kind == "ssh":
            p = _run(
                ["ssh", self.host, "bash", "-s"],
                input=full + "\n",
                text=True,
                stdout=out,
                check=False,
            )
            return p.stdout or "" if capture else ""
        if self.kind == "local":
            p = _run(["bash", "-lc", full], text=True, stdout=out, check=False)
            return p.stdout or "" if capture else ""
        if self.kind == "adb":
            if self.adb_host:
                sh = f'adb -s {self.serial} shell "{full}" </dev/null\n'
                p = _run(
                    ["ssh", self.adb_host, "bash", "-s"],
                    input=sh,
                    text=True,
                    stdout=out,
                    check=False,
                )
            else:
                p = _run(
                    ["adb", "-s", self.serial, "shell", full],
                    stdin=subprocess.DEVNULL,
                    text=True,
                    stdout=out,
                    check=False,
                )
            return (p.stdout or "").replace("\r", "") if capture else ""
        raise ValueError(f"unknown transport kind: {self.kind}")


# -- thin wrappers preserving 03_run_schedule.py's original signatures ----------
def deploy_and_run_ssh(host, binary, schedule, device, dest, n_sched):
    """scp binary+schedule to host:dest, run the executor, return stdout text."""
    t = Target("ssh", host=host, dest=dest)
    t.push([binary, schedule])
    bname, sname = os.path.basename(binary), os.path.basename(schedule)
    cmd = (
        f"LD_LIBRARY_PATH=. ./{bname} --device {device} "
        f"--schedule-file {sname} --n-schedules-to-run {n_sched}"
    )
    # Real weights, fail-loud (deploy-weights.sh first). Sparse stage timings
    # depend on the real CSR pattern, so benchmarking synthetic would be a lie.
    return t.exec(cmd, env={**VK_QUIET_ENV, "BT_WEIGHTS_DIR": f"{t.dest}/weights"})


def deploy_and_run_adb(serial, adb_host, binary, schedule, device, n_sched):
    """adb push + run on the phone; adb itself may run on a remote host (adb_host)."""
    t = Target("adb", serial=serial, adb_host=adb_host)
    t.push([binary, schedule])
    bname, sname = os.path.basename(binary), os.path.basename(schedule)
    cmd = (
        f"LD_LIBRARY_PATH=. ./{bname} --device {device} "
        f"--schedule-file {sname} --n-schedules-to-run {n_sched}"
    )
    return t.exec(cmd, env={"BT_WEIGHTS_DIR": f"{t.dest}/weights"})
