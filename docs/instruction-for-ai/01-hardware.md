# Hardware — the test fleet, specs & how to access

Every backend must be validated on real hardware. This is the single reference for
**what each target is, what it tests, and exactly how to reach and deploy to it.**

> **Deploy convention:** push built binaries, libs, and artifacts to the target's
> **tmp**, never the home dir — `/tmp/bt/` on Linux hosts (Jetson, rocky-ryzen),
> `/data/local/tmp/bt/` on Android. `ssh <host> 'mkdir -p /tmp/bt'` first (scp won't
> create it). Keeps home clean and the staging area self-cleaning across reboots.

## Deploy & run — use the scripts (they encode the gotchas)

Don't hand-write the `scp`/`ssh`/`adb` dance — call the helper for the target. Each
builds nothing (build first with the matching preset, see [`02-building.md`](02-building.md)),
stages to the tmp dir above, runs every target with the right `--device`, and exits
non-zero if any test fails:

```bash
scripts/run-on-jetson.sh                       # build/jetson  → doremy@duck-stable (CUDA tests)
# twin devkit: BT_JETSON_HOST=doremy@duck-naughty BT_JETSON_DEVICE=duck-naughty \
#              BT_CELL_HW=duck-naughty scripts/run-on-jetson.sh
scripts/run-on-rocky.sh                         # build/vulkan  → rocky-ryzen   (Vulkan tests)
# Both phones' adb lives on rocky-ryzen now → copy build/android there & run the script on it:
scripts/run-on-android.sh 3A021JEHN02756        # build/android → Pixel 7a   (adb -s picks the phone)
scripts/run-on-android.sh R5CY21Y3VEV           # build/android → Samsung    (same host, other serial)
scripts/run-mali-oracle.sh                      # build/android → BOTH phones (Mali oracle gate, run from the build box)
# pass explicit targets to narrow:  scripts/run-on-jetson.sh test-tree-cu
```

> **Two gotchas the scripts handle for you — hit them if you hand-roll commands:**
> 1. **rocky-ryzen's login shell is fish** (the reflashed Jetsons are plain bash). An
>    inline `ssh HOST '… for/VAR=… …'` dies with a fish parse error on fish hosts.
>    Pipe a bash script over stdin instead: `ssh HOST bash -s <<'EOF' … EOF` (bypasses
>    the login shell) — the scripts do this for every ssh target, fish or not.
> 2. **`adb shell` swallows the rest of stdin.** Inside a heredoc the first `adb shell`
>    eats the remaining script lines (exit 0, empty output). Suffix **every** `adb`
>    call with `</dev/null`.

The machine-readable CPU topology for each device lives in `../../devices/<id>.json`
(validated against `../../schemas/device-spec.schema.json`); that registry is the
source of truth for core tiers and pinning. This doc is the human/agent-readable
**access + role** layer on top of it. To add a device, see
[`../../devices/README.md`](../../devices/README.md).

## CI runners on this fleet (org-level, live 2026-07-02)

Every machine above doubles as a GitHub Actions self-hosted runner for the
`ucsc-redwood` org, selected by **capability label, never by name**:
`[self-hosted, rdna3]` (Rocky box: iGPU Vulkan, podman, adb host),
`[self-hosted, android, pixel|samsung]` (each phone is its own runner with
`ANDROID_SERIAL` pinned — plain `adb` works, jobs per phone serialize),
`[self-hosted, jetson, cuda]` (+ `duck-stable`/`duck-naughty` to pin a box;
arm64 — build natively, x86 binaries will not run). Runners are persistent and
the runner user has **no sudo** — missing system deps go to the fleet owner's
Ansible role, never into workflows. Fork PRs must never target these runners.
The health check is `.github/workflows/runner-smoke.yml`; the nightly
hardware-in-the-loop gate is `.github/workflows/fleet.yml`.

## Target matrix at a glance

| Target | `--device` id | Backends it runs | Access | Role |
|---|---|---|---|---|
| Old build/dev box | `pc` | OMP (CUDA/Vulkan **build-only**) | (retired from workflow) | no longer needed — rocky builds everything |
| Jetson Orin #1 | `duck-stable` | OMP + CUDA + Vulkan | `ssh doremy@duck-stable` | primary CUDA target, coverage-gated |
| Jetson Orin #2 | `duck-naughty` | OMP + CUDA + Vulkan | `ssh doremy@duck-naughty` | twin devkit, benchmark-only |
| Rocky MiniPC | `minipc` | OMP + Vulkan | `ssh doremy@rocky-ryzen` | **the build host** (x86 Vulkan native + Jetson cross via podman); Vulkan (iGPU) runner; **adb host for both phones** |
| Pixel 7a | `3A021JEHN02756` | OMP + Vulkan (**subgroup 16**) | adb on `rocky-ryzen` (`adb -s`) | Mali subgroup-16 shader variant |
| Samsung Galaxy | `R5CY21Y3VEV` | OMP + Vulkan (**subgroup 32**) | adb on `rocky-ryzen` (`adb -s`) | subgroup-32 shader variant |

Key hardware constraint: `platform/engine/vulkan/base_engine.cpp` **hard-selects
an integrated GPU** (`eIntegratedGpu`) — discrete GPUs throw "No integrated GPU
found", which is why the build box's RTX 4070 Ti never runs Vulkan.

---

## Where builds happen (2026-07-02): rocky-ryzen for everything

The old i9/RTX build box is **no longer part of the workflow**. All builds run on
**rocky-ryzen** (or wherever convenient — nothing is machine-specific anymore):

- **x86 Vulkan** (`vulkan` preset): builds **natively on rocky** (gcc 14 + cmake,
  verified 2026-07-02) — then runs right there (it *is* the minipc target).
- **Jetson CUDA+Vulkan** (`jetson` preset): cross-compiles on rocky inside the
  `bt-cross:7.2` **podman** image (`just build-jetson` with `BT_CONTAINER=podman`,
  or `scripts/build-bench-jetson.sh`); `bt-cross:6.1` remains the legacy image for
  JetPack-6 targets.
- **Android** (`android` preset): needs NDK **29.0.14206865** on the building
  machine (currently a laptop); binaries are then copied to rocky and deployed via
  its adb (`scripts/run-on-android.sh` there — `ANDROID_NDK_HOME=$HOME/ndk-libcxx`
  on rocky supplies just `libc++_shared.so`, no full NDK install needed).
- **OMP oracle / CI**: `ctest -L omp` runs in hosted GitHub CI; any dev machine
  can run it locally with the `pc` preset.

Historical note: the retired build box was an i9-14900K + RTX 4070 Ti S (`pc`
device spec, 8 big + 16 little). Its RTX never ran Vulkan (kiss-vk hard-selects
iGPUs) and CUDA 13 broke its CUDA build via the CUB removal — the same CUB port
that now blocks native CUDA 13.2 builds on the Jetsons.

## Jetson Orin ×2 — `doremy@duck-stable` (`duck-stable`) + `doremy@duck-naughty` (`duck-naughty`)

Two identical **Jetson Orin Nano Devkit "Super"** units, both **reflashed 2026-07-01
to JetPack 7.2** (L4T R39.2.0, kernel 6.8-tegra, Ubuntu 24.04, CUDA **13.2**, power
mode **MAXN_SUPER**). They replace the retired JetPack-6 device id `jetson`; all
pre-2026-07 Jetson numbers are from that old software stack and are **not comparable**
— archived under
[`perf-results/test-runs/archive-pre-2026-07/`](../reports-for-human/perf-results/test-runs/archive-pre-2026-07/).

- **SoC** Orin Nano 8GB, sm_87, 6× Cortex-A78AE @ 1.73 GHz (single tier), GPU max
  1.02 GHz, 7.4 GB.
- **Access** `ssh doremy@duck-stable` / `ssh doremy@duck-naughty`. Login shell is
  **bash** (the old fish gotcha no longer applies here — it still does on rocky).
  Passwordless sudo is available (`nvpmodel`, clock locking).
- **Build** **cross-compile** in the `bt-cross:7.2` container (CUDA 13.2, matching
  the fleet — official SBSA cross toolchain, default since 2026-07-02; all six
  `test-*-{cu,vk}` suites green on duck-stable from it), then `scp` aarch64 binaries
  over ([`02-building.md`](02-building.md)). Legacy `bt-cross:6.1` (CUDA 12.6) also
  produces binaries verified on 7.2. The CUB port (2026-07-02) means native
  on-device CUDA 13.2 builds should work too (compiler-equivalent to the 7.2 cross;
  not yet exercised).
- **Run** `scripts/run-on-jetson.sh` (deploy + run CUDA tests on `/tmp/bt`); env
  overrides select the twin (see the script header).
- **Role** the only targets that run **CUDA**; also run Vulkan. `duck-stable` is the
  coverage-gated primary; `duck-naughty` is benchmark-only (`coverage_backends: []`
  in `fleet.json`).

## Rocky Linux MiniPC — `doremy@rocky-ryzen`

- **GPU** AMD Radeon 780M — **integrated** (RADV). `minipc` device spec, 16 cores.
- **OS** Rocky Linux 10.2, glibc 2.39.
- **Access** `ssh rocky-ryzen`. **Login shell is fish** → for loops / `VAR=…` /
  multi-line, pipe bash via `ssh rocky-ryzen bash -s <<'EOF' … EOF` (a single
  `bash -lc '…'` also works for one-liners).
- **Deploy** copy the x86 Vulkan binary + `libomp.so` to `/tmp/bt/`, run with
  `LD_LIBRARY_PATH=.`. All Vulkan suites pass here.
- **Run** `scripts/run-on-rocky.sh` (deploy + run Vulkan tests).
- **Role** the easiest **Vulkan** path (x86, no cross-compile); also the **adb host
  for both phones** (Pixel 7a + Samsung Galaxy — the build box can't see either).
  Both are plugged in at once, so `adb devices` lists two → always `adb -s <serial>`
  (the `run-on-android.sh` script does this).

## Pixel 7a — `3A021JEHN02756`

- **SoC** Tensor G2, Mali-G710 GPU, **subgroup size 16**.
- **Access** connected via adb **on `rocky-ryzen`** (moved off the build box 2026-06-17).
  It now shares rocky's adb with the Samsung, so `adb devices` shows both — select with
  `adb -s 3A021JEHN02756`. Same workflow as the Samsung below: `scp build/android/`
  binaries + `libc++_shared.so` to `rocky-ryzen:/tmp/bt/`, then `ssh rocky-ryzen` and run
  the script there.
- **Run** `scripts/run-on-android.sh 3A021JEHN02756` (on rocky-ryzen; pushes binary +
  `libc++_shared.so` to `/data/local/tmp/bt`, runs with `</dev/null` on each `adb shell`).
- **Role** uniquely exercises the **subgroup-16** Mali shader variant
  (`tmp_single_radixsort_warp16`) — the Jetson cannot.

## Samsung Galaxy — `R5CY21Y3VEV`

- **SoC** SM-S926B, **subgroup size 32**, 10 cores.
- **Access** connected via adb **on `rocky-ryzen`, NOT the build box** — and as of
  2026-06-17 the **Pixel shares the same rocky adb**, so `adb devices` lists both and
  every call needs `adb -s <serial>`. Run the adb deploy **from rocky**: `scp` the
  `build/android/` binaries + `libc++_shared.so` to `rocky-ryzen:/tmp/bt/`, then
  `ssh rocky-ryzen` and run `scripts/run-on-android.sh R5CY21Y3VEV` there (the script
  self-checks the serial is locally visible and tells you if you're on the wrong host).
- **Role** the **subgroup-32** shader variant on an Android target.

## Android build notes (both phones)

```bash
cmake --preset android        # NDK resolves via $ANDROID_HOME/ndk/29.0.14206865
```
Set `ANDROID_NDK_HOME` to that path. `libc++_shared.so` (from
`…/ndk/.../sysroot/usr/lib/aarch64-linux-android/`) must ride along with every
binary; `libvulkan` is already on the device. Full preset/override details:
[`02-building.md`](02-building.md).

## Common errors — symptom → cause → fix

A fast lookup for failures you'll actually hit on this fleet. Narratives behind the
backend bugs are in [`../reports-for-human/bugs-found.md`](../reports-for-human/bugs-found.md).

| Symptom (what you see) | Cause | Fix |
|---|---|---|
| `No integrated GPU found` (Vulkan throws at startup) | `kiss-vk` hard-selects `eIntegratedGpu`; you ran on a **discrete**-GPU box (the build box's RTX) | Run Vulkan on Jetson, rocky-ryzen, or a phone — never the build box. |
| `fish: Missing end to balance this for loop` (or syntax errors over ssh) | rocky's **login shell is fish**; you sent bash syntax inline | `ssh HOST bash -s <<'EOF' … EOF`, or use the `run-on-*.sh` scripts. |
| `adb` run exits 0 but produces **empty output**; later commands skipped | `adb shell` **ate the heredoc stdin** | Suffix every `adb` call with `</dev/null` (the scripts already do). |
| CUDA build fails with CUB / `cub::` errors | **CUDA 13** removed bundled CUB APIs; the repo's usage was ported 2026-07-02 | Rebuild from current `dev` (CUDA 12.6 **and** 13.2 both compile); if it recurs, the new code reintroduced a removed `cub::` API. |
| `*-cu` tests **red on Jetson** (wrong/zero output) | was the managed-memory visibility defect | **Fixed** (zero-copy pinned, 2026-06-16); see `bugs-found.md` §1. If you still see it, rebuild from current `dev`. |
| `--device <id>` self-skips core-pinning tests | unknown/missing device id (non-fatal by design) | Pass a real id from `devices/*.json`; `adb devices` for a phone serial. |
| Vulkan on Mali was very slow / wrong before a rebuild | old kiss-vk host-coherency defect | Already fixed (HOST_CACHED + flush/invalidate); rebuild from current `dev`. |
