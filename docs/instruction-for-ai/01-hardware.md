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
scripts/run-on-jetson.sh                       # build/jetson  → duck-naughty  (CUDA tests)
scripts/run-on-rocky.sh                         # build/vulkan  → rocky-ryzen   (Vulkan tests)
# Both phones' adb lives on rocky-ryzen now → copy build/android there & run the script on it:
scripts/run-on-android.sh 3A021JEHN02756        # build/android → Pixel 7a   (adb -s picks the phone)
scripts/run-on-android.sh R5CY21Y3VEV           # build/android → Samsung    (same host, other serial)
scripts/run-mali-oracle.sh                      # build/android → BOTH phones (Mali oracle gate, run from the build box)
# pass explicit targets to narrow:  scripts/run-on-jetson.sh test-tree-cu
```

> **Two gotchas the scripts handle for you — hit them if you hand-roll commands:**
> 1. **Jetson and rocky-ryzen login shells are BOTH fish.** An inline
>    `ssh HOST '… for/VAR=… …'` dies with a fish parse error. Pipe a bash script over
>    stdin instead: `ssh HOST bash -s <<'EOF' … EOF` (bypasses the login shell).
> 2. **`adb shell` swallows the rest of stdin.** Inside a heredoc the first `adb shell`
>    eats the remaining script lines (exit 0, empty output). Suffix **every** `adb`
>    call with `</dev/null`.

The machine-readable CPU topology for each device lives in `../../devices/<id>.json`
(validated against `../../schemas/device-spec.schema.json`); that registry is the
source of truth for core tiers and pinning. This doc is the human/agent-readable
**access + role** layer on top of it. To add a device, see
[`../../devices/README.md`](../../devices/README.md).

## Target matrix at a glance

| Target | `--device` id | Backends it runs | Access | Role |
|---|---|---|---|---|
| Build/dev box | `pc` | OMP (CUDA/Vulkan **build-only**) | local | OMP oracle, CI, cross-build host |
| Jetson Orin | `jetson` | OMP + CUDA + Vulkan | `ssh yanwen@duck-naughty` | CUDA + Vulkan self-hosted runner |
| Rocky MiniPC | `minipc` | OMP + Vulkan | `ssh doremy@rocky-ryzen` | Vulkan (iGPU) runner; **adb host for both phones** |
| Pixel 7a | `3A021JEHN02756` | OMP + Vulkan (**subgroup 16**) | adb on `rocky-ryzen` (`adb -s`) | Mali subgroup-16 shader variant |
| Samsung Galaxy | `R5CY21Y3VEV` | OMP + Vulkan (**subgroup 32**) | adb on `rocky-ryzen` (`adb -s`) | subgroup-32 shader variant |

Key hardware constraint: `platform/engine/vulkan/base_engine.cpp` **hard-selects
an integrated GPU** (`eIntegratedGpu`) — discrete GPUs throw "No integrated GPU
found", which is why the build box's RTX 4070 Ti never runs Vulkan.

---

## Build / dev box (local, x86_64)

- **CPU** i9-14900K (24 physical / 32 logical; `pc` spec = 8 big + 16 little).
- **GPU** RTX 4070 Ti SUPER — **discrete**, sm_89. CUDA build **breaks on CUDA-13
  CUB removal**, so the PC is **build-only** for CUDA (cross-compile, run on Jetson).
  Vulkan **rejected** — kiss-vk needs an iGPU.
- **OS** Ubuntu 26.04, glibc 2.43, CUDA 13.3, clang 21, Docker (no sudo).
- **Role** runs the OMP oracle (`ctest -L omp`) and hosts the Jetson cross-build
  container. **No phone is attached here anymore** — both moved to rocky-ryzen.

## Jetson Orin — `yanwen@duck-naughty`

- **SoC** Jetson Orin, sm_87. JetPack 6.2 (L4T R36.4.7), Ubuntu 22.04, glibc 2.35,
  CUDA 12.6, gcc 11.4. 6 cores / 7.4 GB.
- **Access** `ssh duck-naughty` (configured in `~/.ssh/config`, User `yanwen`).
  **Login shell is fish** — use `ssh duck-naughty bash -s <<'EOF' … EOF` for any
  loop / multi-line / `VAR=…` command (see the gotchas above).
- **Build** slow on-device → **cross-compile** in the `bt-cross:6.1` container on the
  build box, then `scp` aarch64 binaries over. Recipe:
  [`02-building.md`](02-building.md) (`jetson` preset) and
  [`jetson cross-build`](../../scripts/cross-build-jetson.sh).
- **Run** `scripts/run-on-jetson.sh` (deploy + run CUDA tests on `/tmp/bt`).
- **Role** the only target that runs **CUDA**; also runs Vulkan. Self-hosted runner.

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
| `fish: Missing end to balance this for loop` (or syntax errors over ssh) | Jetson / rocky **login shell is fish**; you sent bash syntax inline | `ssh HOST bash -s <<'EOF' … EOF`, or use the `run-on-*.sh` scripts. |
| `adb` run exits 0 but produces **empty output**; later commands skipped | `adb shell` **ate the heredoc stdin** | Suffix every `adb` call with `</dev/null` (the scripts already do). |
| CUDA build fails on the build box (CUB / `cub::` errors) | **CUDA 13** removed CUB; PC is **build-only** | Cross-compile in `bt-cross:6.1` and run on the Jetson; don't expect CUDA to run on the PC. |
| `*-cu` tests **red on Jetson** (wrong/zero output) | was the managed-memory visibility defect | **Fixed** (zero-copy pinned, 2026-06-16); see `bugs-found.md` §1. If you still see it, rebuild from current `dev`. |
| `--device <id>` self-skips core-pinning tests | unknown/missing device id (non-fatal by design) | Pass a real id from `devices/*.json`; `adb devices` for a phone serial. |
| Vulkan on Mali was very slow / wrong before a rebuild | old kiss-vk host-coherency defect | Already fixed (HOST_CACHED + flush/invalidate); rebuild from current `dev`. |
