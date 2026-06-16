# Hardware — the test fleet, specs & how to access

Every backend must be validated on real hardware. This is the single reference for
**what each target is, what it tests, and exactly how to reach and deploy to it.**

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
| Rocky MiniPC | `minipc` | OMP + Vulkan | `ssh doremy@rocky-ryzen` | Vulkan (iGPU) runner; Samsung adb host |
| Pixel 7a | `3A021JEHN02756` | OMP + Vulkan (**subgroup 16**) | adb on build box | Mali subgroup-16 shader variant |
| Samsung Galaxy | `R5CY21Y3VEV` | OMP + Vulkan (**subgroup 32**) | adb on `rocky-ryzen` | subgroup-32 shader variant |

Key hardware constraint: `builtin-apps/common/kiss-vk/base_engine.cpp` **hard-selects
an integrated GPU** (`eIntegratedGpu`) — discrete GPUs throw "No integrated GPU
found", which is why the build box's RTX 4070 Ti never runs Vulkan.

---

## Build / dev box (local, x86_64)

- **CPU** i9-14900K (24 physical / 32 logical; `pc` spec = 8 big + 16 little).
- **GPU** RTX 4070 Ti SUPER — **discrete**, sm_89. CUDA build **breaks on CUDA-13
  CUB removal**, so the PC is **build-only** for CUDA (cross-compile, run on Jetson).
  Vulkan **rejected** — kiss-vk needs an iGPU.
- **OS** Ubuntu 26.04, glibc 2.43, CUDA 13.3, clang 21, Docker (no sudo).
- **Role** runs the OMP oracle (`ctest -L omp`), hosts the Jetson cross-build
  container, and has the **Pixel 7a** attached via adb directly.

## Jetson Orin — `yanwen@duck-naughty`

- **SoC** Jetson Orin, sm_87. JetPack 6.2 (L4T R36.4.7), Ubuntu 22.04, glibc 2.35,
  CUDA 12.6, gcc 11.4. 6 cores / 7.4 GB.
- **Access** `ssh duck-naughty` (configured in `~/.ssh/config`, User `yanwen`).
- **Build** slow on-device → **cross-compile** in the `bt-cross:6.1` container on the
  build box, then `scp` aarch64 binaries over. Recipe:
  [`02-building.md`](02-building.md) (`jetson` preset) and
  [`jetson cross-build`](../../scripts/cross-build-jetson.sh).
- **Role** the only target that runs **CUDA**; also runs Vulkan. Self-hosted runner.

## Rocky Linux MiniPC — `doremy@rocky-ryzen`

- **GPU** AMD Radeon 780M — **integrated** (RADV). `minipc` device spec, 16 cores.
- **OS** Rocky Linux 10.2, glibc 2.39.
- **Access** `ssh rocky-ryzen`. **Login shell is fish** → wrap remote commands in
  `bash -lc '…'` (bash `VAR=…` syntax fails in fish).
- **Deploy** copy the x86 Vulkan binary + `libomp.so`, run with
  `LD_LIBRARY_PATH=$PWD`. All Vulkan suites pass here.
- **Role** the easiest **Vulkan** path (x86, no cross-compile); also the **adb host
  for the Samsung Galaxy** (the build box can't see it).

## Pixel 7a — `3A021JEHN02756`

- **SoC** Tensor G2, Mali-G710 GPU, **subgroup size 16**.
- **Access** connected via adb **directly on the build box**.
- **Deploy**
  ```bash
  adb -s 3A021JEHN02756 push <bin> libc++_shared.so /data/local/tmp/bt/
  adb -s 3A021JEHN02756 shell "cd /data/local/tmp/bt && \
    LD_LIBRARY_PATH=. ./<bin> --device 3A021JEHN02756"
  ```
- **Role** uniquely exercises the **subgroup-16** Mali shader variant
  (`tmp_single_radixsort_warp16`) — the Jetson cannot.

## Samsung Galaxy — `R5CY21Y3VEV`

- **SoC** SM-S926B, **subgroup size 32**, 10 cores.
- **Access** connected via adb **on `rocky-ryzen`, NOT the build box**. Workflow:
  `scp` the Android arm64 binaries + `libc++_shared.so` to `doremy@rocky-ryzen:~`,
  then `ssh rocky-ryzen` and adb-push/run there (remember: fish → `bash -lc`).
- **Role** the **subgroup-32** shader variant on an Android target.

## Android build notes (both phones)

```bash
cmake --preset android        # NDK resolves via $ANDROID_HOME/ndk/29.0.14206865
```
Set `ANDROID_NDK_HOME` to that path. `libc++_shared.so` (from
`…/ndk/.../sysroot/usr/lib/aarch64-linux-android/`) must ride along with every
binary; `libvulkan` is already on the device. Full preset/override details:
[`02-building.md`](02-building.md).
