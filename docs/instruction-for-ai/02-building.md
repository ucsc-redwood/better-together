# Building BetterTogether (CMake)

> CMake is **the build system** (xmake was retired 2026-06-16; history:
> [`../reports-for-human/cmake-migration-rfc.md`](../reports-for-human/cmake-migration-rfc.md)).
> It covers the CPU(OpenMP) + CUDA + Vulkan paths for all three apps, plus the
> runners, benchmarks, and tests. Not ported: the volk diagnostics
> (check-vulkan/query-warpsize) and NNAPI targets (sources still under `utility/`).

## TL;DR — just want to build & test on this machine

```bash
cmake --preset pc          # configure (CPU/OpenMP only; deps auto-fetched)
cmake --build --preset pc  # build
ctest --test-dir build/pc  # run the OpenMP unit tests
```

That's it for the CPU path — no system packages to install beyond a compiler.

## Common prerequisites

| Need | Why | Notes |
|------|-----|-------|
| **CMake ≥ 3.25** | presets + `CUDA_STANDARD 20` | `cmake --version` |
| A **C++20 compiler** | gcc 11+ or clang 14+ | the `pc` preset uses your default compiler |
| **Network access** | deps are fetched via [CPM.cmake](../../cmake/CPM.cmake) | spdlog, CLI11, glm, libmorton, gtest (+ Vulkan-Headers/VMA) — no system install needed |
| **Docker** | only for the Jetson cross-build | image built from [`Dockerfile.cross`](../../Dockerfile.cross) |
| **Android NDK** | only for the `android` preset | installed via `sdkmanager` (see below) |

First configure downloads dependency sources into the build tree (CPM); set
`CPM_SOURCE_CACHE` (the `jetson` preset already does) to share them across builds.

> Target hardware, access (ssh/adb), and per-device deploy recipes live in
> [`01-hardware.md`](01-hardware.md).

## The presets / build matrix

| Preset | Target | Backends | Where it builds | Where it runs | Extra prerequisites |
|--------|--------|----------|-----------------|---------------|---------------------|
| `pc` | x86_64 native | OpenMP | this host | this host | none |
| `jetson` | aarch64 (Jetson Orin) | CUDA sm_87 + Vulkan | NVIDIA cross container | a Jetson | Docker |
| `vulkan` | x86_64 native | Vulkan | this host | an **integrated-GPU** box | Vulkan loader at runtime |
| `android` | arm64-v8a (Android) | OpenMP + Vulkan | NDK toolchain | an Android device | Android NDK + adb |

Backends are orthogonal options (`BT_ENABLE_CUDA`, `BT_ENABLE_VULKAN`, both default
combinable) — a hybrid binary links `bt::core` plus the enabled backend lib(s).

---

## `pc` — CPU / OpenMP (works out of the box)

```bash
cmake --preset pc && cmake --build --preset pc
ctest --test-dir build/pc --output-on-failure
```

## `jetson` — CUDA (+ Vulkan), cross-compiled

The Jetson Orin Nano is slow to build on, so we cross-compile on x86 inside
NVIDIA's official cross container (CUDA 12.6, matching JetPack 6.x), then copy the
aarch64 binaries to the device.

```bash
# one-time: build the cross image (adds a current CMake to NVIDIA's base image)
docker build -t bt-cross:6.1 -f Dockerfile.cross .

# build everything for the Jetson (native x86 speed, no emulation)
scripts/cross-build-jetson.sh         # == cmake --preset jetson && cmake --build --preset jetson

# deploy to /tmp and run (handles the fish login shell for you)
scripts/run-on-jetson.sh              # all CUDA tests; or pass targets: ... test-tree-cu
```

> Prefer the script. The manual equivalent must pipe bash over stdin because
> `duck-naughty`'s login shell is fish — `ssh duck-naughty bash -s <<'EOF' …` — an
> inline `ssh duck-naughty '… for …'` fails to parse. Hosts/serials: [`01-hardware.md`](01-hardware.md).

nvcc statically links the CUDA runtime, so the binary needs no extra libraries on
the Jetson. The toolchain finds nvcc from `$CUDACXX`, else the container's
`/usr/local/cuda-12.6`, else `PATH`.

## `vulkan` — Vulkan on an integrated-GPU box

The Vulkan engine selects an **integrated GPU** (`kiss-vk/base_engine.cpp`), so it
throws "No integrated GPU found" on a discrete-GPU desktop. Build natively on x86
and run on a machine with an iGPU (Intel/AMD), or on the Jetson.

```bash
cmake --preset vulkan && cmake --build --preset vulkan
# deploy to the iGPU box (rocky-ryzen) and run; gcc's libgomp is usually present there:
scripts/run-on-rocky.sh               # all Vulkan tests; or pass targets to narrow
```

`libvulkan` is loaded via `dlopen` at runtime, so the binary doesn't link it.
(rocky-ryzen is also fish — the script uses `ssh … bash -s`; see [`01-hardware.md`](01-hardware.md).)

## `android` — arm64-v8a (OpenMP + Vulkan), cross-compiled

Install the NDK (version mirrors Gradle's `ndkVersion`) and point the build at it.

```bash
# install the NDK into the SDK (matches CMakePresets' BT_ANDROID_NDK_VERSION)
sdkmanager "ndk;29.0.14206865"          # -> $ANDROID_HOME/ndk/29.0.14206865

cmake --preset android && cmake --build --preset android

# deploy to the phone and run (finds libc++_shared.so, pushes, runs; serial = device id)
scripts/run-on-android.sh 3A021JEHN02756        # Pixel 7a (adb on build box)
# Samsung is attached to rocky-ryzen — run the script from there (it self-checks the host)
```

> The script suffixes every `adb shell` with `</dev/null` — without that, the first
> `adb shell` eats the rest of a heredoc/script and later commands silently no-op.

NDK resolution order in [`android-arm64.cmake`](../../cmake/toolchains/android-arm64.cmake):
`$ANDROID_NDK_HOME` → `$ANDROID_HOME/ndk/$BT_ANDROID_NDK_VERSION` → `$ANDROID_NDK_ROOT`.
`libc++_shared.so` must ride along with the binary; `libvulkan` is on the device.

---

## Overriding defaults

All environment-specific values are cache variables, not hardcoded:

| Variable | Default | Example |
|----------|---------|---------|
| `BT_ENABLE_CUDA` / `BT_ENABLE_VULKAN` | per preset | `-DBT_ENABLE_VULKAN=ON` |
| `BT_CUDA_ARCH` | `87` (Orin) | `-DBT_CUDA_ARCH=89` (Ada) |
| `BT_TEST_DEVICE` | per preset | `-DBT_TEST_DEVICE=3A021JEHN02756` |
| `BT_ANDROID_NDK_VERSION` | `29.0.14206865` | `-DBT_ANDROID_NDK_VERSION=27.2.12479018` |

The `--device <id>` passed to tests must match an entry in the device registry
(`../../builtin-apps/conf.cpp` / `../../devices/*.json`); find a phone's id with `adb devices`.

## Notes

- **xmake is retired** (2026-06-16); CMake is the only build system. The retirement
  was gated on parity: the OpenMP test output was byte-identical between the old
  xmake (clang/-O2) and CMake (gcc/Release) builds — see
  [RFC Appendix B](../reports-for-human/cmake-migration-rfc.md).
- All Vulkan suites pass on every target GPU including Mali-G710 (the earlier
  `cifar-dense-vk`-on-Mali failure was a kiss-vk host-coherency/perf defect, since
  fixed — HOST_CACHED memory + explicit flush/invalidate).
