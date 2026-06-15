# RFC: Migrate BetterTogether from xmake to CMake (multi-platform, multi-backend, hybrid)

Status: **draft / for discussion** · Branch: `refactor/framework-device-axis` · Author session: 2026-06-15

Related: [`REARCHITECTURE.md`](REARCHITECTURE.md) · [`PLANNING-NOTES-2026-06-15.md`](PLANNING-NOTES-2026-06-15.md)

---

## 1. Goal (the thing the build must enable)

Run **all three applications** (`tree`, `cifar-dense`, `cifar-sparse`) on **all three backends**
(OpenMP/CPU, CUDA, Vulkan), including the **hybrid pipelines that are the point of the paper**:

- **OpenMP + CUDA** in one binary (Jetson Orin — stages split across CPU cores and the iGPU).
- **OpenMP + Vulkan** in one binary (phones via Android, and PC via an integrated GPU).

And Vulkan must build for **two targets**: **Android (arm64, NDK)** and **normal PC (x86_64)**.

The build system therefore has to span a **platform × backend matrix with cross-compilation**, and
it must let backends be **combined** (not selected as mutually exclusive modes), because hybrid
execution links two backend libraries into a single executable. The current xmake build already
does this (`test-tree-cu` links `builtin-apps-cuda` **and** `builtin-apps`; `test-tree-vk` links
`builtin-apps-vulkan` **and** `builtin-apps`) — the CMake design must preserve it.

## 2. The build matrix

| Target platform | Host→target | Toolchain | OMP | CUDA | Vulkan | Hybrid of record |
|---|---|---|---|---|---|---|
| **PC x86_64** | native | system clang/gcc | ✅ | ✅ (dGPU/iGPU) | ✅ (needs ICD; iGPU for the engine's integrated-GPU check) | OMP+CUDA, OMP+Vulkan |
| **Jetson Orin (arm64)** | x86_64 → aarch64 | `jetson-aarch64.cmake` inside NVIDIA cross container | ✅ | ✅ **sm_87** | ✅ | **OMP+CUDA** |
| **Android (arm64)** | x86_64 → aarch64 | NDK `android.toolchain.cmake` | ✅ (static) | ❌ | ✅ | **OMP+Vulkan** (+ NNAPI/TPU) |

Three orthogonal options drive it — `BT_ENABLE_OPENMP` (default ON), `BT_ENABLE_CUDA`,
`BT_ENABLE_VULKAN` — and any combination the platform supports may be enabled at once.

> ⚠️ The Vulkan engine (`kiss-vk/base_engine.cpp`) hard-selects an **integrated GPU**. On a discrete-GPU
> PC every Vulkan test throws "No integrated GPU found" (verified on an RTX 4070 Ti SUPER). PC Vulkan
> testing needs an iGPU (e.g. an AMD/Intel integrated part); binaries built on x86 run cross-host
> wherever an iGPU + Vulkan loader exist (verified: x86-built Vulkan tests pass on an AMD Radeon 780M box).

## 3. Why CMake (and proven evidence, not theory)

The strategic reasons are external and already argued in `PLANNING-NOTES §8` (CMake ubiquity, CTest,
first-class CUDA, `find_package(Vulkan)`, the NDK's native `android.toolchain.cmake`,
`compile_commands.json`). What this session **proved end-to-end**:

- NVIDIA ships `nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:6.1` — a **native x86** image
  with CUDA 12.6 cross-toolkit + `aarch64-linux-gnu-g++-11` + a JetPack sysroot. It matches the
  Jetson's ABI exactly (CUDA 12.6, Ubuntu 22.04, glibc 2.35, gcc 11.4, sm_87).
- A CUDA program **cross-compiled with CMake** (`CMAKE_TOOLCHAIN_FILE` + `CUDA_ARCHITECTURES 87`) in
  that container produced an AArch64 ELF that **ran on `duck-naughty`'s Orin GPU** and executed a real
  kernel. nvcc static-links cudart, so the binary needed **zero extra libraries** on the device.
- Compilation ran at full x86 speed (no QEMU emulation) — which is the entire point (the Orin Nano is
  6 cores / 7.4 GB and "takes forever").
- Building in the **12.6** container also sidesteps the **CUDA 13 `cub::DivideAndRoundUp` removal**
  that breaks the build with the host's CUDA 13.3.

A 10-line CMake toolchain file does cleanly what xmake makes hard (it hardpins `set_toolchains("clang")`
and carries bespoke `cuflags`).

## 4. Target architecture (CMake)

Mirror the existing library decomposition; keep backends as **separate, option-gated static libs** that
binaries combine:

```
libbt_core         (always)  conf, app, pipeline, appdata, resources, CPU/OMP dispatchers
libbt_cuda         (CUDA)    *.cu kernels + dispatchers           ── links CUDA::cudart_static, cub
libbt_vulkan       (Vulkan)  vk dispatchers + kiss-vk + embedded SPIR-V headers
                              ── links Vulkan + VMA + volk
app/pipeline/test/bm binaries  link libbt_core + (libbt_cuda? )+(libbt_vulkan?)  ← hybrid here
```

- **One `add_executable` recipe, parameterised** over (app × backend-set) instead of 67 hand-written
  targets. A CMake function `bt_add_app_binary(app BACKENDS omp cuda)` generates the test/run/bm
  targets, registering tests with `add_test()` (CTest) and passing `--device`.
- `enable_language(CUDA)` only when `BT_ENABLE_CUDA`; `enable_language(CXX)` always.
- **`CMakePresets.json`** is the user-facing entry point — replaces the `just set-*` recipes:
  `cmake --preset pc | pc-cuda | jetson | android`. The `justfile` becomes a thin wrapper.

## 5. Dependency strategy

Recommend **CPM.cmake** (single `.cmake`, no system install, closest to xmake's xrepo ergonomics —
`PLANNING-NOTES §8` already flagged it as the front-runner). Prefer `find_package` for libraries that
should come from the **target sysroot** when cross-compiling (notably libcurl).

| xmake `add_requires` | CMake approach | Notes |
|---|---|---|
| spdlog | CPM or `find_package(spdlog)` | |
| cli11 | CPM | header-only |
| glm | CPM | header-only; needs `--diag-suppress=20012` under nvcc |
| nlohmann_json | CPM | header-only |
| gtest | CPM / FetchContent | CTest integration |
| benchmark | CPM / FetchContent | |
| **libcurl** | `find_package(CURL)` | the one heavyweight; use system/**sysroot** when cross. Confirm `curl_json.hpp` is even needed on the CUDA path — may be optional. |
| libmorton | CPM | header-only |
| cnpy | CPM | small; needs zlib |
| **cub** | bundled with CUDA Toolkit | no fetch; do **not** pin a separate cub (that's what broke on CUDA 13) |
| vulkan-headers / vulkan-hpp | `find_package(Vulkan)` (PC/Jetson) / NDK (Android) | pin vulkan-hpp API version — code was written against 1.3.290; host SDK is 1.4 |
| vulkan-memory-allocator (VMA) | CPM | header-only |
| volk | CPM or system | loader shim (dlopen) |

## 6. Toolchain & cross-compile delivery

`cmake/toolchains/`:

- **`jetson-aarch64.cmake`** (proven):
  ```cmake
  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR aarch64)
  set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
  set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++)
  set(CMAKE_CUDA_ARCHITECTURES 87)
  # find_package(CURL) etc. resolve against the JetPack sysroot
  ```
- **`android-arm64.cmake`** — thin wrapper over the NDK's `android.toolchain.cmake`
  (`-DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29`), carrying the existing
  `-fopenmp -static-openmp` handling for Android OMP.

Ship a **`Dockerfile.cross`** `FROM nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:6.1` that
adds a current CMake (the image's is 3.22.1) + CPM cache. Jetson builds = run CMake with the toolchain
file inside that image; deploy with the existing `scp`/adb-push flow. (Android stays a documented
bring-your-own-NDK flow, consistent with today.)

## 7. Shaders & CUDA specifics (carry-over details)

- **Shaders need no build step.** SPIR-V is committed as pre-baked headers
  (`kiss-vk/shaders/h/*_spv.h`, `#include`d by `all_shaders.hpp`). Optionally add an **off-by-default**
  `bt_regen_shaders` target invoking `glslc` (present in the Vulkan SDK) for reproducibility — but the
  committed `.h` are the source of truth, so migration carries zero shader risk.
- CUDA flags to preserve: `CUDA_ARCHITECTURES` = `87` (Jetson) or `87;89` (PC Ada); glm
  `--diag-suppress=20012`; NVTX (`CUDA::nvToolsExt`/nvtx3); `-Xcompiler -fopenmp` so OMP works inside
  `.cu` TUs; **cudart static** (default) for portable cross binaries.

## 8. Non-negotiable sequencing: characterization net FIRST

Per `PLANNING-NOTES §8`, a build migration perturbs flags / arch codes / optimization / link order and
can **silently change numerical results**. Therefore, **before** any target is ported:

1. **Lock behavior** — golden device topology (already: `validate_devices.py` green), golden schedule
   JSON for the shipped tables, figure regeneration, and the OMP **oracle** correctness tests (already
   green on PC here: cifar-dense 12/12, cifar-sparse 9/9, tree 7/7).
2. **Parity gate** — for each migrated target, build with CMake, run the same tests on the same seeded
   inputs, and diff against the xmake build: exact equality for integer/structural stages (Morton,
   radix tree, octree, sort), `EXPECT_NEAR(tol)` for float stages (conv/linear). No fast-math exists in
   the tree today, which keeps drift risk low — but it must be *proven*, not assumed.

xmake and CMake **build side-by-side** during the transition; xmake is retired per-phase only after the
CMake target passes the parity gate.

## 9. Phased plan (each shippable, test-guarded)

| Phase | Work | Done when |
|---|---|---|
| **A. Char net** | Golden topology/schedule/figure + OMP oracle + parity harness. No build change. | tests green; parity script exists |
| **B. Core + OMP (PC)** | Root `CMakeLists.txt`, options, CPM, deps; build `libbt_core` + OMP tests on PC; `CMakePresets.json`; `compile_commands.json`. | OMP tests pass; **parity vs xmake** |
| **C. CUDA + cross** | `libbt_cuda` + CUDA tests on PC (RTX), then cross to Jetson via toolchain + `Dockerfile.cross`; run on Orin. Validate **OMP+CUDA hybrid** (the mixing tests). | CUDA tests pass on Orin; hybrid parity |
| **D. Vulkan (PC + Android)** | `libbt_vulkan` + kiss-vk; PC Vulkan on an iGPU; Android via NDK toolchain. Validate **OMP+Vulkan hybrid** on both. | Vulkan tests pass on iGPU + device; hybrid parity |
| **E. Finish** | `pipe/` + `bm-*` targets; retire xmake; `justfile` → thin preset wrapper; CI presets; docs. | xmake removed; CI green |

## 10. Risks & open questions

- **Numerical parity** is the headline risk — mitigated by the Phase-A net + matching `-O2`
  (`set_optimize("faster")`) and CUDA arch codes; verified per target by the parity gate.
- **libcurl** cross-build — resolve via the JetPack sysroot's `find_package(CURL)` rather than building
  from source; first confirm whether the CUDA/CPU path actually needs it.
- **Vulkan-Hpp version skew** — code targets 1.3.290; pin the header version to avoid API drift against
  newer SDKs.
- **Android OMP** — preserve `-fopenmp -static-openmp`; the NDK toolchain handles the rest.
- **Container CMake 3.22.1** — upgrade inside `Dockerfile.cross`.
- **Open:** CPM vs vcpkg manifest (recommend CPM); per-platform vs combined `CUDA_ARCHITECTURES`
  (87-only on Jetson speeds builds); keep the discrete-GPU Vulkan rejection or add an opt-in
  `eDiscreteGpu` fallback for CI on dGPU boxes.

## Appendix B — parity gate results (this session)

OMP/CPU path (the oracle) is parity-verified: the three OMP test binaries built
by **xmake (clang, -O2)** and by **CMake (gcc, Release)** produce **byte-identical
output** (28 tests, after normalizing log timestamps and gtest timings). Two
different compilers + optimization levels → identical behavior, which backs the
CMake build on the CPU path. CUDA/Vulkan parity (xmake-on-target vs CMake-cross)
is the remaining gate; the shared source + all-green device runs give high
confidence pending that check.

Verified runs:
- OMP (host, x86): cmake ctest 3/3; **parity vs xmake byte-identical**.
- CUDA (Jetson Orin, cross): tree 13/13 (cifar has a pre-existing test-constant drift).
- Vulkan (Jetson Orin, cross): kiss-vk 2/2, tree 14/14, cifar-dense 10/10, cifar-sparse 9/9.
- Vulkan (rocky-ryzen iGPU): same 35/35.
- Android arm64-v8a (cross): full build (OMP + Vulkan) compiles; on-device run pending.

## Appendix A — verified environment facts (this session)

- **Build/host box:** i9-14900K, RTX 4070 Ti SUPER (sm_89, discrete), Ubuntu 26.04, glibc 2.43,
  CUDA 13.3, clang 21, Docker (no sudo), 186 GB free. OMP tests green; CUDA build broken by CUDA-13
  CUB removal; Vulkan rejected (discrete GPU).
- **Jetson `duck-naughty`:** Orin, JetPack 6.2 (L4T R36.4.7), Ubuntu 22.04, glibc 2.35, CUDA 12.6,
  sm_87, gcc 11.4, 6 cores / 7.4 GB. Runs cross-built CUDA binaries.
- **iGPU box `rocky-ryzen`:** AMD Radeon 780M (integrated, RADV), Rocky 10.2, glibc 2.39. Runs
  x86-built Vulkan binaries (kiss-vk 2/2, tree 14/14, cifar-dense 10/10, cifar-sparse 9/9).
- **Cross container:** `nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86` — newest tag **6.1**
  (CUDA 12.6; matches the 6.2 device). Toolchain at `/usr/local/cuda-12.6/bin/nvcc` +
  `aarch64-linux-gnu-g++-11`; full target rootfs at `/l4t/targetfs.tbz2.*`.
