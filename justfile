# BetterTogether — build & unit-test matrix
#
# Build system is CMake presets (xmake retired 2026-06-16). The previous, larger
# justfile (profiling / schedule / benchmark recipes, many still on `xmake r`) is
# archived as justfile.old.
#
# Four things this file does:
#   build-jetson    1. build aarch64  — cross-compiled in the bt-cross:6.1 docker image
#   build-x86       2. build x86 native (Vulkan + OpenMP)
#   build-android   3. build arm64 Android — NDK toolchain
#   test            4. run the unit tests across the HW x backend matrix
#
# The differential unit tests are OMP-as-oracle: each binary runs every stage of
# an app on its backend and compares against the in-process OpenMP reference, so a
# green run means "all stages compute the correct answer" on that target.
#
#   Target                build         runs on                    backends
#   Jetson Orin (aarch64) build-jetson  ssh duck-naughty           OMP + CUDA + VK
#   mini pc (x86 iGPU)    build-x86     ssh rocky-ryzen            OMP + VK
#   Samsung (arm64)       build-android R5CY21Y3VEV via rocky adb  OMP + VK

# --- config -----------------------------------------------------------------

jetson_host    := "duck-naughty"
minipc_host    := "rocky-ryzen"
samsung_serial := "R5CY21Y3VEV"
ndk            := env_var_or_default("ANDROID_NDK_HOME", env_var("ANDROID_HOME") / "ndk/29.0.14206865")

# Per-app test binaries. OMP+VK builds (x86 / Android) ship six; Jetson adds CUDA.
omp_vk_bins := "test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk"
jetson_bins := "test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp test-tree-cu test-cifar-dense-cu test-cifar-sparse-cu test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk"

_default:
    @just --list

# 1. Build aarch64 (Jetson) — cross-compiled in the bt-cross:6.1 docker image.
build-jetson:
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/workspace/build \
      -v "$PWD:/workspace" -w /workspace bt-cross:6.1 bash -lc \
      'cmake --preset jetson && cmake --build --preset jetson --target {{jetson_bins}}'

# 2. Build x86 native (mini pc) — Vulkan + OpenMP.
build-x86:
    cmake --preset vulkan
    cmake --build --preset vulkan --target {{omp_vk_bins}}

# 3. Build arm64 Android (Samsung) — NDK toolchain.
build-android:
    ANDROID_NDK_HOME={{ndk}} cmake --preset android
    cmake --build --preset android --target {{omp_vk_bins}}

# Build all three packages.
build: build-jetson build-x86 build-android

# 4. Run the unit tests across the whole matrix (build first with `just build`).
test: test-jetson test-minipc test-samsung

# Deploy x86 build to the iGPU mini pc and run OMP + Vulkan.
test-minipc: (_test-ssh minipc_host "minipc" "build/vulkan" omp_vk_bins)

# Deploy aarch64 build to the Jetson and run OMP + CUDA + Vulkan.
test-jetson: (_test-ssh jetson_host "jetson" "build/jetson" jetson_bins)

# Deploy + run on a Linux SSH target (internal helper).
# NOTE: duck-naughty AND rocky-ryzen both use fish as the login shell, so
# `ssh host bash -s` forces bash for the loop; a bare `ssh host 'for ...'` fails
# with "Missing end to balance this for loop".
_test-ssh host device builddir bins:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "===== {{host}} ({{device}}) ====="
    ssh {{host}} 'mkdir -p /tmp/bt'
    scp -q $(for b in {{bins}}; do echo {{builddir}}/$b; done) {{host}}:/tmp/bt/
    script='cd /tmp/bt; for b in {{bins}}; do echo "##### $b"; LD_LIBRARY_PATH=. ./$b --device {{device}} 2>&1 | grep -E "tests ran|PASSED|FAILED|SKIPPED" | tail -8; done'
    ssh {{host}} bash -s <<<"$script"

# NOTE: `adb shell` reads the script's stdin, so every adb call gets `</dev/null`
# — otherwise the first one eats the rest of the loop.

# Deploy the Android build to the Samsung (adb lives on rocky-ryzen) and run OMP + Vulkan.
test-samsung:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "===== {{samsung_serial}} (Samsung) ====="
    libcxx={{ndk}}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so
    ssh {{minipc_host}} 'mkdir -p /tmp/bt-and'
    scp -q $(for b in {{omp_vk_bins}}; do echo build/android/$b; done) "$libcxx" {{minipc_host}}:/tmp/bt-and/
    script='adb -s {{samsung_serial}} shell "mkdir -p /data/local/tmp/bt" </dev/null
    adb -s {{samsung_serial}} push /tmp/bt-and/. /data/local/tmp/bt/ </dev/null >/dev/null
    for b in {{omp_vk_bins}}; do
      echo "##### $b"
      adb -s {{samsung_serial}} shell "cd /data/local/tmp/bt && chmod 755 $b && LD_LIBRARY_PATH=. ./$b --device {{samsung_serial}} 2>&1" </dev/null | grep -E "tests ran|PASSED|FAILED|SKIPPED" | tail -8
    done'
    ssh {{minipc_host}} bash -s <<<"$script"
