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
# Container runtime for the Jetson cross-build: `docker` (build box) or `podman`
# (the Rocky fleet runner — rootless + SELinux → --userns=keep-id + a :z volume).
container      := env_var_or_default("BT_CONTAINER", "docker")

# Per-app test binaries. OMP+VK builds (x86 / Android) ship six; Jetson adds CUDA.
omp_vk_bins := "test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk"
jetson_bins := "test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp test-tree-cu test-cifar-dense-cu test-cifar-sparse-cu test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk"

_default:
    @just --list

# 1. Build aarch64 (Jetson) — cross-compiled in the bt-cross:6.1 container.
# Works with docker (build box) or podman (BT_CONTAINER=podman, the Rocky runner).
build-jetson:
    #!/usr/bin/env bash
    set -euo pipefail
    build='cmake --preset jetson && cmake --build --preset jetson --target {{jetson_bins}}'
    if [ "{{container}}" = "podman" ]; then
      # rootless podman: --userns=keep-id keeps file ownership; :z relabels for SELinux
      podman run --rm --userns=keep-id -e HOME=/workspace/build \
        -v "$PWD:/workspace:z" -w /workspace bt-cross:6.1 bash -lc "$build"
    else
      docker run --rm --user "$(id -u):$(id -g)" -e HOME=/workspace/build \
        -v "$PWD:/workspace" -w /workspace bt-cross:6.1 bash -lc "$build"
    fi

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
# Fail-loud: any failing fleet test makes this go red (see the per-target notes).
test: test-jetson test-minipc test-samsung

# Fail if any expected GPU (app x backend x hardware) cell was never RAN on the
# fleet. Diffs fleet-coverage.log (emitted by the run-on-*.sh deploy scripts as
# BT-CELL markers) against fleet-coverage.json. Run after a fleet sweep; this is
# what the dev->main promotion gate should assert (see CONTRIBUTING.md).
check-fleet:
    scripts/check_fleet_coverage.py

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
    # Fail-loud: capture each binary's exit code and scan for gtest's [  FAILED  ]
    # marker so a failing fleet test makes the recipe (and `just test`) go red. The
    # old `... | grep | tail` pipe discarded the exit code, so the gate could not fail.
    script='cd /tmp/bt; rc=0; for b in {{bins}}; do echo "##### $b"; out=$(LD_LIBRARY_PATH=. ./$b --device {{device}} 2>&1) || rc=1; printf "%s\n" "$out" | grep -E "tests ran|PASSED|FAILED|SKIPPED" | tail -8; printf "%s" "$out" | grep -q "\[  FAILED  \]" && rc=1; done; exit $rc'
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
    # Fail-loud (see test-ssh note): exit non-zero if any binary fails or prints
    # gtest's [  FAILED  ] marker, so a bad Android run makes `just test` go red.
    script='rc=0
    adb -s {{samsung_serial}} shell "mkdir -p /data/local/tmp/bt" </dev/null
    adb -s {{samsung_serial}} push /tmp/bt-and/. /data/local/tmp/bt/ </dev/null >/dev/null
    for b in {{omp_vk_bins}}; do
      echo "##### $b"
      out=$(adb -s {{samsung_serial}} shell "cd /data/local/tmp/bt && chmod 755 $b && LD_LIBRARY_PATH=. ./$b --device {{samsung_serial}} 2>&1" </dev/null) || rc=1
      printf "%s\n" "$out" | grep -E "tests ran|PASSED|FAILED|SKIPPED" | tail -8
      printf "%s" "$out" | grep -q "\[  FAILED  \]" && rc=1
    done
    exit $rc'
    ssh {{minipc_host}} bash -s <<<"$script"

# --- formatting -------------------------------------------------------------
# Formatters: clang-format (C++), ruff (Python), gersemi (CMake), shfmt (shell),
# prettier (JSON). The pip tools run via `uv run` and are version-pinned in
# uv.lock (`uv sync --group dev` to install) — clang-format especially, whose
# output drifts between releases. shfmt is a Go binary
# (`go install mvdan.cc/sh/v3/cmd/shfmt@latest`); prettier runs via `bunx`.
# File sets come from `git ls-files`, so untracked + build/ + _deps are skipped.
# Generated code is excluded — codegen re-emits it unformatted at build time, so
# formatting it is futile and makes fmt-check non-deterministic: the */generated/
# headers (device_specs_embedded.hpp, bt_vocab.hpp), the baked shader headers
# (platform/engine/vulkan/shaders/h/*_spv.h — all .h), and the .comp GLSL. The
# Python codegen output (optimizer/smt/bt_vocab.py) is excluded via pyproject.

# Format the whole tree in place.
fmt:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}"
    echo "▸ clang-format (C++)"
    git ls-files '*.cpp' '*.hpp' '*.cu' '*.cuh' ':(exclude)*/generated/*' | xargs -r uv run clang-format -i
    echo "▸ ruff (Python)"
    uv run ruff format .
    uv run ruff check --fix .
    echo "▸ gersemi (CMake)"
    git ls-files '*CMakeLists.txt' '*.cmake' | xargs -r uv run gersemi -i
    echo "▸ shfmt (shell)"
    git ls-files '*.sh' | xargs -r shfmt -w -i 2 -ci
    echo "▸ prettier (JSON)"
    git ls-files '*.json' | xargs -r bunx prettier@3.8.4 --write --log-level warn

# Verify formatting without writing (non-zero exit if anything is unformatted).
fmt-check:
    #!/usr/bin/env bash
    set -uo pipefail
    cd "{{justfile_directory()}}"
    rc=0
    echo "▸ clang-format (C++)"
    git ls-files '*.cpp' '*.hpp' '*.cu' '*.cuh' ':(exclude)*/generated/*' | xargs -r uv run clang-format --dry-run -Werror || rc=1
    echo "▸ ruff (Python)"
    uv run ruff format --check . || rc=1
    uv run ruff check . || rc=1
    echo "▸ gersemi (CMake)"
    git ls-files '*CMakeLists.txt' '*.cmake' | xargs -r uv run gersemi --check || rc=1
    echo "▸ shfmt (shell)"
    git ls-files '*.sh' | xargs -r shfmt -d -i 2 -ci || rc=1
    echo "▸ prettier (JSON)"
    git ls-files '*.json' | xargs -r bunx prettier@3.8.4 --check --log-level warn || rc=1
    exit $rc
