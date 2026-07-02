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
#   Jetson Orin (aarch64) build-jetson  ssh doremy@duck-stable     OMP + CUDA + VK
#     (twin devkit:                     ssh doremy@duck-naughty — same build)
#   mini pc (x86 iGPU)    build-x86     ssh rocky-ryzen            OMP + VK
#   Samsung (arm64)       build-android R5CY21Y3VEV via rocky adb  OMP + VK

# --- config -----------------------------------------------------------------

jetson_host         := "doremy@duck-stable"
jetson_naughty_host := "doremy@duck-naughty"
minipc_host    := "rocky-ryzen"
samsung_serial := "R5CY21Y3VEV"
ndk            := env_var_or_default("ANDROID_NDK_HOME", env_var("ANDROID_HOME") / "ndk/29.0.14206865")
# Container runtime for the Jetson cross-build: `docker` (build box) or `podman`
# (the Rocky fleet runner — rootless + SELinux → --userns=keep-id + a :z volume).
container      := env_var_or_default("BT_CONTAINER", "docker")

# Per-app test binaries. OMP+VK builds (x86 / Android) ship six; Jetson adds CUDA.
omp_vk_bins := "test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk"
jetson_bins := "test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp test-tree-cu test-cifar-dense-cu test-cifar-sparse-cu test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk"

# Benchmark drivers the fleet e2e (00_run_fleet.py) deploys: profiling (bm-prof) +
# schedule runner (bm-gen-logs). These are the profiler/ targets, NOT the unit-test
# binaries above. Vulkan-only for x86/Android; the Jetson CUDA+VK set is built by
# scripts/build-bench-jetson.sh (on rocky).
bench_vk_bins := "bm-prof-tree-vk bm-prof-cifar-dense-vk bm-prof-cifar-sparse-vk bm-gen-logs-tree-vk bm-gen-logs-cifar-dense-vk bm-gen-logs-cifar-sparse-vk"

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

# --- benchmark binaries (for the fleet e2e: 00_run_fleet.py) ----------------
# bm-prof (profiling) + bm-gen-logs (schedule runner). 00_run_fleet.py's build phase
# shells out to these. x86/Android build locally; Jetson cross-builds on rocky (the
# bt-cross:6.1 podman image lives there, not on this box).

build-bench-x86:
    cmake --preset vulkan
    cmake --build --preset vulkan --target {{bench_vk_bins}}

build-bench-android:
    ANDROID_NDK_HOME={{ndk}} cmake --preset android
    cmake --build --preset android --target {{bench_vk_bins}}

# Cross-build the Jetson benchmark binaries ON rocky-ryzen (the bt-cross:6.1 podman
# image lives there). Heredoc-over-ssh doesn't survive just's recipe parser, so the
# rsync->podman->pull flow lives in a standalone script (cf. scripts/run-on-*.sh).
build-bench-jetson:
    BT_BENCH_JETSON_HOST={{minipc_host}} scripts/build-bench-jetson.sh

# Run the whole fleet benchmark e2e concurrently with live progress (see
# optimizer/orchestrate/00_run_fleet.py --help). Pass-through args, e.g.
#   just fleet-bench --only jetson,samsung --phases profile,schedule,run,summary
#   just fleet-bench --fresh     # start from scratch (wipe old results) after a code change
fleet-bench *args:
    uv run optimizer/orchestrate/00_run_fleet.py {{args}}

# Delete ALL regenerable benchmark results (profiling + z3 schedules + run logs +
# speedup-summary). Use after a kernel/runtime change to start clean; or run
# `fleet-bench --fresh` to wipe + re-benchmark in one go.
bench-clean:
    rm -rf data/profiling data/schedules_btpm data/schedules_isolated data/sched_logs

# 4. Run the unit tests across the whole matrix (build first with `just build`).
# Fail-loud: any failing fleet test makes this go red (see the per-target notes).
test: test-jetson test-minipc test-samsung

# Fail if any expected GPU (app x backend x hardware) cell was never RAN on the
# fleet. Diffs fleet-coverage.log (emitted by the run-on-*.sh deploy scripts as
# BT-CELL markers) against the matrix derived from fleet.json. Run after a fleet sweep; this is
# what the dev->main promotion gate should assert (see CONTRIBUTING.md).
check-fleet:
    scripts/check_fleet_coverage.py

# Deploy x86 build to the iGPU mini pc and run OMP + Vulkan.
test-minipc: (_test-ssh minipc_host "minipc" "build/vulkan" omp_vk_bins)

# Deploy aarch64 build to the Jetsons and run OMP + CUDA + Vulkan.
test-jetson: (_test-ssh jetson_host "duck-stable" "build/jetson" jetson_bins)
test-duck-naughty: (_test-ssh jetson_naughty_host "duck-naughty" "build/jetson" jetson_bins)

# Deploy + run on a Linux SSH target (internal helper).
# NOTE: rocky-ryzen's login shell is fish (the reflashed Jetsons are bash), so
# `ssh host bash -s` forces bash for the loop; a bare `ssh host 'for ...'` fails
# with "Missing end to balance this for loop" on fish hosts.
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
    git ls-files '*.json' ':(exclude)dashboard/*' | xargs -r bunx prettier@3.8.4 --write --log-level warn

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
    git ls-files '*.json' ':(exclude)dashboard/*' | xargs -r bunx prettier@3.8.4 --check --log-level warn || rc=1
    exit $rc

# --- static analysis & sanitizers -------------------------------------------
# These build the OpenMP surface only (no GPU needed), so they run anywhere.

# Build + run the OMP tests under AddressSanitizer + UBSan. Catches heap
# overflow / use-after-free / UB that the optimised oracle build hides. Runs the
# full omp label locally (the fleet has the cores for pipeline-e2e); the hosted CI
# job excludes pipeline-e2e (needs >=9 physical cores to pin).
asan:
    cmake --preset pc-asan
    cmake --build --preset pc-asan
    ASAN_OPTIONS=abort_on_error=1:detect_leaks=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    ctest --test-dir build/pc-asan -L omp --output-on-failure

# Build + run the OMP tests under ThreadSanitizer. Most valuable on the concurrent
# pipeline ring (pipeline-e2e), so run the full label on a machine with the cores.
tsan:
    cmake --preset pc-tsan
    cmake --build --preset pc-tsan
    TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1 \
    ctest --test-dir build/pc-tsan -L omp --output-on-failure

# Lint only the lines changed vs a base ref (default: origin/main) so legacy
# findings never wall the gate -- the same gate CI runs. `build` is the compile-db
# dir: build/pc lints the OMP surface anywhere; pass build/vulkan (or build/jetson)
# on a box that has them to also lint the GPU engine TUs. Fails on first-party check
# findings; ignores clang-diagnostic-* (a backend TU absent from `build`).
tidy base="origin/main" build="build/pc":
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}"
    test -f "{{build}}/compile_commands.json" || cmake --preset pc
    script=$(command -v clang-tidy-diff.py 2>/dev/null \
      || ls /usr/bin/clang-tidy-diff*.py /usr/lib/llvm-*/share/clang/clang-tidy-diff.py 2>/dev/null | head -1)
    out=$(git diff -U0 "{{base}}" -- '*.cpp' '*.hpp' '*.cu' '*.cuh' \
      | python3 "$script" -p1 -path "{{build}}" -j"$(nproc)" 2>&1) || true
    echo "$out"
    findings=$(echo "$out" | grep -E ': (warning|error): ' | grep -vE '\[clang-diagnostic-' || true)
    [ -z "$findings" ] && echo "clang-tidy: clean" || { echo "clang-tidy: findings above"; exit 1; }
