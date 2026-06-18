#!/usr/bin/env bash
# Deploy & run test binaries on an Android phone via adb.
#
# Two gotchas this script handles for you:
#   1) `adb shell` SWALLOWS the rest of the caller's stdin. Inside a heredoc-driven
#      script the first `adb shell` eats the remaining lines -> later commands
#      silently don't run (exit 0, empty output). Every adb call here ends in
#      `</dev/null` so it can't consume the script.
#   2) libc++_shared.so must ride along with the binary (it isn't on the device).
#
# Both phones now hang off rocky-ryzen's adb (the build box has none attached):
#   - Pixel 7a (3A021JEHN02756, subgroup 16)
#   - Samsung  (R5CY21Y3VEV,    subgroup 32)
# `adb devices` lists both, so `adb -s <serial>` (used throughout) picks one. Run
# this script ON rocky-ryzen after copying build/android there.
# Deploy goes to the Android tmp: /data/local/tmp/bt.
#
# Usage:   scripts/run-on-android.sh <serial> [test-target ...]
# Env:     BT_ANDROID_BUILD (default build/android), ANDROID_NDK_HOME / ANDROID_HOME
set -euo pipefail

serial=${1:?usage: run-on-android.sh <serial> [test-target ...]}; shift || true
BUILD=${BT_ANDROID_BUILD:-build/android}
DEST=/data/local/tmp/bt

targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp)

if ! adb -s "$serial" get-state >/dev/null 2>&1; then
  echo "error: '$serial' is not visible to local adb." >&2
  echo "  Both phones (Pixel 7a 3A021JEHN02756, Samsung R5CY21Y3VEV) are attached to" >&2
  echo "  rocky-ryzen, not the build box. Copy build/android there, then run this script" >&2
  echo "  on rocky-ryzen ('adb -s <serial>' selects between the two)." >&2
  exit 1
fi

ndk=${ANDROID_NDK_HOME:-${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/29.0.14206865}
# libc++ ABI dir under the NDK: aarch64 for the arm64 preset; set BT_ANDROID_LIBCXX_ARCH=
# arm-linux-androideabi for an armeabi-v7a (android32) build so the matching .so is shipped.
libcxx_arch=${BT_ANDROID_LIBCXX_ARCH:-aarch64}
libcxx=$(find "$ndk" -name libc++_shared.so -path "*${libcxx_arch}*" 2>/dev/null | head -1)
[ -n "$libcxx" ] || { echo "error: libc++_shared.so not found under $ndk" >&2; exit 1; }

paths=(); for t in "${targets[@]}"; do paths+=("$BUILD/$t"); done

echo ">> staging ${#targets[@]} binaries + libc++_shared.so to $serial:$DEST"
adb -s "$serial" shell "mkdir -p $DEST" </dev/null
adb -s "$serial" push "${paths[@]}" "$libcxx" "$DEST/" </dev/null

echo ">> running on $serial (--device $serial)"
fail=0
for t in "${targets[@]}"; do
  echo "== $t =="
  # </dev/null on EVERY adb shell, or it eats the loop's remaining iterations.
  adb -s "$serial" shell "cd $DEST && chmod 755 $t && LD_LIBRARY_PATH=. ./$t --device $serial" </dev/null || fail=1
done
exit $fail
