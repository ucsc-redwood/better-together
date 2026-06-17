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
# The phone must be visible to THIS host's adb:
#   - Pixel 7a (3A021JEHN02756) is attached to the build box.
#   - Samsung  (R5CY21Y3VEV)    is attached to rocky-ryzen -> run this script there.
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
  echo "  The Samsung (R5CY21Y3VEV) is attached to rocky-ryzen, not the build box;" >&2
  echo "  copy the build there and run this script on that host. The Pixel 7a" >&2
  echo "  (3A021JEHN02756) is on the build box." >&2
  exit 1
fi

ndk=${ANDROID_NDK_HOME:-${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/29.0.14206865}
libcxx=$(find "$ndk" -name libc++_shared.so -path '*aarch64*' 2>/dev/null | head -1)
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
