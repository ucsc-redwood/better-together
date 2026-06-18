#!/usr/bin/env bash
# Deploy & run Vulkan binaries on BOTH Mali phones (Pixel 7a + Samsung) in one go.
# Both phones' adb lives on rocky-ryzen now, so this stages to rocky once and drives
# each phone there with `adb -s <serial>`. It's the Mali differential-oracle gate for
# the Phase-4 §3.2/§3.3 kiss-vk perf work: the HOST_CACHED flush is a no-op on
# rocky/RADV, so a coherency regression (BUGS-FOUND §7, the 47x one) only shows up
# HERE -- run this after every kiss-vk change and require it stays green.
#
# Builds nothing -- build the `android` preset first (see 02-building.md). Two gotchas
# handled: rocky's login shell is fish (we pipe bash over stdin), and `adb shell`
# swallows the rest of stdin (every adb call ends in `</dev/null`).
#
# Usage:   scripts/run-mali-oracle.sh [test-target ...]
# Env:     BT_ANDROID_BUILD (build/android), BT_ROCKY_HOST (rocky-ryzen),
#          BT_MALI_SERIALS ("3A021JEHN02756 R5CY21Y3VEV"), ANDROID_NDK_HOME/ANDROID_HOME
set -euo pipefail

HOST=${BT_ROCKY_HOST:-rocky-ryzen}
BUILD=${BT_ANDROID_BUILD:-build/android}
SERIALS=${BT_MALI_SERIALS:-"3A021JEHN02756 R5CY21Y3VEV"}
STAGE=/tmp/bt-android
DEST=/data/local/tmp/bt

targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk)

ndk=${ANDROID_NDK_HOME:-${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/29.0.14206865}
libcxx_arch=${BT_ANDROID_LIBCXX_ARCH:-aarch64}
libcxx=$(find "$ndk" -name libc++_shared.so -path "*${libcxx_arch}*" 2>/dev/null | head -1)
[ -n "$libcxx" ] || { echo "error: libc++_shared.so not found under $ndk" >&2; exit 1; }

paths=(); for t in "${targets[@]}"; do paths+=("$BUILD/$t"); done

echo ">> staging ${#targets[@]} binaries + libc++_shared.so to $HOST:$STAGE"
ssh "$HOST" "mkdir -p $STAGE" </dev/null
scp "${paths[@]}" "$libcxx" "$HOST:$STAGE/"

fail=0
for serial in $SERIALS; do
  echo ">> ===== phone $serial ====="
  # All adb work runs ON rocky (the phones' adb host). bash over stdin (fish login
  # shell); every `adb shell` ends in </dev/null.
  ssh "$HOST" bash -s "$serial" "$STAGE" "$DEST" "${targets[@]}" <<'REMOTE' || fail=1
set -e
serial=$1; stage=$2; dest=$3; shift 3
adb -s "$serial" shell "mkdir -p $dest" </dev/null
pushlist=""
for t in "$@"; do pushlist="$pushlist $stage/$t"; done
adb -s "$serial" push $pushlist "$stage/libc++_shared.so" "$dest/" </dev/null >/dev/null
rc=0
for t in "$@"; do
  echo "== $t =="
  adb -s "$serial" shell "cd $dest && chmod 755 $t && LD_LIBRARY_PATH=. ./$t --device $serial" </dev/null || rc=1
done
exit $rc
REMOTE
done

exit $fail
