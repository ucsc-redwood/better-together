#!/usr/bin/env bash
# Deploy & run Vulkan test binaries on the Rocky MiniPC (integrated GPU / RADV).
#
# Gotcha handled: rocky-ryzen's LOGIN SHELL IS FISH (same as the Jetson) -> we
# pipe a bash script via `ssh HOST bash -s` rather than `ssh HOST '...'`, so bash
# loops / `VAR=val cmd` prefixes don't trip the fish parser.
#
# Deploy goes to the target's /tmp (never home): /tmp/bt. Vulkan tests link
# OpenMP; gcc's libgomp is normally already present on the box, so we ship only
# the binaries and run with LD_LIBRARY_PATH=. (add libomp.so to the scp list if
# the loader can't find it).
#
# Usage:   scripts/run-on-rocky.sh [test-target ...]
# Env:     BT_ROCKY_HOST (default rocky-ryzen), BT_ROCKY_DEVICE (minipc),
#          BT_ROCKY_BUILD (build/vulkan)
set -euo pipefail

HOST=${BT_ROCKY_HOST:-rocky-ryzen}
DEVICE=${BT_ROCKY_DEVICE:-minipc}
BUILD=${BT_ROCKY_BUILD:-build/vulkan}
DEST=/tmp/bt

targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk)

paths=(); for t in "${targets[@]}"; do paths+=("$BUILD/$t"); done

echo ">> staging ${#targets[@]} binaries to $HOST:$DEST"
ssh "$HOST" bash -s <<EOF
mkdir -p $DEST
EOF
scp "${paths[@]}" "$HOST:$DEST/"

echo ">> running on $HOST (--device $DEVICE)"
ssh "$HOST" bash -s "$DEVICE" "${targets[@]}" <<'EOF'
set -e
device=$1; shift
cd /tmp/bt
fail=0
for t in "$@"; do
  echo "== $t =="
  LD_LIBRARY_PATH=. ./"$t" --device "$device" || fail=1
done
exit $fail
EOF
