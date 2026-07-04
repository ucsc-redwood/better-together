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
# Per-cell coverage markers (RAN/SKIP/FAIL) for scripts/check_fleet_coverage.py.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/lib-cell-marker.sh"
CELL_HW=${BT_CELL_HW:-minipc}
CELL_LOG=${BT_CELL_LOG:-$ROOT/fleet-coverage.log}

targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk)

paths=()
for t in "${targets[@]}"; do paths+=("$BUILD/$t"); done

# Local mode: when this runs ON the Rocky box itself (BT_ROCKY_HOST=localhost, the
# fleet runner), skip ssh/scp and run the binaries straight from $BUILD.
# VK_LOADER_LAYERS_DISABLE keeps RADV's validation-layer stdout noise out of the
# coverage-marker parsing (and off the measured path).
if [ "$HOST" = localhost ] || [ "$HOST" = "$(hostname)" ]; then
  echo ">> running locally (--device $DEVICE)"
  runlog="$(mktemp)"
  set -o pipefail
  (
    cd "$BUILD" && fail=0
    for t in "${targets[@]}"; do
      echo "== $t =="
      VK_LOADER_LAYERS_DISABLE='~all~' LD_LIBRARY_PATH=. ./"$t" --device "$DEVICE" || fail=1
    done
    exit $fail
  ) | tee "$runlog"
  rc=${PIPESTATUS[0]}
  bt_emit_markers_from_log "$runlog" "$CELL_HW" | tee -a "$CELL_LOG"
  echo ">> coverage markers appended to $CELL_LOG"
  rm -f "$runlog"
  exit $rc
fi

echo ">> staging ${#targets[@]} binaries to $HOST:$DEST"
ssh "$HOST" bash -s <<EOF
mkdir -p $DEST
EOF
scp "${paths[@]}" "$HOST:$DEST/"

echo ">> running on $HOST (--device $DEVICE)"
# tee the remote output to a log so we can post-process per-cell coverage markers
# locally; pipefail keeps the remote exit status (not tee's) as the gate result.
runlog="$(mktemp)"
set -o pipefail
rc=0
ssh "$HOST" bash -s "$DEVICE" "${targets[@]}" <<'EOF' | tee "$runlog"
set -e
device=$1; shift
cd /tmp/bt
# Deployed real weights (scripts/deploy-weights.sh rocky) are picked up
# automatically; without a deploy the apps keep their synthetic seeded init.
[ -d /tmp/bt/weights/dense ] && export BT_WEIGHTS_DIR=/tmp/bt/weights
# Same for the real tree corpus (scripts/deploy-tree-data.sh rocky).
[ -d /tmp/bt/tree-data ] && export BT_TREE_DATA_DIR=/tmp/bt/tree-data
fail=0
for t in "$@"; do
  echo "== $t =="
  LD_LIBRARY_PATH=. ./"$t" --device "$device" || fail=1
done
exit $fail
EOF
rc=$?

bt_emit_markers_from_log "$runlog" "$CELL_HW" | tee -a "$CELL_LOG"
echo ">> coverage markers appended to $CELL_LOG (check: scripts/check_fleet_coverage.py)"
rm -f "$runlog"
exit $rc
