#!/usr/bin/env bash
# Deploy & run BetterTogether test binaries on the Jetson Orin (CUDA + Vulkan).
#
# Why this script exists (gotcha it handles for you):
#   duck-naughty's LOGIN SHELL IS FISH. An inline
#     ssh duck-naughty '... for t in ...; do ...; done'
#   fails with a fish parse error ("Missing end to balance this for loop").
#   We pipe a real bash script over stdin via `ssh HOST bash -s` instead, which
#   bypasses the login shell entirely.
#
# Deploy goes to the target's /tmp (never the home dir): /tmp/bt.
#
# Usage:   scripts/run-on-jetson.sh [test-target ...]
#          default targets: the three CUDA test binaries
# Env:     BT_JETSON_HOST (default duck-naughty), BT_JETSON_DEVICE (jetson),
#          BT_JETSON_BUILD (build/jetson)
set -euo pipefail

HOST=${BT_JETSON_HOST:-duck-naughty}
DEVICE=${BT_JETSON_DEVICE:-jetson}
BUILD=${BT_JETSON_BUILD:-build/jetson}
DEST=/tmp/bt
# Per-cell coverage markers (RAN/SKIP/FAIL) for scripts/check_fleet_coverage.py.
# BT_CELL_HW names this fleet host (the fleet.json device key); BT_CELL_LOG is where the
# BT-CELL marker lines are appended (default fleet-coverage.log at the repo root).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/lib-cell-marker.sh"
CELL_HW=${BT_CELL_HW:-jetson}
CELL_LOG=${BT_CELL_LOG:-$ROOT/fleet-coverage.log}

targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(test-tree-cu test-cifar-dense-cu test-cifar-sparse-cu)

paths=()
for t in "${targets[@]}"; do paths+=("$BUILD/$t"); done

echo ">> staging ${#targets[@]} binaries to $HOST:$DEST"
ssh "$HOST" bash -s <<EOF
mkdir -p $DEST
EOF
scp "${paths[@]}" "$HOST:$DEST/"

echo ">> running on $HOST (--device $DEVICE)"
# QUOTED heredoc -> nothing expands locally; args are passed via \`bash -s\`.
# tee the remote output to a log so we can post-process per-cell coverage markers
# locally; pipefail keeps the remote exit status (not tee's) as the gate result.
runlog="$(mktemp)"
set -o pipefail
rc=0
ssh "$HOST" bash -s "$DEVICE" "${targets[@]}" <<'EOF' | tee "$runlog"
set -e
device=$1; shift
cd /tmp/bt
fail=0
for t in "$@"; do
  echo "== $t =="
  ./"$t" --device "$device" || fail=1
done
exit $fail
EOF
rc=$?

# Append RAN/SKIP/FAIL markers for the differential cells to the coverage log.
bt_emit_markers_from_log "$runlog" "$CELL_HW" | tee -a "$CELL_LOG"
echo ">> coverage markers appended to $CELL_LOG (check: scripts/check_fleet_coverage.py)"
rm -f "$runlog"
exit $rc
