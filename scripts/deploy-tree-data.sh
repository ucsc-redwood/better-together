#!/usr/bin/env bash
# Deploy the real Octomap-derived tree corpus (resources/octomap/data/points.npy,
# built by scripts/data_prep/oct.py) to a fleet target.
#
# Pushes points.npy to the target's tree-data dir, where the run-on-*.sh scripts
# auto-export BT_TREE_DATA_DIR when the deployed file exists (without a deploy,
# nothing changes -- the tree app keeps its synthetic seeded input). See
# specs/001-octomap-real-workload/contracts/tree-real-data-contract.md.
#
#   jetson | rocky  -> $HOST:/tmp/bt/tree-data/          (ssh HOST bash -s, fish-proof)
#   android <serial> -> /data/local/tmp/bt/tree-data/    (adb -s <serial> push)
#
# Usage:   scripts/deploy-tree-data.sh jetson
#          scripts/deploy-tree-data.sh rocky
#          scripts/deploy-tree-data.sh android <serial>
# Env:     BT_JETSON_HOST (default doremy@duck-stable), BT_ROCKY_HOST (rocky-ryzen),
#          BT_TREE_DATA_SRC (default resources/octomap/data)
set -euo pipefail

kind=${1:?usage: deploy-tree-data.sh (jetson|rocky|android <serial>)}
SRC=${BT_TREE_DATA_SRC:-resources/octomap/data}

[ -e "$SRC/points.npy" ] || {
  echo "error: $SRC/points.npy not found (build it first: scripts/data_prep/oct.py, see specs/001-octomap-real-workload/quickstart.md)" >&2
  exit 1
}

case "$kind" in
  jetson | rocky)
    if [ "$kind" = jetson ]; then
      HOST=${BT_JETSON_HOST:-doremy@duck-stable}
    else
      HOST=${BT_ROCKY_HOST:-rocky-ryzen}
    fi
    DEST=/tmp/bt/tree-data
    echo ">> staging $SRC/points.npy -> $HOST:$DEST"
    ssh "$HOST" bash -s <<EOF
mkdir -p $DEST
EOF
    scp "$SRC/points.npy" "$HOST:$DEST/"
    ;;
  android)
    serial=${2:?usage: deploy-tree-data.sh android <serial>}
    DEST=/data/local/tmp/bt/tree-data
    echo ">> staging $SRC/points.npy -> $serial:$DEST"
    # </dev/null on every adb call so it cannot swallow this script's stdin.
    adb -s "$serial" shell "mkdir -p $DEST" </dev/null
    adb -s "$serial" push "$SRC/points.npy" "$DEST/" </dev/null
    ;;
  *)
    echo "error: unknown target '$kind' (jetson|rocky|android <serial>)" >&2
    exit 1
    ;;
esac
echo ">> done"
