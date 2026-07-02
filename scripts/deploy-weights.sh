#!/usr/bin/env bash
# Deploy the real AlexNetCIFAR weights (saved_params/export/) to a fleet target.
#
# Pushes dense/, sparse/, test_batch.npy and test_labels.npy to the target's
# weight dir, where the run-on-*.sh scripts auto-export BT_WEIGHTS_DIR when the
# deployed dir exists (without a deploy, nothing changes -- the apps keep their
# synthetic seeded init). See docs/instruction-for-ai/04-alexnet-cifar-spec.md §7.
#
#   jetson | rocky  -> $HOST:/tmp/bt/weights/          (ssh HOST bash -s, fish-proof)
#   android <serial> -> /data/local/tmp/bt/weights/    (adb -s <serial> push)
#
# Usage:   scripts/deploy-weights.sh jetson
#          scripts/deploy-weights.sh rocky
#          scripts/deploy-weights.sh android <serial>
# Env:     BT_JETSON_HOST (default doremy@duck-stable), BT_ROCKY_HOST (rocky-ryzen),
#          BT_WEIGHTS_SRC (default saved_params/export)
set -euo pipefail

kind=${1:?usage: deploy-weights.sh (jetson|rocky|android <serial>)}
SRC=${BT_WEIGHTS_SRC:-saved_params/export}

for p in dense sparse test_batch.npy test_labels.npy; do
  [ -e "$SRC/$p" ] || {
    echo "error: $SRC/$p not found (run the export first: 04-alexnet-cifar-spec.md §6)" >&2
    exit 1
  }
done

case "$kind" in
  jetson | rocky)
    if [ "$kind" = jetson ]; then
      HOST=${BT_JETSON_HOST:-doremy@duck-stable}
    else
      HOST=${BT_ROCKY_HOST:-rocky-ryzen}
    fi
    DEST=/tmp/bt/weights
    echo ">> staging $SRC -> $HOST:$DEST"
    ssh "$HOST" bash -s <<EOF
mkdir -p $DEST
EOF
    scp -r "$SRC"/dense "$SRC"/sparse "$SRC"/test_batch.npy "$SRC"/test_labels.npy "$HOST:$DEST/"
    ;;
  android)
    serial=${2:?usage: deploy-weights.sh android <serial>}
    DEST=/data/local/tmp/bt/weights
    echo ">> staging $SRC -> $serial:$DEST"
    # </dev/null on every adb call so it cannot swallow this script's stdin.
    adb -s "$serial" shell "mkdir -p $DEST" </dev/null
    adb -s "$serial" push "$SRC"/dense "$SRC"/sparse "$SRC"/test_batch.npy "$SRC"/test_labels.npy "$DEST/" </dev/null
    ;;
  *)
    echo "error: unknown target '$kind' (jetson|rocky|android <serial>)" >&2
    exit 1
    ;;
esac
echo ">> done"
