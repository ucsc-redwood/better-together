#!/usr/bin/env bash
# Build the x86-Vulkan or Android BENCHMARK binaries (bm-prof-* + bm-gen-logs-*)
# ON rocky-ryzen and pull them back into build/<preset>/.
#
# Since the old build box retired (2026-07-02), rocky is the build host for
# EVERYTHING: x86 Vulkan natively, Android via the NDK at ~/android-ndk-r29,
# Jetson via podman (that flow lives in build-bench-jetson.sh).
#
# Usage: build-bench-on-rocky.sh (vulkan|android)
# Env:   BT_BENCH_ROCKY_HOST (default doremy@rocky-ryzen), BT_BENCH_SRC (default bt-src)
set -euo pipefail

preset=${1:?usage: build-bench-on-rocky.sh (vulkan|android)}
case "$preset" in vulkan | android) ;; *)
  echo "error: unknown preset '$preset' (vulkan|android)" >&2
  exit 1
  ;;
esac

HOST=${BT_BENCH_ROCKY_HOST:-doremy@rocky-ryzen}
SRC=${BT_BENCH_SRC:-bt-src}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGETS="bm-prof-tree-vk bm-prof-cifar-dense-vk bm-prof-cifar-sparse-vk \
bm-gen-logs-tree-vk bm-gen-logs-cifar-dense-vk bm-gen-logs-cifar-sparse-vk"

echo ">> rsync repo -> $HOST:$SRC"
rsync -az --delete \
  --exclude=build/ --exclude=data/ --exclude=resources/ --exclude=saved_params/ \
  --exclude='.venv*' --exclude='**/node_modules/' --exclude='**/__pycache__/' \
  --exclude='*.zip' --exclude='*.tar.gz' --exclude=Testing/ --exclude=dashboard/manifest/ \
  ./ "$HOST:$SRC/"

echo ">> build --preset $preset on $HOST"
# fish login shell on rocky -> feed bash a script over stdin (never `ssh host '...'`).
ssh "$HOST" bash -s <<EOF
set -euo pipefail
cd ~/$SRC
export ANDROID_NDK_HOME="\$HOME/android-ndk-r29"
cmake --preset $preset
cmake --build --preset $preset --target $TARGETS -j"\$(nproc)"
EOF

echo ">> pull binaries -> build/$preset"
mkdir -p "build/$preset"
rsync -az "$HOST:$SRC/build/$preset/bm-prof-"* "$HOST:$SRC/build/$preset/bm-gen-logs-"* "build/$preset/"
echo ">> done: $(ls build/$preset/bm-prof-* build/$preset/bm-gen-logs-* | wc -l) binaries"
