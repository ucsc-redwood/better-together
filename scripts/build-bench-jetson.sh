#!/usr/bin/env bash
# Cross-build the Jetson BENCHMARK binaries (bm-prof-* + bm-gen-logs-*, CUDA+Vulkan)
# ON rocky-ryzen and pull them back into build/jetson/.
#
# Why on rocky: the bt-cross:7.2 podman image lives there, not on this (x86) box. We
# rsync the source tree to rocky:~/bt-src (incl. .git so bm-prof's git-sha provenance
# is correct), cross-compile in the container, then rsync the binaries back.
#
# Used by `just build-bench-jetson`, which 00_run_fleet.py's build phase invokes.
# Env: BT_BENCH_JETSON_HOST (default rocky-ryzen), BT_BENCH_SRC (default bt-src).
set -euo pipefail

HOST=${BT_BENCH_JETSON_HOST:-rocky-ryzen}
SRC=${BT_BENCH_SRC:-bt-src}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGETS="bm-prof-tree-cu bm-prof-cifar-dense-cu bm-prof-cifar-sparse-cu \
bm-prof-tree-vk bm-prof-cifar-dense-vk bm-prof-cifar-sparse-vk \
bm-gen-logs-tree-cu bm-gen-logs-cifar-dense-cu bm-gen-logs-cifar-sparse-cu \
bm-gen-logs-tree-vk bm-gen-logs-cifar-dense-vk bm-gen-logs-cifar-sparse-vk"

echo ">> rsync repo -> $HOST:$SRC"
rsync -az --delete \
  --exclude=build/ --exclude=data/ --exclude=resources/ --exclude=saved_params/ \
  --exclude='.venv*' --exclude='**/node_modules/' --exclude='**/__pycache__/' \
  --exclude='*.zip' --exclude='*.tar.gz' --exclude=Testing/ --exclude=dashboard/manifest/ \
  ./ "$HOST:$SRC/"

echo ">> podman cross-build on $HOST (bt-cross:7.2)"
# fish login shell on rocky -> feed bash a script over stdin (never `ssh host '...'`).
ssh "$HOST" bash -s <<EOF
set -euo pipefail
cd ~/$SRC
podman run --rm --userns=keep-id -e HOME=/workspace/build \\
  -v "\$HOME/$SRC:/workspace:z" -w /workspace bt-cross:7.2 \\
  bash -lc 'cmake --preset jetson && cmake --build --preset jetson --target $TARGETS -j'
EOF

echo ">> pull binaries -> build/jetson"
mkdir -p build/jetson
rsync -az "$HOST:$SRC/build/jetson/bm-prof-"* "$HOST:$SRC/build/jetson/bm-gen-logs-"* build/jetson/
echo ">> done: $(ls build/jetson/bm-prof-* build/jetson/bm-gen-logs-* | wc -l) binaries"
