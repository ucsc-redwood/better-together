#!/usr/bin/env bash
# Cross-compile the CUDA backend for the Jetson Orin inside the NVIDIA cross
# container, using the `jetson` CMake preset. One command; native x86 speed.
#
#   scripts/cross-build-jetson.sh                 # configure + build everything
#   BT_CROSS_IMAGE=bt-cross:6.1 scripts/cross-build-jetson.sh
#
# Output: build/jetson/test-*-cu (aarch64). scp to the Jetson and run with
#         --device jetson.
set -euo pipefail

IMAGE="${BT_CROSS_IMAGE:-bt-cross:6.1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">> building $IMAGE from Dockerfile.cross"
  docker build -t "$IMAGE" -f "$ROOT/Dockerfile.cross" "$(mktemp -d)"
fi

# Run as the current user so build artifacts in build/ are not root-owned.
# HOME must be writable (CPM/git use it).
docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/build \
  -v "$ROOT:/workspace" -w /workspace \
  "$IMAGE" bash -lc '
    cmake --preset jetson &&
    cmake --build --preset jetson -j"$(nproc)"
  '

echo
echo ">> done. aarch64 binaries in build/jetson/:"
ls -1 "$ROOT"/build/jetson/test-*-cu 2>/dev/null || true
