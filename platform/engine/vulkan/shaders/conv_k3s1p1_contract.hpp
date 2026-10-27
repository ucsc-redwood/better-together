#pragma once

#include <cstddef>
#include <cstdint>

// Launch geometry of the 3x3/stride-1/pad-1 conv shaders new_cifar_dense_conv2d_k3s1p1{,_oc}.comp
// and new_cifar_sparse_conv2d_k3s1p1.comp, shared by the cifar-dense and cifar-sparse dispatchers
// and the conv tests. The dense shader picks its path from the subgroup size it is specialized with
// (spec constant 0, the engine's subgroup_size()); cifar-sparse picks one of two shaders by it. The
// path below 32 (Mali) may compute several adjacent output pixels of a row per invocation. Keep in
// step with OC_TILE and PX in the shaders: a grid that counts more outputs per invocation than the
// shader computes leaves outputs uncomputed.
namespace kiss_vk {

// Output channels per invocation of the dense shader (OC_TILE), on every path.
inline constexpr int kDenseConvChannelsPerInvocation = 4;
// Adjacent output pixels of a row per invocation on the subgroup < 32 path (PX in each shader).
inline constexpr int kDenseConvMaliPixels = 4;
inline constexpr int kSparseConvMaliPixels = 4;

// Invocations for an N x K x H x W output; grid.x = div_ceil(this, 256).
constexpr size_t dense_conv_k3s1p1_invocations(
    const int n, const int k, const int h, const int w, const uint32_t subgroup_size) {
  const int px = subgroup_size < 32 ? kDenseConvMaliPixels : 1;
  return static_cast<size_t>(n) * (k / kDenseConvChannelsPerInvocation) * h * ((w + px - 1) / px);
}

// new_cifar_dense_conv2d_k3s1p1_oc.comp, which cifar-dense runs on subgroups of 32 and above. Its
// tile is spec constants 1-3 (cifar-gpu-kernels C2/C3): output channels and adjacent output pixels
// of a row per invocation, and input channels staged in shared memory per pass (0: none).
struct DenseConvTile {
  int oc = 4, px = 1, c = 0;
};
// Widest map the staging (c > 0) is sized for; the dispatcher does not stage wider ones.
inline constexpr int kDenseConvStagedMaxWidth = 32;

// Shared-memory floats one staged input channel takes: a workgroup's 256 invocations span at most
// 255 / w_tiles + 2 output rows, staged with a halo row above and below, w_tiles * px + 2 floats
// wide (w_tiles = ceil(W / px)); the most over every width up to kDenseConvStagedMaxWidth. The
// shader's LDS_FLOATS (spec constant 4) is c times this.
constexpr int dense_conv_staged_floats_per_channel(const int px) {
  int most = 0;
  for (int w = 1; w <= kDenseConvStagedMaxWidth; ++w) {
    const int w_tiles = (w + px - 1) / px;
    const int floats = (255 / w_tiles + 4) * (w_tiles * px + 2);
    most = floats > most ? floats : most;
  }
  return most;
}

// x = the K / tile.oc blocks of output channels, y = the blocks of 256 invocations.
struct DenseConvOcGrid {
  uint32_t x, y;
};
constexpr DenseConvOcGrid dense_conv_k3s1p1_oc_grid(
    const int n, const int k, const int h, const int w, const DenseConvTile& tile) {
  const size_t invocations = static_cast<size_t>(n) * h * ((w + tile.px - 1) / tile.px);
  return {static_cast<uint32_t>(k / tile.oc), static_cast<uint32_t>((invocations + 255) / 256)};
}

// new_cifar_sparse_conv2d_k3s1p1.comp is the Mali path only: cifar-sparse runs it below subgroup
// 32, and new_cifar_sparse_conv2d_k3s1p1_oc.comp, grid (div_ceil(N*H*W, 256), K, 1), on 32 and
// above.
constexpr size_t sparse_conv_k3s1p1_mali_invocations(const int n,
                                                     const int k,
                                                     const int h,
                                                     const int w) {
  return static_cast<size_t>(n) * k * h * ((w + kSparseConvMaliPixels - 1) / kSparseConvMaliPixels);
}

}  // namespace kiss_vk
