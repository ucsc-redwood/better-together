#pragma once

#include <cstddef>
#include <cstdint>

// Launch geometry of the 3x3/stride-1/pad-1 conv shaders new_cifar_dense_conv2d_k3s1p1.comp and
// new_cifar_sparse_conv2d_k3s1p1.comp, shared by the cifar-dense and cifar-sparse dispatchers and
// the conv tests. The dense shader picks its path from the subgroup size it is specialized with
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
