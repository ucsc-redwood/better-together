#pragma once

// Launch geometry of the shader new_cifar_dense_linear_bt.comp, shared by every dispatcher
// that selects it (cifar-dense and cifar-sparse). Keep in step with ROWS * FPR and MAX_N in
// the shader: a dispatcher that disagrees leaves output features or images uncomputed.
namespace kiss_vk {

// Output features per workgroup (ROWS * FPR): grid.x = div_ceil(out_features, this).
inline constexpr int kLinearBtFeaturesPerGroup = 32;
// Images per workgroup (MAX_N): grid.y = div_ceil(N, this).
inline constexpr int kLinearBtBatchTile = 16;

}  // namespace kiss_vk
