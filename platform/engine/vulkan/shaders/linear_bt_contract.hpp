#pragma once

// Launch geometry of the shader new_cifar_dense_linear_bt.comp, shared by every dispatcher
// that selects it (cifar-dense and cifar-sparse). Keep in step with ROWS and MAX_N in
// the shader: a dispatcher that disagrees leaves output features or images uncomputed.
namespace kiss_vk {

// Rows per workgroup (ROWS); a row owns FPR output features (spec constant 3, 2 by default), so
// grid.x = div_ceil(out_features, kLinearBtRows * FPR).
inline constexpr int kLinearBtRows = 16;
// Images per workgroup (MAX_N): grid.y = div_ceil(N, this).
inline constexpr int kLinearBtBatchTile = 16;

}  // namespace kiss_vk
