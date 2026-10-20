#pragma once

// Launch geometry of the shaders new_cifar_dense_linear_bt{,_portable}.comp (one body,
// new_cifar_dense_linear_bt.glsl), shared by every dispatcher that selects them (cifar-dense and
// cifar-sparse). Keep in step with ROWS and MAX_N in the body: a dispatcher that disagrees leaves
// output features or images uncomputed.
namespace kiss_vk {

// Rows per workgroup (ROWS); a row owns FPR output features (spec constant 3, 2 by default), so
// grid.x = div_ceil(out_features, kLinearBtRows * FPR).
inline constexpr int kLinearBtRows = 16;
// Images per workgroup (MAX_N): grid.y = div_ceil(N, this).
inline constexpr int kLinearBtBatchTile = 16;

// The module to create for a device: the clustered reduction where the device has clustered
// subgroup operations (its pipeline is subgroup_size_dependent()), else the portable module, which
// declares no subgroup capability and takes no subgroup-size constant. The choice has to precede
// module creation: a module that declares GroupNonUniformClustered is invalid on a device without
// it.
[[nodiscard]] constexpr const char* linear_bt_shader(const bool subgroup_clustered) {
  return subgroup_clustered ? "new_cifar_dense_linear_bt" : "new_cifar_dense_linear_bt_portable";
}

}  // namespace kiss_vk
