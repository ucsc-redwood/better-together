#pragma once

#include <string>
#include <unordered_map>
#include <utility>

namespace kiss_vk {

namespace shaders {

#include "h/cifar_conv2d_spv.h"
#include "h/cifar_linear_spv.h"
#include "h/cifar_maxpool2d_spv.h"
#include "h/cifar_sparse_conv2d_spv.h"
#include "h/cifar_sparse_linear_spv.h"
#include "h/cifar_sparse_maxpool_spv.h"
#include "h/hello_multiple_steps_spv.h"
#include "h/hello_vector_add_spv.h"
#include "h/sort_safe_radix_sort_spv.h"
#include "h/tmp_add_base_spv.h"
#include "h/tmp_add_base_v2_16_spv.h"
#include "h/tmp_add_base_v2_32_spv.h"
#include "h/tmp_add_base_v2_64_spv.h"
#include "h/tmp_global_exclusive_scan_spv.h"
#include "h/tmp_global_exclusive_scan_v2_16_spv.h"
#include "h/tmp_global_exclusive_scan_v2_32_spv.h"
#include "h/tmp_global_exclusive_scan_v2_64_spv.h"
#include "h/tmp_local_inclusive_scan_spv.h"
#include "h/tmp_local_inclusive_scan_v2_16_spv.h"
#include "h/tmp_local_inclusive_scan_v2_32_spv.h"
#include "h/tmp_local_inclusive_scan_v2_64_spv.h"
#include "h/tmp_single_radixsort_warp16_spv.h"
#include "h/tmp_single_radixsort_warp32_spv.h"
#include "h/tmp_single_radixsort_warp64_spv.h"
#include "h/tree_build_octree_spv.h"
#include "h/tree_build_radix_tree_spv.h"
#include "h/tree_edge_count_spv.h"
#include "h/tree_find_dups_spv.h"
#include "h/tree_merge_sort_spv.h"
#include "h/tree_morton_spv.h"
#include "h/tree_move_dups_spv.h"
#include "h/tree_naive_prefix_sum_spv.h"

// Helper macro to create shader entry with proper naming convention
#define SHADER_ENTRY(name)                                         \
  {                                                                \
    #name, {                                                       \
      builtin_apps_common_kiss_vk_shaders_spv_##name##_spv,        \
          builtin_apps_common_kiss_vk_shaders_spv_##name##_spv_len \
    }                                                              \
  }

// Map of shader names to their binary data and size
// Key: shader name
// Value: pair of (shader binary data pointer, shader binary size)
static const std::unordered_map<std::string, std::pair<const unsigned char*, size_t>> all_shaders =
    {
        SHADER_ENTRY(cifar_conv2d),
        SHADER_ENTRY(cifar_linear),
        SHADER_ENTRY(cifar_maxpool2d),
        SHADER_ENTRY(cifar_sparse_conv2d),
        SHADER_ENTRY(cifar_sparse_linear),
        SHADER_ENTRY(cifar_sparse_maxpool),
        SHADER_ENTRY(hello_vector_add),
        SHADER_ENTRY(hello_multiple_steps),
        SHADER_ENTRY(sort_safe_radix_sort),
        SHADER_ENTRY(tmp_add_base),
        SHADER_ENTRY(tmp_add_base_v2_16),
        SHADER_ENTRY(tmp_add_base_v2_32),
        SHADER_ENTRY(tmp_add_base_v2_64),
        SHADER_ENTRY(tmp_global_exclusive_scan),
        SHADER_ENTRY(tmp_global_exclusive_scan_v2_16),
        SHADER_ENTRY(tmp_global_exclusive_scan_v2_32),
        SHADER_ENTRY(tmp_global_exclusive_scan_v2_64),
        SHADER_ENTRY(tmp_local_inclusive_scan),
        SHADER_ENTRY(tmp_local_inclusive_scan_v2_16),
        SHADER_ENTRY(tmp_local_inclusive_scan_v2_32),
        SHADER_ENTRY(tmp_local_inclusive_scan_v2_64),
        SHADER_ENTRY(tmp_single_radixsort_warp16),
        SHADER_ENTRY(tmp_single_radixsort_warp32),
        SHADER_ENTRY(tmp_single_radixsort_warp64),
        SHADER_ENTRY(tree_build_octree),
        SHADER_ENTRY(tree_build_radix_tree),
        SHADER_ENTRY(tree_edge_count),
        SHADER_ENTRY(tree_find_dups),
        SHADER_ENTRY(tree_merge_sort),
        SHADER_ENTRY(tree_morton),
        SHADER_ENTRY(tree_move_dups),
        SHADER_ENTRY(tree_naive_prefix_sum),
};

#undef SHADER_ENTRY

}  // namespace shaders

}  // namespace kiss_vk