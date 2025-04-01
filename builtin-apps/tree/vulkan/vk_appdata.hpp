#pragma once

#include "../../common/kiss-vk/vma_pmr.hpp"
#include "../safe_tree_appdata.hpp"
// #include "../tree_appdata.hpp"

namespace tree::vulkan {

// struct VkAppData final : public tree::AppData {
//   explicit VkAppData(kiss_vk::VulkanMemoryResource::memory_resource* vk_mr,
//                      const size_t n_input = kDefaultInputSize)
//       : AppData(vk_mr, n_input),
//         u_contributes(n_input, vk_mr),
//         u_out_idx(n_input, vk_mr),
//         u_sums(n_input, vk_mr),
//         u_prefix_sums(n_input, vk_mr) {}

//   // --------------------------------------------------------------------------
//   // intergrated tmp storage
//   // --------------------------------------------------------------------------

//   // for remove duplicates
//   std::pmr::vector<uint32_t> u_contributes;
//   std::pmr::vector<uint32_t> u_out_idx;

//   // for prefix sum
//   std::pmr::vector<uint32_t> u_sums;
//   std::pmr::vector<uint32_t> u_prefix_sums;
// };

struct VkAppData_Safe final : public tree::SafeAppData {
  explicit VkAppData_Safe(kiss_vk::VulkanMemoryResource::memory_resource* vk_mr)
      : SafeAppData(vk_mr),
        u_contributes(n_input, vk_mr),
        u_out_idx(n_input, vk_mr),
        u_sums(n_input, vk_mr),
        u_prefix_sums(n_input, vk_mr) {}

  // --------------------------------------------------------------------------
  // intergrated tmp storage
  // --------------------------------------------------------------------------

  // for remove duplicates
  std::pmr::vector<uint32_t> u_contributes;
  std::pmr::vector<uint32_t> u_out_idx;

  // for prefix sum
  std::pmr::vector<uint32_t> u_sums;
  std::pmr::vector<uint32_t> u_prefix_sums;
};

}  // namespace tree::vulkan
