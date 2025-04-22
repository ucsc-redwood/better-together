#pragma once

#include "../../common/kiss-vk/vma_pmr.hpp"
#include "../safe_tree_appdata.hpp"

namespace tree::vulkan {

struct VkAppData_Safe final : public tree::SafeAppData {
  explicit VkAppData_Safe(kiss_vk::VulkanMemoryResource::memory_resource* vk_mr)
      : SafeAppData(vk_mr),
        histogram_s2(n_input, vk_mr),
        u_contributes(n_input, vk_mr),
        u_out_idx(n_input, vk_mr),
        u_sums(n_input, vk_mr),
        u_prefix_sums(n_input, vk_mr) {
    spdlog::trace("VkAppData_Safe constructor, address: {}", (void*)this);
  }

  ~VkAppData_Safe() { spdlog::trace("VkAppData_Safe destructor, address: {}", (void*)this); }

  // --------------------------------------------------------------------------
  // intergrated tmp storage
  // --------------------------------------------------------------------------

  // histogram
  std::pmr::vector<uint32_t> histogram_s2;

  // for remove duplicates
  std::pmr::vector<uint32_t> u_contributes;
  std::pmr::vector<uint32_t> u_out_idx;

  // for prefix sum
  std::pmr::vector<uint32_t> u_sums;
  std::pmr::vector<uint32_t> u_prefix_sums;
};

}  // namespace tree::vulkan
