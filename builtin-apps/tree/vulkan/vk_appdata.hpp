#pragma once

#include "../../common/kiss-vk/vma_pmr.hpp"
#include "../safe_tree_appdata.hpp"

namespace tree::vulkan {

struct VkAppData_Safe final : public tree::SafeAppData {
  // Multi-workgroup (device-wide) radix sort scratch sizing.
  // 256 bins; a fixed number of workgroups so the per-workgroup histogram
  // buffer has a deterministic size (RADIX_SORT_BINS * kRadixNumWorkgroups).
  static constexpr uint32_t kRadixBins = 256;
  static constexpr uint32_t kRadixNumWorkgroups = 256;

  explicit VkAppData_Safe(kiss_vk::VulkanMemoryResource::memory_resource* vk_mr)
      : SafeAppData(vk_mr),
        u_contributes(n_input, vk_mr),
        u_out_idx(n_input, vk_mr),
        u_sums(n_input, vk_mr),
        u_prefix_sums(n_input, vk_mr),
        u_sort_tmp(n_input, vk_mr),
        u_sort_histograms(kRadixBins * kRadixNumWorkgroups, vk_mr) {
    spdlog::trace("VkAppData_Safe constructor, address: {}", (void*)this);
  }

  ~VkAppData_Safe() { spdlog::trace("VkAppData_Safe destructor, address: {}", (void*)this); }

  // --------------------------------------------------------------------------
  // intergrated tmp storage
  // --------------------------------------------------------------------------

  // for remove duplicates
  std::pmr::vector<uint32_t> u_contributes;
  std::pmr::vector<uint32_t> u_out_idx;

  // for prefix sum
  std::pmr::vector<uint32_t> u_sums;
  std::pmr::vector<uint32_t> u_prefix_sums;

  // for multi-workgroup radix sort (stage 2): ping-pong scratch + histograms
  std::pmr::vector<uint32_t> u_sort_tmp;
  std::pmr::vector<uint32_t> u_sort_histograms;
};

}  // namespace tree::vulkan
