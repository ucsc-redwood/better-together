#pragma once

#include "apps/tree/safe_tree_appdata.hpp"
#include "apps/tree/tree_appdata.hpp"
#include "platform/engine/vulkan/vma_pmr.hpp"

namespace tree::vulkan {

struct VkAppData_Safe final : public tree::SafeAppData {
  // Multi-workgroup (device-wide) radix sort scratch sizing.
  // 256 bins; a fixed number of workgroups so the per-workgroup histogram
  // buffer has a deterministic size (RADIX_SORT_BINS * kRadixNumWorkgroups).
  static constexpr uint32_t kRadixBins = 256;
  static constexpr uint32_t kRadixNumWorkgroups = 256;

  // Device-wide scan tile size (must match ELEMENTS_PER_WG in the
  // tree_scan_*.comp shaders). One block per tile; the per-block totals are
  // scanned by a single workgroup in pass 2, so #blocks must stay <= the tile
  // size (640x480 -> 150 blocks, FHD 2M -> ~977; well within 2048).
  static constexpr uint32_t kScanElementsPerWg = 256 * 8;

  explicit VkAppData_Safe(kiss_vk::VulkanMemoryResource::memory_resource* vk_mr)
      : SafeAppData(vk_mr),
        u_contributes(n_input, vk_mr),
        u_out_idx(n_input, vk_mr),
        u_sums(n_input, vk_mr),
        u_prefix_sums(n_input, vk_mr),
        u_sort_tmp(n_input, vk_mr),
        u_sort_histograms(kRadixBins * kRadixNumWorkgroups, vk_mr),
        u_scan_block_sums((n_input + kScanElementsPerWg - 1) / kScanElementsPerWg + 1, vk_mr) {
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

  // for the device-wide scan (stages 3 & 6): per-block totals / offsets
  std::pmr::vector<uint32_t> u_scan_block_sums;
};

// ----------------------------------------------------------------------------
// Genuinely-chained variant (no golden/_out split, see
// apps/tree/tree_appdata.hpp) -- mirrors what VkAppData_Safe adds onto
// tree::SafeAppData, but onto plain tree::AppData instead. Every pipeline
// field is single-buffer: stage N+1 reads what stage N's OWN dispatch on this
// instance actually wrote, not a construction-time golden. Unlike
// VkAppData_Safe (whose n_unique/n_brt_nodes/n_octree_nodes are const, known
// upfront from the golden), this variant needs an explicit host readback of
// those counts after stage 3 and after stage 6 -- see
// VulkanDispatcher::dispatch_multi_stage's VkAppData overload.
// ----------------------------------------------------------------------------
// Destructor only logs; the pmr vector members below self-manage, so copy/move are safe
// to leave implicitly deleted (base class already disables them).
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
struct VkAppData final : public tree::AppData {
  static constexpr uint32_t kRadixBins = 256;
  static constexpr uint32_t kRadixNumWorkgroups = 256;
  static constexpr uint32_t kScanElementsPerWg = 256 * 8;

  // Every pmr vector member below IS initialized, in the initializer list; clang-tidy
  // misflags this constructor shape.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit VkAppData(kiss_vk::VulkanMemoryResource::memory_resource* vk_mr,
                     const size_t n_input = tree::kDefaultInputSize)
      : tree::AppData(vk_mr, n_input),
        u_contributes(n_input, vk_mr),
        u_out_idx(n_input, vk_mr),
        u_sums(n_input, vk_mr),
        u_prefix_sums(n_input, vk_mr),
        u_sort_tmp(n_input, vk_mr),
        u_sort_histograms(kRadixBins * kRadixNumWorkgroups, vk_mr),
        u_scan_block_sums((n_input + kScanElementsPerWg - 1) / kScanElementsPerWg + 1, vk_mr) {
    spdlog::trace("VkAppData constructor, address: {}", (void*)this);
  }

  ~VkAppData() { spdlog::trace("VkAppData destructor, address: {}", (void*)this); }

  std::pmr::vector<uint32_t> u_contributes;
  std::pmr::vector<uint32_t> u_out_idx;
  std::pmr::vector<uint32_t> u_sums;
  std::pmr::vector<uint32_t> u_prefix_sums;
  std::pmr::vector<uint32_t> u_sort_tmp;
  std::pmr::vector<uint32_t> u_sort_histograms;
  std::pmr::vector<uint32_t> u_scan_block_sums;
};

}  // namespace tree::vulkan
