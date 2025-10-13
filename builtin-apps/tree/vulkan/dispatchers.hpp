#pragma once

#include "../../common/kiss-vk/engine.hpp"
#include "vk_appdata.hpp"

namespace tree::vulkan {

class VulkanDispatcher final {
 public:
  explicit VulkanDispatcher();

  // disallow copy/move constructor
  VulkanDispatcher(const VulkanDispatcher&) = delete;
  VulkanDispatcher& operator=(const VulkanDispatcher&) = delete;
  VulkanDispatcher(VulkanDispatcher&&) = delete;
  VulkanDispatcher& operator=(VulkanDispatcher&&) = delete;

  kiss_vk::VulkanMemoryResource::memory_resource* get_mr() { return engine.get_mr(); }

  void run_stage_1(VkAppData_Safe& appdata);
  void run_stage_2(VkAppData_Safe& appdata);
  void run_stage_3(VkAppData_Safe& appdata);
  void run_stage_4(VkAppData_Safe& appdata);
  void run_stage_5(VkAppData_Safe& appdata);
  void run_stage_6(VkAppData_Safe& appdata);
  void run_stage_7(VkAppData_Safe& appdata);

  using StageFn = void (VulkanDispatcher::*)(VkAppData_Safe&);

  static constexpr std::array<StageFn, 7> stage_functions = {
      &VulkanDispatcher::run_stage_1,
      &VulkanDispatcher::run_stage_2,
      &VulkanDispatcher::run_stage_3,
      &VulkanDispatcher::run_stage_4,
      &VulkanDispatcher::run_stage_5,
      &VulkanDispatcher::run_stage_6,
      &VulkanDispatcher::run_stage_7,
  };

  void dispatch_stage(VkAppData_Safe& data, const int stage) {
    assert(stage >= 1 && stage <= 7);

    (this->*stage_functions[stage - 1])(data);
  }

  void dispatch_multi_stage(VkAppData_Safe& data, const int start_stage, const int end_stage) {
    if (start_stage < 1 || end_stage > 7) throw std::out_of_range("Invalid stage");

    for (int stage = start_stage; stage <= end_stage; stage++) {
      (this->*stage_functions[stage - 1])(data);
    }
  }

 private:
  kiss_vk::Engine engine;
  std::shared_ptr<kiss_vk::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;

  // --------------------------------------------------------------------------
  // Temporary storages
  // --------------------------------------------------------------------------

  // (n + 255) / 256;
  // std::pmr::vector<uint32_t> tmp_u_sums;
  // std::pmr::vector<uint32_t> tmp_u_prefix_sums;
};

}  // namespace tree::vulkan
