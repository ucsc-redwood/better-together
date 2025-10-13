#pragma once

#include "../../common/kiss-vk/engine.hpp"
#include "../appdata.hpp"

namespace octree::vulkan {

class VulkanDispatcher final {
 public:
  explicit VulkanDispatcher();

  VulkanDispatcher(const VulkanDispatcher&) = delete;
  VulkanDispatcher& operator=(const VulkanDispatcher&) = delete;
  VulkanDispatcher(VulkanDispatcher&&) = delete;
  VulkanDispatcher& operator=(VulkanDispatcher&&) = delete;

  kiss_vk::VulkanMemoryResource::memory_resource* get_mr() const { return engine.get_mr(); }

  void run_stage_1(AppData& app);
  void run_stage_2(AppData& app);
  void run_stage_3(AppData& app);
  void run_stage_4(AppData& app);
  void run_stage_5(AppData& app);
  void run_stage_6(AppData& app);
  void run_stage_7(AppData& app);

  using StageFn = void (VulkanDispatcher::*)(AppData&);

  static constexpr std::array<StageFn, 7> stage_functions = {
      &VulkanDispatcher::run_stage_1,
      &VulkanDispatcher::run_stage_2,
      &VulkanDispatcher::run_stage_3,
      &VulkanDispatcher::run_stage_4,
      &VulkanDispatcher::run_stage_5,
      &VulkanDispatcher::run_stage_6,
      &VulkanDispatcher::run_stage_7,
  };

  void dispatch_multi_stage(AppData& data, const int start_stage, const int end_stage) {
    if (start_stage < 1 || end_stage > 7) throw std::out_of_range("Invalid stage");

    for (int stage = start_stage; stage <= end_stage; stage++) {
      (this->*stage_functions[stage - 1])(data);
    }
  }

 private:
  kiss_vk::Engine engine;
  std::shared_ptr<kiss_vk::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;
};

}  // namespace octree::vulkan
