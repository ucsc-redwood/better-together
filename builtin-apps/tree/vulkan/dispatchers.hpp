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
  std::pmr::vector<uint32_t> tmp_u_sums;
  std::pmr::vector<uint32_t> tmp_u_prefix_sums;
};

// class Singleton {
//  public:
//   // Delete copy constructor and assignment operator to prevent copies
//   Singleton(const Singleton &) = delete;
//   Singleton &operator=(const Singleton &) = delete;

//   static Singleton &getInstance() {
//     static Singleton instance;
//     return instance;
//   }

//   kiss_vk::VulkanMemoryResource::memory_resource *get_mr() { return engine.get_mr(); }

//   void process_stage_1(VkAppData_Safe &appdata);
//   void process_stage_2(VkAppData_Safe &appdata);
//   void process_stage_3(VkAppData_Safe &appdata);
//   void process_stage_4(VkAppData_Safe &appdata);
//   void process_stage_5(VkAppData_Safe &appdata);
//   void process_stage_6(VkAppData_Safe &appdata);
//   void process_stage_7(VkAppData_Safe &appdata);

//   template <int Stage>
//     requires(Stage >= 1 && Stage <= 7)
//   void run_stage(VkAppData_Safe &appdata) {
//     if constexpr (Stage == 1) {
//       process_stage_1(appdata);
//     } else if constexpr (Stage == 2) {
//       process_stage_2(appdata);
//     } else if constexpr (Stage == 3) {
//       process_stage_3(appdata);
//     } else if constexpr (Stage == 4) {
//       process_stage_4(appdata);
//     } else if constexpr (Stage == 5) {
//       process_stage_5(appdata);
//     } else if constexpr (Stage == 6) {
//       process_stage_6(appdata);
//     } else if constexpr (Stage == 7) {
//       process_stage_7(appdata);
//     }
//   }

//  private:
//   Singleton();
//   ~Singleton() { spdlog::info("Singleton instance destroyed."); }

//   kiss_vk::Engine engine;
//   std::shared_ptr<kiss_vk::Sequence> seq;
//   std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;

//   // --------------------------------------------------------------------------
//   // Temporary storages
//   // --------------------------------------------------------------------------

//   // (n + 255) / 256;
//   std::pmr::vector<uint32_t> tmp_u_sums;
//   std::pmr::vector<uint32_t> tmp_u_prefix_sums;
// };

}  // namespace tree::vulkan
