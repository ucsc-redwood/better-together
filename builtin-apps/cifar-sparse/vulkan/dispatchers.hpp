#pragma once

#include "../../common/kiss-vk/engine.hpp"
#include "../sparse_appdata.hpp"

namespace cifar_sparse::vulkan {

class VulkanDispatcher final {
 public:
  explicit VulkanDispatcher();

  // disallow copy/move constructor
  VulkanDispatcher(const VulkanDispatcher&) = delete;
  VulkanDispatcher& operator=(const VulkanDispatcher&) = delete;
  VulkanDispatcher(VulkanDispatcher&&) = delete;
  VulkanDispatcher& operator=(VulkanDispatcher&&) = delete;

  kiss_vk::VulkanMemoryResource::memory_resource* get_mr() { return engine.get_mr(); }

  void run_stage_1(cifar_sparse::AppData& appdata);
  void run_stage_2(cifar_sparse::AppData& appdata);
  void run_stage_3(cifar_sparse::AppData& appdata);
  void run_stage_4(cifar_sparse::AppData& appdata);
  void run_stage_5(cifar_sparse::AppData& appdata);
  void run_stage_6(cifar_sparse::AppData& appdata);
  void run_stage_7(cifar_sparse::AppData& appdata);
  void run_stage_8(cifar_sparse::AppData& appdata);
  void run_stage_9(cifar_sparse::AppData& appdata);

  using StageFn = void (VulkanDispatcher::*)(cifar_sparse::AppData&);

  static constexpr std::array<StageFn, 9> stage_functions = {
      &VulkanDispatcher::run_stage_1,
      &VulkanDispatcher::run_stage_2,
      &VulkanDispatcher::run_stage_3,
      &VulkanDispatcher::run_stage_4,
      &VulkanDispatcher::run_stage_5,
      &VulkanDispatcher::run_stage_6,
      &VulkanDispatcher::run_stage_7,
      &VulkanDispatcher::run_stage_8,
      &VulkanDispatcher::run_stage_9,
  };

  void dispatch_multi_stage(cifar_sparse::AppData& data,
                            const int start_stage,
                            const int end_stage) {
    if (start_stage < 1 || end_stage > 9) throw std::out_of_range("Invalid stage");

    for (int stage = start_stage; stage <= end_stage; stage++) {
      (this->*stage_functions[stage - 1])(data);
    }
  }

 private:
  kiss_vk::Engine engine;
  std::shared_ptr<kiss_vk::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;
};

namespace v2 {

class VulkanDispatcher final {
 public:
  explicit VulkanDispatcher();

  // disallow copy/move constructor
  VulkanDispatcher(const VulkanDispatcher&) = delete;
  VulkanDispatcher& operator=(const VulkanDispatcher&) = delete;
  VulkanDispatcher(VulkanDispatcher&&) = delete;
  VulkanDispatcher& operator=(VulkanDispatcher&&) = delete;

  kiss_vk::VulkanMemoryResource::memory_resource* get_mr() { return engine.get_mr(); }

  void run_stage_1(cifar_sparse::v2::AppData& appdata);
  void run_stage_2(cifar_sparse::v2::AppData& appdata);
  void run_stage_3(cifar_sparse::v2::AppData& appdata);
  void run_stage_4(cifar_sparse::v2::AppData& appdata);
  void run_stage_5(cifar_sparse::v2::AppData& appdata);
  void run_stage_6(cifar_sparse::v2::AppData& appdata);
  void run_stage_7(cifar_sparse::v2::AppData& appdata);
  void run_stage_8(cifar_sparse::v2::AppData& appdata);
  void run_stage_9(cifar_sparse::v2::AppData& appdata);

  using StageFn = void (VulkanDispatcher::*)(cifar_sparse::v2::AppData&);

  static constexpr std::array<StageFn, 9> stage_functions = {
      &VulkanDispatcher::run_stage_1,
      &VulkanDispatcher::run_stage_2,
      &VulkanDispatcher::run_stage_3,
      &VulkanDispatcher::run_stage_4,
      &VulkanDispatcher::run_stage_5,
      &VulkanDispatcher::run_stage_6,
      &VulkanDispatcher::run_stage_7,
      &VulkanDispatcher::run_stage_8,
      &VulkanDispatcher::run_stage_9,
  };

  void dispatch_multi_stage(cifar_sparse::v2::AppData& data,
                            const int start_stage,
                            const int end_stage) {
    if (start_stage < 1 || end_stage > 9) throw std::out_of_range("Invalid stage");

    for (int stage = start_stage; stage <= end_stage; stage++) {
      (this->*stage_functions[stage - 1])(data);
    }
  }

 private:
  kiss_vk::Engine engine;
  std::shared_ptr<kiss_vk::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;
};

}  // namespace v2

}  // namespace cifar_sparse::vulkan
