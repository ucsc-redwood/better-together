#pragma once

#include "../../common/kiss-vk/engine.hpp"
#include "../dense_appdata.hpp"

namespace cifar_dense::vulkan {

class Singleton {
 public:
  // Delete copy constructor and assignment operator to prevent copies
  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;

  static Singleton &getInstance() {
    static Singleton instance;
    return instance;
  }

  kiss_vk::VulkanMemoryResource::memory_resource *get_mr() { return engine.get_mr(); }

  void process_stage_1(cifar_dense::AppData &app_data);
  void process_stage_2(cifar_dense::AppData &app_data);
  void process_stage_3(cifar_dense::AppData &app_data);
  void process_stage_4(cifar_dense::AppData &app_data);
  void process_stage_5(cifar_dense::AppData &app_data);
  void process_stage_6(cifar_dense::AppData &app_data);
  void process_stage_7(cifar_dense::AppData &app_data);
  void process_stage_8(cifar_dense::AppData &app_data);
  void process_stage_9(cifar_dense::AppData &app_data);

  template <int Stage>
    requires(Stage >= 1 && Stage <= 9)
  void run_stage(cifar_dense::AppData &app_data) {
    if constexpr (Stage == 1) {
      process_stage_1(app_data);
    } else if constexpr (Stage == 2) {
      process_stage_2(app_data);
    } else if constexpr (Stage == 3) {
      process_stage_3(app_data);
    } else if constexpr (Stage == 4) {
      process_stage_4(app_data);
    } else if constexpr (Stage == 5) {
      process_stage_5(app_data);
    } else if constexpr (Stage == 6) {
      process_stage_6(app_data);
    } else if constexpr (Stage == 7) {
      process_stage_7(app_data);
    } else if constexpr (Stage == 8) {
      process_stage_8(app_data);
    } else if constexpr (Stage == 9) {
      process_stage_9(app_data);
    }
  }

 private:
  Singleton();
  ~Singleton() { spdlog::info("Singleton instance destroyed."); }

  kiss_vk::Engine engine;
  std::shared_ptr<kiss_vk::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;
};

}  // namespace cifar_dense::vulkan