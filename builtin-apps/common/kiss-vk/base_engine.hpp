#pragma once

#include "vk.hpp"

namespace kiss_vk {

class BaseEngine {
 public:
  explicit BaseEngine(bool enable_validation_layer = true);

  ~BaseEngine();

  [[nodiscard]] vk::Device &get_device() { return device_; }
  [[nodiscard]] vk::Queue &get_compute_queue() { return compute_queue_; }
  [[nodiscard]] uint32_t get_compute_queue_family_index() const {
    return compute_queue_family_index_;
  }

 protected:
  void initialize_dynamic_loader();
  void request_validation_layer();

  void create_instance();
  void create_physical_device(vk::PhysicalDeviceType type = vk::PhysicalDeviceType::eIntegratedGpu);
  void create_device(vk::QueueFlags queue_flags = vk::QueueFlagBits::eCompute);

  void initialize_vma_allocator() const;

  // Handles
  vk::Instance instance_;
  vk::PhysicalDevice physical_device_;
  vk::Device device_;
  vk::Queue compute_queue_;

 private:
  uint32_t compute_queue_family_index_;
  std::vector<const char *> enabled_layers_;

  vk::DynamicLoader dl_;
  vk::DispatchLoaderDynamic dldi_;
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_;
};

}  // namespace kiss_vk
