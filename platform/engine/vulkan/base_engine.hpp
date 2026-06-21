#pragma once

#include "vk.hpp"

// Vulkan-Hpp version-safe DynamicLoader
#if defined(VK_HEADER_VERSION) && VK_HEADER_VERSION >= 300
// Vulkan-Hpp >= 1.3.243 → DynamicLoader moved to detail
using DynamicLoader = vk::detail::DynamicLoader;
using DispatchLoaderDynamic = vk::detail::DispatchLoaderDynamic;
#else
// Vulkan-Hpp <= 1.3.242 → DynamicLoader is public
using DynamicLoader = vk::DynamicLoader;
using DispatchLoaderDynamic = vk::DispatchLoaderDynamic;
#endif

namespace kiss_vk {

class BaseEngine {
 public:
  explicit BaseEngine(bool enable_validation_layer = true);

  ~BaseEngine();

  // Owns vk::Instance/Device + the VMA allocator, all torn down by hand in ~BaseEngine
  // (and a single shared g_vma_allocator). Copying would double-free those handles; there
  // is no meaningful move (the derived Engine is held by value in each dispatcher and
  // never relocated). Make both ill-formed so a stray by-value use is a compile error,
  // not a runtime VK_ERROR_DEVICE_LOST.
  BaseEngine(const BaseEngine&) = delete;
  BaseEngine& operator=(const BaseEngine&) = delete;
  BaseEngine(BaseEngine&&) = delete;
  BaseEngine& operator=(BaseEngine&&) = delete;

  [[nodiscard]] vk::Device& get_device() { return device_; }
  [[nodiscard]] vk::PhysicalDevice& get_physical_device() { return physical_device_; }
  [[nodiscard]] vk::Queue& get_compute_queue() { return compute_queue_; }
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
  std::vector<const char*> enabled_layers_;

  DynamicLoader dl_;
  DispatchLoaderDynamic dldi_;
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_;
};

// Probe (once, cached) whether this machine has an integrated GPU the engine can select.
// Used by the differential tests' Runner::Available() so they GTEST_SKIP (like the CUDA
// suites) on a discrete-GPU-only box instead of crashing. noexcept: any failure -> false.
[[nodiscard]] bool has_integrated_gpu() noexcept;

}  // namespace kiss_vk
