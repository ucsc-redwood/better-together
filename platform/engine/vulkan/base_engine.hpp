#pragma once

#include <array>
#include <string>
#include <vector>

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

// What a compute pipeline may use on the selected device, fixed at device creation and
// handed to every Algorithm (design D5).
struct ComputeLimits {
  // Device-reported default subgroup size (VkPhysicalDeviceSubgroupProperties), passed to
  // subgroup-dependent shaders as spec constant 0. Never taken from devices/*.json.
  uint32_t subgroup_size = 0;
  uint32_t min_subgroup_size = 0;  // the range subgroup_size may be pinned to
  uint32_t max_subgroup_size = 0;
  // subgroupSizeControl is enabled and COMPUTE is in requiredSubgroupSizeStages, so
  // subgroup-dependent pipelines require subgroup_size instead of letting the driver pick.
  bool subgroup_size_pinned = false;
  // computeFullSubgroups is enabled, so subgroup-dependent pipelines require full subgroups
  // (design A5). Either way their X workgroup size must be a multiple of subgroup_size.
  bool full_subgroups = false;
  uint32_t max_compute_workgroup_subgroups = 0;  // 0 unless pinned
  std::array<uint32_t, 3> max_work_group_size{};
  uint32_t max_work_group_invocations = 0;
};

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
  [[nodiscard]] const ComputeLimits& compute_limits() const { return compute_limits_; }
  [[nodiscard]] uint32_t subgroup_size() const { return compute_limits_.subgroup_size; }
  // VK_KHR_push_descriptor's vkCmdPushDescriptorSetKHR when the extension is enabled, else
  // null: Algorithm::record_bind then uses the Sequence's transient pool (design D6).
  [[nodiscard]] PFN_vkCmdPushDescriptorSetKHR push_descriptor_fn() const {
    return push_descriptor_fn_;
  }
  // One (device tick, CLOCK_MONOTONIC ns) pair via VK_KHR/EXT_calibrated_timestamps; false
  // when the device lacks them (design D1).
  [[nodiscard]] bool calibrate(uint64_t& device_ticks, uint64_t& host_ns) const;
  // Turns calibration off (with a warning) if a queue timestamp does not fall between two
  // calibrated device ticks around it: the domains are then not correlated (design D1).
  void check_calibration();

 protected:
  void initialize_dynamic_loader();
  void request_validation_layer();

  void create_instance();
  void create_physical_device(vk::PhysicalDeviceType type = vk::PhysicalDeviceType::eIntegratedGpu);
  void create_device();

  void initialize_vma_allocator() const;

  // Pins subgroup-dependent pipelines to `size` instead of the device default. Throws
  // unless pinning is available and `size` is a power of two in [min, max].
  void override_subgroup_size(uint32_t size);

  [[nodiscard]] uint32_t queue_family_timestamp_bits() const;

  // Handles
  vk::Instance instance_;
  vk::PhysicalDevice physical_device_;
  vk::Device device_;
  vk::Queue compute_queue_;

 private:
  // Releases whatever has been created so far (allocator, device, instance). Shared by the
  // destructor and the constructor's failure path, which the destructor never sees.
  void destroy() noexcept;

  // g_vma_allocator is a process global: free it only if this engine created it.
  bool owns_vma_allocator_ = false;
  uint32_t compute_queue_family_index_;
  ComputeLimits compute_limits_;
  PFN_vkCmdPushDescriptorSetKHR push_descriptor_fn_ = nullptr;
  PFN_vkGetCalibratedTimestampsKHR calibrated_timestamps_fn_ = nullptr;
  std::vector<const char*> enabled_layers_;

  DynamicLoader dl_;
  DispatchLoaderDynamic dldi_;
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_;
};

// Probe (once, cached) whether this machine has an integrated GPU the engine can select.
// Used by the differential tests' Runner::Available() so they GTEST_SKIP (like the CUDA
// suites) on a discrete-GPU-only box instead of crashing. noexcept: any failure -> false.
[[nodiscard]] bool has_integrated_gpu() noexcept;

// Names of the engine's required device features that the supported sets lack (empty:
// none). Device creation throws with these names instead of running with a feature disabled.
[[nodiscard]] std::vector<std::string> missing_required_features(
    const vk::PhysicalDeviceVulkan12Features& supported_12,
    const vk::PhysicalDeviceVulkan13Features& supported_13);

}  // namespace kiss_vk
