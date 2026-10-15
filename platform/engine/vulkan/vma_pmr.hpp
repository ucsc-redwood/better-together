#pragma once

#include <spdlog/spdlog.h>

// Vulkan Memory Allocator
#include <vk_mem_alloc.h>

// Standard Library
#include <memory_resource>
#include <mutex>
#include <unordered_map>

#include "vk.hpp"

extern VmaAllocator g_vma_allocator;

namespace kiss_vk {

// Structure to keep track of the buffer allocation details
struct VulkanAllocationRecord {
  VkBuffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo allocInfo;
};

// Which memory types a VulkanMemoryResource allocates from (design B1/B2). Every placement
// is host-visible and persistently mapped: the pmr allocator must return a host pointer.
struct MemoryPlacement {
  const char* name = "default";
  // Today's placement: host-visible, host-cached where the device offers it (do_allocate).
  VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  VkMemoryPropertyFlags preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  uint32_t memory_type_bits = 0;  // 0: any type with `required`
};

// The weights class's placement on this device (design B1/B2). BT_VK_WEIGHTS_PLACEMENT
// (experiment knob) overrides it: `default` = today's placement; `device_local` =
// DEVICE_LOCAL|HOST_VISIBLE without HOST_CACHED (on an AMD APU the carve-out); `uncached` =
// HOST_VISIBLE without HOST_CACHED, and without DEVICE_LOCAL where the device has such a type
// (write-combined system memory). On a phone, where every type is DEVICE_LOCAL, both resolve
// to the coherent uncached type. Unset: `uncached` on the devices where the fc1/fc2
// measurement showed a gain (the Pixel 7a's Mali-G710), `default` elsewhere. A placement the
// device has no type for warns and keeps the default; any other value throws.
[[nodiscard]] MemoryPlacement weights_class_placement(vk::PhysicalDevice physical_device);

// ----------------------------------------------------------------------------
// VulkanMemoryResource
// ----------------------------------------------------------------------------

class VulkanMemoryResource : public std::pmr::memory_resource {
 public:
  explicit VulkanMemoryResource(vk::Device device, MemoryPlacement placement = {});

  ~VulkanMemoryResource() override;

  // Owns the VMA buffer allocations torn down in the dtor and is held behind a unique_ptr
  // (Engine::mr_ptr_). A std::pmr::memory_resource is also identified by ADDRESS
  // (do_is_equal), so it must never be copied or moved. Make both ill-formed.
  VulkanMemoryResource(const VulkanMemoryResource&) = delete;
  VulkanMemoryResource& operator=(const VulkanMemoryResource&) = delete;
  VulkanMemoryResource(VulkanMemoryResource&&) = delete;
  VulkanMemoryResource& operator=(VulkanMemoryResource&&) = delete;

  [[nodiscard]] vk::Device get_device() const { return device_; }
  [[nodiscard]] const MemoryPlacement& placement() const { return placement_; }

  // The buffer behind a pointer this resource returned, or a null handle.
  [[nodiscard]] vk::Buffer find_buffer(const void* p) const;
  // The allocation behind one of this resource's buffers, or null. Sequence uses it for
  // host<->device cache maintenance of exactly the buffers a submit binds (flush before
  // the submit, invalidate after its wait); both are no-ops on coherent memory.
  [[nodiscard]] VmaAllocation find_allocation(VkBuffer buffer) const;

 protected:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override;

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override;

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;

 private:
  vk::Device device_;
  MemoryPlacement placement_;

  mutable std::mutex mutex_;
  std::unordered_map<const void*, VulkanAllocationRecord> allocations_;
  std::unordered_map<VkBuffer, VmaAllocation> allocation_of_buffer_;
};

}  // namespace kiss_vk