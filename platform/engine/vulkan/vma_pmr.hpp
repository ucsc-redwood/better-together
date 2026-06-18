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

// ----------------------------------------------------------------------------
// VulkanMemoryResource
// ----------------------------------------------------------------------------

class VulkanMemoryResource : public std::pmr::memory_resource {
 public:
  // We use the requested defaults for usage flags and allocation flags.
  explicit VulkanMemoryResource(
      vk::Device device,
      vk::BufferUsageFlags buffer_usage = vk::BufferUsageFlagBits::eStorageBuffer,
      VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlags flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                       VMA_ALLOCATION_CREATE_MAPPED_BIT);

  ~VulkanMemoryResource() override;

  [[nodiscard]] vk::Device get_device() const { return device_; }

  [[nodiscard]] vk::Buffer get_buffer_from_pointer(void *p);

  // Host<->device cache maintenance for NON-coherent (HOST_CACHED) memory; no-ops
  // on coherent memory. flush_all() makes host writes visible to the GPU (call
  // before submitting GPU work); invalidate_all() makes GPU writes visible to the
  // host (call after the GPU fence is signaled). See do_allocate's heap choice.
  void flush_all();
  void invalidate_all();

  //   [[nodiscard]] vk::DescriptorBufferInfo make_descriptor_buffer_info(vk::Buffer buffer) const;

 protected:
  void *do_allocate(std::size_t bytes, std::size_t alignment) override;

  void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override;

  bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override;

 private:
  vk::Device device_;
  vk::BufferUsageFlags bufferUsage_;
  VmaMemoryUsage memoryUsage_;
  VmaAllocationCreateFlags allocationFlags_;

  mutable std::mutex mutex_;
  std::unordered_map<void *, VulkanAllocationRecord> allocations_;
};

}  // namespace kiss_vk