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

  [[nodiscard]] vk::Buffer get_buffer_from_pointer(void* p);

  // Host<->device cache maintenance for NON-coherent (HOST_CACHED) memory; no-ops
  // on coherent memory. flush_all() makes host writes visible to the GPU (call
  // before submitting GPU work); invalidate_all() makes GPU writes visible to the
  // host (call after the GPU fence is signaled). See do_allocate's heap choice.
  void flush_all();
  void invalidate_all();

  // Scoped cache maintenance: flush/invalidate ONLY the buffers touched (bound) since
  // the last clear_touched(), instead of the whole allocation map. The allocation map
  // holds every pooled AppData's buffers (the memory resource is shared across the
  // pool), so flush_all() does pool_size x more cache maintenance than a single task
  // needs -- a real cost on non-coherent Mali. get_buffer_from_pointer() records each
  // bound buffer; the dispatch records a chunk's stages (binding exactly that task's
  // buffers) between clear_touched() and submit, so the touched set == the buffers this
  // chunk actually reads/writes. Conservative + correct: it is the union of inputs and
  // outputs, so flushing it before submit and invalidating it after covers both.
  void flush_touched();
  void invalidate_touched();
  void clear_touched();

  //   [[nodiscard]] vk::DescriptorBufferInfo make_descriptor_buffer_info(vk::Buffer buffer) const;

 protected:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override;

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override;

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;

 private:
  vk::Device device_;
  vk::BufferUsageFlags bufferUsage_;
  VmaMemoryUsage memoryUsage_;
  VmaAllocationCreateFlags allocationFlags_;

  mutable std::mutex mutex_;
  std::unordered_map<void*, VulkanAllocationRecord> allocations_;

  // Buffers bound since the last clear_touched() (for scoped flush/invalidate). Only
  // mutated by the single GPU-chunk worker thread during command-buffer recording.
  std::vector<VmaAllocation> touched_;
};

}  // namespace kiss_vk